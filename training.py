import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.nn.utils.clip_grad import clip_grad_norm_
from utils import metrics_dict


def train(self, epoch_idx):
    self.model.train()
    sum_loss = 0.0
    sum_bpr_loss, sum_cl_loss = 0.0, 0.0
    sum_diff_loss = 0.0
    sum_causal_loss = 0.0

    step = 0.0
    for batch_idx, interactions in enumerate(self.train_data):
        self.optimizer.zero_grad()
        loss, bpr_loss, cl_loss, diff_loss, causal_loss = self.model.loss(interactions[0],
                                                                          interactions[1])
        if torch.isnan(loss):
            self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
            return loss, torch.tensor(0.0)

        loss.backward()
        self.optimizer.step()

        step += 1.0
        sum_loss += loss
        sum_bpr_loss += bpr_loss
        sum_cl_loss += cl_loss
        sum_diff_loss += diff_loss
        sum_causal_loss += causal_loss
    mean_loss = sum_loss / step
    mean_bpr_loss = sum_bpr_loss / step
    mean_cl_loss = sum_cl_loss / step
    mean_diff_loss = sum_diff_loss / step
    mean_causal_loss = sum_causal_loss / step

    if self.writer is not None:
        self.writer.add_scalar('loss/train', mean_loss, epoch_idx)
        self.writer.add_scalar('loss/bpr_loss', mean_bpr_loss, epoch_idx)
        self.writer.add_scalar('loss/cl_loss', mean_cl_loss, epoch_idx)
        self.writer.add_scalar('loss/diff_loss', mean_diff_loss, epoch_idx)
        self.writer.add_scalar('loss/causal_loss', mean_causal_loss, epoch_idx)

    return [mean_loss, mean_bpr_loss, mean_cl_loss, mean_diff_loss, mean_causal_loss]


def evaluate_model(self, epoch, eval_data, t_or_v):
    self.model.eval()
    with torch.no_grad():
        batch_matrix_list = []
        for batch_idx, batched_data in enumerate(eval_data):
            scores = self.model.full_sort_predict(batched_data)
            masked_items = batched_data[1]
            scores[masked_items[0], masked_items[1] - self.model.n_users] = -1e10
            _, top_k_index = torch.topk(scores, max(self.topk), dim=-1)
            batch_matrix_list.append(top_k_index)

    pos_items = eval_data.eval_items_per_u
    pos_len_list = eval_data.eval_len_list
    top_k_index = torch.cat(batch_matrix_list, dim=0).cpu().numpy()
    assert len(pos_len_list) == len(top_k_index)
    bool_rec_matrix = []

    for m, n in zip(pos_items, top_k_index):
        bool_rec_matrix.append([True if i in m else False for i in n])
    bool_rec_matrix = np.asarray(bool_rec_matrix)

    metric_dict = {}
    result_list = cal_metrics(self.metrics, pos_len_list, bool_rec_matrix)
    list_key = []
    for metric, value in zip(self.metrics, result_list):
        for k in self.topk:
            key = '{}@{}'.format(metric, k)
            list_key.append(key) if k == self.topk[-1] else None
            metric_dict[key] = round(value[k - 1], 4)
    valid_score = metric_dict[self.valid_metric] if self.valid_metric else metric_dict['NDCG@20']
    if self.writer is not None:
        for idx in list_key:
            self.writer.add_scalar(t_or_v + "_" + idx, metric_dict[idx], epoch)
        self.writer.add_histogram(t_or_v + '_user_visual_distribution', self.model.user_v_prefer, epoch)
        self.writer.add_histogram(t_or_v + '_user_textual_distribution', self.model.user_t_prefer, epoch)
        self.writer.add_embedding(self.model.user_id_embedding.weight, global_step=epoch,
                                  tag=t_or_v + "user_id_embedding")
        self.writer.add_embedding(self.model.item_id_embedding.weight, global_step=epoch,
                                  tag=t_or_v + "item_id_embedding")

    return valid_score, metric_dict


def cal_metrics(topk_metrics, pos_len_list, topk_index):
    result_list = []
    for metric in topk_metrics:
        metric_fuc = metrics_dict[metric]
        result = metric_fuc(topk_index, pos_len_list)
        result_list.append(result)
    return np.stack(result_list, axis=0)