import os
import torch
import argparse
import multiprocessing
from utils import init_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run TCDRec.")
    parser.add_argument('--seed', type=int, default=2026, help='Seed init.')
    parser.add_argument('--model_name', default='TCDRec', help='Model name.')
    parser.add_argument('--use_gpu', type=bool, default=True, help='enable CUDA training.')
    parser.add_argument('--gpu_id', type=int, default=0, help='The model of the device running the program')
    parser.add_argument('--dataset', nargs='?', default='baby',
                        help='Choose a dataset from {baby, sports, clothing}')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size.')
    parser.add_argument('--eval_batch_size', type=int, default=8192, help='The data size of batch evaluation')
    parser.add_argument('--metrics', type=list, default=["Precision", "Recall", "NDCG"],
                        help='Choose some from {"Precision", "Recall", "NDCG", "MAP"}')
    parser.add_argument('--topk', type=list, default=[10, 20], help='Metrics scale')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Latent dimension 64.')
    parser.add_argument('--num_epoch', type=int, default=2000, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=0, help='Workers number.')
    parser.add_argument('--stopping_step', type=int, default=20, help='early stopping strategy.')
    parser.add_argument('--valid_metric', type=str, default="Recall@20", help='valid metric')
    parser.add_argument('--with_tensorboard', action='store_true', default=False, help='with tensorboard analysis ')

    parser.add_argument('--l_r', type=float, default=5e-5, help='Learning rate.')
    parser.add_argument('--learning_rate_scheduler', type=list, default=[1.0, 50], help='learning rate scheduler.')
    parser.add_argument('--reg_weight', type=float, default=5e-4, help='regularization weight.')
    parser.add_argument('--num_layer', type=int, default=4, help='Layer number.')
    parser.add_argument('--s_drop', type=float, default=0.4, help='self_attention_dropout.')
    parser.add_argument('--m_drop', type=float, default=0.6, help='mutual_attention_dropout.')
    parser.add_argument('--cl_tmp', type=float, default=0.6, help='Contrast learning temperature coefficient')
    parser.add_argument('--cl_loss_weight', type=float, default=5e-5, help='contrast loss weight.')
    parser.add_argument('--user_knn_k', type=int, default=40,
                        help='Select the 10 users most similar to the target users to build the users graph')
    parser.add_argument('--item_knn_k', type=int, default=10,
                        help='Select the 10 items most similar to the target item to build the item graph')

    parser.add_argument('--i_mm_image_weight', type=float, default=0,
                        help='The proportion of visual feat in item graph.')
    parser.add_argument('--u_mm_image_weight', type=float, default=0.2,
                        help='The proportion of visual feat in user graph.')
    parser.add_argument('--n_ii_layers', type=int, default=1,
                        help='Number of layers of item feature propagation in the item graph')
    parser.add_argument('--n_uu_layers', type=int, default=1,
                        help='Number of layers of user feature propagation in the user graph')
    parser.add_argument('--user_aggr_mode', type=str, default='softmax',
                        help='Choose a modedataset from {softmax, mean}')

    parser.add_argument('--uu_co_weight', type=float, default=0.4,
                        help='the proportion of user co-occurrence graphs to user homographs')
    parser.add_argument('--ii_co_weight', type=float, default=0.2,
                        help='the proportion of item co-occurrence graphs to user homographs')

    parser.add_argument('--timesteps', type=int, default=10, help='Diffusion timesteps.')
    parser.add_argument('--beta_start', type=float, default=1e-4, help='Beta schedule start.')
    parser.add_argument('--beta_end', type=float, default=0.02, help='Beta schedule end.')
    parser.add_argument('--beta_sche', type=str, default='linear',
                        help='Beta schedule from {linear, cosine, exp, sqrt}.')
    parser.add_argument('--w', type=float, default=0.3, help='Convex mix weight for diffusion refinement.')
    parser.add_argument('--neighbors', type=int, default=3, help='Top-k neighbors for retrieval condition.')
    parser.add_argument('--diff_weight', type=float, default=0.2, help='Diffusion reconstruction loss weight.')
    parser.add_argument('--causal_weight', type=float, default=5e-5, help='Causal loss weight.')

    return parser.parse_args()


class Config(object):
    def __init__(self, args):
        self.model_name = args.model_name
        self.dataset = args.dataset
        self.learning_rate = args.l_r
        self.learning_rate_scheduler = args.learning_rate_scheduler
        self.embedding_dim = args.embedding_dim
        self.num_epoch = args.num_epoch
        self.reg_weight = args.reg_weight
        self.use_gpu = args.use_gpu
        self.gpu_id = args.gpu_id
        self.seed = args.seed

        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.topk = args.topk
        self.valid_metric = args.valid_metric
        self.metrics = args.metrics
        self.stopping_step = args.stopping_step
        self.n_layers = args.num_layer

        self.s_drop = args.s_drop
        self.m_drop = args.m_drop
        self.cl_tmp = args.cl_tmp
        self.item_knn_k = args.item_knn_k
        self.user_knn_k = args.user_knn_k
        self.num_user_co = args.user_knn_k
        self.num_item_co = args.item_knn_k
        self.n_ii_layers = args.n_ii_layers
        self.n_uu_layers = args.n_uu_layers
        self.writer = args.with_tensorboard
        self.uu_co_weight = args.uu_co_weight
        self.ii_co_weight = args.ii_co_weight
        self.cl_loss_weight = args.cl_loss_weight
        self.user_aggr_mode = args.user_aggr_mode
        self.i_mm_image_weight = args.i_mm_image_weight
        self.u_mm_image_weight = args.u_mm_image_weight

        self.timesteps = args.timesteps
        self.beta_start = args.beta_start
        self.beta_end = args.beta_end
        self.beta_sche = args.beta_sche
        self.w = args.w
        self.neighbors = args.neighbors
        self.diff_weight = args.diff_weight
        self.causal_weight = args.causal_weight

        self._init_device(args)
        init_seed(self.seed)

    def _init_device(self, args):
        if self.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.use_gpu else "cpu")

        max_cpu_count = multiprocessing.cpu_count()
        self.num_workers = max_cpu_count // 2 if max_cpu_count // 2 < args.num_workers else args.num_workers

    def __str__(self):
        args_info = '\nModel arguments: '
        args_info += ',\n'.join(["{} = {}".format(arg, value) for arg, value in self.__dict__.items()])
        args_info += '.\n'
        return args_info