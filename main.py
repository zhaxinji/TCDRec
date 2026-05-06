import os
import warnings
import torch
import platform
from time import time
from tqdm import tqdm
from training import train, evaluate_model
import torch.optim as optim
from params import parse_args, Config
from utils import init_logger, early_stopping, plot_curve, res_output, stop_log, update_result, sele_para

warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from dataloader import Load_dataset, Load_eval_dataset
from model import Model


class Net:
    def __init__(self, args):
        self.config = Config(args)
        self.logger = init_logger(self.config)
        self.logger.info(self.config)
        self.device = self.config.device
        self.model_name = self.config.model_name
        self.dataset_name = self.config.dataset
        self.batch_size = self.config.batch_size
        self.num_workers = self.config.num_workers
        self.learning_rate = self.config.learning_rate
        self.num_epoch = self.config.num_epoch
        self.topk = self.config.topk
        self.metrics = self.config.metrics
        self.valid_metric = self.config.valid_metric
        self.stopping_step = self.config.stopping_step
        self.reg_weight = self.config.reg_weight
        self.cur_step = 0
        self.best_valid_score = -1
        self.best_valid_result = {}
        self.best_test_upon_valid = {}
        self.writer = SummaryWriter() if self.config.writer else None

        Dataset = Load_dataset(self.config)
        valid_dataset, test_dataset = Dataset.load_eval_data()
        self.train_data = DataLoader(Dataset, batch_size=self.batch_size, shuffle=True,
                                     num_workers=self.num_workers)

        (self.valid_data, self.test_data) = (Load_eval_dataset("Validation", self.config, valid_dataset),
                                             Load_eval_dataset("Testing", self.config, test_dataset))
        self.model = Model(self.config, Dataset).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), self.learning_rate, weight_decay=self.reg_weight)
        lr_scheduler = self.config.learning_rate_scheduler
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler

    def plot_train_loss(self):
        plot_curve(self)

    def run(self):
        run_start_time = time()
        for epoch_idx in range(self.num_epoch):
            train_start_time = time()
            train_loss = train(self, epoch_idx)
            if torch.isnan(train_loss[0]):
                ret_value = {"Recall@20": -1} if self.best_test_upon_valid == {} else self.best_test_upon_valid
                stop_output = '\n ' + str(self.config.dataset) + '  key parameter: ' + sele_para(self.config)
                self.logger.info(stop_output)
                self.logger.info('Loss is nan at epoch: {}; last value is {}Exiting.'.format(epoch_idx, ret_value))
                return ret_value

            self.lr_scheduler.step()

            train_output = res_output(epoch_idx, train_start_time, time(), train_loss, "train")
            self.logger.info(train_output)

            self.model.sample()

            valid_start_time = time()
            valid_score, valid_result = evaluate_model(self, epoch_idx, self.valid_data, t_or_v="valid")
            valid_output = res_output(epoch_idx, valid_start_time, time(), valid_result, t_or_v="valid")

            test_start_time = time()
            test_score, test_result = evaluate_model(self, epoch_idx, self.test_data, t_or_v="test")
            test_score_output = res_output(epoch_idx, test_start_time, time(), test_result, t_or_v="test")
            self.logger.info(test_score_output)

            self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                test_score, self.best_valid_score, self.cur_step, self.stopping_step)

            self.best_valid_result[epoch_idx] = self.best_valid_score

            if update_flag:
                update_result(self, test_result)

            if stop_flag:
                break
            else:
                print()
        return self.best_test_upon_valid


if __name__ == '__main__':
    _args = parse_args()
    model = Net(_args)
    best_score = model.run()