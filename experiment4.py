import logging
import sys

import numpy as np
import torch
import tqdm
from torch import nn

from arguments.act_args import ActArgs
from arguments.env_args import EnvArgs
from arguments.gumbel_args import GumbelArgs
from utils.load_data import load_dataset, get_optimizer, get_metric_type, get_folds, generate_split_mask_data
from utils.set_seeds import set_seed
from model.CoGNN4 import CoGNN


class Experiment4(object):
    def __init__(self, args):
        super().__init__()
        for arg in vars(args):
            value_arg = getattr(args, arg)
            self.__setattr__(arg, value_arg)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            filename=f'logs/4/{self.dataset_name}.log',
            filemode='w')
        self.logger = logging.getLogger(__name__)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.CrossEntropyLoss()
        self.metric = None
        self.out_dim = None

        self.logger.info("=" * 100)
        for arg in vars(args):
            value_arg = getattr(args, arg)
            self.logger.info(f'{arg}: {value_arg}')

        print(f'dataset: {self.dataset_name}')

    def run(self):
        folds = get_folds()
        dataset, dataset_type, env_act_type = load_dataset(self.dataset_name)

        in_dim = dataset.x.shape[1]
        self.out_dim = int(np.max(dataset.y.numpy()) + 1)

        dataset = dataset.to(device=self.device)

        # metric
        self.metric = get_metric_type(self.out_dim).to(device=self.device)

        gumbel_args = \
            GumbelArgs(tau0=self.tau0, temp=self.temp, learn_temp=self.learn_temp, k=self.k, alpha=self.alpha)

        act_args = \
            ActArgs(model_type=self.act_model, num_layers=self.act_num_layers, env_dim=self.env_dim,
                    dropout=self.dropout, act_type=env_act_type, hidden_dim=self.act_dim, out_dim=2)
        env_args = \
            EnvArgs(model_type=self.env_model, num_layers=self.env_num_layers, dropout=self.dropout,
                    act_type=env_act_type, in_dim=in_dim, hidden_dim=self.env_dim, out_dim=self.out_dim)

        metric_list = []
        for fold in folds:
            set_seed(self.seed)
            # data = generate_split_mask_data(dataset, self.dataset_name, fold, self.out_dim)
            # data = generate_single_random_split(dataset, self.out_dim)
            data = dataset
            sota = self.single_fold(fold + 1, data, env_args, act_args, gumbel_args)
            print(
                f'Split {fold + 1}, train_metric: {sota[0]:.4f}, val_metric: {sota[1]:.4f}, test_metric: {sota[2]:.4f}\n')
            self.logger.info(
                f'Split {fold + 1}, train_metric: {sota[0]:.4f}, val_metric: {sota[1]:.4f}, test_metric: {sota[2]:.4f}\n')
            metric_list.append(sota[2])
        print(f'\nFinal metric: {np.mean(metric_list):.4f} \u00B1 {np.std(metric_list):.4f}')
        self.logger.info(f'Final metric: {np.mean(metric_list):.4f} \u00B1 {np.std(metric_list):.4f}')
        return sota[2]

    def single_fold(self, fold, data, env_args, act_args, gumbel_args):
        model = CoGNN(gumbel_args, env_args, act_args).to(device=self.device)
        optimizer = get_optimizer(model, self.lr, self.weight_decay)

        with tqdm.tqdm(total=self.max_epochs, file=sys.stdout) as pbar:
            test_metric_list, sota = self.train_and_test(data=data, model=model, optimizer=optimizer, pbar=pbar,
                                                         fold=fold)
        return sota

    def train_and_test(self, data, model, optimizer, pbar, fold):
        best_train_metric = 0.0
        best_test_metric = 0.0
        best_val_metric = 0.0
        patience_counter = 0
        train_loss_list, train_metric_list = [], []
        test_loss_list, test_metric_list = [], []
        val_loss_list, val_metric_list = [], []
        for epoch in range(self.max_epochs):
            self.train(data=data, model=model, optimizer=optimizer)
            train_loss, train_metric = self.test(data=data, model=model, mask='train')
            val_loss, val_metric = self.test(data=data, model=model, mask='val')
            test_loss, test_metric = self.test(data=data, model=model, mask='test')

            # === 早停判断 ===
            improved = (val_metric - best_val_metric) > 1e-4
            if improved:
                best_val_metric = val_metric
                best_test_metric = test_metric
                best_train_metric = train_metric
                patience_counter = 0
            else:
                patience_counter += 1

            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_test_metric = test_metric
                best_train_metric = train_metric

            train_loss_list.append(train_loss), train_metric_list.append(train_metric)
            test_loss_list.append(test_loss), test_metric_list.append(test_metric)
            val_loss_list.append(val_loss), val_metric_list.append(val_metric)

            if epoch + 1 == self.max_epochs or patience_counter >= self.patience:
                self.logger.info(f'Split {fold}: epoch {epoch}, '
                                 f'train_loss: {train_loss:.2f}, val_loss: {val_loss:.4f}, test_loss: {test_loss:.4f}, '
                                 f'train_metric: {train_metric:.4f}, val_metric: {val_metric:.4f}, '
                                 f'test_metric: {test_metric:.4f}({best_test_metric:.4f})')

            pbar.set_description(f'Split {fold}: epoch {epoch}, '
                                 f'train_loss: {train_loss:.2f}, val_loss: {val_loss:.4f}, test_loss: {test_loss:.4f}, '
                                 f'train_metric: {train_metric:.4f}, val_metric: {val_metric:.4f}, '
                                 f'test_metric: {test_metric:.4f}({best_test_metric:.4f})')
            pbar.update(1)

            if patience_counter >= self.patience:
                return test_metric_list, [best_train_metric.item(), best_val_metric.item(), best_test_metric.item()]

        return test_metric_list, [best_train_metric.item(), best_val_metric.item(), best_test_metric.item()]

    def train(self, data, model, optimizer):
        model.train()
        optimizer.zero_grad()
        logits, edge_weight, embedding = model.forward(x=data.x, edge_index=data.edge_index)

        task_loss = self.criterion(logits[data.train_mask], data.y[data.train_mask])

        row, col = data.edge_index
        z_src, z_dst = embedding[row], embedding[col]
        reg_loss = (edge_weight * torch.norm(z_src - z_dst, dim=1) ** 2).mean()

        loss = task_loss + self.reg_weight * reg_loss

        loss.backward()
        optimizer.step()

    def test(self, data, model, mask):
        model.eval()
        with torch.no_grad():
            logits, _, _ = model(x=data.x, edge_index=data.edge_index)
            node_mask = getattr(data, f'{mask}_mask')
            loss = self.criterion(logits[node_mask], data.y[node_mask])
            metric = self.metric(logits[node_mask], data.y[node_mask])

        return loss, metric
