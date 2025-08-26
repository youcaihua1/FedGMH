"""
    消融研究算法：FedGMH-avghead
"""
import copy
import time
import torch
import os
import numpy as np
import torch.nn as nn
from flcore.clients.clientgmha import ClientGMHa
from flcore.servers.serverbase import Server
from collections import defaultdict


def read_data(dataset, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join('../dataset', dataset, 'train/')
        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()
        return train_data

    else:
        test_data_dir = os.path.join('../dataset', dataset, 'test/')
        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()
        return test_data


def read_client_data(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)
        train_data = [(x, y) for x, y in zip(X_train, y_train)]

        y_train_np = y_train.numpy()
        # 使用 numpy.bincount 来统计元素出现的次数
        counts = np.bincount(y_train_np, minlength=y_train_np.max() + 1)
        weight = counts / float(len(y_train_np))
        label_weight = {i: weight[i] for i in range(len(weight))}
        label_weight = {k: v for k, v in label_weight.items() if v != 0}
        return train_data, label_weight
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


class FedGMHa(Server):
    """消融研究算法：FedGMH-avghead"""

    def __init__(self, args, times):
        super().__init__(args, times)
        self.global_model = None

        self.set_clients(ClientGMHa)
        print(f"\nJoin ratio | total clients: {self.join_ratio} | {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.CEloss = nn.CrossEntropyLoss()
        self.server_learning_rate = args.server_learning_rate

        self.heads = defaultdict()
        self.opt_heads = defaultdict()
        for i in range(args.num_classes):
            self.heads[i] = copy.deepcopy(self.clients[0].model.head)
            self.opt_heads[i] = torch.optim.SGD(self.heads[i].parameters(), lr=self.server_learning_rate)

    def set_clients(self, clientObj):
        for i in range(self.num_clients):
            train_data, label_weight = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, id=i, train_samples=len(train_data), test_samples=len(test_data),
                               label_weight=label_weight)
            self.clients.append(client)

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                # 打印当前学习率
                print(f'Current Learning Rate: {self.selected_clients[0].optimizer.param_groups[0]["lr"]:.4f}')
                if self.auto_break or i > (self.global_rounds - 20):
                    print("\nEvaluate personalized models")
                    self.evaluate()

            for client in self.selected_clients:
                client.train()
                client.collect_protos()

            self.receive_protos()
            self.train_head()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_csv()

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.heads)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            for cc in client.protos.keys():
                y = torch.tensor(cc, dtype=torch.int64, device=self.device)
                self.uploaded_protos.append((client.protos[cc], y))

    def train_head(self):
        for x, y in self.uploaded_protos:
            label = int(y.item())
            out = self.heads[label](x)
            loss = self.CEloss(out, y)
            self.opt_heads[label].zero_grad()
            loss.backward()
            self.opt_heads[label].step()
