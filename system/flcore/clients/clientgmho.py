import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
from collections import defaultdict
from sklearn.preprocessing import label_binarize


class ClientGMHo(Client):
    def __init__(self, args, id, train_samples, test_samples, label_weight, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.label_weight = label_weight
        self.args = args
        self.global_mask, self.local_mask = None, None  # global_mask 对应标签头部；local_mask 对应本地头部

    def train(self):
        trainloader = self.load_train_data()

        initial_head = copy.deepcopy(self.model.head)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.global_mask, self.local_mask = self.evaluate_critical_parameter(
            initial_head, self.model.head, self.args.tau
        )

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def evaluate_critical_parameter(self, prevhead, head, tau):
        global_mask = []  # mark critical parameter
        local_mask = []  # mark non-critical parameter

        # select critical parameters in each layer
        for (name1, prevparam), (name2, param) in zip(prevhead.named_parameters(), head.named_parameters()):
            g = (param.data - prevparam.data)
            v = param.data
            c = torch.abs(g * v)

            metric = c.view(-1)
            num_params = metric.size(0)
            nz = int(tau * num_params)
            top_values, _ = torch.topk(metric, nz)
            thresh = top_values[-1] if len(top_values) > 0 else np.inf
            # if threshold equals 0, select minimal nonzero element as threshold
            if thresh <= 1e-10:
                new_metric = metric[metric > 1e-20]
                if len(new_metric) == 0:  # this means all items in metric are zero
                    print(f'Abnormal!!! metric:{metric}')
                else:
                    thresh = new_metric.sort()[0][0]

            # Get the local mask and global mask
            mask = (c >= thresh).int().to('cpu')
            global_mask.append((c < thresh).int().to('cpu'))
            local_mask.append(mask)
        head.zero_grad()

        return global_mask, local_mask

    def set_parameters(self, heads):
        if self.global_mask is None or self.local_mask is None:
            for param in self.model.head.parameters():
                param.data.zero_()
            for client_param, server_param in zip(self.model.head.parameters(), heads.parameters()):
                client_param.data += server_param.data.clone()
        else:
            index = 0
            head = copy.deepcopy(self.model.head)
            gobal_head = copy.deepcopy(self.model.head)
            for param in gobal_head.parameters():
                param.data.zero_()
            for client_param, server_param in zip(gobal_head.parameters(), heads.parameters()):
                client_param.data += server_param.data.clone()
            for param1, param2, param3 in zip(self.model.head.parameters(), gobal_head.parameters(),
                                              head.parameters()):
                param1.data = self.local_mask[index].to(self.device).float() * param3.data.clone() + \
                              self.global_mask[index].to(self.args.device).float() * param2.data.clone()
                index += 1

    def collect_protos(self):
        trainloader = self.load_train_data()
        self.model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        self.protos = agg_func(protos)

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        return test_acc, test_num


def agg_func(protos):
    """
    Returns the average of the weights.
    """
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]
    return protos
