import copy
import torch
import time
from flcore.clients.clientbase import Client
from collections import defaultdict


class ClientGMHa(Client):
    def __init__(self, args, id, train_samples, test_samples, label_weight, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.label_weight = label_weight
        self.args = args

    def train(self):
        trainloader = self.load_train_data()
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

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, heads):
        gobal_head = copy.deepcopy(self.model.head)
        for param in gobal_head.parameters():
            param.data.zero_()
        for label in self.label_weight:
            for client_param, server_param in zip(gobal_head.parameters(), heads[label].parameters()):
                client_param.data += server_param.data.clone() * self.label_weight[label]
        for param1, param2 in zip(self.model.head.parameters(), gobal_head.parameters()):
            param1.data = param2.data.clone()

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
