import os
import ujson
import numpy as np
from sklearn.model_selection import train_test_split

batch_size = 10
train_ratio = 0.75  # merge original training set and test set, then split it manually.
alpha = 5  # for Dirichlet distribution. 100 for exdir


def get_alpha():
    return str(alpha)


def check(config_path, train_path, test_path, num_clients):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
                config['alpha'] == alpha and \
                config['batch_size'] == batch_size:
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False


def separate_data(data, num_clients, num_classes, class_per_client=None):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data

    C = class_per_client
    min_size_per_label = 0
    # You can adjust the `min_require_size_per_label` to meet you requirements
    min_require_size_per_label = max(C * num_clients // num_classes // 2, 1)
    if min_require_size_per_label < 1:
        raise ValueError
    clientidx_map = {}
    while min_size_per_label < min_require_size_per_label:
        # initialize
        for k in range(num_classes):
            clientidx_map[k] = []
        # allocate
        for i in range(num_clients):
            labelidx = np.random.choice(range(num_classes), C, replace=False)
            for k in labelidx:
                clientidx_map[k].append(i)
        min_size_per_label = min([len(clientidx_map[k]) for k in range(num_classes)])

    '''The second level: allocate data idx'''
    dataidx_map = {}
    y_train = dataset_label
    min_size = 0
    min_require_size = 10
    K = num_classes
    N = len(y_train)
    print("\n*****clientidx_map*****")
    print(clientidx_map)
    print("\n*****Number of clients per label*****")
    print([len(clientidx_map[i]) for i in range(len(clientidx_map))])

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array(
                [p * (len(idx_j) < N / num_clients and j in clientidx_map[k]) for j, (p, idx_j) in
                 enumerate(zip(proportions, idx_batch))])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            if proportions[-1] != len(idx_k):
                for w in range(clientidx_map[k][-1], num_clients - 1):
                    proportions[w] = len(idx_k)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_clients):
        np.random.shuffle(idx_batch[j])
        dataidx_map[j] = idx_batch[j]

    # assign data
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client] == i))))

    del data

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic


def split_data(X, y):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train': [], 'test': []}

    for i in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_ratio, shuffle=True)

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y

    return train_data, test_data


def save_file(config_path, train_path, test_path, train_data, test_data, num_clients,
              num_classes, statistic):
    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'Size of samples for labels in clients': statistic,
        'alpha': alpha,
        'batch_size': batch_size,
    }

    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")
