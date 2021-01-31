import unicodedata
import torch
import numpy as np
from random import randrange

def string_to_tensor(string, dict):
    list = [[dict[letter] for letter in unicodedata.normalize("NFD",string.lower())
            if unicodedata.category(letter) != "Mn"
            and letter in dict]]
    return torch.tensor(list).float()

def create_w2v_dict(alphabet):
    '''
    :param alphabet: list of characters or words in string format
    :return: PyTorch Tensor of one-hot encoded characters
    '''

    alphabet = list(set(alphabet))
    alphabet.sort()
    w2v_dict = {}
    for i in range(len(alphabet)):
        one_hot_vector = np.zeros(len(alphabet))
        one_hot_vector[i] = 1
        w2v_dict[alphabet[i]] = list(one_hot_vector)
    return w2v_dict

def alphabetical_data_to_tensor(data, alphabet):
    '''
    :param data:
    :return:
    '''
    w2v_dict = create_w2v_dict(alphabet)
    return [string_to_tensor(string, w2v_dict).transpose(0,1) for string in data] #Transpose needed because PyTorch takes format seq_length x batch_size x input_size

def labels_to_tensor(list_of_labels):
    '''
    :param list_of_labels: labels
    :return: Tensor
    '''
    labels = list(set(list_of_labels))
    labels.sort()
    return torch.tensor([[labels.index(instance)] for instance in list_of_labels])

def split_data(features, labels, train = 0.7, val = 0.1, test = 0.2):
    '''Inputs:
    - data: List of Tensors
    - labels: List of Label Tensor
    '''

    if len(features) != len(labels):
        raise ValueError("The lists features and labels must be of equal length")
    if train <= 0 or train > 1 or test <= 0 or test > 1:
        raise ValueError("Both train and test must be in range [0,1)")

    shuffled_labels = labels
    shuffled_features = features

    val_size = int(len(features)*val)
    test_size = int(len(features)*test)
    train_size = len(features)-test_size-val_size

    seed = np.random.randint(0,100000)
    np.random.seed(seed)
    np.random.shuffle(shuffled_features)
    np.random.seed(seed)
    np.random.shuffle(shuffled_labels)

    x_train, x_val, x_test = (shuffled_features[:train_size],
                              shuffled_features[train_size:(len(shuffled_features)-test_size)],
                              shuffled_features[(len(shuffled_features)-test_size):])

    y_train, y_val, y_test = (shuffled_labels[:train_size],
                              shuffled_labels[train_size:(len(shuffled_labels)-test_size)],
                              shuffled_labels[(len(shuffled_labels)-test_size):])

    if x_val == []:
        return x_train, y_train, x_test, y_test
    else:
        return x_train, y_train, x_val, y_val, x_test, y_test


#predict new input
def infer_language(inf_name, model, label_key, allowed_letters):
    test_tensor = alphabetical_data_to_tensor([inf_name], allowed_letters)

    model.eval()
    hidden = model.initial_hidden()
    with torch.no_grad():
        for i in range(len(test_tensor[0])):
            output, hidden = model(test_tensor[0][i], hidden)

    index = torch.argmax(output, 1).item()
    pred = label_key[index]
    return pred