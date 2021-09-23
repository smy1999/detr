import torch
import torch.nn as nn


def test_cpu_or_gpu():
    model = nn.LSTM(input_size=10, hidden_size=4, num_layers=1, batch_first=True)
    # model = model.cuda()
    print(next(model.parameters()).device)


def test_directory():
    import glob
    path = './outputs/*'
    file_list = glob.glob(path)
    print(file_list)


if __name__ == '__main__':
    # test_cpu_or_gpu()
    test_directory()
