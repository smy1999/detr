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


def test_torch():
    x = torch.rand((3, 4, 5, 6))
    print(x.size())
    x = x.unsqueeze(1).repeat(1, 7, 1, 1, 1).flatten(0,1)
    print(x.size())

if __name__ == '__main__':
    # test_cpu_or_gpu()
    # test_directory()
    test_torch()