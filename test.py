import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path


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


def test_checkpoint():
    # checkpoint = torch.load("./outputs/checkpoint.pth", map_location='cpu')
    checkpoint = torch.load("./detr-r50_2.pth", map_location='cpu')
    print(2)


def test_d():
    dic = {"train_lr": 0.0001, "train_class_error": 0.7155924710360441, "train_loss": 7.599573807282881, "train_loss_ce": 0.08765341693122702, "train_loss_bbox": 0.18215506845577198, "train_loss_giou": 0.9405036690560254, "train_loss_ce_0": 0.10179406506094066, "train_loss_bbox_0": 0.21456225436519494, "train_loss_giou_0": 1.09054873612794, "train_loss_ce_1": 0.0939756432780996, "train_loss_bbox_1": 0.19551564380526543, "train_loss_giou_1": 0.9993628900159489, "train_loss_ce_2": 0.0928933957951482, "train_loss_bbox_2": 0.19035557014021007, "train_loss_giou_2": 0.9701324945146387, "train_loss_ce_3": 0.08959895735394886, "train_loss_bbox_3": 0.18561143343421546, "train_loss_giou_3": 0.9492876340042461, "train_loss_ce_4": 0.08884960181735964, "train_loss_bbox_4": 0.18364166739312085, "train_loss_giou_4": 0.9431316310709174, "train_loss_ce_unscaled": 0.08765341693122702, "train_class_error_unscaled": 0.7155924710360441, "train_loss_bbox_unscaled": 0.03643101341599091, "train_loss_giou_unscaled": 0.4702518345280127, "train_cardinality_error_unscaled": 91.86363636363636, "train_loss_ce_0_unscaled": 0.10179406506094066, "train_loss_bbox_0_unscaled": 0.04291245104237036, "train_loss_giou_0_unscaled": 0.54527436806397, "train_cardinality_error_0_unscaled": 95.43181818181819, "train_loss_ce_1_unscaled": 0.0939756432780996, "train_loss_bbox_1_unscaled": 0.039103129116648976, "train_loss_giou_1_unscaled": 0.49968144500797446, "train_cardinality_error_1_unscaled": 93.70454545454545, "train_loss_ce_2_unscaled": 0.0928933957951482, "train_loss_bbox_2_unscaled": 0.03807111393490976, "train_loss_giou_2_unscaled": 0.48506624725731934, "train_cardinality_error_2_unscaled": 92.86363636363636, "train_loss_ce_3_unscaled": 0.08959895735394886, "train_loss_bbox_3_unscaled": 0.03712228685617447, "train_loss_giou_3_unscaled": 0.47464381700212305, "train_cardinality_error_3_unscaled": 92.52272727272727, "train_loss_ce_4_unscaled": 0.08884960181735964, "train_loss_bbox_4_unscaled": 0.03672833317382769, "train_loss_giou_4_unscaled": 0.4715658155354587, "train_cardinality_error_4_unscaled": 92.22727272727273, "test_class_error": 0.0, "test_loss": 7.662118593851726, "test_loss_ce": 0.025051836234827835, "test_loss_bbox": 0.19102699806292853, "test_loss_giou": 0.9998189806938171, "test_loss_ce_0": 0.024893892618517082, "test_loss_bbox_0": 0.21668010453383127, "test_loss_giou_0": 1.1937936345736186, "test_loss_ce_1": 0.022937055366734665, "test_loss_bbox_1": 0.20297900338967642, "test_loss_giou_1": 1.0797017415364583, "test_loss_ce_2": 0.021750126034021378, "test_loss_bbox_2": 0.1947764828801155, "test_loss_giou_2": 1.0395078460375469, "test_loss_ce_3": 0.024203140599032242, "test_loss_bbox_3": 0.19284919649362564, "test_loss_giou_3": 1.0124972065289815, "test_loss_ce_4": 0.024072468901673954, "test_loss_bbox_4": 0.19196157405773798, "test_loss_giou_4": 1.0036176045735676, "test_loss_ce_unscaled": 0.025051836234827835, "test_class_error_unscaled": 0.0, "test_loss_bbox_unscaled": 0.03820539948840936, "test_loss_giou_unscaled": 0.49990949034690857, "test_cardinality_error_unscaled": 68.66666666666667, "test_loss_ce_0_unscaled": 0.024893892618517082, "test_loss_bbox_0_unscaled": 0.04333602202435335, "test_loss_giou_0_unscaled": 0.5968968172868093, "test_cardinality_error_0_unscaled": 68.66666666666667, "test_loss_ce_1_unscaled": 0.022937055366734665, "test_loss_bbox_1_unscaled": 0.04059580030540625, "test_loss_giou_1_unscaled": 0.5398508707682291, "test_cardinality_error_1_unscaled": 68.66666666666667, "test_loss_ce_2_unscaled": 0.021750126034021378, "test_loss_bbox_2_unscaled": 0.038955296079317726, "test_loss_giou_2_unscaled": 0.5197539230187734, "test_cardinality_error_2_unscaled": 68.66666666666667, "test_loss_ce_3_unscaled": 0.024203140599032242, "test_loss_bbox_3_unscaled": 0.03856983967125416, "test_loss_giou_3_unscaled": 0.5062486032644907, "test_cardinality_error_3_unscaled": 68.66666666666667, "test_loss_ce_4_unscaled": 0.024072468901673954, "test_loss_bbox_4_unscaled": 0.038392314687371254, "test_loss_giou_4_unscaled": 0.5018088022867838, "test_cardinality_error_4_unscaled": 68.66666666666667, "test_coco_eval_bbox": [0.043258445179013116, 0.15047311070952502, 0.010754702942781668, 0.04325847556879242, -1.0, -1.0, 0.0008036739380022963, 0.015958668197474168, 0.09150401836969001, 0.09150401836969001, -1.0, -1.0], "epoch": 3, "n_parameters": 41279495}
    for (key, value) in dic.items():
        print(key, value)


def test_chkpt():
    # pretrained_weights = torch.load('detr-r50_p2.pth')
    pretrained_weights = torch.load('detr-r50-panoptic-00ce5173.pth')
    s = pretrained_weights['model'].keys()
    s = list(s)

    for k in s:
        print(k)


def test_softmax():
    x = np.ones((2, 3))
    k = 0
    for i in range(2):
        for j in range(3):
            x[i][j] = k
            k += 1
    mask = torch.Tensor(x)
    print(mask)
    mask = mask.flatten()
    print(mask)
    mask = mask.softmax(0)
    print(mask)
    mask = mask.reshape((2, 3))
    print(mask)


def test_log_txt():
    log_name = "./outputs/segm_model/log.txt"
    dfs = pd.read_json(Path(log_name), lines=True)
    print(list(dfs.keys()))
    txt_path = 'C:/ima/large_scale_test_image_list.txt'
    f = open(txt_path, 'r')
    data = f.read()
    print(data)


if __name__ == '__main__':
    test_log_txt()