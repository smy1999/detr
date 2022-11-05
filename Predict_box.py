import math

from PIL import Image
import requests
import glob
import matplotlib.pyplot as plt

#import ipywidgets as widgets
#from IPython.display import display, clear_output

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
from hubconf import *
from util.misc import nested_tensor_from_tensor_list
torch.set_grad_enabled(False);

# COCO classes
CLASSES = [
    'N/A', 'ore'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741]]
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=5,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


def predict(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.95

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled


def visualize_encoder_decoder_weight(model, im):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    # use lists to store the outputs via up-values
    conv_features, enc_attn_weights, dec_attn_weights = [], [], []

    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
            lambda self, input, output: enc_attn_weights.append(output[1])
        ),
        model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
            lambda self, input, output: dec_attn_weights.append(output[1])
        ),
    ]

    # propagate through the model
    outputs = model(img)

    for hook in hooks:
        hook.remove()

    # don't need the list anymore
    conv_features = conv_features[0]
    enc_attn_weights = enc_attn_weights[0]
    dec_attn_weights = dec_attn_weights[0]

    # get the feature map shape
    h, w = conv_features['0'].tensors.shape[-2:]

    # Visutalize encoder-decoder multi-head attention weights
    # fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
    # colors = COLORS * 100
    # for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
    #     ax = ax_i[0]
    #     ax.imshow(dec_attn_weights[0, idx].view(h, w))
    #     ax.axis('off')
    #     ax.set_title(f'query id: {idx.item()}')
    #     ax = ax_i[1]
    #     ax.imshow(im)
    #     ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
    #                                fill=False, color='blue', linewidth=3))
    #     ax.axis('off')
    #     ax.set_title(CLASSES[probas[idx].argmax()])
    #     break
    # fig.tight_layout()
    # plt.show()

    # for idx in keep.nonzero():
    #     plt.imshow(dec_attn_weights[0, idx].view(h, w))
    #     plt.axis('off')
    #     plt.title(f'query id: {idx.item()}')
    #     plt.show()

    # for idx, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), bboxes_scaled):
    #     fig = plt.figure()
    #     plt.axis('off')
    #     plt.title(CLASSES[probas[idx].argmax()])
    #     ax = fig.add_subplot(111)
    #     rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='blue', linewidth=3)
    #     ax.add_patch(rect)
    #     plt.imshow(im)
    #     plt.show()


    # # Visualize encoder self-attention weights
    # # out of the CNN
    # f_map = conv_features['0']
    #
    # # get the HxW shape of the feature maps of the CNN
    # shape = f_map.tensors.shape[-2:]
    # # and reshape the self-attention to a more interpretable shape
    # sattn = enc_attn_weights[0].reshape(shape + shape)
    #
    # # downsampling factor for the CNN, is 32 for DETR and 16 for DETR DC5
    # fact = 32
    #
    # # let's select 4 reference points for visualization
    # idxs = [(200, 200), (280, 400), (200, 600), (440, 800), ]
    # idxs = [(200, 200), (520, 220), (200, 600), (440, 800), ]
    # # idxs = [(800, 80), (600, 220), (580, 780), (300, 500), ]
    #
    # # here we create the canvas
    # fig = plt.figure(constrained_layout=True, figsize=(25 * 0.7, 8.5 * 0.7))
    # # and we add one plot per reference point
    # gs = fig.add_gridspec(2, 4)
    # axs = [
    #     fig.add_subplot(gs[0, 0]),
    #     fig.add_subplot(gs[1, 0]),
    #     fig.add_subplot(gs[0, -1]),
    #     fig.add_subplot(gs[1, -1]),
    # ]
    #
    # # for each one of the reference points, let's plot the self-attention
    # # for that point
    # for idx_o, ax in zip(idxs, axs):
    #     idx = (idx_o[0] // fact, idx_o[1] // fact)
    #     ax.imshow(sattn[..., idx[0], idx[1]], cmap='cividis', interpolation='nearest')
    #     ax.axis('off')
    #     ax.set_title(f'self-attention{idx_o}')
    #
    # # and now let's add the central image, with the reference points as red circles
    # fcenter_ax = fig.add_subplot(gs[:, 1:-1])
    # fcenter_ax.imshow(im)
    # for (y, x) in idxs:
    #     scale = im.height / img.shape[-2]
    #     x = ((x // fact) + 0.5) * fact
    #     y = ((y // fact) + 0.5) * fact
    #     fcenter_ax.add_patch(plt.Circle((x * scale, y * scale), fact // 2, color='r'))
    #     fcenter_ax.axis('off')
    plt.show()


if __name__ == "__main__":
    model = detr_resnet50(False, 2)
    state_dict = torch.load("outputs/checkpoint0399.pth", map_location='cpu')
    model.load_state_dict(state_dict["model"])
    model.eval()

    p = 'data/coco/val2017/IMG_4897.jpg'
    im = Image.open(p)
    scores, boxes = predict(im, model, transform)

    plot_results(im, scores, boxes)

    # p = 'data/coco/val2017/'
    # im_list = ['IMG_4861.jpg', 'IMG_4937.jpg', 'IMG_4965.jpg', 'IMG_5438.jpg', 'IMG_9755.jpg']
    # for k in im_list:
    #     im = Image.open(p + k)
    #     scores, boxes = predict(im, model, transform)
    #     plot_results(im, scores, boxes)

    # visualize_encoder_decoder_weight(model, im)
