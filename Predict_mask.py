from PIL import Image
import requests
import io
import math
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
import numpy

import glob
from hubconf import *
from util.misc import nested_tensor_from_tensor_list
torch.set_grad_enabled(False);

import panopticapi
from panopticapi.utils import id2rgb, rgb2id

# COCO classes
CLASSES = [
    'background', 'ore'
]

# Detectron2 uses a different numbering scheme, we build a conversion table
coco2d2 = {}
count = 0
for i, c in enumerate(CLASSES):
    if c != 'N/A':
        coco2d2[i] = count
        count += 1

# colors for visualization
COLORS = [[0.000, 0.447, 0.741]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


model, postprocessor = torch.hub.load('', 'detr_mine', pretrained=True,
                                      return_postprocessor=True, num_classes=2, source='local')
model.eval()


def img_read():
    path = 'data/coco/val2017/'
    img_list = glob.glob(path + '*.jpg')
    # for i in img_list:
    #     im = Image.open(i)
    #     im.show()
    im = path + "IMG_4861.jpg"
    img = Image.open(im)
    return img


def predict(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)
    out = model(img)
    return img, out


def predict_mask(out):
    # compute the scores, excluding the "no-object" class (the last one)
    scores = out["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]
    # threshold the confidence
    keep = scores > 0.85

    # Plot all the remaining masks
    ncols = 5
    fig, axs = plt.subplots(ncols=ncols, nrows=math.ceil(keep.sum().item() / ncols), figsize=(18, 10))
    for line in axs:
        for a in line:
            a.axis('off')
    for i, mask in enumerate(out["pred_masks"][keep]):
        ax = axs[i // ncols, i % ncols]
        ax.imshow(mask, cmap="cividis")
        ax.axis('off')
    fig.tight_layout()
    plt.show()


def predict_total_mask(img, out):
    import itertools
    import seaborn as sns

    # the post-processor expects as input the target size of the predictions (which we set here to the image size)
    result = postprocessor(out, torch.as_tensor(img.shape[-2:]).unsqueeze(0))[0]

    palette = itertools.cycle(sns.color_palette())

    # The segmentation is stored in a special-format png
    panoptic_seg = Image.open(io.BytesIO(result['png_string']))
    panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8).copy()
    # We retrieve the ids corresponding to each mask
    panoptic_seg_id = rgb2id(panoptic_seg)

    # Finally we color each mask individually
    panoptic_seg[:, :, :] = 0
    for id in range(panoptic_seg_id.max() + 1):
        panoptic_seg[panoptic_seg_id == id] = numpy.asarray(next(palette)) * 255
    plt.figure(figsize=(15, 15))
    plt.imshow(panoptic_seg)
    plt.axis('off')
    plt.show()


def visualization():
    from detectr


if __name__ == '__main__':
    im = img_read()
    img, out = predict(im, model, transform)
    # predict_mask(out)
    predict_total_mask(img, out)









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
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        # ax.text(xmin, ymin, text, fontsize=15,
        #         bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


def detect(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.00001

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled


def predict(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    anImg = transform(im)
    data = nested_tensor_from_tensor_list([anImg])

    # propagate through the model
    outputs = model(data)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    
    keep = probas.max(-1).values > 0.7
    #print(probas[keep])

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled


def predict_bounding_box():
    model = detr_resnet50(False, 2);
    state_dict = torch.load("outputs/checkpoint.pth", map_location='cpu')
    model.load_state_dict(state_dict["model"])
    model.eval()

    path = 'data/coco/val2017/'
    val_list = glob.glob(path + '*.jpg')
    for p in val_list:
        im = Image.open(p)
        scores, boxes = predict(im, model, transform)
        plot_results(im, scores, boxes)
    # im = Image.open('data/coco/val2017/IMG_4861.jpg')
    # scores, boxes = predict(im, model, transform)
    # plot_results(im, scores, boxes)

#
# if __name__ == "__main__":
#     # predict_bounding_box()
#     model = detr_resnet50(False, 2);
#     state_dict = torch.load("outputs/segm_model/checkpoint.pth", map_location='cpu')
#     model.load_state_dict(state_dict["model"])
#     model.eval()
#
#     p = 'data/coco/val2017/IMG_4861.jpg'
#     im = Image.open(p)
#     scores, boxes = predict(im, model, transform)
#     plot_results(im, scores, boxes)
