from PIL import Image
import requests
import io
import math
import matplotlib.pyplot as plt
import cv2

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision
import torchvision.transforms as T
import numpy
numpy.set_printoptions(threshold=numpy.inf)

import glob
from hubconf import *
from util.misc import nested_tensor_from_tensor_list

torch.set_grad_enabled(False);

import panopticapi
from panopticapi.utils import id2rgb, rgb2id


def rgb2id(color):
    if isinstance(color, numpy.ndarray) and len(color.shape) == 3:
        if color.dtype == numpy.uint8:
            color = color.astype(numpy.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])

# COCO classes
CLASSES = [
    'N/A',
    'ore'
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
# model, postprocessor = torch.hub.load('', 'detr_pan', pretrained=True,
#                                       return_postprocessor=True, num_classes=2, source='local')
model.eval()


def img_read():
    path = 'data/coco/val2017/'
    im = path + "IMG_9755.jpg"
    # path = 'C:/ima/'
    # im = path + 'IMG_0398.jpg'
    # img_list = glob.glob(path + '*.jpg')
    # for i in img_list:
    #     im = Image.open(i)
    #     im.show()

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
    keep = scores > 0.99

    # Plot all the remaining masks
    ncols = 7
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

    for mask in out["pred_masks"][keep]:
        plt.imshow(mask, cmap="cividis")
    plt.show()


def predict_total_mask(img, out):
    import itertools
    import seaborn as sns

    # the post-processor expects as input the target size of the predictions (which we set here to the image size)
    result = postprocessor(out, torch.as_tensor(img.shape[-2:]).unsqueeze(0))[0]

    palette = itertools.cycle(sns.color_palette())

    # The segmentation is stored in a special-format png
    panoptic_seg = Image.open(io.BytesIO(result['png_string']))

    plt.imshow(panoptic_seg)
    plt.show()
    panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8).copy()

    # print(panoptic_seg)
    # We retrieve the ids corresponding to each mask
    panoptic_seg_id = rgb2id(panoptic_seg)

    # numpy.set_printoptions(threshold=numpy.inf)
    # print(panoptic_seg_id.shape)

    # Finally we color each mask individually
    panoptic_seg[:, :, :] = 0
    for id in range(panoptic_seg_id.max() + 1):
        panoptic_seg[panoptic_seg_id == id] = numpy.asarray(next(palette)) * 255
        # panoptic_seg[panoptic_seg_id == id] = numpy.asarray((0.9, 0.1, 0.1)) * 255

        # # plt.figure(figsize=(15, 15))
        # plt.imshow(panoptic_seg)
        # image = cv2.imread("data/coco/val2017/" + "IMG_9755.jpg")
        # # image = cv2.imread("IMG_9755.jpg")
        # b, g, r = cv2.split(image)
        # image = cv2.merge([r, g, b])
        # image = cv2.resize(image, (1066, 800))
        # # plt.imshow(panoptic_seg + image)
        # plt.axis('off')
        # plt.show()
    plt.figure(figsize=(15, 15))
    plt.imshow(panoptic_seg)
    plt.axis('off')
    plt.show()


def predict_attention(out):

    scores = out["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]
    keep = scores > 0.99
    for i, mask in enumerate(out["pred_masks"][keep]):

        # subplot
        plt.subplot(121)
        plt.imshow(mask, cmap='cividis')

        plt.subplot(122)
        plt.imshow(mask > -8, cmap='cividis')
        plt.show()

        # softmax
        # mask_shape = mask.shape
        # mask = mask.flatten()
        # mask = mask.softmax(0)
        # mask = mask.reshape(mask_shape)
        # plt.subplot(122)
        # plt.imshow(mask, cmap="cividis")


def predict_attention_full_image(img, out, img_name):
    scores = out["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]
    keep = scores > 0.99
    total = torch.zeros((200, 267), dtype=torch.bool)

    for i, mask in enumerate(out["pred_masks"][keep]):

        x = mask > -8
        total = total | x
        # plt.imshow(x, cmap='cividis')
        # plt.show()
        # break
    total = total.unsqueeze(0).unsqueeze(0).float()
    total = torch.nn.functional.interpolate(total, (3000, 4000))
    total = total.squeeze(0).squeeze(0)
    # plt.imshow(total, cmap='cividis')
    # plt.show()

    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(total, cmap='cividis')
    # plt.savefig(img_name)
    plt.show()


def predict_attention_full_image_big_size_slow(img, out, img_name):
    scores = out["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]
    keep = scores > 0.99
    target_size = (3000, 4000)
    total = torch.zeros(target_size, dtype=torch.bool)

    alpha = -8

    for i, mask in enumerate(out["pred_masks"][keep]):
        mask = mask.unsqueeze(0).unsqueeze(0).float()
        mask = torch.nn.functional.interpolate(mask, target_size)
        mask = mask.squeeze(0).squeeze(0)
        x = mask > alpha
        total = total | x
        # plt.imshow(x, cmap='cividis')
        # plt.show()
        # break
    # plt.imshow(total, cmap='cividis')
    # plt.show()

    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(total, cmap='cividis')
    # plt.savefig(str(alpha) + '.jpg')
    plt.show()


def test(img, out):

    is_thing_map = {i: i == 1 for i in range(2)}
    result = forward(out, torch.as_tensor(img.shape[-2:]).unsqueeze(0), self_is_thing_map=is_thing_map)[0]


def forward(outputs, processed_sizes, self_is_thing_map, target_sizes=None, self_threshold=0.85):
    import util.box_ops as box_ops
    from util.misc import NestedTensor, interpolate, nested_tensor_from_tensor_list
    from collections import defaultdict

    target_sizes = processed_sizes
    out_logits, raw_masks, raw_boxes = outputs["pred_logits"], outputs["pred_masks"], outputs["pred_boxes"]
    assert len(out_logits) == len(raw_masks) == len(target_sizes)
    preds = []

    def to_tuple(tup):
        if isinstance(tup, tuple):
            return tup
        return tuple(tup.cpu().tolist())

    for cur_logits, cur_masks, cur_boxes, size, target_size in zip(
        out_logits, raw_masks, raw_boxes, processed_sizes, target_sizes
    ):
        # we filter empty queries and detection below threshold
        scores, labels = cur_logits.softmax(-1).max(-1)
        keep = labels.ne(outputs["pred_logits"].shape[-1] - 1) & (scores > self_threshold)
        cur_scores, cur_classes = cur_logits.softmax(-1).max(-1)
        cur_scores = cur_scores[keep]
        cur_classes = cur_classes[keep]
        cur_masks = cur_masks[keep]
        cur_masks = interpolate(cur_masks[:, None], to_tuple(size), mode="bilinear").squeeze(1)
        cur_boxes = box_ops.box_cxcywh_to_xyxy(cur_boxes[keep])

        h, w = cur_masks.shape[-2:]
        assert len(cur_boxes) == len(cur_classes)

        # It may be that we have several predicted masks for the same stuff class.
        # In the following, we track the list of masks ids for each stuff class (they are merged later on)
        cur_masks = cur_masks.flatten(1)
        stuff_equiv_classes = defaultdict(lambda: [])
        for k, label in enumerate(cur_classes):
            if not self_is_thing_map[label.item()]:
                stuff_equiv_classes[label.item()].append(k)

        def get_ids_area(masks, scores, dedup=False):
            # This helper function creates the final panoptic segmentation image
            # It also returns the area of the masks that appears on the image
            m_id = masks.transpose(0, 1).softmax(-1)

            if m_id.shape[-1] == 0:
                # We didn't detect any mask :(
                m_id = torch.zeros((h, w), dtype=torch.long, device=m_id.device)
            else:
                m_id = m_id.argmax(-1).view(h, w)

            final_h, final_w = to_tuple(target_size)
            # print(m_id)
            # m_id shows every pixel belongs to which query
            seg_img = Image.fromarray(id2rgb(m_id.view(h, w).cpu().numpy()))
            seg_img = seg_img.resize(size=(final_w, final_h), resample=Image.NEAREST)

            np_seg_img = (
                torch.ByteTensor(torch.ByteStorage.from_buffer(seg_img.tobytes())).view(final_h, final_w, 3).numpy()
            )
            m_id = torch.from_numpy(rgb2id(np_seg_img))

            area = []
            for i in range(len(scores)):
                area.append(m_id.eq(i).sum().item())
            return area, seg_img

        area, seg_img = get_ids_area(cur_masks, cur_scores, dedup=True)
        if cur_classes.numel() > 0:
            # We know filter empty masks as long as we find some
            while True:
                filtered_small = torch.as_tensor(
                    [area[i] <= 4 for i, c in enumerate(cur_classes)], dtype=torch.bool, device=keep.device
                )
                if filtered_small.any().item():
                    cur_scores = cur_scores[~filtered_small]
                    cur_classes = cur_classes[~filtered_small]
                    cur_masks = cur_masks[~filtered_small]
                    area, seg_img = get_ids_area(cur_masks, cur_scores)
                else:
                    break

        else:
            cur_classes = torch.ones(1, dtype=torch.long, device=cur_classes.device)

        segments_info = []
        for i, a in enumerate(area):
            cat = cur_classes[i].item()
            segments_info.append({"id": i, "isthing": self_is_thing_map[cat], "category_id": cat, "area": a})
        del cur_classes

        with io.BytesIO() as out:
            seg_img.save(out, format="PNG")
            predictions = {"png_string": out.getvalue(), "segments_info": segments_info}
        preds.append(predictions)
    return preds


def test_pipeline():
    path = 'C:/ima/'
    path = 'data/coco/val2017/'
    img_list = glob.glob(path + '*.jpg')
    for i in img_list:
        img_name = 'data/' + i.split('\\')[-1]
        print(img_name)
        im = Image.open(i)
        img, out = predict(im, model, transform)
        predict_attention_full_image(im, out, img_name)


def test_large_scale_mine():
    path = 'C:/ima/'
    txt_name = 'large_scale_test_image_list.txt'
    f = open(path + txt_name, 'r')
    name_list = f.read().split('\n')
    for name in name_list:
        img_name = path + 'IMG_' + name + '.JPG'
        if name == str(2723):
            im = Image.open(img_name)
            img, out = predict(im, model, transform)
            predict_total_mask_instance(img, out, im, img_name.split('_')[-1])


def predict_total_mask_instance(img, out, ori_img, img_name):
    # the post-processor expects as input the target size of the predictions (which we set here to the image size)
    # size.shape = batch_size * 2
    # type(size) = torch.Tensor
    target_size = torch.as_tensor(img.shape[-2:]).unsqueeze(0)
    w, h = target_size.squeeze().numpy().tolist()
    result = list()
    result = postprocessor(result, out, target_size, target_size)

    total = torch.zeros((w, h), dtype=torch.bool)
    tar = numpy.asarray(result[0]['masks'], dtype=numpy.float)
    for i in range(tar.shape[0]):
        x = tar[i][0] != 0
        total = total | x
        # plt.imshow(tar[i][0], cmap='cividis')
        # plt.show()
    # plt.imshow(total, cmap='cividis')
    plt.subplot(121)
    plt.imshow(ori_img)
    plt.subplot(122)
    plt.imshow(total, cmap='cividis')
    # plt.savefig(img_name, dpi=300)
    plt.show()

if __name__ == '__main__':
    im = img_read()
    img, out = predict(im, model, transform)

    # test(img, out)
    # predict_mask(out)
    # predict_total_mask(img, out)
    # predict_attention(out)
    # predict_attention_full_image(im, out, "")
    # predict_attention_full_image_big_size_slow(im, out, "")
    predict_total_mask_instance(img, out, im, 'IMG_9755.jpg')

    # test_pipeline()
    # test_large_scale_mine()
