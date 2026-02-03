# Systems & Technology Research LLC (DBA STR)
# BRIAR Program <briar@str.us>
# Copyright 2021-2023 STR
# Use of this software is governed by the LICENSE file.

import os
import time
import sys

import numpy as np  # type: ignore
import torch  # type: ignore
from PIL import Image  # type: ignore
from torch import nn

# import argparse
# import torch
# import numpy as np
import cv2
import torch.nn.functional as F
from itertools import product as product
from math import ceil
import torchvision.models._utils as _utils

from loguru import logger

# import time


def initialize_model(model_path):

    checkpoint_filename = model_path
    if not os.path.exists(checkpoint_filename):
        logger.error(f'This is broken - {checkpoint_filename} not found')
        sys.exit(0)

    cfg = {
        'name': 'Resnet50',
        'min_sizes': [[16, 32], [64, 128], [256, 512]],
        'steps': [8, 16, 32],
        'variance': [0.1, 0.2],
        'clip': False,
        'loc_weight': 2.0,
        'gpu_train': True,
        'batch_size': 24,
        'ngpu': 2,
        'epoch': 100,
        'decay1': 70,
        'decay2': 90,
        'image_size': 840,
        'pretrain': True,
        'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
        'in_channel': 256,
        'out_channel': 256,
        'multi_task': False,
        'visualize': True,
    }

    model = RetinaFace(cfg=cfg, phase='test')
    checkpoint = torch.load(checkpoint_filename)
    if "state_dict" in checkpoint.keys():
        checkpoint = remove_prefix(checkpoint['state_dict'], 'module.')
    else:
        checkpoint = remove_prefix(checkpoint, 'module.')

    check_keys(model, checkpoint)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    model.cuda()
    return model, cfg


# def compute_detections(model, imcv, cfg):
#     resize_scale = 2
#     nms_threshold = 0.4
#     confidence_threshold = 0.02
#     vis_thres = 0.5
#     imcv_ms = imcv - (104, 117, 123)
#     # imcv -= (104, 117, 123)

#     pf = torch.from_numpy(imcv_ms).float().permute(2, 0, 1)
#     pf = pf.unsqueeze(0)
#     pf = pf.cuda()
#     img_resize = F.upsample(
#         pf, size=(pf.size(2) // resize_scale, pf.size(3) // resize_scale), mode='bilinear'
#     )
#     im_height, im_width = img_resize.size(2), img_resize.size(3)
#     scale = torch.Tensor([im_width, im_height, im_width, im_height]).cuda()
#     with torch.no_grad():
#         loc, conf, landms = model(img_resize)  # forward pass

#     priorbox = PriorBox(cfg, image_size=(im_height, im_width))
#     priors = priorbox.forward()
#     priors = priors.to("cuda")
#     prior_data = priors.data
#     boxes = []
#     boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
#     boxes = boxes * scale * resize_scale
#     boxes = boxes.cpu().detach().numpy()
#     scores = conf.squeeze(0).data.cpu().detach().numpy()[:, 1]

#     ## keypoint processing
#     # landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
#     # scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
#     #                    img.shape[3], img.shape[2], img.shape[3], img.shape[2],
#     #                    img.shape[3], img.shape[2]])
#     # scale1 = scale1.to("cuda")
#     # landms = landms * scale1 / resize
#     # landms = landms.cpu().detach().numpy()

#     # ignore low scores
#     inds = np.where(scores > confidence_threshold)[0]
#     boxes = boxes[inds]
#     scores = scores[inds]
#     # keep top-K before NMS
#     order = scores.argsort()[::-1]
#     boxes = boxes[order]
#     scores = scores[order]

#     # do NMS
#     dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
#     keep = py_cpu_nms(dets, nms_threshold)
#     dets = dets[keep, :]
#     bbox = []
#     score = []

#     for b in dets:
#         if b[4] < vis_thres:
#             continue
#         score.append(float(b[4]))
#         b = list(map(int, b[:4]))
#         b1 = b
#         b1[2] = b[2] - b[0]
#         b1[3] = b[3] - b[1]
#         bbox.append(b1)

#     # if none of the detections have confidence higher than vis_thres,
#     # then take the highest confidence score detection
#     b = []
#     if len(bbox) == 0:
#         if len(dets) > 0:
#             # there should be atleast one detection
#             b = dets[0, :]
#             score.append(float(b[4]))
#             final_det = list(map(int, b[:4]))
#             final_det[2] = int(b[2]) - int(b[0])
#             final_det[3] = int(b[3]) - int(b[1])
#             bbox.append(final_det)
#         # else:
#         #    bbox.append(b)
#         #    score.append(b)

#     return bbox, score


def resize_constant_factor(image_ms, resize_scale):
    """Simple constant factor scaling.

    Problem - images and videos were diff sizes
    """
    rescale_method = Image.BOX if resize_scale > 1 else Image.BILINEAR
    w_img, h_img = image_ms.width, image_ms.height
    h_scaled, w_scaled = h_img // resize_scale, w_img // resize_scale
    img_resize = image_ms.resize((w_scaled, h_scaled), resample=rescale_method)
    return img_resize, 1.0 / resize_scale


def resize_to_target_size(image_ms, target_size, max_size):
    """This approach was used in Briar Phase 1."""
    w_img, h_img = image_ms.width, image_ms.height
    im_size_min = np.min((w_img, h_img))
    im_size_max = np.max((w_img, h_img))
    resize = float(target_size) / float(im_size_min)
    if np.round(resize * im_size_max) > max_size:
        resize = float(max_size) / float(im_size_max)
    if resize != 1:
        img_resize = cv2.resize(
            np.asarray(image_ms),
            None,
            None,
            fx=resize,
            fy=resize,
            interpolation=cv2.INTER_LINEAR,
        )
        return Image.fromarray(img_resize), resize
    else:
        return image_ms, resize


def resize_to_target_size_with_padding(image_ms, target_size, max_size):
    """This approach was used in Briar Phase 2."""
    w_img, h_img = image_ms.width, image_ms.height
    # im_size_max = np.max((w_img, h_img))
    # resize = float(target_size) / float(im_size_max)
    im_size_min = np.min((w_img, h_img))
    im_size_max = np.max((w_img, h_img))
    resize = float(target_size) / float(im_size_min)
    if np.round(resize * im_size_max) > max_size:
        resize = float(max_size) / float(im_size_max)
    if resize < 1:  # downsample the image
        img_resize = cv2.resize(
            np.asarray(image_ms),
            None,
            None,
            fx=resize,
            fy=resize,
            interpolation=cv2.INTER_LINEAR,
        )
        return Image.fromarray(img_resize), resize
    # elif resize > 1:  # pad the image
    #     new_h = np.round(h_img * resize)
    #     new_w = np.round(w_img * resize)
    #     img_resize = image_io.pad_image_to_size(image_ms, new_h, new_w)
    #     resize = 1.0
    else:
        # TODO, may need to test for super small images and pad them, but for now just
        # push them through
        img_resize = image_ms
        resize = 1.0

    return img_resize, resize


def compute_detections_batch(model, imcv, priorbox, batch_size, resize_func):
    """This function computes the detections in batches.

    :param model: The model call.
    :param imcv_batch: list of images to process.
    :return: list of detections
    """
    nms_threshold = 0.4
    confidence_threshold = 0.02
    vis_thres = 0.5

    imcv_batch = []
    for indx in range(len(imcv)):
        image_ms = imcv[indx]
        image_ms = Image.fromarray(image_ms)
        # Phase 1 - first implementation
        # img_resize, resize_factor = resize_constant_factor(image_ms, 2)
        # Phase 1 - end of phase implementation
        # img_resize, resize_factor = resize_to_target_size(image_ms, 1600, 2150)
        # Phase 2 - added padding
        # img_resize, resize_factor = resize_to_target_size_with_padding(image_ms, 1600, 2150)
        img_resize, resize_factor = resize_func(image_ms, 1600, 2150)

        # 2000, 3000 for jpg
        im_height, im_width = img_resize.height, img_resize.width
        img_resize = np.asarray(img_resize) - (104, 117, 123)
        img_resize = torch.from_numpy(img_resize).float().permute(2, 0, 1)
        img_resize = img_resize.unsqueeze(0)
        im_height, im_width = img_resize.size(2), img_resize.size(3)
        if indx == 0:
            imcv_batch = img_resize.cuda()
        else:
            imcv_batch = torch.cat((imcv_batch, img_resize.cuda()), 0)

    with torch.no_grad():
        loc_0, conf_0, landms_0 = model(imcv_batch)  # forward pass
        imcv_batch = None

    priorbox.set_image_size(image_size=(im_height, im_width))
    priors = priorbox.forward("cuda")
    prior_data = priors.data

    bbox_final = []
    score_final = []
    keypts_final = []
    scale1 = None
    scale = [im_width, im_height, im_width, im_height]

    for indy in range(batch_size):
        loc = loc_0[indy, :]
        conf = conf_0[indy, :]
        landms = landms_0[indy, :]

        boxes = []
        boxes = decode(loc.data.squeeze(0), prior_data, priorbox.variance)
        boxes = boxes.cpu().detach().numpy()
        boxes = boxes * scale / resize_factor
        scores = conf.squeeze(0).data.cpu().detach().numpy()[:, 1]

        # keypoint processing
        landms = decode_landm(landms.data.squeeze(0), prior_data, priorbox.variance)
        if scale1 is None:
            scale1 = [
                im_width,
                im_height,
                im_width,
                im_height,
                im_width,
                im_height,
                im_width,
                im_height,
                im_width,
                im_height,
            ]
            # scale1 = scale1.to("cuda")
        landms = landms.cpu().detach().numpy()
        landms = landms * scale1 / resize_factor

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]
        landms = landms[inds]
        # keep top-K before NMS
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        scores = scores[order]
        landms = landms[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        dets = np.concatenate((dets, landms), axis=1)

        bbox = []
        score = []
        keypts = []

        for b in dets:
            if b[4] < vis_thres:
                continue
            score.append(float(b[4]))
            k = list(map(float, b[5:]))
            b0 = list(map(int, b[:4]))
            b1 = b0
            b1[2] = b0[2] - b0[0]
            b1[3] = b0[3] - b0[1]
            bbox.append(b1)
            keypts.append(k)

        # if none of the detections have confidence higher than vis_thres,
        # then take the highest confidence score detection
        b = []
        if len(bbox) == 0:
            if len(dets) > 0:
                # there should be atleast one detection
                b = dets[0, :]
                score.append(float(b[4]))
                final_det = list(map(int, b[:4]))
                final_det[2] = int(b[2]) - int(b[0])
                final_det[3] = int(b[3]) - int(b[1])
                bbox.append(final_det)
                keypts.append(list(map(float, b[5:])))

        bbox_final.append(bbox)
        score_final.append(score)
        keypts_final.append(keypts)

    return bbox_final, score_final, keypts_final


def compute_detections_batch_ver3(model, imcv, priorbox, batch_size):
    """This function computes the detections in batches. This version does.

    :param model: The model call.
    :param imcv_batch: list of images to process.
    :return: list of detections
    """
    nms_threshold = 0.4
    confidence_threshold = 0.02
    vis_thres = 0.5
    target_size = 1600
    max_size = 2150

    # reference_pt = get_reference_facial_points(default_square=True)
    imcv_batch = []
    for indx in range(len(imcv)):
        image_ms = imcv[indx]
        image_ms = Image.fromarray(image_ms)
        w_img, h_img = image_ms.width, image_ms.height
        im_size_min = np.min((w_img, h_img))
        im_size_max = np.max((w_img, h_img))
        resize = float(target_size) / float(im_size_min)
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
            # if args.origin_size:
            #     resize = 1
        if resize != 1:
            img_resize = cv2.resize(
                np.asarray(image_ms),
                None,
                None,
                fx=resize,
                fy=resize,
                interpolation=cv2.INTER_LINEAR,
            )

        # h_scaled, w_scaled = h_img // resize_scale, w_img // resize_scale
        # img_resize = image_ms.resize((w_scaled, h_scaled), resample=rescale_method)

        # pf = torch.from_numpy(image_ms).float().permute(2, 0, 1)
        # pf = pf.unsqueeze(0)
        # pf = pf.cuda()
        # img_resize = F.upsample(pf, size=(pf.size(2) // resize_scale, pf.size(3) // resize_scale),
        #                         mode='bilinear')

        # 2000, 3000 for jpg
        img_resize = np.asarray(img_resize) - (104, 117, 123)
        img_resize = torch.from_numpy(img_resize).float().permute(2, 0, 1)
        img_resize = img_resize.unsqueeze(0)
        im_height, im_width = img_resize.size(2), img_resize.size(3)
        if indx == 0:
            imcv_batch = img_resize.cuda()
        else:
            imcv_batch = torch.cat((imcv_batch, img_resize.cuda()), 0)

    with torch.no_grad():
        loc_0, conf_0, landms_0 = model(imcv_batch)  # forward pass
        imcv_batch = None

    priorbox.set_image_size(image_size=(im_height, im_width))
    priors = priorbox.forward("cuda")
    prior_data = priors.data

    bbox_final = []
    score_final = []
    keypts_final = []
    scale1 = None
    scale = [im_width, im_height, im_width, im_height]

    for indy in range(batch_size):
        loc = loc_0[indy, :]
        conf = conf_0[indy, :]
        landms = landms_0[indy, :]

        boxes = []
        boxes = decode(loc.data.squeeze(0), prior_data, priorbox.variance)
        boxes = boxes.cpu().detach().numpy()
        boxes = boxes * scale / resize
        scores = conf.squeeze(0).data.cpu().detach().numpy()[:, 1]

        # keypoint processing
        landms = decode_landm(landms.data.squeeze(0), prior_data, priorbox.variance)
        if scale1 is None:
            scale1 = [
                im_width,
                im_height,
                im_width,
                im_height,
                im_width,
                im_height,
                im_width,
                im_height,
                im_width,
                im_height,
            ]
            # scale1 = scale1.to("cuda")
        landms = landms.cpu().detach().numpy()
        landms = landms * scale1 / resize

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]
        landms = landms[inds]
        # keep top-K before NMS
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        scores = scores[order]
        landms = landms[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        dets = np.concatenate((dets, landms), axis=1)

        bbox = []
        score = []
        keypts = []

        for b in dets:
            if b[4] < vis_thres:
                continue
            score.append(float(b[4]))
            k = list(map(float, b[5:]))
            b0 = list(map(int, b[:4]))
            b1 = b0
            b1[2] = b0[2] - b0[0]
            b1[3] = b0[3] - b0[1]
            bbox.append(b1)
            keypts.append(k)

        # if none of the detections have confidence higher than vis_thres,
        # then take the highest confidence score detection
        b = []
        if len(bbox) == 0:
            if len(dets) > 0:
                # there should be atleast one detection
                b = dets[0, :]
                score.append(float(b[4]))
                final_det = list(map(int, b[:4]))
                final_det[2] = int(b[2]) - int(b[0])
                final_det[3] = int(b[3]) - int(b[1])
                bbox.append(final_det)
                keypts.append(list(map(float, b[5:])))

        bbox_final.append(bbox)
        score_final.append(score)
        keypts_final.append(keypts)

    return bbox_final, score_final, keypts_final


def conv_bn(inp, oup, stride=1, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if out_channel <= 64:
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        leaky = 0
        if out_channels <= 64:
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride=1, leaky=leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky=0.1),  # 3
            conv_dw(8, 16, 1),  # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1),  # 59 + 32 = 91
            conv_dw(128, 128, 1),  # 91 + 32 = 123
            conv_dw(128, 128, 1),  # 123 + 32 = 155
            conv_dw(128, 128, 1),  # 155 + 32 = 187
            conv_dw(128, 128, 1),  # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2),  # 219 +3 2 = 241
            conv_dw(256, 256, 1),  # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x


class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(
            inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0
        )

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(
            inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0
        )

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(
            inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0
        )

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 10)


class HeadPoseHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(HeadPoseHead, self).__init__()
        self.conv1x1 = nn.Conv2d(
            inchannels, num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0
        )

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 2)


class GenderHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(GenderHead, self).__init__()
        self.conv1x1 = nn.Conv2d(
            inchannels, num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0
        )

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 2)


class AgeHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(AgeHead, self).__init__()
        self.conv1x1 = nn.Conv2d(
            inchannels, num_anchors * 1, kernel_size=(1, 1), stride=1, padding=0
        )

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 1)


class RestoreHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=3):
        super(RestoreHead, self).__init__()
        self.conv1x1 = nn.Conv2d(
            inchannels, num_anchors * 128, kernel_size=(1, 1), stride=1, padding=0
        )

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 1)


class RetinaFace(nn.Module):
    def __init__(self, cfg=None, phase='train', restore=False):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace, self).__init__()
        self.phase = phase
        self.multi_task = cfg['multi_task']
        self.restore = restore
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load(
                    "./weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu')
                )
                from collections import OrderedDict

                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models

            # backbone = models.resnet50(pretrained=cfg['pretrain'])
            if cfg['pretrain']:
                backbone = models.resnet50(weights="DEFAULT")
            else:
                backbone = models.resnet50(weights=None)

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])
        if self.multi_task:
            self.HeadPoseHead = self._make_headpose_head(fpn_num=3, inchannels=cfg['out_channel'])
            self.GenderHead = self._make_gender_head(fpn_num=3, inchannels=cfg['out_channel'])
            self.AgeHead = self._make_age_head(fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels, anchor_num))
        return classhead

    def _make_bbox_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels, anchor_num))
        return bboxhead

    def _make_landmark_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def _make_headpose_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        headposehead = nn.ModuleList()
        for i in range(fpn_num):
            headposehead.append(HeadPoseHead(inchannels, anchor_num))
        return headposehead

    def _make_gender_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        genderhead = nn.ModuleList()
        for i in range(fpn_num):
            genderhead.append(GenderHead(inchannels, anchor_num))
        return genderhead

    def _make_age_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        agehead = nn.ModuleList()
        for i in range(fpn_num):
            agehead.append(AgeHead(inchannels, anchor_num))
        return agehead

    def _make_restore_head(self, fpn_num=3, inchannels=64, anchor_num=2):
        restorehead = nn.ModuleList()
        for i in range(fpn_num):
            restorehead.append(RestoreHead(inchannels, anchor_num))
        return restorehead

    def forward(self, inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat(
            [self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1
        )
        classifications = torch.cat(
            [self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1
        )
        ldm_regressions = torch.cat(
            [self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1
        )
        if self.multi_task:
            headpose_regressions = torch.cat(
                [self.HeadPoseHead[i](feature) for i, feature in enumerate(features)], dim=1
            )
            gender_regressions = torch.cat(
                [self.GenderHead[i](feature) for i, feature in enumerate(features)], dim=1
            )
            age_regressions = torch.cat(
                [self.AgeHead[i](feature) for i, feature in enumerate(features)], dim=1
            )

        if self.phase == 'train':
            if self.multi_task:
                output = (
                    bbox_regressions,
                    classifications,
                    ldm_regressions,
                    headpose_regressions,
                    gender_regressions,
                    age_regressions,
                )
            else:
                output = (bbox_regressions, classifications, ldm_regressions)
        else:
            if self.multi_task:
                output = (
                    bbox_regressions,
                    F.softmax(classifications, dim=-1),
                    ldm_regressions,
                    headpose_regressions,
                    F.softmax(gender_regressions, dim=-1),
                    age_regressions,
                )
            else:
                output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def clear(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo the encoding we
    did for offset regression at train time.

    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
        ),
        1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo the encoding we did
    for offset regression at train time.

    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = torch.cat(
        (
            priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
        ),
        dim=1,
    )
    return landms


def py_cpu_nms(dets, thresh, is_merge=False):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    if is_merge:
        merge_keep = []
        order_full = []
    # merge_order = order.copy()
    while order.size > 0:
        if is_merge:
            order_full.append(order)
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        # merge_inds = np.where(ovr > thresh)[0]
        if is_merge:
            merge_keep.append(np.setdiff1d(order, order[inds + 1]))
        order = order[inds + 1]
    if is_merge:
        return keep, merge_keep, order_full
    else:
        return keep


class PriorBox(object):
    def __init__(self, cfg, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.variance = cfg['variance']
        self.clip = cfg['clip']
        self.name = "s"
        self.anchors = None
        self.image_size = (2000, 3000)
        self.feature_maps = [
            [ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)]
            for step in self.steps
        ]

    def set_image_size(self, image_size):
        if self.image_size[0] != image_size[0] or self.image_size[1] != image_size[1]:
            self.image_size = image_size
            self.feature_maps = [
                [ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)]
                for step in self.steps
            ]
            self.anchors = None

    def forward(self, device="cuda"):
        if self.anchors is not None:
            return self.anchors

        # else we need to recompute them
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)

        self.anchors = output.to("cuda")
        return self.anchors


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    """Old style model is stored with all names of parameters sharing common
    prefix 'module.'."""
    print('remove prefix \'{}\''.format(prefix))

    def f(x):
        return x.split(prefix, 1)[-1] if x.startswith(prefix) else x

    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device)
        )
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model
