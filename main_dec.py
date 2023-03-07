import sys
import time

sys.argv.extend(['--common.config-file', 'config/parcnet_dec_infer.yaml'])

sys.path.append('parcnet')
from parcnet.cvnets.models.detection.ssd import *
from parcnet.cvnets.models.classification.edgeformer import *
from parcnet.options.opts import get_eval_arguments
from parcnet.utils.common_utils import device_setup
from parcnet.cvnets.models import get_model
from parcnet.engine.eval_detection import *
from parcnet.engine.utils import print_summary
from parcnet.utils.tensor_utils import tensor_size_from_opts

import torch
from torch.nn import functional as F
import torchvision.transforms as transforms
from torch.cuda.amp import autocast
import numpy as np
import json

from PIL import Image, ImageDraw

opts = get_eval_arguments()
opts = device_setup(opts)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

res_h, res_w = tensor_size_from_opts(opts)
img_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((res_h, res_w)),
    transforms.ToTensor(),
])

img_pil_transforms = transforms.Compose([
    transforms.Resize((res_h, res_w)),
    transforms.ToTensor(),
])

def init_model():    
    model = get_model(opts)
    model.to(device)
    model.eval()

    if model.training:
        model.eval()
    
    return model

def predict_image_batch(model, batch, _dir):

    images = []
    for file in batch:
        image = Image.open(_dir + '/' + file)
        image = img_pil_transforms(image)
        image = image.cpu().numpy()
        images.append(image)

    labels_batch = []
    scores_batch = []
    boxes_batch = []

    for i, img in enumerate(images):
        labels, scores, boxes = predict_image(model, img, is_batch=True)
        labels_batch.append(labels)
        scores_batch.append(scores)
        boxes_batch.append(boxes)
    
    return labels_batch, scores_batch, boxes_batch

def predict_image(model, image, is_batch=False):
    
    with torch.no_grad():
        image = np.array(image)
        orig_h, orig_w = image.shape[0], image.shape[1]
        
        if is_batch:
            image = torch.from_numpy(image)
            image = image.unsqueeze(0)
        else:
            image = img_transforms(image)
            image = image.unsqueeze(0)

        output_stride = 32
        curr_height, curr_width = image.shape[2:]
        new_h = (curr_height // output_stride) * output_stride
        new_w = (curr_width // output_stride) * output_stride

        if new_h != curr_height or new_w != curr_width:
            image = F.interpolate(input=image, size=(new_h, new_w), mode="bilinear", align_corners=False)

        image.to(device)

        mixed_precision_training = getattr(opts, "common.mixed_precision", False)
        with autocast(enabled=mixed_precision_training if torch.cuda.is_available() else False):
            img = image.cuda() if torch.cuda.is_available() else image.cpu()
            prediction: DetectionPredTuple = model.predict(img, is_scaling=True)
        
        boxes = prediction.boxes.cpu().numpy()
        labels = prediction.labels.cpu().numpy()
        scores = prediction.scores.cpu().numpy()

        boxes[..., 0::2] = boxes[..., 0::2] * orig_w
        boxes[..., 1::2] = boxes[..., 1::2] * orig_h
        boxes[..., 0::2] = np.clip(a_min=0, a_max=orig_w, a=boxes[..., 0::2])
        boxes[..., 1::2] = np.clip(a_min=0, a_max=orig_h, a=boxes[..., 1::2])

        boxes = boxes.astype(np.int16)
        
    return labels, scores, boxes
        
# if __name__ == '__main__':

    # with open('labels/ms_coco_81_classes.json') as f:
    #     CLASSES = json.load(f)

    # model = init_model()

    # torch.onnx.export(model, torch.randn(1, 3, 224, 224), "parcnet.onnx")
    
    # CLASSES = ['_', 'robot', 'ball', 'goal']
    
    # COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    # COLORS = COLORS.astype(np.uint8)

    # img_path = 'images_test/krsbi1.jpg'
    # image = Image.open(img_path)
    # orig = image.copy()
    # draw = ImageDraw.Draw(orig)
    
    # labels, scores, boxes = predict_image(image)

    # for idx, score, coords in zip(labels, scores, boxes):
    #     if score > 0.2:
    #         label = "{}: {:.2f}%".format(CLASSES[idx], score * 100)
    #         startX, startY, endX, endY = coords
    #         # print('label:', label)
    #         # print('coords:', (startX, startY, endX, endY))
    #         draw.rectangle(
    #             [(startX, startY), (endX, endY)],
    #             outline=tuple(COLORS[idx]),
    #             width=3
    #         )
    #         y = startY - 15 if startY - 15 > 15 else startY + 15
    #         draw.text((startX, y), label, tuple(COLORS[idx]))

    # orig.show()
    