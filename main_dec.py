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

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    opts = get_eval_arguments()
    opts = device_setup(opts)

    # with open('labels/ms_coco_81_classes.json') as f:
    #     CLASSES = json.load(f)
    
    CLASSES = ['robot', 'ball', 'goal']
    
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    COLORS = COLORS.astype(np.uint8)
    
    res_h, res_w = tensor_size_from_opts(opts)

    img_transforms = transforms.Compose([
        transforms.Resize((res_h, res_w)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     [0.485, 0.456, 0.406],
        #     [0.229, 0.224, 0.225]
        # )
    ])

    model = get_model(opts)
    print(model)
    model.to(device)
    model.eval()

    if model.training:
        model.eval()
    
    with torch.no_grad():

        img_path = 'images_test/krsbi5.jpg'
        image = Image.open(img_path)
        orig = image.copy()
        draw = ImageDraw.Draw(orig)
        
        orig_h, orig_w = orig.size[1], orig.size[0]
        
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
            prediction: DetectionPredTuple = model.predict(img, is_scaling=False)
        
        boxes = prediction.boxes.cpu().numpy()
        labels = prediction.labels.cpu().numpy()
        scores = prediction.scores.cpu().numpy()

        boxes[..., 0::2] = boxes[..., 0::2] * orig_w
        boxes[..., 1::2] = boxes[..., 1::2] * orig_h
        boxes[..., 0::2] = np.clip(a_min=0, a_max=orig_w, a=boxes[..., 0::2])
        boxes[..., 1::2] = np.clip(a_min=0, a_max=orig_h, a=boxes[..., 1::2])

        boxes = boxes.astype(np.uint32)

        for idx, score, coords in zip(labels, scores, boxes):
            idx = idx - 1
            label = "{}: {:.2f}%".format(CLASSES[idx], score * 100)
            startX, startY, endX, endY = coords
            print('label:', label)
            print('coords:', (startX, startY, endX, endY))
            if score > 0.2:
                draw.rectangle(
                    [(startX, startY), (endX, endY)],
                    outline=tuple(COLORS[idx]),
                    width=3
                )
                y = startY - 15 if startY - 15 > 15 else startY + 15
                draw.text((startX, y), label, tuple(COLORS[idx]))

        orig.show()
        
    
    
    