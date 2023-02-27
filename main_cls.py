import json
import sys
import time

sys.argv.extend(['--common.config-file', 'config/parcnet_cls.yaml'])

sys.path.append('parcnet')
from parcnet.cvnets.models.classification.edgeformer import *
from parcnet.options.opts import get_eval_arguments
from parcnet.utils.common_utils import device_setup
from parcnet.cvnets.models import get_model
from parcnet.utils.tensor_utils import tensor_size_from_opts

import torch
from torch.cuda.amp import autocast
import torchvision.transforms as transforms

from PIL import Image

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open('labels/imagenet_classes.json') as f:
        CLASSES = json.load(f)

    opts = get_eval_arguments()
    opts = device_setup(opts)

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

    model.to(device)
    model.eval()  
    
    if model.training:
        model.eval()
    
    with torch.no_grad():
        image = Image.open('images_test/pizza.jpg')
        tensor = img_transforms(image).unsqueeze(0)
        tensor.to(device)

        # start = time.time()

        mixed_precision_training = getattr(opts, "common.mixed_precision", False)
        with autocast(enabled=mixed_precision_training if torch.cuda.is_available() else False):
            output = model(tensor)
        
        pred = output.cpu().squeeze().argmax().item()
        print(pred)

        predicted_idx = CLASSES[str(pred)]
        print(predicted_idx)

        # end = time.time()
        # print(end - start)
    
    
    