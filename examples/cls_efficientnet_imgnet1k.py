import json
import time

import torch
import torchvision.transforms as transforms
from torchvision import models

from PIL import Image

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
    
    model.to(device)
    model.eval()  

    data_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    image = Image.open('images_test/basketball.jpg')
    tensor = data_transforms(image).unsqueeze(0)
    tensor.to(device)

    start = time.time()

    outputs = model(tensor)
    _, preds = torch.max(outputs, 1)
    
    with open('labels/imagenet_classes.json') as f:
        classes = json.load(f)
        classes = [v[1] for k, v in classes.items()]
    print(classes[preds.item()])

    end = time.time()
    print(str(round(end - start, 5)) + ' seconds')
    
    
    