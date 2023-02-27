import json
from torchvision.models import detection
import torchvision.transforms as transforms
import numpy as np
import torch

from PIL import Image, ImageDraw

if __name__ == "__main__":

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('labels/ms_coco_91_classes.json') as f:
        CLASSES = json.load(f)
    
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    COLORS = COLORS.astype(np.uint8)
    
    model = detection.ssdlite320_mobilenet_v3_large(
        weights='COCO_V1',
        progress=True,
        weights_backbone='IMAGENET1K_V1',
    )

    model.to(DEVICE)
    model.eval()

    image = Image.open('images_test/catdog2.jpg')
    orig = image.copy()
    draw = ImageDraw.Draw(orig)

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    image = data_transforms(image)
    image = image.unsqueeze(0)
    image.to(DEVICE)

    outputs = model(image)[0]
    
    for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
        
        if score > 0.5:
            
            idx = int(label)-1
            box = box.detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("uint16")
            
            label = "{}: {:.2f}%".format(CLASSES[str(idx)], score * 100)
            
            draw.rectangle(
                [(startX, startY), (endX, endY)],
                outline=tuple(COLORS[idx]),
                width=3
            )
            y = startY - 15 if startY - 15 > 15 else startY + 15
            draw.text((startX, y), label, tuple(COLORS[idx]))
    
    orig.show()