import json
from torchvision.models import detection
import torchvision.transforms as transforms
import numpy as np
import torch

from PIL import Image, ImageDraw

class ObjectDetector(torch.nn.Module):
    def __init__(self, baseModel, numClasses):
        super(ObjectDetector, self).__init__()
		# initialize the base model and the number of classes
        self.baseModel = baseModel
        self.numClasses = numClasses
        # build the regressor head for outputting the bounding box
		# coordinates
        self.regressor = torch.nn.Sequential(
			torch.nn.Linear(baseModel.fc.in_features, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 64),
			torch.nn.ReLU(),
			torch.nn.Linear(64, 32),
			torch.nn.ReLU(),
			torch.nn.Linear(32, 4),
			torch.nn.Sigmoid()
        )
        # build the classifier head to predict the class labels
        self.classifier = torch.nn.Sequential(
			torch.nn.Linear(baseModel.fc.in_features, 512),
			torch.nn.ReLU(),
			torch.nn.Dropout(),
			torch.nn.Linear(512, 512),
			torch.nn.ReLU(),
			torch.nn.Dropout(),
			torch.nn.Linear(512, self.numClasses)
        )
		# set the classifier of our base model to produce outputs
		# from the last convolution block
        self.baseModel.fc = torch.nn.Identity()
        
    def forward(self, x):
		# pass the inputs through the base model and then obtain
		# predictions from two different branches of the network
        features = self.baseModel(x)
        bboxes = self.regressor(features)
        classLogits = self.classifier(features)
		# return the outputs as a tuple
        return (bboxes, classLogits)

if __name__ == "__main__":

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    CLASSES = ['cat', 'dog']
    
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    COLORS = COLORS.astype(np.uint8)
    
    model = torch.load('pretrained/dec_modelresnet_tlcatdog.pth', map_location=DEVICE)

    model.to(DEVICE)
    model.eval()

    image = Image.open('images_test/catdog3.jpg')
    orig = image.copy()
    draw = ImageDraw.Draw(orig)

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    image = data_transforms(image)
    image = image.unsqueeze(0)
    image.to(DEVICE)

    (boxPreds, labelPreds) = model(image)
    print(len(boxPreds), 'boxes found')
    print(len(labelPreds), 'labels found')
    (startX, startY, endX, endY) = boxPreds[0]
    (startX, startY, endX, endY) = (startX.item(), startY.item(), endX.item(), endY.item())
    height, width = orig.size
    startX = int(startX * width)
    startY = int(startY * height)
    endX = int(endX * width)
    endY = int(endY * height)

    print(startX, startY, endX, endY) 

    # determine the class label with the largest predicted
	# probability
    labelPreds = torch.nn.Softmax(dim=-1)(labelPreds)
    i = labelPreds.argmax(dim=-1).cpu()
    print(i)
    
    score = 1
    for i in range(len(boxPreds)):
        
        if score > 0.5:
            
            idx = i
            # box = box.detach().cpu().numpy()
            # (startX, startY, endX, endY) = box.astype("uint16")
            
            label = "{}: {:.2f}%".format(CLASSES[idx], score * 100)
            
            draw.rectangle(
                [(startX, startY), (endX, endY)],
                outline=tuple(COLORS[idx]),
                width=3
            )
            y = startY - 15 if startY - 15 > 15 else startY + 15
            draw.text((startX, y), label, tuple(COLORS[idx]))
    
    orig.show()