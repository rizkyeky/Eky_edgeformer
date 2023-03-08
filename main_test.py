import os
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint
import torch
import main_dec
import random
import xml.etree.ElementTree as ET
import cv2
import numpy as np

def extract_xml(file, _dir):

    boxes = []
    scores = []
    labels = []

    xml_file = file[:-3] + 'xml'
    tree = ET.parse(_dir + '/' + xml_file)
    root = tree.getroot()

    for obj in root.findall('object'):
        name = obj.find('name').text
        label = 0
        if name == 'robot':
            label = 1
        elif name == 'ball':
            label = 2
        elif name == 'goal':
            label = 3

        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        boxes.append([xmin, ymin, xmax, ymax])
        scores.append(1.0)
        labels.append(label)

    return boxes, scores, labels

if __name__ == '__main__':

    preds = []
    targets = []

    target_robot = 0
    target_ball = 0
    target_goal = 0

    pred_robot = 0
    pred_ball = 0
    pred_goal = 0

    CLASSES = ['_', 'robot', 'ball', 'goal']
    COLORS = [(0,0,0), (0, 0, 255), (0, 255, 0), (255, 0, 0)]

    _dir = '/Users/eky/Documents/_SKRIPSI/_dataset/_balance/images/test'
    
    file_list = sorted(os.listdir(_dir))
    image_list = [file for file in file_list if file.endswith('.jpg')]
    random.shuffle(image_list)

    model = main_dec.init_model()
    metric = MeanAveragePrecision()

    with torch.no_grad():
        
        for i, file in enumerate(image_list):
            print(i, file)
            target_boxes, target_scores, target_labels = extract_xml(file, _dir)
            target_robot += target_labels.count(1) 
            target_ball += target_labels.count(2) 
            target_goal += target_labels.count(3) 

            target_boxes = np.array(target_boxes, dtype=np.int16)
            target_scores = np.array(target_scores)
            target_labels = np.array(target_labels)
            
            targets.append({
                'boxes': torch.tensor(target_boxes).to(torch.int16),
                'scores': torch.tensor(target_scores).to(torch.float16),
                'labels': torch.tensor(target_labels).to(torch.int16),
            })

            img = cv2.imread(_dir + '/' + file)
            pred_labels, pred_scores, pred_boxes = main_dec.predict_image(model, img)
            pred_robot += list(pred_labels).count(1) 
            pred_ball += list(pred_labels).count(2) 
            pred_goal += list(pred_labels).count(3) 

            pred_boxes = np.array(pred_boxes, dtype=np.int16)
            pred_scores = np.array(pred_scores)
            pred_labels = np.array(pred_labels)

            preds.append({
                'boxes': torch.tensor(pred_boxes).to(torch.int16),
                'scores': torch.tensor(pred_scores).to(torch.float16),
                'labels': torch.tensor(pred_labels).to(torch.int16),
            })

            # pprint(targets[i])
            # pprint(preds[i])

            # for i, (idx, score, coords) in enumerate(zip(target_labels, target_scores, target_boxes)):
            #     if score > 0.2:
            #         label = "{}: {:.2f}% {}".format(CLASSES[idx], score * 100, i)
            #         startX, startY, endX, endY = coords
            #         cv2.rectangle(img,
            #             (startX, startY), (endX, endY),
            #             COLORS[1], 3
            #         )
            #         y = startY - 15
            #         cv2.putText(img, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tuple(COLORS[1]), 1)

            # for i, (idx, score, coords) in enumerate(zip(pred_labels, pred_scores, pred_boxes)):
            #     if score > 0.2:
            #         label = "{}: {:.2f}% {}".format(CLASSES[idx], score * 100, i)
            #         startX, startY, endX, endY = coords
            #         cv2.rectangle(img,
            #             (startX, startY), (endX, endY),
            #             COLORS[2], 3
            #         )
            #         y = startY - 15
            #         cv2.putText(img, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tuple(COLORS[2]), 1)

            # cv2.imshow('Frame', img)
            # cv2.waitKey(0)

    metric.update(preds, targets)
    result = metric.compute()
    pprint(result)

    print('target_robot:', target_robot)
    print('target_ball:', target_ball)
    print('target_goal:', target_goal)

    print('pred_robot:', pred_robot)
    print('pred_ball:', pred_ball)
    print('pred_goal:', pred_goal)