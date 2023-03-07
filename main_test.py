import os
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint
import torch
# from tqdm import tqdm
import main_dec
from torchvision import transforms
import random
import xml.etree.ElementTree as ET
import cv2
import numpy as np

def extract_xml(batch, _dir):
    boxes_batch = []
    scores_batch = []
    labels_batch = []
    for file in batch:
        xml_file = file[:-3] + 'xml'
        tree = ET.parse(_dir + '/' + xml_file)
        root = tree.getroot()

        boxes = []
        scores = []
        labels = []

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
            boxes.append([xmax, ymax, xmin, ymin])
            scores.append(1.0)
            labels.append(label)
        
        # print(xml_file)
        # print(np.array(boxes).shape)

        boxes_batch.append(boxes)
        scores_batch.append(scores)
        labels_batch.append(labels)

    return boxes_batch, scores_batch, labels_batch

if __name__ == '__main__':

    preds = []
    targets = []
        
    BATCH_SIZE = 32
        
    _dir = '/Users/eky/Documents/_SKRIPSI/_dataset/_balance/images/test'
    # ann_dir = '/Users/eky/Documents/_SKRIPSI/_dataset/_balance/annotations/test'

    file_list = sorted(os.listdir(_dir))
    image_list = [file for file in file_list if file.endswith('.jpg')]
    random.shuffle(image_list)

    image_batchs = [image_list[i:i + BATCH_SIZE] for i in range(0, len(image_list), BATCH_SIZE)]
    random.shuffle(image_batchs)

    # print(len(image_batchs))

    model = main_dec.init_model()
    metric = MeanAveragePrecision()

    with torch.no_grad():
        
        for i, batch in enumerate(image_batchs):
            print('batch:', i+1)

            target_boxes, target_scores, target_labels = extract_xml(batch, _dir)

            for boxes, scores, labels in zip(target_boxes, target_scores, target_labels):
                targets.append({
                    'boxes': torch.tensor(boxes).to(torch.int16),
                    'scores': torch.tensor(scores),
                    'labels': torch.tensor(labels),
                })

            pred_labels, pred_scores, pred_boxes = main_dec.predict_image_batch(model, batch, _dir)
            
            # print(type(labels), type(scores), type(boxes))
            for boxes, scores, labels in zip(pred_boxes, pred_scores, pred_labels):
                preds.append({
                    'boxes': torch.tensor(boxes),
                    'scores': torch.tensor(scores),
                    'labels': torch.tensor(labels),
                })

            pprint(preds[0])
            pprint(targets[0])


    metric.update(preds, targets)
    result = metric.compute()
    print(result['map'])
    # pprint()