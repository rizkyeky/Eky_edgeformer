import json
from torchvision.models import detection
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2
from ultralytics import YOLO


# data_transforms = transforms.Compose([
#     transforms.ToTensor(),
# ])
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_model():

    # Load a model
    # model = YOLO("examples/yolov8n.onnx")
    
    # model = detection.fasterrcnn_mobilenet_v3_large_320_fpn(
    #     weights='COCO_V1',
    #     progress=True,
    #     weights_backbone='IMAGENET1K_V1',
    # )
    model = detection.ssdlite320_mobilenet_v3_large(
        weights='COCO_V1',
        progress=True,
        weights_backbone='IMAGENET1K_V1',
    )

    # model.to(DEVICE)
    # model.eval()
    return model

def predict_image(model, image):

    # image = np.array(image)
    
    # image = data_transforms(image)
    # image = image.unsqueeze(0)
    # image.to(DEVICE)

    outputs = model(image)
    
    boxes = outputs[0].boxes
    labels = boxes.cls
    scores = boxes.conf
    
    return labels, scores, boxes

if __name__ == "__main__":

    cap = cv2.VideoCapture('images_test/video_test.mp4')

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    with open('labels/ms_coco_81_classes.json') as f:
        CLASSES = json.load(f)
    
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    COLORS = COLORS.astype(np.uint8).tolist()

    model = init_model()

    fps_list = []

    while (cap.isOpened()):

        t1 = cv2.getTickCount()
        ret, frame = cap.read()

        if ret == True:

            # orig = frame
            labels, scores, boxes = predict_image(model, frame)

            for box, label, score in zip(boxes, labels, scores):
        
                if score > 0.5:
                    idx = int(label)+1
                    (startX, startY, endX, endY) = box.xyxy.numpy().squeeze().astype("uint16")
                    
                    text = "{}: {:.2f}%".format(CLASSES[str(idx)], score * 100)
                    cv2.rectangle(frame,
                        (startX, startY), (endX, endY),
                        tuple(COLORS[idx]), 3
                    )
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1, tuple(COLORS[idx]), 2)

            t2 = cv2.getTickCount()
            time_diff = (t2 - t1) / cv2.getTickFrequency()
            fps = 1 / time_diff
            fps_list.append(fps)
            print('{:.2f}'.format(np.mean(fps_list)))

            cv2.putText(frame,'FPS: {:.2f}'.format(fps), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)

            cv2.imshow('Frame', frame)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else: 
            break
        
    cap.release()
    cv2.destroyAllWindows()