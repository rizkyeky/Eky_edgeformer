import cv2
import main_dec
import numpy as np

cap = cv2.VideoCapture('images_test/video_msl.mp4')

if (cap.isOpened()== False): 
    print("Error opening video stream or file")

CLASSES = ['_', 'robot', 'ball', 'goal']

COLORS = [(0,0,0), (0, 0, 255), (0, 255, 0), (255, 0, 0)]

model = main_dec.init_model()

fps_list = []

while (cap.isOpened()):

    t1 = cv2.getTickCount()
    ret, frame = cap.read()

    if ret == True:

        # orig = frame
        labels, scores, boxes = main_dec.predict_image(model, frame)

        # for idx, score, coords in zip(labels, scores, boxes):
        #     if score > 0.2:
        #         label = "{}: {:.2f}%".format(CLASSES[idx], score * 100)
        #         startX, startY, endX, endY = coords
        #         cv2.rectangle(orig,
        #             (startX, startY), (endX, endY),
        #             COLORS[idx], 3
        #         )
        #         y = startY - 15 if startY - 15 > 15 else startY + 15
        #         cv2.putText(orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1, tuple(COLORS[idx]), 2)

        t2 = cv2.getTickCount()
        time_diff = (t2 - t1) / cv2.getTickFrequency()
        fps = 1 / time_diff
        fps_list.append(fps)
        print(np.mean(fps_list))

        # cv2.putText(orig,'FPS: {:.2f}'.format(fps), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)

        # cv2.imshow('Frame', orig)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else: 
        break
    
cap.release()
cv2.destroyAllWindows()

print(np.mean(fps_list))