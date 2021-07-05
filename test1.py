import numpy as np
import cv2
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
class_labels = ["Speed limit (20km/h)","Speed limit (30km/h)","Speed limit (50km/h)","Speed limit (60km/h)",
"Speed limit (70km/h)","Speed limit (80km/h)","End of speed limit (80km/h)","Speed limit (100km/h)","Speed limit (120km/h)","No passing","No passing for vechiles over 3.5 metric tons","Right-of-way at the next intersection","Priority road","Yield","Stop","No vechiles","Vechiles over 3.5 metric tons prohibited","No entry","General caution"
,"Dangerous curve to the left","Dangerous curve to the right","Double curve","Bumpy road","Slippery road","Road narrows on the right","Road work","Traffic signals","Pedestrians","Children crossing","Bicycles crossing","Beware of ice/snow","Wild animals crossing","End of all speed and passing limits","Turn right ahead","Turn left ahead"
,"Ahead only,Go straight or right","Go straight or left","Keep right","Keep left","Roundabout mandatory","End of no passing","End of no passing by vechiles over 3.5 metric tons"]
cap = cv2.VideoCapture('./test.mp4')
model = tf.keras.models.load_model("./TSC_model.h5")
writer = None
(W, H) = (None, None)
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque()
# loop over frames from the video file stream
i=0
while True:
    grabbed,frame = cap.read()
    
    (H, W) = frame.shape[:2]
    output = frame.copy()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (32,32)).astype("float32")
    frame -= mean
    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    Q.append(preds)
    results = np.array(Q).mean(axis=0)
    i = np.argmax(results)
    label = class_labels[i]
    text = "activity: {}".format(label)
    cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
        1.25, (0, 255, 0), 5)
    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("output.mp4", fourcc, 1,
            (W, H), True)
    # write the output frame to disk
    writer.write(output)
    #cv2.imsave("Output.mp4", output)
    # show the output image
    cv2.imshow("Output", output)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    # frame = cv2.resize(frame,(400,400))
    # w,h = frame.shape[0:2]

    # w = int(w-(w%4))
    # h = int(h-(h%4))
    
    # temp = np.copy(frame[w//4:w//2,h//4:h//2])
    # frame[w//4:w//2,h//4:h//2] = frame[w//2:(3*w)//4,h//2:(3*h)//4]
    # frame[w//2:(3*w)//4,h//2:(3*h)//4] = temp 

    # img = cv2.rectangle(frame,(),(300,300),(0, 150, 0),4)
    # img = cv2.rectangle(frame,(h//4,w//4),((3*h)//4,(3*w)//4),(0, 150, 0),4)
    # font = cv2.FONT_HERSHEY_COMPLEX
    # img = cv2.putText(img,'Sarat', (h//4,w//4-10),font,1,(0,255,0),2,cv2.LINE_AA)

    # img = np.zeros(frame.shape, dtype='uint8')

    # for i in frame:
    #     for j in i:
    #         j = [j[0],0,0]

    # img = frame
    
    # cv2.imshow('Video', frame)

    # if cv2.waitKey(1) == ord('q'):
    #     # cv2.imwrite('pic.jpg',frame)
    #     cv2.destroyAllWindows()
    #     break

cap.release()