import cv2
from ultralytics import YOLO
model = YOLO("yolo26n-pose.pt")

cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model(frame, verbose=False)
    r = results[0]
    messages = []
    xy = r.keypoints.xy.cpu().numpy() #converting the pytorch tensor to NumPy plain array
    conf = r.keypoints.conf.cpu().numpy()

    if len(r.keypoints) > 0:
        # extracting coordinates for body parts
        nose = xy[0, 0]
        nose_conf = conf[0, 0]

        right_wrist = xy[0, 10]
        right_wrist_conf = conf[0, 10]

        left_wrist = xy[0, 9]
        left_wrist_conf = conf[0, 9]

        right_shoulder = xy[0, 6]
        right_shoulder_conf = conf[0, 6]

        left_shoulder = xy[0, 5]
        left_shoulder_conf = conf[0, 5]


        if right_wrist_conf > 0.5 and right_shoulder_conf > 0.5:  # right arm raised/down
            if right_wrist[1] < right_shoulder[1]:
                messages.append("right hand raised!")
                
            else:
                messages.append("right hand down")
        else:
            messages.append("can't see right arm clearly")

        if left_wrist_conf > 0.5 and left_shoulder_conf > 0.5:  # left arm raised_down
            if left_wrist[1] < left_shoulder[1]:
                messages.append("left hand raised!")
            else:
                messages.append("left hand down")
        else:
            messages.append("can't see left arm clearly")
    else:
        messages.append("nobody in frame")


    annotated = cv2.flip(r.plot(labels=False),1)
    for i, msg in enumerate(messages):
        y = i *40 + 40
        cv2.putText(annotated, messages[i],(20,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("frame", annotated)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()





