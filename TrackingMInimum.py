import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)

prevTime = 0
currentTime = 0

mpHands = mp.solutions.hands
hands = mpHands.Hands()  # changed the complexity value of hands to 0 to increase FPS.
mpDraw = mp.solutions.drawing_utils


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                height,width,channels = img.shape
                cx,cy = int(lm.x*width),int(lm.y*height)
                #if id==0:
                print(id,cx,cy)
                #//if id ==0: if we want to highlight a particular point then this will circle the thingy.
                cv2.circle(img,(cx,cy),12,(255,255,255),cv2.FILLED)

            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)

    currentTime = time.time()
    fps = 1/(currentTime-prevTime)
    prevTime = currentTime

    cv2.putText(img,str(int(fps)),(15,55),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),4)

    cv2.imshow("WebCamFeed", img)
    cv2.waitKey(1)
