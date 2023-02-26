import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity,
                                        self.detectionCon, self.trackCon, )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
        # for id, lm in enumerate(handLms.landmark):
        # print(id,lm)
        # height, width, channels = img.shapea
        # cx, cy = int(lm.x * width), int(lm.y * height)
        # # if id==0:
        # print(id, cx, cy)
        # # //if id ==0: if we want to highlight a particular point then this will circle the thingy.
        # cv2.circle(img, (cx, cy), 12, (255, 255, 255), cv2.FILLED)


def main():
    prevTime = 0
    currentTime = 0
    cap = cv2.VideoCapture(0)

    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)

        currentTime = time.time()
        fps = 1 / (currentTime - prevTime)
        prevTime = currentTime
        cv2.putText(img, str(int(fps)), (15, 55), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 4)

        cv2.imshow("WebCamFeedLive", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
