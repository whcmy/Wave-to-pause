from cvzone.HandTrackingModule import HandDetector
import cv2

cap = cv2.VideoCapture(0)

detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

while True:

    success, img = cap.read()

    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:

        hand1 = hands[0]
        lmList1 = hand1["lmList"]
        bbox1 = hand1["bbox"]
        center1 = hand1['center']
        handType1 = hand1["type"]
        fingers1 = detector.fingersUp(hand1)
        print(f'H1 = {fingers1.count(1)}', end=" ")


        if len(hands) == 2:
            hand2 = hands[1]
            lmList2 = hand2["lmList"]
            bbox2 = hand2["bbox"]
            center2 = hand2['center']
            handType2 = hand2["type"]


            fingers2 = detector.fingersUp(hand2)
            print(f'H2 = {fingers2.count(1)}', end=" ")

        print("")

    cv2.imshow("Image", img)

    cv2.waitKey(1)