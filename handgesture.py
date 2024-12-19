import mediapipe as mp
import cv2
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

tip_co = [8,12,16,20]
knuck_co = [6,10,14,18]
pts = [[]]

while cap.isOpened():
    stop = []
    _stop = False
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(image)


    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) < 2:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(4):
                    tx,ty = int(hand_landmarks.landmark[tip_co[i]].x * frame.shape[1]), int(hand_landmarks.landmark[tip_co[i]].y * frame.shape[0])
                    kx,ky = int(hand_landmarks.landmark[knuck_co[i]].x * frame.shape[1]), int(hand_landmarks.landmark[knuck_co[i]].y * frame.shape[0])

                    if ty < ky:
                        stop.append(1)

                if not stop:
                    pts = [[]]

                print(stop)

                fx, fy = int(hand_landmarks.landmark[8].x * frame.shape[1]), int(
                    hand_landmarks.landmark[8].y * frame.shape[0]
                )
                tx, ty = int(hand_landmarks.landmark[4].x * frame.shape[1]), int(
                    hand_landmarks.landmark[4].y * frame.shape[0]
                )

                distance = int(math.sqrt((fx - tx) ** 2 + (fy - ty) ** 2))

                cv2.circle(frame, (fx, fy), 5, (255, 0, 255), cv2.FILLED)

                if distance < 80:
                    pts[-1].append((fx, fy))
                    if len(pts[-1]) > 1:
                        for i in range(1, len(pts[-1])):
                            cv2.line(frame, pts[-1][i - 1], pts[-1][i], (255, 0, 0), 5)
                else:
                    if len(pts[-1]) != 0:
                        pts.append([])

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

    for i in pts:
        if len(i) > 1:
            for j in range(1, len(i)):
                cv2.line(frame, i[j - 1], i[j], (255, 0, 0), 5)

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(10) == ord("q") and stop:
        break

cap.release()
cv2.destroyAllWindows()
