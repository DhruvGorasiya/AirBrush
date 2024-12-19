# import cv2
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# import numpy as np
# from mediapipe import solutions
# from mediapipe.framework.formats import landmark_pb2
# from datetime import datetime


# MARGIN = 10  # pixels
# FONT_SIZE = 1
# FONT_THICKNESS = 1
# HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
# mp_drawing = mp.solutions.drawing_utils

# cap = cv2.VideoCapture(0)

# # fps = cap.get(cv2.CAP_PROP_FPS)
# # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

# BaseOptions = mp.tasks.BaseOptions
# HandLandmarker = mp.tasks.vision.HandLandmarker
# HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
# HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
# VisionRunningMode = mp.tasks.vision.RunningMode
# lm_list = HandLandmarkerResult([], [], [])
# def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
#     global lm_list
#     print('hand landmarker result: {}'.format(result))
#     lm_list = result
#     # print("========================================================================================================================================")
#     # print("lm_list: ", lm_list)
#     # print("========================================================================================================================================")
#     # return result

# # Create a hand landmarker instance with the video mode:
# options = HandLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
#     running_mode=VisionRunningMode.LIVE_STREAM,
#     result_callback=print_result)

# def numpy_frame_from_opencv(i):
#     return np.array(i)

# def draw_landmarks_on_image(rgb_image, detection_result: HandLandmarkerResult):
# #   print("hand_landmarks: ---------------------------", detection_result[0])
#   hand_landmarks_list = detection_result.hand_landmarks
#   handedness_list = detection_result.handedness
#   # rgb_image = np.array(rgb_image)
#   # print("rgb_image", rgb_image)
#   annotated_image = np.copy(rgb_image)

#   # Loop through the detected hands to visualize.
#   for idx in range(len(hand_landmarks_list)):
#     hand_landmarks = hand_landmarks_list[idx]
#     handedness = handedness_list[idx]

#     # Draw the hand landmarks.
#     hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#     hand_landmarks_proto.landmark.extend([
#       landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
#     ])
#     solutions.drawing_utils.draw_landmarks(
#       annotated_image,
#       hand_landmarks_proto,
#       solutions.hands.HAND_CONNECTIONS,
#       solutions.drawing_styles.get_default_hand_landmarks_style(),
#       solutions.drawing_styles.get_default_hand_connections_style())

#     # Get the top left corner of the detected hand's bounding box.
#     height, width, _ = annotated_image.shape
#     x_coordinates = [landmark.x for landmark in hand_landmarks]
#     y_coordinates = [landmark.y for landmark in hand_landmarks]
#     text_x = int(min(x_coordinates) * width)
#     text_y = int(min(y_coordinates) * height) - MARGIN

#     # Draw handedness (left or right hand) on the image.
#     cv2.putText(annotated_image, f"{handedness[0].category_name}",
#                 (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
#                 FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

#   return annotated_image

# start_time = datetime.now()

# while True:

#     ret,frame= cap.read()
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv(frame))

#     with HandLandmarker.create_from_options(options) as landmarker:
#         landmarker.detect_async(mp_image, int(((start_time - datetime.now()).total_seconds())*1000))
#         frame_to_show = None

#         print("========================================================================================================================================")
#         print(lm_list, type(lm_list))
#         if lm_list.hand_landmarks:
#             print("mp_image",mp_image)
#             print("========================================================================================================================================")
#             annotated_image = draw_landmarks_on_image(frame, lm_list)
#             print(annotated_image)
#             frame_to_show = annotated_image
#         else:
#             frame_to_show = frame

#         cv2.imshow('Our live sketch',frame_to_show)

#     if cv2.waitKey(1) == 27:
#         break

# cap.release()

import mediapipe as mp
import cv2
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

pts = [[]]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # image = cv2.flip(_image,1)

    results = hands.process(image)

    # Draw the hand landmarks on the image

    # print(frame.shape)

    if results.multi_hand_landmarks:
        # print(len(results.multi_hand_landmarks))
        if len(results.multi_hand_landmarks) < 2:
            for hand_landmarks in results.multi_hand_landmarks:

                fx, fy = int(hand_landmarks.landmark[8].x * frame.shape[1]), int(
                    hand_landmarks.landmark[8].y * frame.shape[0]
                )
                tx, ty = int(hand_landmarks.landmark[4].x * frame.shape[1]), int(
                    hand_landmarks.landmark[4].y * frame.shape[0]
                )

                distance = int(math.sqrt((fx - tx) ** 2 + (fy - ty) ** 2))

                print(distance)

                cv2.circle(frame, (fx, fy), 5, (255, 0, 255), cv2.FILLED)

                if distance < 80:
                    pts[-1].append((fx, fy))
                    if len(pts[-1]) > 1:
                        for i in range(1, len(pts[-1])):
                            cv2.line(frame, pts[-1][i - 1], pts[-1][i], (255, 0, 0), 5)
                else:
                    if len(pts[-1]) != 0:
                        pts.append([])

                # print(hand_landmarks.landmark[1])
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

    for i in pts:
        if len(i) > 1:
            for j in range(1, len(i)):
                cv2.line(frame, i[j - 1], i[j], (255, 0, 0), 5)

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(10) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
