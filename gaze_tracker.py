import cv2
import numpy as np
import dlib
import csv
import time
from datetime import datetime

class GazeTracker:

    capture = cv2.VideoCapture(0)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    font = cv2.FONT_ITALIC

    gaze_left = 0.7
    gaze_right = 2


    def nothing(x):
        print()


    def get_gaze_ratio(frame, gray, eye_points, facial_landmarks):
        left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                    (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                    (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                    (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                    (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                    (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)],
                                   np.int32)

        cv2.polylines(frame, [left_eye_region], True, 255, 2)

        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.polylines(mask, [left_eye_region], True, 255, 2)
        cv2.fillPoly(mask, [left_eye_region], 255)
        eye = cv2.bitwise_and(gray, gray, mask=mask)

        min_x = np.min(left_eye_region[:, 0])
        max_x = np.max(left_eye_region[:, 0])
        min_y = np.min(left_eye_region[:, 1])
        max_y = np.max(left_eye_region[:, 1])
        gray_eye = eye[min_y: max_y, min_x: max_x]
        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
        height, width = threshold_eye.shape
        left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
        left_side_white = cv2.countNonZero(left_side_threshold)
        right_side_threshold = threshold_eye[0: height, int(width / 2): width]
        right_side_white = cv2.countNonZero(right_side_threshold)

        if left_side_white == 0:
            gaze_ratio = 1
        elif right_side_white == 0:
            gaze_ratio = 5
        else:
            gaze_ratio = left_side_white / right_side_white

        return gaze_ratio


    def analyse_capture(self):
        cv2.namedWindow('Frame')

        cv2.createTrackbar('L', 'Frame', 0, 200, GazeTracker.nothing)
        cv2.createTrackbar('R', 'Frame', 200, 350, GazeTracker.nothing)

        cv2.setTrackbarPos('L', 'Frame', 70)
        cv2.setTrackbarPos('R', 'Frame', 200)
        time_right = 0
        time_center = 0
        time_left = 0

        while True:
            _, frame = capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            pos_a = cv2.getTrackbarPos('L', 'Frame')
            pos_b = cv2.getTrackbarPos('R', 'Frame')

            faces = detector(gray)
            for face in faces:
                x, y = face.left(), face.top()
                x1, y1 = face.right(), face.bottom()
                t = cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 1)

                cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 1)
                landmarks = predictor(gray, face)
                # Gaze detection
                gaze_ratio_left_eye = get_gaze_ratio(frame, gray, [36, 37, 38, 39, 40, 41], landmarks)
                gaze_ratio_right_eye = get_gaze_ratio(frame, gray, [42, 43, 44, 45, 46, 47], landmarks)
                gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

                gaze_left = cv2.getTrackbarPos('L', 'Frame') / 100
                gaze_right = cv2.getTrackbarPos('R', 'Frame') / 100
                # cv2.putText(frame, str(gaze_ratio), (0, 100), font, 1, (0, 0, 255), 2)
                new_frame = np.zeros((500, 500, 3), np.uint8)
                start = time.time()
                print(gaze_ratio)
                if gaze_ratio >= gaze_right:
                    cv2.putText(frame, "LEFT", (x - 100, y), font, 1, (0, 0, 255), 2)
                    new_frame[:] = (0, 0, 255)
                    end1 = time.time()
                    time_right += (end1 - start) * 100
                elif gaze_right > gaze_ratio > gaze_left:
                    cv2.putText(frame, "CENTER", (x - 100, y), font, 1, (0, 0, 255), 2)
                    end2 = time.time()
                    time_center += (end2 - start) * 1000
                else:
                    new_frame[:] = (255, 0, 0)
                    cv2.putText(frame, "RIGHT", (x - 100, y), font, 1, (0, 0, 255), 2)
                    end3 = time.time()
                    time_left += (end3 - start) * 100

            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                now = datetime.now()
                d3 = dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
                with open(d3 + '.csv', 'w', newline='') as f:
                    thewriter = csv.writer(f)
                    thewriter.writerow(['Gaze direction:', 'Time (s):'])
                    thewriter.writerow(['Left', time_left])
                    thewriter.writerow(['Right', time_right])
                    thewriter.writerow(['Center', time_center])

                    break
                    capture.release()
                    cv2.destroyAllWindows()


    def analyse_video(filename):
        time_right = 0
        time_center = 0
        time_left = 0
        cap = cv2.VideoCapture(filename)  # load the video
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while cap.isOpened():  # play the video by reading frame by frame
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = GazeTracker.detector(gray)
                for face in faces:
                    x, y = face.left(), face.top()
                    x1, y1 = face.right(), face.bottom()
                    t = cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 1)

                    cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 1)
                    landmarks = GazeTracker.predictor(gray, face)
                    # Gaze detection
                    gaze_ratio_left_eye = GazeTracker.get_gaze_ratio(frame, gray, [36, 37, 38, 39, 40, 41], landmarks)
                    gaze_ratio_right_eye = GazeTracker.get_gaze_ratio(frame, gray, [42, 43, 44, 45, 46, 47], landmarks)
                    gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2
                    new_frame = np.zeros((500, 500, 3), np.uint8)
                    start = time.time()
                    if gaze_ratio >= GazeTracker.gaze_right:
                        cv2.putText(frame, "LEFT", (x - 100, y), GazeTracker.font, 1, (0, 0, 255), 2)
                        new_frame[:] = (0, 0, 255)
                        end1 = time.time()
                        time_right += (end1 - start) * 100
                    elif GazeTracker.gaze_right > gaze_ratio > GazeTracker.gaze_left:
                        cv2.putText(frame, "CENTER", (x - 100, y), GazeTracker.font, 1, (0, 0, 255), 2)
                        end2 = time.time()
                        time_center += (end2 - start) * 1000
                    else:
                        new_frame[:] = (255, 0, 0)
                        cv2.putText(frame, "RIGHT", (x - 100, y), GazeTracker.font, 1, (0, 0, 255), 2)
                        end3 = time.time()
                        time_left += (end3 - start) * 100

                    print([time_right, time_center, time_left])
            else:

                value = time_center / (time_right+time_left+time_center)

                if value > 0.6:
                    return 'Low'
                else:
                    return 'High'


    def analyse_csv(filename):
        n = []
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                n.append(row[1])
        value = float(n[3]) / (float(n[1])+float(n[2])+float(n[3]))
        if value > 0.6:
            return 'Low'
        else:
            return 'High'
