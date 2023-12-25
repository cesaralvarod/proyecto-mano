import re
import cv2 as cv
import math
import numpy as np
from mediapipe.python.solutions import hands as mp_hands, drawing_utils as mp_drawing
import matplotlib.pyplot as plt

HAND_LABEL = "Right"
HAND_LANDMARKS = [9, 10, 11, 12]

CLASSIFICATION_REGEX_PATTERN = r'label:\s*"(.*?)"'

alpha = 0.15
alpha_min = 0.01
alpha_max = 0.2
angle_smoothed = 0
previous_angle = None

# suavizados


def smooth_angle(current_angle, dt):
    global previous_angle

    if previous_angle is None:
        return current_angle

    velocity = abs((current_angle - previous_angle) / dt)
    alpha = alpha_min + (alpha_max - alpha_min) * (1.0 * (1.0 + np.exp(-velocity)))
    smoothed_angle = alpha * current_angle + (1 - alpha) * previous_angle
    previous_angle = smoothed_angle

    return smoothed_angle


def calc_bending_angle(base_joint, middle_joint, tip_joint):
    vector_base_to_middle = middle_joint - base_joint
    vector_middle_to_tip = tip_joint - middle_joint
    dot_product = np.dot(vector_base_to_middle, vector_middle_to_tip)
    norm_product = np.linalg.norm(vector_base_to_middle) * np.linalg.norm(
        vector_middle_to_tip
    )
    angle_radians = (
        np.arccos(dot_product / norm_product)
        if np.arccos(dot_product / norm_product) is not None
        else 0
    )
    angle_degrees = np.degrees(angle_radians)
    normalized_angle = angle_degrees if angle_degrees <= 180 else 360 - angle_degrees
    normalized_angle = 0 if normalized_angle is np.NaN else normalized_angle
    normalized_angle = round(normalized_angle, 2)

    return normalized_angle


def calc_hand_dim(keypoints):
    if keypoints is None:
        return 0, 0

    width = abs(min(keypoints[2:4])[0] - keypoints[20][0])
    height = abs(keypoints[0][1] - min(keypoints[4:20])[1])

    return height, width


s_angles = []
m_angles = []
time = list(range(200))


def tracking_landmarks(results, frame):
    global angle_smoothed, s_angles, m_angles

    if results.multi_hand_landmarks is not None:
        # width and height frame
        frame_height, frame_width, _ = frame.shape

        for hand_index, hand in enumerate(results.multi_handedness):
            hand_classification = str(hand.classification[0])
            hand_label_matches = re.findall(
                CLASSIFICATION_REGEX_PATTERN, hand_classification
            )

            # right hand
            if hand_label_matches[mp_hands.HandLandmark.WRIST] != HAND_LABEL:
                cv.putText(
                    frame,
                    "Use la mano derecha",
                    (20, 20),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    mp_drawing.BLUE_COLOR,
                    2,
                )
                continue

            keypoints = []

            hand_landmarks = results.multi_hand_landmarks[hand_index].landmark

            # correct position
            if (
                hand_landmarks[0].y < hand_landmarks[1].y
                or hand_landmarks[0].y < hand_landmarks[5].y
                or hand_landmarks[0].y < hand_landmarks[9].y
                or hand_landmarks[0].y < hand_landmarks[13].y
                or hand_landmarks[0].y < hand_landmarks[17].y
            ):
                cv.putText(
                    frame,
                    "Coloque la mano en una posicion correcta",
                    (20, 20),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    mp_drawing.BLUE_COLOR,
                    2,
                )
                return

            for idx, points in enumerate(hand_landmarks):
                x_real = int(points.x * frame_width)
                y_real = int(points.y * frame_height)

                keypoints.append((x_real, y_real))

                if idx in HAND_LANDMARKS:
                    cv.circle(frame, (x_real, y_real), 2, mp_drawing.RED_COLOR, 2)

            # fingers coordinates
            thumb_coords = np.array([keypoints[2], keypoints[3], keypoints[4]])
            index_coords = np.array([keypoints[5], keypoints[6], keypoints[8]])
            middle_coords = np.array([keypoints[9], keypoints[10], keypoints[12]])
            ring_coords = np.array([keypoints[13], keypoints[14], keypoints[16]])
            pinky_coords = np.array([keypoints[17], keypoints[18], keypoints[20]])

            # angle
            thumb_angle = calc_bending_angle(*thumb_coords)
            index_angle = calc_bending_angle(*index_coords)
            middle_angle = calc_bending_angle(*middle_coords)
            ring_angle = calc_bending_angle(*ring_coords)
            pinky_angle = calc_bending_angle(*pinky_coords)

            # middle_angle = math.ceil(middle_angle/10)
            # middle_angle = middle_angle*10

            angle_smoothed = round(
                alpha * middle_angle + (1 - alpha) * angle_smoothed, 2
            )

            # write angles in screen
            # cv.putText(frame, f"Pulgar: {thumb_angle}",(20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, mp_drawing.BLUE_COLOR, 2)
            # cv.putText(frame, f"Indice: {index_angle}",(20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.6, mp_drawing.BLUE_COLOR, 2)
            # cv.putText(frame, f"Medio: {middle_angle}",(20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, mp_drawing.BLUE_COLOR, 2)
            # cv.putText(frame, f"Anular: {ring_angle}",(20, 80), cv.FONT_HERSHEY_SIMPLEX, 0.6, mp_drawing.BLUE_COLOR, 2)
            # cv.putText(frame, f"Menique: {pinky_angle}",(20, 100), cv.FONT_HERSHEY_SIMPLEX, 0.6, mp_drawing.BLUE_COLOR, 2)
            cv.putText(
                frame,
                f"Suavizado: {angle_smoothed}",
                (20, 120),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                mp_drawing.BLUE_COLOR,
                2,
            )
            cv.putText(
                frame,
                f"Medio: {middle_angle}",
                (200, 120),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                mp_drawing.RED_COLOR,
                2,
            )
            print(angle_smoothed, middle_angle)
            s_angles.append(angle_smoothed)
            m_angles.append(middle_angle)


def main():
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5) as hands:
        cap = cv.VideoCapture(0)

        while True:
            success, cv_frame = cap.read()
            cv_frame = cv.flip(cv_frame, 1)  # mirror cam
            results = hands.process(cv_frame)

            tracking_landmarks(results, cv_frame)

            if success is False:
                print("Error while showing video")
                break

            cv.imshow("Video", cv_frame)

            if cv.waitKey(33) & 0xFF == ord("q"):
                break

        cap.release()

        plt.plot(time, s_angles[0:200], label="Suavizado")
        plt.plot(time, m_angles[0:200], label="Normal")

        plt.show()

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
