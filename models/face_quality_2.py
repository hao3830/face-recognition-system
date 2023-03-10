import cv2
import utils

import mediapipe as mp
import numpy as np

settings = utils.get_face_quality_assessment_config()

mp_face_mesh = mp.solutions.face_mesh

class FaceQuality:
    def __init__(self, min_detection_confidence, threshold_angle, close_eye_thres):

        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=min_detection_confidence,
        )

        self.right_eye = [
            [33, 133],
            [160, 144],
            [159, 145],
            [158, 153],
        ]  # right eye landmark positions
        self.left_eye = [[263, 362], [387, 373], [386, 374], [385, 380]]

        # RULE THRESHOLD
        self.threshold_angle = threshold_angle
        self.close_eye_thres = close_eye_thres

    def distance(self, p1, p2):
        """Calculate distance between two points
        :param p1: First Point
        :param p2: Second Point
        :return: Euclidean distance between the points. (Using only the x and y coordinates).
        """
        return (((p1[:2] - p2[:2]) ** 2).sum()) ** 0.5

    def eye_aspect_ratio(self, landmarks, eye):
        """Calculate the ratio of the eye length to eye width.
        :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
        :param eye: List containing positions which correspond to the eye
        :return: Eye aspect ratio value
        """
        N1 = self.distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
        N2 = self.distance(landmarks[eye[2][0]], landmarks[eye[2][1]])
        N3 = self.distance(landmarks[eye[3][0]], landmarks[eye[3][1]])
        D = self.distance(landmarks[eye[0][0]], landmarks[eye[0][1]])
        return (N1 + N2 + N3) / (3 * D)

    def is_eyes_closed(self, landmarks):
        return (
            self.eye_aspect_ratio(landmarks, self.left_eye)
            + self.eye_aspect_ratio(landmarks, self.right_eye)
        ) / 2 < self.close_eye_thres

    def is_face_tilt(self, face_landmarks, image):
        img_h, img_w, _ = image.shape
        face_3d = []
        face_2d = []
        for idx, lm in enumerate(face_landmarks.landmark):
            if (
                idx == 33
                or idx == 263
                or idx == 1
                or idx == 61
                or idx == 291
                or idx == 199
            ):

                x, y = int(lm.x * img_w), int(lm.y * img_h)

                # Get the 2D Coordinates
                face_2d.append([x, y])

                # Get the 3D Coordinates
                face_3d.append([x, y, lm.z])

        # Convert it to the NumPy array
        face_2d = np.array(face_2d, dtype=np.float64)

        # Convert it to the NumPy array
        face_3d = np.array(face_3d, dtype=np.float64)

        # The camera matrix
        focal_length = 1 * img_w

        cam_matrix = np.array(
            [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
        )

        # The distortion parameters
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        _, rot_vec, _ = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        # Get rotational matrix
        rmat, _ = cv2.Rodrigues(rot_vec)

        # Get angles
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        # Get the y rotation degree
        x = angles[0] * 360
        y = angles[1] * 360
        return abs(x) > self.threshold_angle or abs(y) > self.threshold_angle

    def is_smile(self, face_landmarks):
        nose_tip = face_landmarks.landmark[6]
        mouth_landmarks = []
        for i in range(48, 68):
            mouth_landmarks.append(face_landmarks.landmark[i])
        mouth_width = (mouth_landmarks[6].x - mouth_landmarks[0].x) / (
            mouth_landmarks[4].x - mouth_landmarks[10].x
        )
        mouth_height = (mouth_landmarks[3].y - mouth_landmarks[13].y) / (
            nose_tip.y - mouth_landmarks[19].y
        )
        mouth_open = mouth_height / mouth_width
        if mouth_open > 0.5:
            print("mouth_open")
        return mouth_open > 0.5

    def run(self, image):
        landmarks_positions = []
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for data_point in face_landmarks.landmark:
                    landmarks_positions.append(
                        [data_point.x, data_point.y, data_point.z]
                    )  # saving normalized landmark positions
                landmarks_positions = np.array(landmarks_positions)
                landmarks_positions[:, 0] *= image.shape[1]
                landmarks_positions[:, 1] *= image.shape[0]

                
                if not (
                    self.is_eyes_closed(landmarks_positions)
                    or self.is_face_tilt(face_landmarks, image)
                ):
                   
                    return "good"
                
        return "bad"

    @staticmethod
    def get_face_quality_assessment():
        face_quality_assessment_settings = utils.get_face_quality_assessment_config()
        return dict(
            min_detection_confidence=face_quality_assessment_settings.min_detection_confidence,
            face_angle=face_quality_assessment_settings.face_angle,
            ear=face_quality_assessment_settings.ear,
        )

    @staticmethod
    def set_face_quality_assessment(min_detection_confidence, face_angle, ear):
        pass
