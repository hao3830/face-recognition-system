import cv2

import mediapipe as mp


class FaceQuality:
    def __init__(self):
        mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
        )

    def is_eyes_closed(self, face_landmarks):

        left_eye = []
        right_eye = []
        for i in range(36, 42):
            left_eye.append(face_landmarks.landmark[i])
        for i in range(42, 48):
            right_eye.append(face_landmarks.landmark[i])
        left_eye_open = (left_eye[4].x - left_eye[0].x) / (
            left_eye[3].x - left_eye[1].x
        )
        right_eye_open = (right_eye[4].x - right_eye[0].x) / (
            right_eye[3].x - right_eye[1].x
        )
        if left_eye_open < 0.15 or right_eye_open < 0.15:
            print("is_eyes_closed")
        return left_eye_open < 0.15 or right_eye_open < 0.15

    def is_face_tilt(self, face_landmarks):

        left_eye_inner = face_landmarks.landmark[159]
        right_eye_inner = face_landmarks.landmark[248]
        nose_tip = face_landmarks.landmark[6]

        left_eye_inner = face_landmarks.landmark[159]
        right_eye_inner = face_landmarks.landmark[248]
        nose_tip = face_landmarks.landmark[6]
        eye_distance = abs(left_eye_inner.x - right_eye_inner.x)
        nose_distance = abs(nose_tip.y - left_eye_inner.y)
        face_angle = abs(nose_tip.z - left_eye_inner.z)
        if eye_distance / nose_distance < 0.25 or face_angle > 0.15:
            print("is_face_tilt")

        return eye_distance / nose_distance < 0.25 or face_angle > 0.15

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
        if (mouth_open > 0.5) :
            print("mouth_open")
        return mouth_open > 0.5

    def run(self, image):

        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if (
                    self.is_eyes_closed(face_landmarks)
                    or self.is_face_tilt(face_landmarks)
                    or self.is_smile(face_landmarks)
                ):
                    return "bad"

        return "good"
