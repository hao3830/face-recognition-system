import cv2
import numpy as np

from PIL import Image, ImageDraw


class ImageProcess:
    light_grey = (211, 211, 211)
    red = (255, 0, 0)
    green = (124, 252, 0)
    blue = (255, 0, 0)
    yellow = (255, 255, 0)

    def __init__(self) -> None:
        pass

    @staticmethod
    def isContain(bbox, roi):
        if roi[0] < bbox[0] and roi[1] < bbox[1]:
            if bbox[2] < roi[2] and bbox[3] < roi[3]:
                return True

        return False

    @staticmethod
    def getArea(bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    @staticmethod
    def draw_4_rounded_conner_bbox(image, bbox, color, radius=20, thickness=1.5):
        if ImageProcess.getArea(bbox) < 2000:
            radius = 15
        if ImageProcess.getArea(bbox) < 1000:
            radius = 5
        x, y, x2, y2 = bbox
        w = x2 - x
        h = y2 - y
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        draw.arc(
            (
                x + thickness,
                y + thickness,
                x + 2 * radius - thickness,
                y + 2 * radius - thickness,
            ),
            start=180,
            end=270,
            fill=color,
            width=thickness,
        )
        draw.arc(
            (
                x + w - 2 * radius + thickness,
                y + thickness,
                x + w - thickness,
                y + 2 * radius - thickness,
            ),
            start=270,
            end=0,
            fill=color,
            width=thickness,
        )
        draw.arc(
            (
                x + w - 2 * radius + thickness,
                y + h - 2 * radius + thickness,
                x + w - thickness,
                y + h - thickness,
            ),
            start=0,
            end=90,
            fill=color,
            width=thickness,
        )
        draw.arc(
            (
                x + thickness,
                y + h - 2 * radius + thickness,
                x + 2 * radius - thickness,
                y + h - thickness,
            ),
            start=90,
            end=180,
            fill=color,
            width=thickness,
        )
        open_cv_image = np.array(img)
        # Convert RGB to BGR
        return open_cv_image[:, :, ::-1].copy()

    @staticmethod
    def get_bbox_from_tracklet(t, frame):
        roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
        x1 = int(roi.topLeft().x)
        y1 = int(roi.topLeft().y)
        x2 = int(roi.bottomRight().x)
        y2 = int(roi.bottomRight().y)
        return x1, y1, x2, y2

    @staticmethod
    def draw_track_status(status, id, frame, bbox):
        x1, y1, _, _ = bbox
        cv2.putText(
            frame,
            f"ID:{id}",
            (x1 + 10, y1 + 35),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            ImageProcess.blue,
        )
        cv2.putText(
            frame,
            status,
            (x1 + 10, y1 + 50),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            ImageProcess.blue,
        )
        return frame

    @staticmethod
    def draw_roi_area(frame, limit_roi):
        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (limit_roi[0], limit_roi[1]),
            (limit_roi[2], limit_roi[3]),
            (0, 200, 0),
            -1,
        )

        alpha = 0.4

        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        return frame
    
