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
