class ImageProcess:
    def __init__(self) -> None:
        pass

    @staticmethod
    def isContain(bbox, roi):
        if bbox[0] < roi[0] or bbox[1] < roi[1] or bbox[2] > roi[2] or bbox[3] > roi[3]:
            return False

        return True
