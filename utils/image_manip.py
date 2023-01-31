class ImageProcess:
    def __init__(self) -> None:
        pass

    @staticmethod
    def isContain(bbox, roi):
        if roi[0] < bbox[0] and roi[1] < bbox[1]:
            if  bbox[2] < roi[2] and bbox[3] < roi[3]:
                return True
        
        return False
    
