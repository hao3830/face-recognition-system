def gen_frame(video_controller,is_default):
     """Video streaming generator function."""
     while True:
        if is_default:
            frame = video_controller.get_default()

        else:
            frame=video_controller.get() 
        if frame is not None:
#            image_buffer = cv2.imencode(".jpg", frame)[1].tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")