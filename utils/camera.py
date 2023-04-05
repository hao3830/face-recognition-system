import asyncio
def gen_frame(video_controller,is_default):
     """Video streaming generator function."""
     while True:
        frame_def = video_controller.get_default()
        frame_draw = video_controller.get() 
        if is_default:
            frame = frame_def
        else:
            frame =frame_draw 
        if frame is not None:
#            image_buffer = cv2.imencode(".jpg", frame)[1].tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        
        # time.sleep(1/30)

async def streamer(gen):
    try:
        for i in gen:
            yield i
    except asyncio.CancelledError:
        print("caught cancelled error")