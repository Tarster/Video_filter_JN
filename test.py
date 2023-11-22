# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# model_path = 'face_landmarker.task'

import cv2 as cv
import mediapipe as mp
video = cv.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
connection_drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
tm = cv.TickMeter()
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        ret, image = video.read()
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        tm.start()
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                         landmark_drawing_spec=drawing_spec) #, connection_drawing_spec=drawing_spec)
        tm.stop()
        cv.putText(image, 'FPS: {:.2f}'.format(tm.getFPS()), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv.imshow('Video', image)
        k = cv.waitKey(1)
        if k == ord('q'):
            break
        tm.reset()
    video.release()
    cv.destroyAllWindows()



# BaseOptions = mp.tasks.BaseOptions
# FaceLandmarker = mp.tasks.vision.FaceLandmarker
# FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
# FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
# VisionRunningMode = mp.tasks.vision.RunningMode

# # Create a face landmarker instance with the live stream mode:
# def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
#     print('face landmarker result: {}'.format(result))

# options = FaceLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path=model_path),
#     running_mode=VisionRunningMode.LIVE_STREAM,
#     result_callback=print_result)

# with FaceLandmarker.create_from_options(options) as landmarker:
#     mp_image = mp.Image(image_format=mp.ImageFormat.RGB, data=numpy_frame_from_opencv)
#     landmarker.detect_async(mp_image, frame_timestamp_ms)
