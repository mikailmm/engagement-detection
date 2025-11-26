import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import cv2
import time

PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
PoseLandmarkOptions = mp.tasks.vision.PoseLandmarkerOptions
BaseOption = mp.tasks.BaseOptions
VisionTaskRunningMode = mp.tasks.vision.RunningMode
PoseLandmarker = mp.tasks.vision.PoseLandmarker

LATEST_RESULT = None


def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global LATEST_RESULT
    LATEST_RESULT = result
    # Untuk mendapatkan bahu saja [0][11:13]
    print('pose landmarker result: {}'.format(result.pose_landmarks))


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
drawing_spec = mp_drawing.DrawingSpec(
    thickness=1, circle_radius=1, color=(0, 255, 0))

options = PoseLandmarkOptions(
    base_options=BaseOption(model_asset_path="./../pose_landmarker_lite.task"),
    running_mode=VisionTaskRunningMode.LIVE_STREAM,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_segmentation_masks=False,
    result_callback=print_result
)

landmarker = PoseLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Eror: Tidak bisa buka webcam.")
    exit()

while cap.isOpened():
    # Baca frame dari webcam
    success, frame = cap.read()
    if not success:
        print("Mengabaikan frame kosong.")
        continue

    # --- 4. PROSES FRAME DENGAN MEDIAPIPE ---
    # Balik frame secara horizontal seperti kamera selfie umumnya
    frame = cv2.flip(frame, 1)

    # Konversi gambar BGR ke RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Timestamp dalam milidetik
    frame_timestamp_ms = time.time_ns() // 1_000_000
    landmarker.detect_async(
        image=mp_image, timestamp_ms=frame_timestamp_ms)

    if LATEST_RESULT and LATEST_RESULT.pose_landmarks:
        pose_landmarks_list = LATEST_RESULT.pose_landmarks[0]

        # PENTING: Konversi list biasa (Task API) ke NormalizedLandmarkList (Protobuf)
        # agar bisa dibaca oleh mp_drawing.draw_landmarks yang lama
        proto_landmarks = landmark_pb2.NormalizedLandmarkList()
        proto_landmarks.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in pose_landmarks_list
        ])

        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=proto_landmarks,  # Gunakan hasil konversi
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec
        )
    # if LATEST_RESULT:
    #     # Draw landmarks.
    #     for pose_landmarks in LATEST_RESULT.pose_landmarks:
    #         # Draw the pose landmarks.
    #         pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    #         pose_landmarks_proto.landmark.extend([
    #             landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
    #                                             z=landmark.z) for landmark
    #             in pose_landmarks
    #         ])
    #         mp_drawing.draw_landmarks(
    #             frame,
    #             pose_landmarks_proto,
    #             mp_pose.POSE_CONNECTIONS,
    #             mp.solutions.drawing_styles.get_default_pose_landmarks_style())

    cv2.imshow('Demo MediaPipe Pose', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
