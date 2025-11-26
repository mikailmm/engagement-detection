import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
# Import protobuf to convert Task API output to format needed by drawing_utils
from mediapipe.framework.formats import landmark_pb2

SHOW_IMAGE = True

# --- 1. SETUP MEDIAPIPE TASKS API ---
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Lokasi file model yang sudah didownload
MODEL_PATH = './../face_landmark.task'

# Konfigurasi options
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
)

# Inisialisasi landmarker
landmarker = FaceLandmarker.create_from_options(options)

# --- 2. INISIALISASI DRAWING UTILS ---
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(
    thickness=1, circle_radius=1, color=(0, 255, 0))


# -- n. DETEKSI LANDMARK --
# Muat input image/gambar
image = mp.Image.create_from_file("./test_image_half.jpg")
rgb_image = None

# Detect landmark
result = landmarker.detect(image)

face_landmarks_list = result.face_landmarks[0]
# [lm.x, lm.y lm.z] for lm in face_landmark_list
# repeat until dapet semua 478 titik, then masukin dalam 1 row

face_features_flattened = np.array(
    [[lm.x, lm.y, lm.z] for lm in face_landmarks_list]).flatten()

# We need names like: x_0, y_0, z_0, x_1, y_1, z_1, ... x_477, y_477, z_477
# Total columns = 478 landmarks * 3 coords = 1434 columns
column_names = []
for i in range(478):
    column_names.extend([f'x_{i}', f'y_{i}', f'z_{i}'])

all_coordinates = [face_features_flattened, face_features_flattened]
data = np.vstack(all_coordinates)

df = pd.DataFrame(data, columns=column_names)


if SHOW_IMAGE:
    rgb_image = cv2.cvtColor(np.copy(image.numpy_view()), cv2.COLOR_BGR2RGB)
    proto_landmarks = landmark_pb2.NormalizedLandmarkList()
    proto_landmarks.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
        for lm in face_landmarks_list
    ])

    mp_drawing.draw_landmarks(
        image=rgb_image,
        landmark_list=proto_landmarks,  # Gunakan hasil konversi
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec
    )

while True and SHOW_IMAGE:
    cv2.imshow("Demo Landmark Image", rgb_image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

if SHOW_IMAGE:
    cv2.destroyAllWindows()

print(len(face_features_flattened))
print(len(column_names))
print(face_features_flattened)
print(column_names)
print(df)
print(df.loc[:, ["x_58", "y_58", "z_58"]])
