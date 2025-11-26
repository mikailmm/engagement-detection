import time
import cv2
import mediapipe as mp
import numpy as np
# Import protobuf to convert Task API output to format needed by drawing_utils
from mediapipe.framework.formats import landmark_pb2

# --- 1. SETUP MEDIAPIPE TASKS API ---
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Lokasi file model yang sudah didownload
MODEL_PATH = '../face_landmarker.task'

# Global variable untuk menyimpan hasil deteksi dari callback
LATEST_RESULT = None

# Callback function: Dipanggil setiap kali MediaPipe selesai memproses frame


def result_callback(result, output_image, timestamp_ms):
    global LATEST_RESULT
    LATEST_RESULT = result


# Konfigurasi Options
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback,
    output_face_blendshapes=True,  # Optional: jika ingin data blendshapes
    output_facial_transformation_matrixes=True,  # Optional
    num_faces=1,
)

# --- 2. INISIALISASI OPENCV DAN DRAWING UTILS ---
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(
    thickness=1, circle_radius=1, color=(0, 255, 0))

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Eror: Tidak bisa buka webcam.")
    exit()

prev_frame_time = 0
new_frame_time = 0
i = 0

print("\033[0;32mMemulai feed (Task API)... Tekan 'q' untuk berhenti\033[0m")

# --- 3. LOOP UTAMA DENGAN CONTEXT MANAGER ---
with FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Mengabaikan frame kosong.")
            continue

        # Balik frame & Convert ke RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- 4. PROSES FRAME (ASYNCHRONOUS) ---
        # Convert ke mp.Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Timestamp dalam milidetik (wajib untuk LIVE_STREAM mode)
        frame_timestamp_ms = time.time_ns() // 1_000_000

        # Kirim ke MediaPipe (Non-blocking, hasil dikirim ke result_callback)
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        # --- 5. MENGGAMBAR HASIL (DARI GLOBAL VAR) ---
        if LATEST_RESULT and LATEST_RESULT.face_landmarks:
            # Ambil wajah pertama, ambil tanpa iris [:468]
            face_landmarks_list = LATEST_RESULT.face_landmarks[0][:]

            # PENTING: Konversi list biasa (Task API) ke NormalizedLandmarkList (Protobuf)
            # agar bisa dibaca oleh mp_drawing.draw_landmarks yang lama
            proto_landmarks = landmark_pb2.NormalizedLandmarkList()
            proto_landmarks.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
                for lm in face_landmarks_list
            ])

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=proto_landmarks,  # Gunakan hasil konversi
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )
            # mp_drawing.draw_landmarks(
            #     image=frame,
            #     landmark_list=proto_landmarks,
            #     connections=mp_face_mesh.FACEMESH_CONTOURS,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp.solutions.drawing_styles
            #     .get_default_face_mesh_contours_style())
            # mp_drawing.draw_landmarks(
            #     image=frame,
            #     landmark_list=proto_landmarks,
            #     connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp.solutions.drawing_styles
            #     .get_default_face_mesh_iris_connections_style())
        # -------------------
        # FPS Calculation
        # -------------------
        new_frame_time = time.time()
        time_diff = new_frame_time - prev_frame_time
        if time_diff > 0:
            fps = 1 / time_diff
        else:
            fps = 0
        prev_frame_time = new_frame_time

        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (100, 255, 0), 2, cv2.LINE_AA)

        # --- 6. MENAMPILKAN HASIL ---
        cv2.imshow('Demo MediaPipe Face Mesh (Tasks API)', frame)
        print(f"{fps_text} Frame {i}", end='\r')
        i += 1

        # Exit logic
        if cv2.waitKey(5) & 0xFF == ord('q'):
            print("\nBerhenti...")
            if LATEST_RESULT and LATEST_RESULT.face_landmarks:
                print("\n--- Debug Info ---")
                # Task API mengembalikan List of Lists, bukan NamedTuple
                print(f"Type Result: {type(LATEST_RESULT.face_landmarks)}")
                print(f"Type Face 0: {type(LATEST_RESULT.face_landmarks[0])}")
                print(
                    f"First Landmark X: {LATEST_RESULT.face_landmarks[0][0].x}")

                landmarks_coords = []
                # Akses langsung list standard Python
                for coord in LATEST_RESULT.face_landmarks[0]:
                    landmarks_coords.append(coord.x)
                    landmarks_coords.append(coord.y)
                    landmarks_coords.append(coord.z)

                # print(landmarks_coords) # Uncomment jika ingin print semua
            else:
                print("Tidak ada wajah terdeteksi saat keluar.")
            break

# --- 7. CLEANUP ---
print("Closing application.")
cap.release()
cv2.destroyAllWindows()
