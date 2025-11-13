import time
import numpy as np
import cv2
import mediapipe as mp

# --- 1. INISIALISASI MEDIAPIPE FACE MESH ---
mp_face_mesh = mp.solutions.face_mesh
# Statement 'with' memastikan resource secara otomatis dilepas
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,    # Mendapatkan 478 titik, termasuk iris
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    # --- 2. INISIALISASI OPENCV DRAWING UTILS DAN VIDEO CAPTURE ---
    mp_drawing = mp.solutions.drawing_utils
    # Mendefinisikan tampilan landmark dan koneksi
    drawing_spec = mp_drawing.DrawingSpec(
        thickness=1, circle_radius=1, color=(0, 255, 0))  # Green color

    # Mendapatkan feed video dari index 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Eror: Tidak bisa buka webcam.")
        exit()

    prev_frame_time = 0
    new_frame_time = 0

    print("\033[0;32mMemulai feed... Tekan 'q' untuk berhenti\033[0m")
    i = 0
    # --- 3. LOOP UTAMA PEMROSESAN FEED VIDEO ---
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

        # Not writable (pass by reference) untuk meningkatkan performa
        image_rgb.flags.writeable = False
        results = face_mesh.process(image_rgb)

        # Mengaktifkan writable untuk menggambar anotasi face mesh
        image_rgb.flags.writeable = True

        # --- 5. MENGGAMBAR ANOTASI FACE MESH PADA FRAME ---
        if results.multi_face_landmarks:
            # Iterasi setiap wajah yang terdeteksi (meskipun max_num_faces=1)
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,  # Gambar segitiga mesh
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec
                )

        # -------------------
        # FPS Calculation Logic
        # -------------------

        # Time when we start processing the current frame
        new_frame_time = time.time()

        # Calculate time difference and then FPS
        time_diff = new_frame_time - prev_frame_time

        # Avoid division by zero if time difference is extremely small
        if time_diff > 0:
            fps = 1 / time_diff
        else:
            fps = 0  # or some other indication

        prev_frame_time = new_frame_time

        # Convert FPS to an integer string for display
        fps_text = f"FPS: {int(fps)}"

        # Add the FPS text to the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, fps_text, (10, 30), font,
                    1, (100, 255, 0), 2, cv2.LINE_AA)

        # -------------------
        #
        # -------------------

        # --- 6. MENAMPILKAN HASIL FRAME ---
        cv2.imshow('Demo MediaPipe Face Mesh', frame)

        print(f"{fps_text} Frame {i}", end='\r')
        i += 1

        # Exit loop ketika 'q' ditekan
        if cv2.waitKey(5) & 0xFF == ord('q'):
            print(type(results.multi_face_landmarks))
            print(type(results.multi_face_landmarks[0]))
            print(results.multi_face_landmarks[0].landmark[0].x)
            landmarks_coords = []
            for coord in results.multi_face_landmarks[0].landmark:
                landmarks_coords.append(coord.x)
                landmarks_coords.append(coord.y)
                landmarks_coords.append(coord.z)
            # print(landmarks_coords)
            break

# --- 7. MENUTUP APLIKASI ---
print("Closing application.")
cap.release()
cv2.destroyAllWindows()
