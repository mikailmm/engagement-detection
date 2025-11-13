import time
import joblib
import numpy as np
import cv2
import mediapipe as mp
import json
from flask import Flask, render_template, Response

# --- 1. INISIALISASI APLIKASI FLASK ---
app = Flask(__name__)

# --- 2. INISIALISASI MEDIAPIPE DAN MODEL (DILAKUKAN SEKALI SAAT SERVER START) ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(
    thickness=1, circle_radius=1, color=(0, 255, 0))

# Muat model machine learning
try:
    model = joblib.load('engagement-model2-rf')
    print("\033[0;32mModel 'engagement-model2-rf' berhasil dimuat.\033[0m")
except FileNotFoundError:
    print("\033[0;31mError: File model 'engagement-model2-rf' tidak ditemukan. Pastikan file berada di direktori yang sama.\033[0m")
    exit()

# Variabel global untuk logika inferensi
arr_hasil_deteksi = []
jumlah_frame_mov_avg = 150  # Sekitar 5 detik jika FPS ~30

# Variabel untuk FPS
prev_frame_time = 0

# Menyimpan statistik terbaru untuk dikirim ke klien
stats = {
    "fps": 0,
    "level": "N/A",
    "modus": "N/A"
}


def generate_frames():
    """Generator function untuk streaming frame video."""
    global stats
    global prev_frame_time, arr_hasil_deteksi
    show_mesh = True

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Eror: Tidak bisa buka webcam.")
        return

    print("\033[0;32mMemulai feed video...\033[0m")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Mengabaikan frame kosong.")
            break

        # Balik frame secara horizontal
        frame = cv2.flip(frame, 1)

        # Konversi warna untuk MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Proses dengan Face Mesh
        results = face_mesh.process(image_rgb)

        image_rgb.flags.writeable = True

        if show_mesh:
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

        # --- Logika Kalkulasi FPS ---
        new_frame_time = time.time()
        time_diff = new_frame_time - prev_frame_time
        fps = 1 / time_diff if time_diff > 0 else 0
        prev_frame_time = new_frame_time
        # fps_text = f"FPS: {int(fps)}"
        stats['fps'] = int(fps)
        # cv2.putText(frame, fps_text, (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)

        # --- Logika Inferensi Keterlibatan ---
        current_level = "N/A"
        current_modus = "N/A"
        try:
            # Ekstraksi landmark jika wajah terdeteksi
            landmarks_coords = []
            for coord in results.multi_face_landmarks[0].landmark:
                landmarks_coords.extend([coord.x, coord.y, coord.z])

            landmarks_coords_array = np.array(landmarks_coords).reshape(1, -1)
            predicted_level = model.predict(landmarks_coords_array)

            current_level = str(predicted_level[0])

            # Menampilkan level di frame
            # cv2.putText(frame, f"Level engagement: {predicted_level[0]}", (
            #     10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)

            # Logika moving average (majority vote)
            if len(arr_hasil_deteksi) < jumlah_frame_mov_avg:
                arr_hasil_deteksi.append(predicted_level[0])
            else:
                arr_hasil_deteksi.pop(0)
                arr_hasil_deteksi.append(predicted_level[0])

                # Hitung modus (nilai yang paling sering muncul)
                np_arr = np.array(arr_hasil_deteksi)
                counts = np.bincount(np_arr)
                mode_value = np.argmax(counts)

                current_modus = str(mode_value)

                # cv2.putText(frame, f"Modus {int(jumlah_frame_mov_avg/30)}dtk: {mode_value}",
                #             (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)

        except (TypeError, IndexError):
            # Jika tidak ada wajah yang terdeteksi
            # cv2.putText(frame, "Wajah tak terdeteksi", (10, 60),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            current_level = "Wajah tak terdeteksi"
            current_modus = "N/A"
            pass

        # --- Update 'stats' global untuk SSE ---
        stats['level'] = current_level
        stats['modus'] = current_modus

        # Encode frame ke format JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame dalam format multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


def get_stats():
    """Generator untuk Server-Sent Events (SSE)."""
    while True:
        # Kirim data 'stats' global dalam format JSON
        json_data = json.dumps(stats)
        # yield f"data: {json_data}\n\n"
        time.sleep(0.2)  # Update setiap 200ms


@app.route('/')
def index():
    """Halaman utama yang akan menampilkan video stream."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Endpoint untuk video streaming."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats_feed')
def stats_feed():
    """Endpoint untuk data statistik via SSE."""
    return Response(get_stats(), mimetype='text/event-stream')


if __name__ == '__main__':
    # Jalankan server Flask
    # host='0.0.0.0' agar bisa diakses dari perangkat lain di jaringan yang sama
    text = """
      WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit  
    """
    app.run(host='0.0.0.0', port=5000, debug=True)
    print(text)
