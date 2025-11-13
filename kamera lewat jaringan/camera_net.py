import numpy as np
import cv2
from flask import Flask, Response
import threading

app = Flask(__name__)
camera = cv2.VideoCapture(1)  # 0 is usually the default webcam


def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # Run the Flask app on your local IP address, on a specific port
    # You can find your IP by running `ipconfig` (Windows) or `ifconfig` (macOS/Linux)
    app.run(host='0.0.0.0', port=5000, threaded=True)
