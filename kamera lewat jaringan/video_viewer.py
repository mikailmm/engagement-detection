from flask import Flask, render_template_string

app = Flask(__name__)

# --- Configuration ---
# The URL of the MJPEG stream provided by the first server (mjpeg_streamer_redirect.py)
# If your stream server is running on a different port or IP, change this URL.
STREAM_SOURCE_URL = "http://127.0.0.1:5000/feed"

# HTML template for the stream viewer page
# This template uses the <img> tag pointing directly to the network stream URL.
VIEWER_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Stream Viewer Client</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .container {
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background-color: #f0f8f8; /* Light blue-gray background */
            padding: 2rem;
            font-family: 'Inter', sans-serif;
        }
        .stream-card {
            background-color: white;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 15px 30px -5px rgba(0, 0, 0, 0.1), 
                        0 6px 15px -6px rgba(0, 0, 0, 0.08);
            max-width: 95vw;
            width: 768px; /* Max width for desktop viewing */
            border: 1px solid #e2e8f0;
        }
        #video_stream {
            width: 100%;
            height: auto;
            border-radius: 10px;
            border: 4px solid #3b82f6; /* Blue border */
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            background-color: #1f2937; /* Dark background when loading */
        }
        .text-info {
            color: #4b5563;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="stream-card">
            <h1 class="text-3xl font-extrabold text-gray-900 mb-2 text-center">
                Stream Viewer
            </h1>
            <p class="text-md text-info mb-6 text-center">
                Video stream:  
                <code class="bg-gray-100 p-1 rounded text-blue-600 font-mono">
                    {{ stream_url }}
                </code>
            </p>
            
            <!-- The crucial <img> tag -->
            <img id="video_stream" src="{{ stream_url }}" alt="Live Camera Stream">
            
            <p class="text-sm text-gray-500 mt-6 text-center">
            Port client: 5001
            </p>
        </div>
    </div>
</body>
</html>
"""


@app.route('/')
def index():
    """
    Root path handler for the client. 
    Renders the HTML template, injecting the stream URL into the HTML.
    """
    return render_template_string(VIEWER_HTML, stream_url=STREAM_SOURCE_URL)


if __name__ == '__main__':
    host_ip = '0.0.0.0'
    # IMPORTANT: Use a different port than the stream server (5000)
    client_port = 3438

    print("\n--- Starting Stream Viewer Client ---")
    print(f"Client is listening on: http://127.0.0.1:{client_port}")
    print("-" * 50)
    print(f"It is configured to read the stream from: {STREAM_SOURCE_URL}")
    print("-" * 50)

    # Run the Flask app on a different port (5001)
    app.run(host=host_ip, port=client_port)
