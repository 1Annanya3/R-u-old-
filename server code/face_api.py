from flask import Flask, request, jsonify, Response
import threading
import numpy as np
import cv2

def load_cnn():
    # TODO: replace with your actual loader
    # from tensorflow.keras.models import load_model
    # return load_model("face_age_cnn.h5")
    return None

def preprocess_for_cnn(img_bgr):
    img = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

class FaceAPI:
    def __init__(self, on_face_age_ready):
        self.app = Flask(__name__)
        self.on_face_age_ready = on_face_age_ready
        self.cnn = load_cnn()

        @self.app.post("/upload_face")
        def upload_face():
            if 'image' not in request.files:
                return jsonify({"error": "missing image"}), 400
            raw = request.files['image'].read()
            if not raw:
                return jsonify({"error": "empty file"}), 400

            arr = np.frombuffer(raw, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return jsonify({"error": "decode failed"}), 400

            # TODO: replace with real model inference
            # pred_age = float(self.cnn.predict(preprocess_for_cnn(img))[0])
            pred_age = float(30.0)

            try:
                self.on_face_age_ready(pred_age)
            except Exception as e:
                print(f"Face logging error: {e}")

            return jsonify({"age": pred_age})

        @self.app.get("/upload_face_form")
        def upload_face_form():
            html = """
<!DOCTYPE html><html><head>
<meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Upload Face</title>
</head><body style="font-family:sans-serif;margin:24px;">
<h3>Upload Face Photo</h3>
<form action="/upload_face" method="post" enctype="multipart/form-data">
  <input type="file" accept="image/*" name="image" capture="environment">
  <button type="submit">Upload</button>
</form>
</body></html>"""
            return Response(html, mimetype="text/html")

    def run_async(self, host="0.0.0.0", port=8000):
        th = threading.Thread(target=lambda: self.app.run(host=host, port=port, debug=False, use_reloader=False))
        th.daemon = True
        th.start()
        print(f"Face API listening on http://{host}:{port}")
