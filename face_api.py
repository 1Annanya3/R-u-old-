# face_api.py
from flask import Flask, request, jsonify, Response
import threading
import numpy as np
import cv2

# Import the predictor implemented above
from cnn_predicter import predict_face_age

class FaceAPI:
    def __init__(self, on_face_age_ready):
        self.app = Flask(__name__)
        self.on_face_age_ready = on_face_age_ready

        @self.app.post("/upload_face")
        def upload_face():
            if "image" not in request.files:
                return jsonify(error="missing image"), 400
            raw = request.files["image"].read()
            if not raw:
                return jsonify(error="empty file"), 400

            arr = np.frombuffer(raw, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return jsonify(error="decode failed"), 400

            try:
                pred_age = predict_face_age(img)
                try:
                    # Notify host app for logging to Sheets
                    self.on_face_age_ready(pred_age)
                except Exception as e:
                    print(f"[Face logging error] {e}")
                return jsonify(age=float(pred_age))
            except Exception as e:
                return jsonify(error=f"inference failed: {e}"), 500

        @self.app.get("/upload_face_form")
        def upload_face_form():
            return Response(
                """<!DOCTYPE html>
<html>
<head><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Upload Face</title></head>
<body style="font-family:sans-serif;margin:24px">
  <h3>Upload Face Photo</h3>
  <form action="/upload_face" method="post" enctype="multipart/form-data">
    <input type="file" accept="image/*" name="image" capture="environment">
    <button type="submit">Upload</button>
  </form>
</body>
</html>""",
                mimetype="text/html"
            )

    def run_async(self, host="0.0.0.0", port=5000):
        th = threading.Thread(target=lambda: self.app.run(host=host, port=port, debug=False, use_reloader=False))
        th.daemon = True
        th.start()
        print(f"[Face API] listening on http://{host}:{port}")
