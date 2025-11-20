# face_api.py (Simplified Form)
from flask import Flask, request, jsonify, Response
import threading
import numpy as np
import cv2

from cnn_predicter import predict_face_age

class FaceAPI:
    def __init__(self, on_face_age_ready):
        self.app = Flask(__name__)
        self.on_face_age_ready = on_face_age_ready

        @self.app.post("/upload_face")
        def upload_face():
            if "image" not in request.files:
                return jsonify(error="missing image"), 400
            
            participant_id = request.form.get("participant_id", "UNKNOWN")
            gender = request.form.get("gender", "UNKNOWN")
            
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
                    self.on_face_age_ready(
                        pred_age, 
                        form_participant_id=participant_id, 
                        form_gender=gender
                    )
                except Exception as e:
                    print(f"[Face logging error] {e}")
                
                return jsonify(age=float(pred_age), status="Prediction received by server, check console for confirmation")
            except Exception as e:
                print(f"[Inference error] {e}") 
                return jsonify(error=f"inference failed: {e}"), 500

        @self.app.get("/upload_face_form")
        def upload_face_form():
            return Response(
                """
<!doctype html>
<html>
  <head><meta charset="utf-8"><title>Upload Face</title></head>
  <body>
    <h3>Upload Face Image</h3>
    <p>Using participant: Check console for ID and confirmation.</p>
    <form method="post" action="/upload_face" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/*" required />
      <button type="submit">Upload</button>
    </form>
  </body>
</html>
""",
                mimetype="text/html",
            )

    def run_async(self, host="0.0.0.0", port=5000):
        thr = threading.Thread(target=self.app.run, kwargs={"host": host, "port": port}, daemon=True)
        thr.start()
        return thr
