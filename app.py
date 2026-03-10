from flask import Flask, render_template, request, redirect, url_for
from deepface import DeepFace
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
KNOWN_FOLDER = "known_faces"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Compare with known faces
            match_found = None
            for known_file in os.listdir(KNOWN_FOLDER):
                known_path = os.path.join(KNOWN_FOLDER, known_file)
                try:
                    result = DeepFace.verify(file_path, known_path, model_name='VGG-Face')
                    if result["verified"]:
                        match_found = known_file
                        break
                except Exception as e:
                    print("Error:", e)

            return render_template("result.html", match=match_found, uploaded=filename)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

