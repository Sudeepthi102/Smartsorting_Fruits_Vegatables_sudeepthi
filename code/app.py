from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# -------- Load trained model --------
with open("model/fruit_classifier.pkl", "rb") as f:
    model, class_names = pickle.load(f)

# -------- Image preprocessing --------
def preprocess(img):
    img = img.resize((64, 64))
    img = np.array(img) / 255.0
    img = img.reshape(1, -1)
    return img

# -------- Routes --------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        file = request.files.get("image")

        if not file:
            return render_template("predict.html", error="No image selected")

        # Read image
        img = Image.open(file.stream).convert("RGB")

        # Convert image to base64 for display
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

        # Prediction
        processed_img = preprocess(img)
        pred_index = model.predict(processed_img)[0]
        label = class_names[pred_index]

        # GOOD / BAD condition
        if "Healthy" in label:
            condition = "GOOD (Fresh)"
            status_color = "green"
        else:
            condition = "BAD (Rotten)"
            status_color = "red"

        return render_template(
            "result.html",
            prediction=label,
            condition=condition,
            status_color=status_color,
            image_data=image_base64
        )

    return render_template("predict.html")

# -------- Run App --------
if __name__ == "__main__":
    app.run(debug=True)