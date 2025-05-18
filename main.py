from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model("model/trained_plant_disease_model.h5")

class_names = ["Disease A", "Disease B", "Healthy"]  # Replace with your actual class names

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    img = image.load_img(file, target_size=(224, 224))  # Adjust as per model
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    return jsonify({"prediction": predicted_class})

if __name__ == "__main__":
    app.run(debug=True)