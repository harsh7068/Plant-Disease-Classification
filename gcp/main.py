from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
from keras.models import load_model

BUCKET_NAME = 'harsh-tf-model'

class_name = ["Early Blight", "Late Blight", "Healthy"]
model = None


def download_blob(bucket_name, source_blob_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_blob_name)


def predict(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/potatoes.h5",
            "/tmp/potatoes.h5"
        )
        model = load_model("/tmp/potatoes.h5")
    image = request.files["file"]
    image = np.array(
        Image.open(image).convert("RGB").resize((256, 256))
    )

    image = image/255

    img_arr = tf. expand_dims(image, 0)
    predictions = model.predict(img_arr)
    print("Predictions:", predictions)

    predicted_class = class_name[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    return {
        "clas": predicted_class,
        "confidence": confidence
    }
