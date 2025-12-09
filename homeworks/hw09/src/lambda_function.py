import onnxruntime as ort
import os

try:
    from model_utils import download_image, prepare_image, preprocess_input
except ImportError:
    from .model_utils import download_image, prepare_image, preprocess_input

# Use the model located in the same directory (container root) or specific path
# adjusted for Docker
MODEL_FILE = "hair_classifier_empty.onnx"
IMAGE_URL = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"


def main():
    # Load model
    if not os.path.exists(MODEL_FILE):
        print(f"Model file {MODEL_FILE} not found!")
        return

    session = ort.InferenceSession(MODEL_FILE)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Prepare data
    img = download_image(IMAGE_URL)
    img = prepare_image(img, (200, 200))
    X = preprocess_input(img)

    # Predict
    result = session.run([output_name], {input_name: X})
    print(f"Result: {result[0][0][0]}")


if __name__ == "__main__":
    main()
