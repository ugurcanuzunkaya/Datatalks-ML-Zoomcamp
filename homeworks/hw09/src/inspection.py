import onnxruntime as ort
import os

try:
    from model_utils import download_image, prepare_image, preprocess_input
except ImportError:
    from .model_utils import download_image, prepare_image, preprocess_input

# --- Setup ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_FILE = os.path.join(DATA_DIR, "hair_classifier_v1.onnx")
IMAGE_URL = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"


def main():
    if not os.path.exists(MODEL_FILE):
        print(f"Model file {MODEL_FILE} not found. Please download it first.")
        return

    # --- Q1 & Q2: Model Inspection ---
    session = ort.InferenceSession(MODEL_FILE)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape

    print(f"Q1: Output name: {output_name}")
    print(
        f"Q2: Input shape: {input_shape} (Target size is {input_shape[2]}x{input_shape[3]})"
    )

    # --- Q3 & Q4: Processing and Inference ---
    target_size = (input_shape[2], input_shape[3])

    img = download_image(IMAGE_URL)
    img_prep = prepare_image(img, target_size)
    X = preprocess_input(img_prep)

    # R channel is index 0 in the channel dimension (Batch, Channel, Height, Width)
    # We want the first pixel (0, 0)
    first_pixel_R = X[0, 0, 0, 0]
    print(f"Q3: First pixel R channel value: {first_pixel_R:.3f}")

    # Run inference
    outputs = session.run([output_name], {input_name: X})
    output_val = outputs[0][0][0]
    print(f"Q4: Model output: {output_val:.3f}")


if __name__ == "__main__":
    main()
