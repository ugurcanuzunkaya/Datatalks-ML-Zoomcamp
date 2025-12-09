FROM agrigorev/model-2025-hairstyle:v1

# Install dependencies (using pip as uv might not be in the base image, forcing pip is safer for this minimal env)
RUN pip install numpy onnxruntime pillow

# Copy our script and utils
COPY src/lambda_function.py .
COPY src/model_utils.py .

# Override entrypoint to run the script directly for testing
ENTRYPOINT ["python", "lambda_function.py"]
