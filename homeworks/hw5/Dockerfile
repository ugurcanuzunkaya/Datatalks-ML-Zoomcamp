FROM --platform=linux/amd64 agrigorev/zoomcamp-model:2025

# Set working directory
WORKDIR /code

# Copy the dependency files
COPY pyproject.toml uv.lock ./

# Install uv in the container and sync dependencies
RUN pip install uv
RUN uv pip install --system -r pyproject.toml

# Copy our application file
COPY q6_app.py ./

# Expose the port
EXPOSE 9696

# Run the app
CMD ["uvicorn", "q6_app:app", "--host", "0.0.0.0", "--port", "9696"]