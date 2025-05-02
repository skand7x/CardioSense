FROM python:3.9-slim

WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make scripts executable
RUN chmod +x scripts/*.py

# Create necessary directories
RUN mkdir -p models data static

# Instead of training during build, provide pre-trained models
# We'll include sample models in the repository or download them at runtime

# Expose Hugging Face's default port
EXPOSE 7860

# Run the start script
CMD ["./start.sh"]
