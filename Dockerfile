FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Generate synthetic data
RUN python -c "from pipeline.synthetic import generate_synthetic_data; import json; data = generate_synthetic_data(250, 42); json.dump(data, open('coords.json', 'w')); print(f'Generated {len(data[\"repos\"])} repos')"

# Expose port
EXPOSE 7860

# Run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
