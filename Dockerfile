FROM python:3.14-slim
WORKDIR /app
COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt
RUN useradd --create-home churn
COPY src ./src
COPY models ./models
COPY results/dashboard ./results/dashboard
RUN mkdir runtime && chown churn:churn runtime
USER churn
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
