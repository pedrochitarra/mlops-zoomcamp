FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Install pipenv and compilation dependencies
RUN pip install -U pip
RUN pip install pipenv pandas pyarrow numpy==1.26.4

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

# this is our entrypoint, use parameters to pass arguments
ENTRYPOINT ["python", "starter.py"]
CMD ["year", "month"]

# docker run -it --rm -p 9696:9696  ride-duration-prediction-service:v1 year=2023 month=5

COPY starter.py /app/starter.py