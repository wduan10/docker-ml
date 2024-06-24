FROM python:3.8

WORKDIR /test

COPY . .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "./train.py"]
