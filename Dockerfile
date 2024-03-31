FROM python:3.10-slim

WORKDIR /image2seq

COPY requirements.txt /image2seq

RUN mkdir -p config
RUN mkdir -p protos


RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN echo "successfully installed"

EXPOSE 8080

ENTRYPOINT ["python", "img_server.py"]

