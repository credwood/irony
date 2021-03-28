# backend/Dockerfile
FROM nvidia/cuda:10.2-base
CMD nvidia-smi

WORKDIR /app

#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3.8.5

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8084

CMD ["python", "main.py"]