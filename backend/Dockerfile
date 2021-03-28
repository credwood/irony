# backend/Dockerfile
FROM nvidia/cuda:10.2-base
CMD nvidia-smi

WORKDIR /app

#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3.8.5
RUN apt-get -y install python3-pip

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

ENTRYPOINT ["python3"]

EXPOSE 8084

CMD ["python3", "main.py"]