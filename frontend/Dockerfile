FROM nv-backend:latest

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --upgrade pip
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN apt-get install cmake --no-install-recommends --no-install-suggests -y curl
RUN pip3 install cython
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8501
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
CMD ["streamlit", "run", "main.py"]
