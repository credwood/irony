FROM python:3.8.5

WORKDIR /app

#set up environment

COPY requirements.txt .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8501
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
CMD ["streamlit", "run", "main.py"]
