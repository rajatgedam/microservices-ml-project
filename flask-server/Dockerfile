FROM python:3.10.11
ENV PYTHONBUFFERED 1
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY . /app

CMD python server.py 
# runserver 0.0.0.0:8000