FROM python:3.10.11
ENV PYTHONBUFFERED 1
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY . /app
RUN echo "Folder creation start"
# RUN if [ ! -d /app/processedFiles ]; then \ 
#         mkdir /app/processedFiles; \ 
#     fi
#RUN mkdir -p /app/processedFiles
RUN echo "Folder creation End"


CMD python server.py 
# runserver 0.0.0.0:8000