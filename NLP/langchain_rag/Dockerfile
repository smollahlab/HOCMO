FROM python:3.10
WORKDIR /home
COPY ./ /home
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install
RUN apt-get install -y libgdal-dev
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
EXPOSE 8501:8501

RUN pip3 install -r requirements.txt
