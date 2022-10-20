FROM python:3.8-slim-bullseye

RUN apt-get update -y \
    && apt-get install git -y \
    && apt-get install python-dev build-essential -y \
    && git clone https://github.com/tkgsn/GG-I

WORKDIR /GG-I

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

ENV DATA_DIR /data

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''"]