FROM python:latest

RUN apt-get update
RUN apt-get install -y pip ffmpeg
RUN pip install --upgrade pip
RUN pip install standard-aifc standard-sunau
RUN pip install pandas matplotlib seaborn numpy librosa
WORKDIR /host

CMD bash
