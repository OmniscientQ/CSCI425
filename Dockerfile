FROM python:latest

RUN apt-get update
RUN apt-get install -y pip
RUN pip install pandas matplotlib seaborn numpy keras
WORKDIR /host

CMD bash
