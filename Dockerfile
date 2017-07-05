FROM golang:1.9

MAINTAINER Zex Li <top_zlynch@yahoo.com>

# Platform 
RUN apt-get update -y ;apt-get install -y \
  mongodb \
  lighttpd \
  python3 \
  libc-dev \
  gcc \
  bash \
  python3-dev \
  python3-pip
  
RUN mkdir -p /var/data
RUN pip3 install -U pip; pip3 install \
  pandas==0.19.2 \
  scipy==0.19.0 \
  numpy==1.12.1 \
  pymongo==3.4.0 \
  pytorch

ENV CITRONEN_ROOT "/opt/citronen"
COPY README.md $CITRONEN_ROOT/README.md
COPY apps $CITRONEN_ROOT/apps
COPY docs $CITRONEN_ROOT/docs
COPY find_news $CITRONEN_ROOT/find_news
COPY passenger_screening $CITRONEN_ROOT/passenger_screening
COPY quora_question $CITRONEN_ROOT/quora_question
COPY russan_housing $CITRONEN_ROOT/russian_housing
COPY zillow $CITRONEN_ROOT/zillow

