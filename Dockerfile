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

COPY . /opt/citronen

