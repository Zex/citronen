FROM alpine:edge

MAINTAINER Zex Li <top_zlynch@yahoo.com>

# Platform 
RUN apk add --upgrade apk-tools
RUN apk upgrade --available
RUN apk add --no-cache \
  mongodb \
  lighttpd \
  python3 \
  libc-dev \
  gcc \
  python3-dev \
  bash
  
RUN ln -s /usr/include/locale.h /usr/include/xlocale.h
RUN mkdir -p /var/data
RUN pip3 install \
  numpy==1.12.1 \
  pymongo==3.4.0


