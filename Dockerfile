FROM fedora:latest

MAINTAINER Zex Li <top_zlynch@yahoo.com>

# Platform 
RUN dnf update -y; dnf install -y \
  redhat-rpm-config \
  mongodb \
  libc-devel \
  gcc \
  python3-devel 
  
RUN mkdir -p /var/data
RUN pip3 install -U pip; pip3 install \
  pandas==0.19.2 \
  numpy==1.12.1 \
  pymongo==3.4.0 
ENV CONDA_INSTALLER /tmp/Anaconda3-4.4.0-Linux-x86_64.sh
RUN curl https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh -o $CONDA_INSTALLER
RUN /bin/bash $CONDA_INSTALLER; conda install pytorch sklearn tensorflow
RUN rm -f $CONDA_INSTALLER 

ENV CITRONEN_ROOT "/opt/citronen"

COPY README.md $CITRONEN_ROOT/README.md
COPY apps $CITRONEN_ROOT/apps
COPY docs $CITRONEN_ROOT/docs
COPY find_news $CITRONEN_ROOT/find_news
COPY passenger_screening $CITRONEN_ROOT/passenger_screening
COPY quora_question $CITRONEN_ROOT/quora_question
COPY russan_housing $CITRONEN_ROOT/russian_housing
COPY zillow $CITRONEN_ROOT/zillow

