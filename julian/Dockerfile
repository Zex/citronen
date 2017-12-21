#FROM local-dtr.zhihuiya.com/buildenv/python-base:Rel.3.6.2-2
#FROM fedora:27
FROM python:3

MAINTAINER Zex Li <top_zlynch@yahoo.com>

ENV JULIAN_ROOT "/opt/julian"
ENV ETL_ROOT "/opt/360-etl"

RUN apt-get update
RUN apt-get install unzip -y
RUN apt-get install openjdk-7-jdk openjdk-7-jre -y --fix-missing

COPY requirements.txt $JULIAN_ROOT/requirements.txt

WORKDIR $JULIAN_ROOT
RUN pip3 install -U pip
#RUN -r requirements.txt
RUN pip3 install nltk==3.2.4
RUN pip3 install Cython==0.26.1
RUN pip3 install numpy==1.13.1
RUN pip3 install pandas==0.20.3
RUN pip3 install scikit-learn==0.19.0
RUN pip3 install scipy==0.19.1
RUN pip3 install tensorflow==1.4.0
RUN pip3 install tensorflow-tensorboard==0.4.0rc3
RUN pip3 install boto3==1.4.7
RUN pip3 install msgpack-python==0.4.8
RUN pip3 install kafka-python==1.3.5
RUN pip3 install ujson==1.35
COPY tools $JULIAN_ROOT/tools
RUN ./tools/install_zk.sh
RUN ./tools/install_kafka.sh

COPY README.rst $JULIAN_ROOT/README.rst
COPY config $JULIAN_ROOT/config
COPY externs/360-etl/src $ETL_ROOT/src
COPY externs/360-etl/dev_env_rawdata.sh $ETL_ROOT/dev_env_rawdata.sh
COPY julian $JULIAN_ROOT/julian
COPY version.txt $JULIAN_ROOT/version.txt

EXPOSE 17811 17310
