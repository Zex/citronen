#FROM local-dtr.zhihuiya.com/buildenv/python-base:Rel.3.6.2-2
#FROM fedora:27
FROM python:3

MAINTAINER Zex Li <top_zlynch@yahoo.com>

ENV JULIAN_ROOT "/opt/julian"
ENV ETL_ROOT "/opt/360-etl"

RUN apt-get update
RUN apt-get install unzip -y
RUN apt-get install openjdk-7-jdk openjdk-7-jre -y --fix-missing

COPY julian $JULIAN_ROOT/julian
COPY README.rst $JULIAN_ROOT/README.rst
COPY externs/360-etl/src $ETL_ROOT/src
COPY externs/360-etl/dev_env_rawdata.sh $ETL_ROOT/dev_env_rawdata.sh
COPY config $JULIAN_ROOT/config
COPY requirements.txt $JULIAN_ROOT/requirements.txt
COPY tools $JULIAN_ROOT/tools

WORKDIR $JULIAN_ROOT
RUN pip3 install -U pip
RUN pip3 install -r requirements.txt
RUN ./tools/install_zk.sh
RUN ./tools/install_kafka.sh

EXPOSE 17811 17310
