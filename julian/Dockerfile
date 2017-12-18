#FROM local-dtr.zhihuiya.com/buildenv/python-base:Rel.3.6.2-2
FROM fedora:27

MAINTAINER Zex Li <top_zlynch@yahoo.com>

ENV JULIAN_ROOT "/opt/julian"
ENV ETL_ROOT "/opt/360-etl"

COPY julian $JULIAN_ROOT/julian
COPY tools $JULIAN_ROOT/tools
COPY config $JULIAN_ROOT/config
COPY requirements.txt $JULIAN_ROOT/requirements.txt
COPY README.rst $JULIAN_ROOT/README.rst
COPY externs/360-etl/src $ETL_ROOT/src
COPY externs/360-etl/dev_env_rawdata.sh $ETL_ROOT/dev_env_rawdata.sh

WORKDIR $JULIAN_ROOT
#RUN dnf update -y;
RUN dnf install python3-pip python3 unzip -y
RUN ./tools/setup_kafka.sh 
RUN pip3 install -U pip; pip3 install -r requirements.txt

EXPOSE 17811 17310
