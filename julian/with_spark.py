# Data processing with Spark
# Author: Zex Li <top_zlynch@yahoo.com>
import os
import pyspark as spark

psql_driver_path = "/media/sf_patsnap/dl/postgresql-42.1.4.jar"
#os.environ['PYSPARK_SUBMIT_ARGS'] = \
#        '--driver-class-path {} --jars {} pyspark-shell'.format(psql_driver_path, psql_driver_path)

def run():
    data_path = "/home/zex/lab/citronen/data/springer/lang/english_tiny.csv"
    builder = spark.sql.SparkSession.builder\
            .master('local')\
            .appName("Linear Regression")\
            .config("spark.executor.memory", "1gb")\
            .getOrCreate()
    context = builder.sparkContext
    rdd = context.textFile(data_path)
    print("[dummy] {}".format(rdd.top(2)))

    df = builder.read.format('csv').option('header','true').option('delimiter','#').load(data_path)
    print(df)


def conn_postgres(context):

    properties = {
            "user":"root",
            "password":"patsnap360",
            "driver": "org.postgresql.Driver"
            }
    url = "jdbc:postgresql://192.168.44.101:5432"
    table_name = "data_services_etl"
    sql_context = spark.sql.SQLContext(context)
    df = spark.sql.DataFrameReader(sql_context).jdbc(
            url=url,
            properties=properties,
            table=table_name,
            )
    print(df)

if __name__ == '__main__':
    run()


