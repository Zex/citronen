# Data processing with Spark
# Author: Zex Li <top_zlynch@yahoo.com>
import pyspark as spark

def run():
    data_path = "/home/zex/lab/citronen/data/springer/lang/english_tiny.csv"

    builder = SparkSession.builder\
            .master('local')\
            .appName("Linear Regression")\
            .config("spark.executor.memory", "1gb")\
            .getOrCreate()
    context = builder.sparkContext
    rdd = context.textFile(data_path)
    print("[dummy] {}".format(rdd.top(2)))

    df = builder.read.format('csv').option('header','true').option('delimiter','#').load(data_path)
    print(df)

