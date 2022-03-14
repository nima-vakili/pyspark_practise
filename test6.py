#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 14:31:55 2022

@author: nvakili
"""

import pandas as pd
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext

sc = SparkContext()

spark = SparkSession.builder.appName('Python Spark DataFrame basic example')\
        .config('spark.some.config.option', 'some_value')\
        .getOrCreate()
        
mtcars = pd.read_csv('mtcars.csv')
mtcars.head()

mtcars.rename(columns={'Unnamed: 0':'name'}, inplace=True)
sdf = spark.createDataFrame(mtcars)

sdf.printSchema()

sdf.createTempView('cars')

spark.sql('SELECT * FROM cars').show()
spark.sql('SELECT mpg FROM cars').show()

spark.sql("SELECT * FROM cars where mpg>20 AND cyl < 6").show(5)
spark.sql("SELECT count(*), cyl from cars GROUP BY cyl").show()

from pyspark.sql.functions import pandas_udf, PandasUDFType

@pandas_udf("float")
def convert_wt(s: pd.Series) -> pd.Series:
    # The formula for converting from imperial to metric tons
    return s * 0.45

spark.udf.register("convert_weight", convert_wt)

spark.sql('SELECT *, wt AS weight_imperial, convert_weight(wt) as weight_metric FROM cars')\
    .show()

spark.sql("SELECT * FROM cars WHERE name like 'Merc%'").show()

@pandas_udf("float")
def convert_mileage(s: pd.Series) -> pd.Series:
    return s*.425
spark.udf.register('convert_mileage', convert_mileage)
spark.sql('SELECT *, mpg AS mpg, convert_mileage(mpg) as kmpl From cars').show()













