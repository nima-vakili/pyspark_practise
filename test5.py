#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 13:35:50 2022

@author: nvakili
"""

import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

sc = SparkContext()
spark = SparkSession.builder.appName('Python Spark DataFrame basic example') \
    .config('spark.some.config.option', 'some-value') \
    .getOrCreate()
    
mtcars = pd.read_csv('mtcars.csv')
mtcars.head()

sdf = spark.createDataFrame(mtcars)    
sdf.printSchema()

sdf.show() 
sdf.select('mpg').show(5)
sdf.filter(sdf['mpg'] < 18).show()

sdf.withColumn('wtTon', sdf['wt']*.45).show(5)
sdf.groupBy(['cyl']).agg({'wt':'AVG'}).show(5)
    
car_counts = sdf.groupBy(['cyl']).agg({'wt':'count'})\
            .sort('count(wt)', ascending=False) \
            .show(5)

sdf.filter(sdf['cyl']>=5).show(5)

sdf.withColumn('wtTon', sdf['wt']*.45)\
    .groupBy(['cyl']).agg({'wtTon':'AVG'}).show(5)
    
sdf.withColumn('kmpl', sdf['mpg']*.425).sort('mpg', ascending=False).show()
    
    
    
    
    
    