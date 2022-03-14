#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 14:53:14 2022

@author: nvakili
"""

from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.ml.feature import VectorAssembler, Normalizer, StandardScaler
from pyspark.ml.stat import Correlation
from pyspark.ml.regression import LinearRegression

import matplotlib.pyplot as plt


sc = SparkContext()

spark = SparkSession.builder.appName('Python Spark DataFrames basic example')\
        .config('spark.some.config.option', 'some-value')\
        .getOrCreate()

cars = spark.read.csv('cars.csv', header=True, inferSchema=True)       
cars.show()
cars.printSchema()

#  number of rows
cars.count()
cars.distinct().count()
cars.columns()
len(cars.columns)

# to check if there is a Null value
cars.filter("horsepower is NULL").show()
cars.filter(cars.mpg.isNull()).show()
cars.where('mpg is Null').show()

import sys
sys.exit()

cars = cars.na.drop('any')

cars.withColumn('hp', cars.horsepower.cast('double')).show()
cars = cars.withColumn('hp', cars.horsepower.cast('int'))
cars.printSchema()

# or with panda
if False:
    import pandas as pd
    cars2 = pd.read_csv('cars2.csv', header=None, names=["mpg", "hp", "weight"])
    cars2.head()
    sdf = spark.createDataFrame(cars2)
    sdf.printSchema()  


assembler = VectorAssembler(inputCols=['hp', 'weight'],
                            outputCol='features')

output = assembler.transform(cars).select('features', 'mpg')

train, test = output.randomSplit([.75, .25])

r1 = Correlation.corr(train, 'features').head()
print('Pearson correlation matrix:\n' + str(r1[0]))

r2 = Correlation.corr(train, 'features', 'spearman').head()
print('Spearman correlation matrix:\n' + str(r2[0]))
    
plt.figure()
plt.scatter(cars.select('hp').head(100), cars.select("weight").head(100))
plt.xlabel("horsepower")
plt.ylabel("weight")
plt.title("Correlation between Horsepower and Weight")
plt.show()    
    
normalizer = Normalizer(inputCol='features', outputCol='features_normalized', p=1.0)
train_norm = normalizer.transform(train)
print('Normalized using L^1 norm')    
train_norm.show(5, truncate=False)

standard_scaler = StandardScaler(inputCol="features", outputCol="features_scaled")
train_model = standard_scaler.fit(train)
train_scaled = train_model.transform(train)
train_scaled.show(5, truncate=False)    

test_scaled = train_model.transform(test)
test_scaled.show(5, truncate=False)    
    
lr = LinearRegression(featuresCol='features_scaled', labelCol='mpg', maxIter=100)
lrModel = lr.fit(train_scaled)

print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))

trainingSummary = lrModel.summary

print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("R-squared: %f" % trainingSummary.r2)    
    
lrModel.transform(test_scaled).show(5)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
