#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 15:19:23 2022

@author: nvakili
"""

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('Missing').getOrCreate()

training_data = spark.read.csv('test2.csv', header=True, inferSchema=True)
training_data.show()
training_data.columns
training_data = training_data.na.drop('any')

from pyspark.ml.feature import VectorAssembler
featureassembler = VectorAssembler(inputCols=['age', 'Experience'], outputCol='Independent Features')
output = featureassembler.transform(training_data)
finalized_data = output.select('Independent Features', 'Salary')

from pyspark.ml.regression import LinearRegression
train_data, test_data = finalized_data.randomSplit([.75, .25])
regressor = LinearRegression(featuresCol='Independent Features', labelCol='Salary')
regressor =regressor.fit(train_data)

regressor.coefficients
pred_results = regressor.evaluate(test_data)
pred_results.predictions.show()
pred_results.meanAbsoluteError, pred_results.meanSquaredError
