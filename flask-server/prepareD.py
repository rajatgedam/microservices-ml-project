
import sys

from flask import Flask, request
import pandas as pd
import json as js
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder


from typing import List
import os


from abc import ABC, ABCMeta, abstractmethod

class PreparingData(metaclass=ABCMeta):
	#Component

	@abstractmethod
	def prepare_file(self) -> int:
		pass

#Called first to instantiate
class ConcreteComp(PreparingData):
	#Concrete Component

	def prepare_file(self)-> int:
		return pd.read_csv('breast-cancer.csv')

class prepDecorator(PreparingData):
	#Decorator
	def __init__(self, prepareData: PreparingData):
		self.__prepareData = prepareData


	@abstractmethod
	def prepare_file(self) -> int:
		return self.__prepareData.prepare_file()
	
	#Test Method returns string
	@abstractmethod
	def selectfeaturing(self) -> str:
		return self.__prepareData.selectfeaturing()


class categoricalEncoding(prepDecorator):

	###				Concrete Decorator

	def __init__(self, prepareData: PreparingData):
		super().__init__(prepareData)
		self.labelCol = None
		self.cols = None
	
	def prepare_file(self, labelCol) -> str:
		df = super().prepare_file()
		le = LabelEncoder()
		dfCols =  df.columns.values.tolist()
		df2=df.copy()

		df2[labelCol] = le.fit_transform(df2[labelCol])
		return df2, dfCols

class cleanData(prepDecorator):

	###				Concrete Decorator

	def __init__(self, prepareData: PreparingData):
		super().__init__(prepareData)


	def prepare_file(self, nanThreshold) -> str:

		df = super().prepare_file()
		df2=df.copy()

		#Replace ? with NaN
		df2 = df2.replace("?", np.nan)

		#Drop the columns with more than x % NaN values
		df2.dropna(thresh=nanThreshold*len(df2),axis=1,inplace=True)


class ImputeData(prepDecorator):

	###				Concrete Decorator

	def __init__(self, prepareData: PreparingData):
		super().__init__(prepareData)


	def prepare_file(self) -> str:

		df = super().prepare_file()
		df2=df.copy()

		#Imputation

		imputer = SimpleImputer(strategy="mean")

		#Encountered error before adding categorical encoding

		imputer.fit(df2)
		imputedDF = imputer.transform(df2)

		idf = pd.DataFrame(imputedDF)
		
		return idf


#Main
if __name__=='__main__':

	PORT = os.environ.get('PORT',5011)
	labelCol = ''
	
	labelCol = 'diagnosis'
	nanThreshold = 0.6
	#Read file
	df = pd.read_csv('breast-cancer.csv')

	prepdata(df,labelCol,nanThreshold)

 

#-----------------------------------------#-----------------------------------------#-----------------------------------------#-----------------------------------------

#app = Flask(__name__)

#@app.route('/prepdata', methods=['POST'])
#def prepdata(file, labelCol, nanThreshold):

#	#initialize variables
#	#nanThreshold = 0.6
#	enc = OrdinalEncoder()
#	le = LabelEncoder()
#	imputer = SimpleImputer(strategy="mean")

#	df=file.copy()
#	df2=file.copy()

#	labelCol_cat=labelCol+'_cat'

#	#Label categorical data
#	#get label from user 

#	#df[labelCol] = enc.fit_transform(df[[labelCol]])
#	df2[labelCol] = le.fit_transform(df2[labelCol])

#	#ordinal encoder

#	#print(enc.categories_)
#	#print(df2.shape)
	
#	#Replace ? with NaN
#	df2 = df2.replace("?", np.nan)
#	print(df2)
#	print(df2.shape)
#	#Check the % of NaN in each column
#	x=df2.isna().sum()/len(df2)*100
#	#print(x[2])
#	#print(type(x))
	
#	#Drop the columns with more than x % NaN values
#	df2.dropna(thresh=nanThreshold*len(df2),axis=1,inplace=True)

#	#print(df2)

#	#Imputation
#	#Encountered error before adding categorical encoding

#	imputer.fit(df2)
#	imputedDF = imputer.transform(df2)

#	#print(imputedDF)
#	#print(pd.DataFrame(imputedDF))

#	idf = pd.DataFrame(imputedDF)

#	print(idf)

#	return 0

##Main


#labelCol = ''

	
#labelCol = 'diagnosis'
#nanThreshold = 0.6
##Read file
#df = pd.read_csv('breast-cancer.csv')

#prepdata(df,labelCol,nanThreshold)
