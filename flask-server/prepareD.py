from flask import Flask, jsonify, request
import pandas as pd
import json as js
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from typing import List
import os
import sys

from abc import ABC, ABCMeta, abstractmethod

inputPath = r'./InputDataset'
prepPath = r'./PreparedDataset' 
flag = -1
#flag for selecting input
class PreparingData(metaclass=ABCMeta):
	#Component

	@abstractmethod
	def prepare_file(self) -> int:
		pass

#Called first to instantiate
class ConcreteComp(PreparingData):
	#Concrete Component

	def prepare_file(self)-> int:
		if (flag == -1):
			return pd.read_csv(inputPath+'/process_data.csv')
		else:
			return pd.read_csv(prepPath+'/prepared_data.csv')

class prepDecorator(PreparingData):
	#Decorator
	def __init__(self, prepareData: PreparingData):
		self.__prepareData = prepareData


	@abstractmethod
	def prepare_file(self) -> int:
		return self.__prepareData.prepare_file()
	
	#Test Method returns string
	# @abstractmethod
	# def selectfeaturing(self) -> str:
	# 	return self.__prepareData.selectfeaturing()


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
		print('Before Save - categoricalEncoding Decorator')
		print("df.shape",df.shape)
		print(df2)
		df2.to_csv(prepPath+'/prepared_data.csv', encoding='utf-8', index=False)
		global flag 
		flag = 1
		print('After Save - categoricalEncoding Decorator/ flag value: ', flag)
		
		#return df2, dfCols

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
		
		print('Before Save - cleanData Decorator')
		print("df.shape",df.shape)
		print(df2)
		df2.to_csv(prepPath+'/prepared_data.csv', encoding='utf-8', index=False)
		global flag 
		flag = 2

		print('After Save - cleanData Decorator / flag value: ', flag)
		


class ImputeData(prepDecorator):

	###				Concrete Decorator

	def __init__(self, prepareData: PreparingData):
		super().__init__(prepareData)


	def prepare_file(self) -> str:
		try:
			df = super().prepare_file()
			df2=df.copy()

			#Imputation
			print('Start - ImputeData Decorator')
			imputer = SimpleImputer(strategy="mean")

			#Encountered error before adding categorical encoding

			imputer.fit(df2)
			imputedDF = imputer.transform(df2)

			idf = pd.DataFrame(imputedDF)
			print("df.shape",df.shape)
			print('Before Save - ImputeData Decorator')
			idf.to_csv(prepPath+'/prepared_data.csv', encoding='utf-8', index=False)
			
			print(idf)

			global flag 
			flag = 3

			print('After Save - ImputeData Decorator/ flag value: ', flag)
			#return idf
		except Exception as e:
			ErrMsg = 'Error Processing the request: '.format(str(e))
			print(ErrMsg, e)
			#return jsonify({'error': ErrMsg}), 500




from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app,resources={r'*':{'origins':'http://localhost:5006'}})


@app.route('/prepdata', methods=['POST'])
def prepdata():
	try:
		
		payload = request.get_json()
		labelCol = payload.get('labelCol')
		labelPresent = False
		if  labelCol:
			labelPresent = True			
			labelCol = labelCol.rstrip('\r\n')
			print('^.^ ^.^ ^.^ ^.^ Label Present ^.^ ^.^ ^.^ ^.^')
		else:			
			print('>.< >.< >.< >.< Label Absent >.< >.< >.< >.<')

		#Threshold for cleaning data
		#Can be made dynamic by adding to url payload
		nanThreshold = 0.6
	
		#Check if directory path exists
		
		if not os.path.exists(prepPath):
			os.makedirs(prepPath)

		#Read file
		#df = pd.read_csv('breast-cancer.csv')

		#Instantiation
		concrete = ConcreteComp()

		if labelPresent:
			decoratorCatEnc = categoricalEncoding(concrete)
			decoratorCatEnc.prepare_file(labelCol)

		print('prepdata flag value: ', flag)

		decoratorcleanData = cleanData(concrete)
		decoratorcleanData.prepare_file(nanThreshold)

		print('prepdata flag value: ', flag)

		decoratorImputeData = ImputeData(concrete)
		decoratorImputeData.prepare_file()

		print('prepdata flag value: ', flag)

		return 'Yaha Tak' 

	except:
		e = sys.exc_info()[0]
		return jsonify({'error': str(e)})

#Main
if __name__=='__main__':

	PORT = os.environ.get('PORT',5007)
	app.run(debug=True, host='0.0.0.0', port=PORT)














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
