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

	# @abstractmethod
	# def applyDataPreparation(self)->int:
	# 	pass

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
	
	@abstractmethod
	def applyDataPreparation(self) -> int:
		pass
		#return self.__prepareData.applyDataPreparation()
	
	

class categoricalEncoding(prepDecorator):

	###				Concrete Decorator
	

		def __init__(self, prepareData: PreparingData):
			super().__init__(prepareData)
			self.labelCol = None
			self.cols = None
		
		def applyDataPreparation(self, labelCol) -> str:
			try:
				df = super().prepare_file()
				le = LabelEncoder()
				dfCols =  df.columns.values.tolist()
				df2=df.copy()

				df2[labelCol] = le.fit_transform(df2[labelCol])
				df2.to_csv(prepPath+'/prepared_data.csv', encoding='utf-8', index=False)
				global flag 
				flag = 1
			except Exception as e:
				ErrMsg = 'Error Processing the request: '.format(str(e))
				print(ErrMsg, e)
				return jsonify({'error': ErrMsg}), 500



class cleanData(prepDecorator):

	###				Concrete Decorator

	def __init__(self, prepareData: PreparingData):
		super().__init__(prepareData)


	def applyDataPreparation(self, nanThreshold) -> str:
		try:
			df = super().prepare_file()
			df2=df.copy()
			
			df2 = df2.replace("?", np.nan)
			df2.dropna(thresh=nanThreshold*len(df2),axis=1,inplace=True)
			df2.to_csv(prepPath+'/prepared_data.csv', encoding='utf-8', index=False)
			global flag 
			flag = 2
			
		except Exception as e:
			ErrMsg = 'Error Processing the request: '.format(str(e))
			print(ErrMsg, e)
			return jsonify({'error': ErrMsg}), 500
	
	
		# try:
		# 	print("123")
			
        # except Exception as e:

	    # try:
	    #     df = super().prepare_file()
		#     df2=df.copy()

        #     #Replace ? with NaN
        #     df2 = df2.replace("?", np.nan)

        #     #Drop the columns with more than x % NaN values
        #     df2.dropna(thresh=nanThreshold*len(df2),axis=1,inplace=True)
            
        #     df2.to_csv(prepPath+'/prepared_data.csv', encoding='utf-8', index=False)
        #     global flag 
        #     flag = 2
	    
	    # except Exception as e:
	    
		# 	ErrMsg = 'Error Processing the request: '.format(str(e))
	    
		# 	print(ErrMsg, e)
	    


		


class ImputeData(prepDecorator):

	###				Concrete Decorator

	def __init__(self, prepareData: PreparingData):
		super().__init__(prepareData)


	def applyDataPreparation(self) -> str:
		try:
			df = super().prepare_file()
			df2=df.copy()

			#Imputation
			imputer = SimpleImputer(strategy="mean")

			
			imputer.fit(df2)
			imputedDF = imputer.transform(df2)

			idf = pd.DataFrame(imputedDF)

			idf.to_csv(prepPath+'/prepared_data.csv', encoding='utf-8', index=False)
			

			global flag 
			flag = 3

			#return idf
		except Exception as e:
			ErrMsg = 'Error Processing the request: '.format(str(e))
			print(ErrMsg, e)
			return jsonify({'error': ErrMsg}), 500




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
			decoratorCatEnc.applyDataPreparation(labelCol)


		print('prepdata flag value: ', flag)

		decoratorcleanData = cleanData(concrete)
		decoratorcleanData.applyDataPreparation(nanThreshold)

		print('prepdata flag value: ', flag)

		decoratorImputeData = ImputeData(concrete)
		decoratorImputeData.applyDataPreparation()

		print('prepdata flag value: ', flag)

		return 'Yaha Tak' 

	except:
		e = sys.exc_info()[0]
		return jsonify({'error': str(e)})

#Main
if __name__=='__main__':

	PORT = os.environ.get('PORT',5007)
	app.run(debug=True, host='0.0.0.0', port=PORT)













