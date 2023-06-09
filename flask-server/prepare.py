from flask import Flask, request, jsonify
import pandas as pd
import json as js
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app,resources={r'*':{'origins':'http://localhost:5006'}})


@app.route('/prepdata/<labelCol>', methods=['POST'])
def prepdata(labelCol):

	try:
		print('Start')
		inputPath = r'./InputDataset'
		#initialize variables
		nanThreshold = 0.6
		enc = OrdinalEncoder()
		
		imputer = SimpleImputer(strategy="mean")
		#labelCol = ''
		prepPath = r'./PreparedDataset' 
		if not os.path.exists(prepPath):
			os.makedirs(prepPath)
	    
		#Read file
		df = pd.read_csv(inputPath+'/process_data.csv')
		#flask-server\dataset\breast-cancer.csv
		#labelCol = 'diagnosis'
		
		#Label categorical data
		#get label from user 
		le = LabelEncoder()
		dfCols =  df.columns.values.tolist()
		df2=df.copy()

		df[labelCol] = le.fit_transform(df2[labelCol])

		#df[labelCol] = enc.fit_transform(df[[labelCol]])

		#ordinal encoder

		#print(enc.categories_)
		print(df.shape)
		
		#Replace ? with NaN
		df = df.replace("?", np.nan)
		print(df)
		print(df.shape)
		#Check the % of NaN in each column
		x=df.isna().sum()/len(df)*100
		#print(x[2])
		#print(type(x))
		
		#Drop the columns with more than x % NaN values
		df.dropna(thresh=nanThreshold*len(df),axis=1,inplace=True)

		#print(df2)

		#Imputation
		
		imputer.fit(df)
		imputedDF = imputer.transform(df)
		#print(imputedDF)
		#print(pd.DataFrame(imputedDF))

		idf = pd.DataFrame(imputedDF)

		print('Before Save')
		idf.to_csv(prepPath+'/prepared_data.csv', encoding='utf-8', index=False)
		print('After Save')
		print(idf)

		return 'Yaha Tak'
	
	except Exception as e:
		ErrMsg = 'Error Processing the request: '.format(str(e))
		print(e)
		return jsonify({'error': ErrMsg}), 500


#Main
if __name__ == "__main__":
    Port = os.environ.get('PORT', 5007)
    app.run(debug=True, host='0.0.0.0', port=Port)
	#prepdata()
