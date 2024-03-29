from flask import Flask, request,send_file, render_template, jsonify
import pandas as pd
import json as js
from io import StringIO
from werkzeug.utils import secure_filename
import os
import requests
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app,resources={r'*':{'origins':'http://localhost:3000'}})

@app.route('/handleInput', methods=['POST'])
def handleinput():

    try:

        inputPath = r'./InputDataset' 
        if not os.path.exists(inputPath):
            os.makedirs(inputPath)

        print("In here")
        # Get the uploaded file from React frontend
        file = request.files['dataset']
        labelColumn = ''
        #Get label from request
        if 'labelCol' in request.form:
            labelColumn = request.form['labelCol']
            labelColumn = labelColumn.rstrip('\r\n')
        if not labelColumn:
            print('>.< >.< >.< >.< Label Absent >.< >.< >.< >.<')

        
        df = pd.read_csv(file)
        
        df.to_csv(inputPath+'/process_data.csv', encoding='utf-8', index=False)

        #Calling Next Microservice here
        #PORT 5007 for PrepareData 
        
        # if not labelColumn:
        #     MSprepUrl = 'http://localhost:5007/prepdata'
        # else:            
        #     MSprepUrl = 'http://localhost:5007/prepdata/{}'.format(labelColumn)
        # print('POST URL rn: ',MSprepUrl)
        # response = requests.post(MSprepUrl)

        MSprepUrl = 'http://localhost:5007/prepdata'

        payload = {'labelCol':labelColumn}

        response = requests.post(MSprepUrl, json=payload)

        print('******* Response from PrepData ******** : ',response.reason)

        if response.status_code == 200:
            print('POST Success')
            return {"members": ["M1","M2","M4"]}
        else:
            print('POST Failed')
            return {"Dismembers": ["D1","D2","D3"]}
        
    except Exception as e:
        ErrMsg = 'Error Processing the request: '.format(str(e))
        return jsonify({'error': e.args, 'ErrMsg': ErrMsg}), 500

    
    # return jsonify({
    #     #'csv': df.to_dict(),
    #     #'first_column': first_column,
    #     'selected_column': selected_column
    # })

if __name__ == "__main__":
    Port = os.environ.get('PORT',5006)
    app.run(debug=True, host='0.0.0.0', port=Port)