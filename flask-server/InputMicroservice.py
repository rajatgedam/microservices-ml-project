import os
from io import StringIO
import requests
import json as js
from flask import Flask, request,send_file, render_template, jsonify
import pandas as pd
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app,resources={r'*':{'origins':'http://localhost:3000'}})

@app.route('/sendInputDataset', methods=['POST'])
def sendInputDataset():

    try:

        inputPath = r'./InputDataset' 
        if not os.path.exists(inputPath):
            os.makedirs(inputPath)

        
        # Get the uploaded file from React frontend
        filename=secure_filename(request.files['dataset'].filename)
        file = request.files['dataset']
        file.save(os.path.join(inputPath, filename))
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
        MSprepUrl = 'http://localhost:5007/prepdata'

        payload = {'labelCol':labelColumn}

        response = requests.post(MSprepUrl, json=payload)

        print('******* Response from PrepData ******** : ',response.reason)

        if response.status_code == 200:
            print('POST Success')
            return {"success": "File Received Successfully!"}
        else:
            print('POST Failed')
            return {"failed": response.reason }
        
    except Exception as e:
        ErrMsg = 'Error Processing the request: '.format(str(e))
        return jsonify({'error': e.args, 'ErrMsg': ErrMsg}), 500


if __name__ == "__main__":
    Port = os.environ.get('PORT',5006)
    app.run(debug=True, host='0.0.0.0', port=Port)