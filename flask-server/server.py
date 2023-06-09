from flask import Flask, request,send_file, render_template, jsonify
import pandas as pd
import json as js
from io import StringIO
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

#Input Microservice

@app.route("/members")
def members():
    newpath = r'./test' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        

    file = request.files['file']

  
    selected_column = request.form['labelCol']
    print('HERE IT ISSS!!!!!!!!!!!!!!!!!!!!!!: ',selected_column)
    return {"members": ["M1","M2","M4"]}


# @app.route('/')
# def index():
#     return 


if __name__ == "__main__":
    Port = os.environ.get('PORT',5006)
    app.run(debug=True, host='0.0.0.0', port=Port)