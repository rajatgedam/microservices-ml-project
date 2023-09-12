import sys
import os
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect
from werkzeug.utils import secure_filename
import io
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import IsolationForest
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import lightgbm as lgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

file_path = "./prepared_data.csv" #location where the preprocessed file has been stored by preprocess_data microservice

app=Flask(__name__)

# Context class
class ModelContext:
    def __init__(self, strategy):
        self.strategy = strategy

    def execute_strategy(self):
        return self.strategy.execute()

# Abstract Strategy class
class AbstractStrategy:
    def execute(self):
        pass

# Concrete Strategy class for Random Forest
class RandomForestStrategy(AbstractStrategy):
    def execute(self):
        try:
            data = pd.read_csv(file_path)
            X = data.drop("2", axis=1)
            y = pd.cut(data["2"], bins=3, labels=[0, 1, 2])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            y_pred = rf_model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='weighted')
            accuracy = accuracy_score(y_test, y_pred)
            print("F1 score:", f1)
            print("Accuracy:", accuracy)
        except:
            e = sys.exc_info()[0]
            return jsonify({'error': str(e)})

# Concrete Strategy class for Isolation Forest
class IsolationForestStrategy(AbstractStrategy):
    def execute(self):
        try:
            data = pd.read_csv(file_path)
            X = data.iloc[:, :1]
            y = data.iloc[:, 1]
            isofor = IsolationForest(random_state=42)
            isofor.fit(X)
            y_pred = isofor.predict(X)
            y_pred_binary = [1 if x == 1 else 0 for x in y_pred]
            accuracy = accuracy_score(y, y_pred_binary)
            f1 = f1_score(y, y_pred_binary, average='weighted')
            print("Accuracy: {:.2f}%".format(accuracy*100))
            print("F1 score: {:.2f}".format(f1))
        except:
            e = sys.exc_info()[0]
            return jsonify({'error': str(e)})

# Concrete Strategy class for Logistic Regression
class LogisticRegressionStrategy(AbstractStrategy):
    def execute(self):
        try:
            df = pd.read_csv(file_path)
            X = df.drop('1', axis=1)
            y = df['1']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
            lr = LogisticRegression(random_state=42, solver='liblinear')
            grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='f1')
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            best_lr = grid_search.best_estimator_
            y_pred = best_lr.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            print("Best parameters:", best_params)
            print("F1 score:", f1)
            print("Accuracy:", accuracy)
            perf = PerformanceMetrics()
            perf.saveToTextFile(f1,accuracy)
        except:
            e = sys.exc_info()[0]
            return jsonify({'error': str(e)})
        
class SVMStrategy(AbstractStrategy):
    def execute(self):
        try:
            data = pd.read_csv(file_path)
            data.dropna(inplace=True)
            X = data[['4', '5', '6']] 
            y = data['2'] 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            svm_model = SVR(kernel='linear', C=1)
            svm_model.fit(X_train, y_train)
            y_pred = svm_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            print("Mean Squared Error: {:.2f}".format(mse)) 
        except:
            e = sys.exc_info()[0]
            return jsonify({'error': str(e)})
           
class lightGBMStrategy(AbstractStrategy):
    def execute(self):
        try:
            df = pd.read_csv(file_path)
            X = df.drop('2', axis=1)
            y = df['2']
            threshold = 0.5
            y = np.where(y > threshold, 1, 0)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = lgb.LGBMClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            print("F1 score: ", f1)
            print("Accuracy: ", accuracy)
            perf = PerformanceMetrics()
            perf.saveToTextFile(f1,accuracy*100)

        except:
            e = sys.exc_info()[0]
            return jsonify({'error': str(e)})
              
class LDAStrategy(AbstractStrategy):
    def execute(self):
        try:
            data = pd.read_csv(file_path)
            X = data.iloc[:, :1]
            y = data.iloc[:, 1]
            le = LabelEncoder()
            y = le.fit_transform(y)
            lda = LinearDiscriminantAnalysis()
            lda.fit(X, y)
            y_pred = lda.predict(X)
            accuracy = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, average='weighted')
            print("Accuracy: {:.2f}%".format(accuracy*100))
            print("F1 score: {:.2f}".format(f1))
            perf = PerformanceMetrics()
            perf.saveToTextFile(f1,accuracy*100)
        except:
            e = sys.exc_info()[0]
            return jsonify({'error': str(e)})
                
class KMeanStrategy(AbstractStrategy):
    def execute(self):
        try:
            print('Kmeans Triggered.')
            data = pd.read_csv(file_path)
            X = data.iloc[:, :1]
            y = data.iloc[:, 1]
            kmeans = KMeans(n_clusters=2, random_state=42)
            kmeans.fit(X)
            y_pred = kmeans.predict(X)
            accuracy = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, average='weighted')
            print("Accuracy: {:.2f}%".format(accuracy*100))
            print("F1 score: {:.2f}".format(f1))
            perf = PerformanceMetrics()
            perf.saveToTextFile(f1,accuracy*100)
            
            return 1
        except:
            e = sys.exc_info()[0]
            return jsonify({'error': str(e)})

class PerformanceMetrics():
    def __init__(self, outputPath = r'./outputFiles'):
        self.outputPath = outputPath
        if not os.path.exists(outputPath):
            os.makedirs(self.outputPath)

    def saveToTextFile(self,f1,avg):
        #outputPath = r'./outputFiles' 
        #if not os.path.exists(outputPath):
        #    os.makedirs(outputPath)

        newFile = os.path.join(self.outputPath,'Scores.txt')
        #with open('./Presentation/Scores.txt','w') as file:
        with open(newFile,'w') as file:
            file.write(f'F1 score: {f1}')
            file.write(f'Avg: {avg}')
        print('Performance Metrics Saved.')
        return 1

    def saveAUROCPlot(self,lbl_true, score_pred):
        plt.figure()
        plt.title('AUC ROC')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.grid()
        #plt.savefig('./Presentation/plot_AUROC.png')
        #pltPath=os.path.join(self.outputPath, 'plot_AUROC.png')
        #plt.savefig('plot_AUROC.png')

        pltPath=os.path.join(self.outputPath, 'plot_AUROC.png')
        plt.savefig(pltPath)

#app = Flask(__name__) 

@app.route('/execute_strategy', methods=['POST'])
def execute_strategy():
    #strategy_name = request.json.get('strategy_name')
    strategy_name = request.form['strategy_name']
    if not strategy_name:
        return jsonify({'error': 'Strategy name not provided'})

    if strategy_name == "RandomForest":
        strategy = RandomForestStrategy()
    elif strategy_name == "IsolationForest":
        strategy = IsolationForestStrategy()
    elif strategy_name == "LogisticRegression":
        strategy = LogisticRegressionStrategy()
    elif strategy_name == "SVM":
        strategy = SVMStrategy()
    elif strategy_name == "lightGBM":
        strategy = lightGBMStrategy()
    elif strategy_name == "LDA":
        strategy = LDAStrategy()
    elif strategy_name == "KMeans":
        strategy = KMeanStrategy()    
    else:
        return jsonify({'error': 'Invalid strategy name'})

    result = strategy.execute()
    #result = strategy.execute(filepath)
    #removed file path as a param as it is globally declared.
    return jsonify({'successMsg':'Metrics Calculated Successfully.'}), 200

if __name__ == '__main__':
    PORT = os.environ.get('PORT', 5010)
    app.run(debug=True, host='0.0.0.0', port=PORT)