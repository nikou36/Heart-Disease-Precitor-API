import cv2
import numpy as np
import urllib.request as rq
import pandas as pd
from flask import Flask,request
from flask_restful import Resource, Api
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as tree

app = Flask(__name__)
api = Api(app)
'''
Service that predicts if a person has heart disease based on their medical data

'''
class Predict(Resource):
    '''
    Takes in a Json array where the values are the data for a patient's medical examinination.
    From left to right the entries should be:
    [age, sex(0 for female and 1 for male),chest pain type(1 : no pain, 2 : typical-agina,3: Non-agina,4:asymptotic), resting blood pressure, 
    serum cholesterol in mg/dl, is fasting blood sugar > 120mg/dl (0 for no , 1 for yes ), electrocardigraph results ( 0,1 or 2), max heart rate, 
    exercise induced chest pain(0: no, 1: yes) ]

    '''

    def post(self):
      event = request.get_json(force = True)
      data = pd.read_csv("shuffled_heart.csv")
      data.append(event)
      cols = ['age','sex','cp','trestbps','chol','fbs','restecg','exang','oldpeak','slope','ca','thal']
      goal = ['target']
      lda = LinearDiscriminantAnalysis()
      lda.fit(data[cols],data[goal])

     #Transform data using Linear Discriminant analysis
      transformed = lda.transform(data[cols])
      transformed_df = pd.concat([pd.DataFrame(transformed), data['target']], axis=1)
      
      #Logistic Regression
      logistic = LogisticRegression().fit(transformed[:-1],data['target'][:-1])
      prediction = logistic.predict(transformed[-1].reshape(-1,1))
      return prediction


'''
Service that returns model accuracy

'''

class Accuracy(Resource):
    def get(self):
      data = pd.read_csv("shuffled_heart.csv")
      cols = ['age','sex','cp','trestbps','chol','fbs','restecg','exang','oldpeak','slope','ca','thal']
      goal = ['target']
      train,test = np.split(data,[data['age'].count()//2],axis = 0)

      x_train = train[cols]
      y_train = train[goal]
      x_test = test[cols]
      y_test = test[goal]
    
      #Linear discriminant analysis
      lda = LinearDiscriminantAnalysis()
      lda.fit(x_train,y_train)
      transformed = lda.transform(data[cols])
      transformed_df = pd.concat([pd.DataFrame(transformed),data['target']] ,axis = 1)
      # 5- fold evaluation
      lda_logic_score = cross_val_score(LogisticRegression(), transformed, data[goal], cv = 5,scoring = 'accuracy')
      return lda_logic_score.mean()
    
        
        

api.add_resource(Predict, "/predict")
api.add_resource(Accuracy, "/accuracy")

if(__name__) == '__main__':
    app.run(host='0.0.0.0')