# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import flask
import gunicorn
import numpy as np
import pandas as pd
import uvicorn
from flask import Flask, request
import pickle
import statsmodels.api as sm
from pandas import json_normalize
from typing import Optional
from fastapi import FastAPI
#from starlette.requests import Request
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
from logging.config import dictConfig
import os
from waitress import serve
import json

app = Flask(__name__)
#app = FastAPI()
model = loaded_model = pickle.load(open("finalized_model.sav", 'rb'))
variables = ['x5_saturday', 'x81_July', 'x81_December', 'x31_japan', 'x81_October', 'x5_sunday', 'x31_asia',
             'x81_February', 'x91', 'x81_May', 'x5_monday', 'x81_September', 'x81_March', 'x53', 'x81_November', 'x44',
             'x81_June', 'x12', 'x5_tuesday', 'x81_August', 'x81_January', 'x62', 'x31_germany', 'x58', 'x56']

threshold = .75


def investigate_object(df):
    """
    This function prints the unique categories of all the object dtype columns.
    It prints '...' if there are more than 13 unique categories.
    """
    col_obj = df.columns[df.dtypes == 'object']

    for i in range(len(col_obj)):
        if len(df[col_obj[i]].unique()) > 13:
            print(col_obj[i] + ":", "Unique Values:", np.append(df[col_obj[i]].unique()[:13], "..."))
        else:
            print(col_obj[i] + ":", "Unique Values:", df[col_obj[i]].unique())

    del col_obj


@app.route('/predict/', methods=['POST'])
#@app.post('/predict/')
def predict():
    #print(request.get_json(force=True))
    df =json_normalize(request.get_json(force=True))
    # print(df.dtypes.to_frame('dtypes').to_json("test.json"))
    df = df.replace(r'^\s*$', np.nan, regex=True)
    # investigate_object(df)

    test_val = df
    # investigate_object(raw_test)
    test_val['x12'] = test_val['x12'].str.replace('$', '', regex=True)
    test_val['x12'] = test_val['x12'].str.replace(',', '', regex=True)
    test_val['x12'] = test_val['x12'].str.replace(')', '', regex=True)
    test_val['x12'] = test_val['x12'].str.replace('(', '-', regex=True)
    test_val['x12'] = test_val['x12'].astype(float)
    test_val['x63'] = test_val['x63'].str.replace('%', '', regex=True)
    test_val['x63'] = test_val['x63'].astype(float)
    # print(type(test_val))
    df8 = test_val

    imputer = pickle.load(open("crunchyImputer.sav", 'rb'))
    std_scaler = pickle.load(open("crunchyScaler.sav", 'rb'))
    df8.to_csv("df8.csv")
    test_imputed = pd.DataFrame(imputer.transform(df8.drop(columns=['x5', 'x31', 'x81', 'x82'])),
                                columns=df8.drop(columns=['x5', 'x31', 'x81', 'x82']).columns)
    test_imputed_std = pd.DataFrame(std_scaler.transform(test_imputed), columns=test_imputed.columns)

    dumb5 = pd.get_dummies(df8['x5'], drop_first=True, prefix='x5', prefix_sep='_', dummy_na=True)
    df9 = pd.concat([test_imputed_std, dumb5], axis=1, sort=False)

    dumb31 = pd.get_dummies(df8['x31'], drop_first=True, prefix='x31', prefix_sep='_', dummy_na=True)
    df9 = pd.concat([test_imputed_std, dumb31], axis=1, sort=False)

    dumb81 = pd.get_dummies(df8['x81'], drop_first=True, prefix='x81', prefix_sep='_', dummy_na=True)
    df9 = pd.concat([test_imputed_std, dumb81], axis=1, sort=False)

    dumb82 = pd.get_dummies(df8['x82'], drop_first=True, prefix='x82', prefix_sep='_', dummy_na=True)
    df9 = pd.concat([test_imputed_std, dumb82], axis=1, sort=False)

    for i in variables:
        if i not in df9.columns:
            df9[i] = 0

    prob = pd.DataFrame(model.predict(df9[variables])).rename(columns={0: 'phat'})
    # print(prob)
    # prob['prob_bin'] = pd.qcut(prob['phat'], q=1, duplicates='drop')
    # print(prob['prob_bin'])
    # print("prob ", prob.groupby(['prob_bin']).sum())
    # print("params :", model.params)

    prob['business_outcome'] = (prob['phat'] >= threshold).astype(int)
    # prob['business_outcome'] = prob['business_outcome'].replace(np.nan, 0)
    # print(type(prob['phat']))
    df = prob
    # df9 = pd.concat([df9, df['phat']], axis=1, sort=False)
    # df.drop(df[df['phat'] < threshold].index, inplace=True)
    # ph = model.params.axes
    #print("pickle rick")
    # print(df['phat'].mean())
    # newlist
    modelJSON = json.loads(df9[variables].to_json(orient='records'))
    outcomeJSON = json.loads(df['business_outcome'].to_json(orient='records'))
    phatJSON = json.loads(df['phat'].to_json(orient='records'))
    if len(df) > 1:
        dataSet = {"business_outcome": outcomeJSON, "phat": phatJSON,
                   "model_inputs": modelJSON}
    else:
        dataSet = {"business_outcome": outcomeJSON, "phat": phatJSON,
                   "model_inputs": modelJSON}
    toReturn = json.dumps(dataSet, sort_keys=True, indent=4, separators=(',', ': '))

    return toReturn



if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=1314, threads=16)

