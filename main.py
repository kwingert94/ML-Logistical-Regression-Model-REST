from datetime import datetime
from typing import Any, List
import numpy as np
import pandas as pd
import uvicorn
import pickle
import statsmodels.api as sm
from pandas import json_normalize
from fastapi.encoders import jsonable_encoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import json
import yaml
from find_variables import *
from loguru import logger
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Load Config Data
from inputData import SingleRow, MultipleRows

with open(r'config/config.yml') as file:
    config_data = yaml.load(file, Loader=yaml.FullLoader)

app = FastAPI(
    title="GLM 26",
    description='REST endpoint for predicting if customer will make purchase.',
    version="0.0.2",)


# Unpickle required objects
model = pickle.load(open(config_data["pickled_model_path"], 'rb'))
imputer = pickle.load(open(config_data["pickled_imputer_path"], 'rb'))
std_scaler = pickle.load(open(config_data["pickled_scaler_path"], 'rb'))

# Setup variables needed to evaluate incoming data
variables = find_variables(model.params.axes)
rawVars = []
for i in range(len(variables)):
    rawVars.append(variables[i].split('_')[0])
threshold = config_data["cuttof_threshold"]


@app.post('/predict/multi')
async def predict_multiple(request: Request):
    logger.info("Prediction Fired at: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    # Load JSON File and clean up data.
    try:
        df = json_normalize(await request.json())
        #df = pd.DataFrame(jsonable_encoder(input_data))
        #logger.info("Input_Data: " + str(input_data))
        #for j in range(len(input_data)):
        #    df.loc[j] = list(input_data[j].dict().values())

        print(df.head())

    except Exception as e:
        logger.info("Error: " + str(e))
        return "Invalid Input Data", 400
    logger.info("Made it here at: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))


    # Replace empty strings with NaNs, and convert currency and percentages to floats.
    # Todo find dynamic way to do this, remove hard coded references
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df['x12'] = (df['x12'].replace('[\$,)]', '', regex=True)
                 .replace('[(]', '-', regex=True).astype(float))
    df['x63'] = df['x63'].str.replace('%', '', regex=True).astype(float)

    # Run imputer to fill in gaps for data and then scale the values up
    test_imputed = pd.DataFrame(imputer.transform(df.drop(columns=['x5', 'x31', 'x81', 'x82'])),
                                columns=df.drop(columns=['x5', 'x31', 'x81', 'x82']).columns)
    test_imputed_std = pd.DataFrame(std_scaler.transform(test_imputed), columns=test_imputed.columns)

    # Converts columns of strings into a boolean matrix.
    # Todo find dynamic way to do this, remove hard coded references
    dumb5 = pd.get_dummies(df['x5'], drop_first=True, prefix='x5', prefix_sep='_', dummy_na=True)
    df_imputed = pd.concat([test_imputed_std, dumb5], axis=1, sort=False)

    dumb31 = pd.get_dummies(df['x31'], drop_first=True, prefix='x31', prefix_sep='_', dummy_na=True)
    df_imputed = pd.concat([test_imputed_std, dumb31], axis=1, sort=False)

    dumb81 = pd.get_dummies(df['x81'], drop_first=True, prefix='x81', prefix_sep='_', dummy_na=True)
    df_imputed = pd.concat([test_imputed_std, dumb81], axis=1, sort=False)

    dumb82 = pd.get_dummies(df['x82'], drop_first=True, prefix='x82', prefix_sep='_', dummy_na=True)
    df_imputed = pd.concat([test_imputed_std, dumb82], axis=1, sort=False)

    # Not all data sets will have all of the variables required for the model
    # Append columns of 0s for the missing data
    for j in variables:
        if j not in df_imputed.columns:
            df_imputed[j] = 0

    # Perform the model prediction using the unpickled Logistical Regression finalized model
    # Store values under phat
    model_predict = pd.DataFrame(model.predict(df_imputed[variables])).rename(columns={0: 'phat'})

    # Determine outcome by comparing phat to threshold provided by business partner
    # See config file for threshold value
    model_predict['business_outcome'] = (model_predict['phat'] >= threshold).astype(int)

    # Convert Pandas to the JSON format, json.loads makes the end result human readable
    finalDf = pd.concat([df_imputed[variables], model_predict], axis=1, sort=False).astype('O')
    dataSet = {}
    for j in finalDf.index:
        dataSet[str(j)] = json.loads(finalDf.loc[j].to_json(orient='columns'))
    toReturn = json.dumps(dataSet, sort_keys=True, separators=(',', ': '))
    #logger.info("Predictions: " + str(toReturn))
    return JSONResponse(content=toReturn)


@app.post('/predict/single')
async def predict_single(input_data: SingleRow) -> Any:
    logger.info("Prediction Fired at: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    # Load JSON File and clean up data.
    try:

        df = pd.DataFrame(columns=list(input_data.dict().keys()))
        df.loc[0] = list(input_data.dict().values())
        logger.info("Input_Data: " + str(input_data))
        print(df.head())

    except Exception as e:
        logger.info("Error: " + str(e))
        return "Invalid Input Data", 400
    logger.info("Made it here at: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

    # Replace empty strings with NaNs, and convert currency and percentages to floats.
    # Todo find dynamic way to do this, remove hard coded references
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df['x12'] = (df['x12'].replace('[\$,)]', '', regex=True)
                 .replace('[(]', '-', regex=True).astype(float))
    df['x63'] = df['x63'].str.replace('%', '', regex=True).astype(float)

    # Run imputer to fill in gaps for data and then scale the values up
    test_imputed = pd.DataFrame(imputer.transform(df.drop(columns=['x5', 'x31', 'x81', 'x82'])),
                                columns=df.drop(columns=['x5', 'x31', 'x81', 'x82']).columns)
    test_imputed_std = pd.DataFrame(std_scaler.transform(test_imputed), columns=test_imputed.columns)

    # Converts columns of strings into a boolean matrix.
    # Todo find dynamic way to do this, remove hard coded references
    dumb5 = pd.get_dummies(df['x5'], drop_first=True, prefix='x5', prefix_sep='_', dummy_na=True)
    df_imputed = pd.concat([test_imputed_std, dumb5], axis=1, sort=False)

    dumb31 = pd.get_dummies(df['x31'], drop_first=True, prefix='x31', prefix_sep='_', dummy_na=True)
    df_imputed = pd.concat([test_imputed_std, dumb31], axis=1, sort=False)

    dumb81 = pd.get_dummies(df['x81'], drop_first=True, prefix='x81', prefix_sep='_', dummy_na=True)
    df_imputed = pd.concat([test_imputed_std, dumb81], axis=1, sort=False)

    dumb82 = pd.get_dummies(df['x82'], drop_first=True, prefix='x82', prefix_sep='_', dummy_na=True)
    df_imputed = pd.concat([test_imputed_std, dumb82], axis=1, sort=False)

    # Not all data sets will have all of the variables required for the model
    # Append columns of 0s for the missing data
    for j in variables:
        if j not in df_imputed.columns:
            df_imputed[j] = 0

    # Perform the model prediction using the unpickled Logistical Regression finalized model
    # Store values under phat
    model_predict = pd.DataFrame(model.predict(df_imputed[variables])).rename(columns={0: 'phat'})

    # Determine outcome by comparing phat to threshold provided by business partner
    # See config file for threshold value
    model_predict['business_outcome'] = (model_predict['phat'] >= threshold).astype(int)

    # Convert Pandas to the JSON format, json.loads makes the end result human readable
    finalDf = pd.concat([df_imputed[variables], model_predict], axis=1, sort=False).astype('O')
    dataSet = {}
    for j in finalDf.index:
        dataSet[str(j)] = json.loads(finalDf.loc[j].to_json(orient='columns'))

    toReturn = json.dumps(dataSet, sort_keys=True, separators=(',', ': '))
    logger.info("Predictions: " + str(toReturn))
    return JSONResponse(content=toReturn)


@app.get('/fets')
def model_features():
    logger.info("Features Fired: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    columnNames = [x.split('_')[0] for x in variables]
    toReturn = list(set(columnNames))
    return JSONResponse(content=json.dumps({'Features Required': toReturn}, sort_keys=True, separators=(',', ': ')))


if __name__ == "__main__":
    logger.info("Starting Webserver")
    uvicorn.run(app, host="0.0.0.0", port=8089)
