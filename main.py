import numpy as np
import pandas as pd
from flask import Flask, request
import pickle
import statsmodels.api as sm
from pandas import json_normalize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from waitress import serve
import json
import yaml
from find_variables import *

# Load Config Data
with open(r'config/config.yml') as file:
    config_data = yaml.load(file, Loader=yaml.FullLoader)


app = Flask(__name__)

# Unpickle required objects
model = pickle.load(open(config_data["pickled_model_path"], 'rb'))
imputer = pickle.load(open(config_data["pickled_imputer_path"], 'rb'))
std_scaler = pickle.load(open(config_data["pickled_scaler_path"], 'rb'))

# Setup variables needed to evaluate incoming data
variables = find_variables(model.params.axes)
threshold = config_data["cuttof_threshold"]



@app.route('/predict/', methods=['POST'])
def predict():

    # Load JSON File and clean up data.
    # Replace empty strings with NaNs, and convert currency and percentages to floats.
    # Todo find dynamic way to do this, remove hard coded references
    df = json_normalize(request.get_json(force=True))
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
    for i in variables:
        if i not in df_imputed.columns:
            df_imputed[i] = 0

    # Perform the model prediction using the unpickled Logistical Regression finalized model
    # Store values under phat
    model_predict = pd.DataFrame(model.predict(df_imputed[variables])).rename(columns={0: 'phat'})

    # Determine outcome by comparing phat to threshold provided by business partner
    # See config file for threshold value
    model_predict['business_outcome'] = (model_predict['phat'] >= threshold).astype(int)

    # Convert Pandas to the JSON format, json.loads makes the end result human readable
    modelJSON = json.loads(df_imputed[variables].to_json(orient='columns'))
    outcomeJSON = json.loads(model_predict['business_outcome'].to_json(orient='records'))
    phatJSON = json.loads(model_predict['phat'].to_json(orient='records'))

    # Build JSON to return to caller
    dataSet = {"business_outcome": outcomeJSON, "phat": phatJSON}
    for i in range(len(variables)):
        dataSet[variables[i]] = (modelJSON[variables[i]])
    toReturn = json.dumps(dataSet, sort_keys=True, indent=4, separators=(',', ': '))
    return toReturn


if __name__ == "__main__":
    # serve(app, host="0.0.0.0", port=1313, threads=16)
    app.run(host='0.0.0.0', port=1312, debug=True)
