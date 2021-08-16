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

app = Flask(__name__)

model = loaded_model = pickle.load(open("finalized_model.sav", 'rb'))


def find_variables(axesNames):
    """
    This function takes in the axes of the model parameters and finds the column anmes.
    It returns a list of strings with the names
    """
    variables = []
    for i in range(0, len(axesNames[0])):
        variables.append(axesNames[0][i])
    variables.sort()
    return variables


variables = find_variables(model.params.axes)

imputer = pickle.load(open("crunchyImputer.sav", 'rb'))
std_scaler = pickle.load(open("crunchyScaler.sav", 'rb'))

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
def predict():
    df = json_normalize(request.get_json(force=True))
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df['x12'] = (df['x12'].replace('[\$,)]', '', regex=True)
                 .replace('[(]', '-', regex=True).astype(float))
    df['x63'] = df['x63'].str.replace('%', '', regex=True).astype(float)

    test_imputed = pd.DataFrame(imputer.transform(df.drop(columns=['x5', 'x31', 'x81', 'x82'])),
                                columns=df.drop(columns=['x5', 'x31', 'x81', 'x82']).columns)
    test_imputed_std = pd.DataFrame(std_scaler.transform(test_imputed), columns=test_imputed.columns)

    dumb5 = pd.get_dummies(df['x5'], drop_first=True, prefix='x5', prefix_sep='_', dummy_na=True)
    df_imputed = pd.concat([test_imputed_std, dumb5], axis=1, sort=False)

    dumb31 = pd.get_dummies(df['x31'], drop_first=True, prefix='x31', prefix_sep='_', dummy_na=True)
    df_imputed = pd.concat([test_imputed_std, dumb31], axis=1, sort=False)

    dumb81 = pd.get_dummies(df['x81'], drop_first=True, prefix='x81', prefix_sep='_', dummy_na=True)
    df_imputed = pd.concat([test_imputed_std, dumb81], axis=1, sort=False)

    dumb82 = pd.get_dummies(df['x82'], drop_first=True, prefix='x82', prefix_sep='_', dummy_na=True)
    df_imputed = pd.concat([test_imputed_std, dumb82], axis=1, sort=False)

    # Not all data sets will have all the variables required for the model
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

    modelJSON = json.loads(df_imputed[variables].to_json(orient='columns'))
    outcomeJSON = json.loads(model_predict['business_outcome'].to_json(orient='records'))
    phatJSON = json.loads(model_predict['phat'].to_json(orient='records'))
    print(variables)

    dataSet = {"business_outcome": outcomeJSON, "phat": phatJSON}
    for i in range(len(variables)):
        dataSet[variables[i]] = (modelJSON[variables[i]])

    toReturn = json.dumps(dataSet, sort_keys=True, indent=4, separators=(',', ': '))
    return toReturn


if __name__ == "__main__":
    # serve(app, host="0.0.0.0", port=1313, threads=16)
    app.run(host='0.0.0.0', port=1312, debug=True)
