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
    # investigate_object(df)

    test_val = df
    test_val['x12'] = test_val['x12'].str.replace('$', '', regex=True)
    test_val['x12'] = test_val['x12'].str.replace(',', '', regex=True)
    test_val['x12'] = test_val['x12'].str.replace(')', '', regex=True)
    test_val['x12'] = test_val['x12'].str.replace('(', '-', regex=True)
    test_val['x12'] = test_val['x12'].astype(float)
    test_val['x63'] = test_val['x63'].str.replace('%', '', regex=True)
    test_val['x63'] = test_val['x63'].astype(float)
    df8 = test_val
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

    prob['business_outcome'] = (prob['phat'] >= threshold).astype(int)
    df = prob

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
    # serve(app, host="0.0.0.0", port=1313, threads=16)
    app.run(host='0.0.0.0', port=1312, debug=True)
