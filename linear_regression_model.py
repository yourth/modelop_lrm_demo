import json
import math
import numpy as np
import pickle
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

print("Starting program", flush=True)

#modelop.init
def begin():
    global model_artifact
    model_artifact = pickle.load(open("lr_model.pkl", "rb"))
    print("pass", flush=True)
    pass

#modelop.score
def action(datum):
    prediction = compute_prediction(datum)
    print("modelop.score.action:", prediction, flush=True)
    yield prediction

def compute_prediction(datum):
    x = datum['x']
    print("x:", x, flush=True)
    prediction = model_artifact.predict([[x]])[0]
    return prediction

#modelop.metrics
def metrics(data):
    actuals = data.y.tolist()
    data = data.to_dict(orient='records')
    predictions = list(map(compute_prediction, data))
    diffs = [x[0] - x[1] for x in zip(actuals, predictions)]
    rmse = math.sqrt(sum(list(map(lambda x: x**2, diffs))) / len(diffs))
    mae = sum(list(map(abs, diffs))) / len(diffs)
    yield dict(MAE=mae, RMSE=rmse)
 