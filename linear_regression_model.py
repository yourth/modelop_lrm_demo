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
    datum = np.array([[datum]])
    prediction = model_artifact.predict(datum)[0]
    return prediction

