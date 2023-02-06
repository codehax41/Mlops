import joblib
import json
import numpy as np

from azureml.core.model import Model

def init():
    global model_3
    model_3_path = Model.get_model_path(model_name='model_estimator_500')
    model_3 = joblib.load(model_3_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        data = np.array(data)
        result_1 = model_3.predict(data)
        
        return {"prediction1": result_1.tolist()}
    except Exception as e:
        result = str(e)
        return result
