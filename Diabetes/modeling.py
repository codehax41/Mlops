from azureml.core import Workspace, Dataset, Datastore
import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime, date, timedelta
import sklearn
import joblib
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, f1_score, recall_score, precision_score
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import Run

#------------------------------Auth-------------------------------#
interactive_auth = InteractiveLoginAuthentication(tenant_id="48dbabfc-37fa-4dd4-913c-cd7091a8ea08", force=True)

ws = Workspace(subscription_id = "82cfc0a9-b871-4f48-bb9a-0f9e00faacf9",
                workspace_name = "ram",
                resource_group = "ram",
                auth = interactive_auth)

#------------------------------End Auth-------------------------------#
#------------------------------Data Import-------------------------------#

data_store_name = "workspaceblobstore"
container_name = os.getenv("BLOB_CONTAINER", "ram5308115514")
account_name = os.getenv("BLOB_ACCOUNTNAME", "ram5308115514")

datastore = Datastore.get(ws, data_store_name)

#------------------------------Argparser-------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--train", type=str)
args = parser.parse_args()
#------------------------------Modeling-------------------------------#
#------------------------------Read_data-------------------------------#
#df = Dataset.Tabular.from_delimited_files(path=[(datastore, "diabetes.csv")]).to_pandas_dataframe()
df = Dataset.Tabular.from_delimited_files(path=[(datastore, args.train)]).to_pandas_dataframe()

target_col = "Outcome"
test_size = .2
y = df[target_col]
X = df.drop(target_col, axis=1)

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=0)

#define estimators
n_estimators = [100, 200, 500]
#------------------------------Run-------------------------------#
run = Run.get_context()
for i in n_estimators:
    model_rf = RandomForestClassifier(n_estimators=i).fit(X_train, y_train)
    pred_rf = model_rf.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_true = y_test, y_pred = pred_rf))
    run.log("rmse", rmse)
    run.log("train_size", X_train.size)
    run.log("split_size", test_size)
    run.log("estimators", i)
    run.log("precision", precision_score(y_test, pred_rf))
    run.log("f1-score", f1_score(y_test, pred_rf))
    run.log("recall", recall_score(y_test, pred_rf))

    model_name = "model_estimator_" + str(i) + ".pkl"
    filename = "outputs/" + model_name
    joblib.dump(value = model_rf, filename = filename)
    run.upload_file(name=model_name, path_or_stream = filename)

    run.complete()
#------------------------------Export Data-------------------------------#
path = "tmp/"
try:
    os.mkdir(path)
except OSError:
    print("Creation of directory %s failed!" % path)
else:
    print("Sucessfully created the directory %s" % path)

temp_path = path + "training.csv"
X_train.to_csv(temp_path)

filename1 = "tmp/" + model_name
joblib.dump(value=model_rf, filename = filename1)
#------------------------------To Datastore-------------------------------#
datastr = Datastore.get(ws, data_store_name)
datastr.upload(src_dir = path, target_path = "", overwrite=True)
    
print("Model training completed!")




























