from azureml.core import Workspace, Dataset, Datastore
import pandas as pd
import numpy as np
import math
import joblib
import sklearn
import os
from datetime import datetime, date, timedelta
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt #to plot charts
#import seaborn as sns #used for data visualization
import warnings #avoid warning flash
warnings.filterwarnings('ignore')
from azureml.core.authentication import InteractiveLoginAuthentication

#-----------------------------XX-----------------------------------#
############ only place in python script not required in pipeline
interactive_auth = InteractiveLoginAuthentication(tenant_id="48dbabfc-37fa-4dd4-913c-cd7091a8ea08", force=True)

ws = Workspace(subscription_id= "82cfc0a9-b871-4f48-bb9a-0f9e00faacf9",
    resource_group= "ram",
    workspace_name= "ram", auth = interactive_auth) 
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')
 
#Name of blob datastore
data_store_name = 'workspaceblobstore'
#Name of Azure blob container 
container_name = os.getenv("BLOB_CONTAINER", "ram5308115514") 
#Name of Storage Account
account_name = os.getenv("BLOB_ACCOUNTNAME", "ram5308115514")

#Key of Blob Account
#account_key = os.getenv("BLOB_ACCOUNT_KEY", "")

parser = argparse.ArgumentParser()
parser.add_argument("--train", type=str)
args = parser.parse_args()


datastore = Datastore.get(ws, 'workspaceblobstore')

from azureml.core import Run
run = Run.get_context()

#-----------------------------XX-----------------------------------#
#Read csv file
df = Dataset.Tabular.from_delimited_files(path=[(datastore, args.train)]).to_pandas_dataframe()
print("Shape of Dataframe", df.shape)
#-----------------------------XX-----------------------------------#

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

target_name='Outcome'
y = df[target_name]#given predictions - training data 
X = df.drop(target_name,axis=1)#dropping the Outcome column and keeping all other columns as X

#Splitting
test_size = 0.2
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=test_size,random_state=0)#splitting data in 80% train, 20%test

# define models and parameters
n_estimators = [100, 200, 500]

# define grid search
for i in n_estimators:
    best_model = RandomForestClassifier(n_estimators=i).fit(X_train, y_train)
    rf_pred=best_model.predict(X_test)
    rmse = math.sqrt(mean_squared_error(y_true = y_test, y_pred = rf_pred))
    run.log("rmse", rmse)
    run.log("train_split_size", X_train.size)
    run.log("test_split_size", test_size)
    run.log("n_estimators", i)
    run.log("precision",sklearn.metrics.precision_score(y_test,rf_pred))
    run.log("recall",sklearn.metrics.recall_score(y_test,rf_pred))
    run.log("f1-score",sklearn.metrics.f1_score(y_test,rf_pred))
    #sns.heatmap(confusion_matrix(y_test,rf_pred))
    model_name = "model_estimator_" + str(i) + ".pkl"
    filename = "outputs/" + model_name
    joblib.dump(value=best_model, filename=filename)
    run.upload_file(name=model_name, path_or_stream=filename)

    #export model to local
    run.complete()
#-----------------------------XX-----------------------------------#
#Exporting the file
path = "tmp/"
try:
    os.mkdir(path)
except OSError:
    print("Creation of directory %s failed" % path)
else:
    print("Sucessfully created the directory %s " % path)
    
temp_path = path + "training.csv"
df.to_csv(temp_path)

filename1 = "tmp/" + model_name
joblib.dump(value=best_model, filename=filename1)

#Now to datastore
datastr = Datastore.get(ws, "workspaceblobstore")
datastr.upload(src_dir = path, target_path="", overwrite=True)
#-----------------------------XX-----------------------------------#
print("Completed Training Process!")