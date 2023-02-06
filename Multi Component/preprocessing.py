from azureml.core import Workspace, Dataset, Datastore
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from azureml.core.authentication import InteractiveLoginAuthentication
import numpy as np
import os
from datetime import datetime, date, timedelta
import argparse
import matplotlib.pyplot as plt #to plot charts
import warnings #avoid warning flash
warnings.filterwarnings('ignore')

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
#account_key = os.getenv("BLOB_ACCOUNT_KEY", "dvJ13jUkTTPEoXMAoQ+fege/nevKL/Qa0urybvP0TT1xVRB2RTpjuOyjnKh4HdWahdKo/ufdfuIQ+AStDA3LvA==")

parser = argparse.ArgumentParser()
parser.add_argument("--prep", type=str)
args = parser.parse_args()

datastore = Datastore.get(ws, 'workspaceblobstore')

from azureml.core import Run
run = Run.get_context()

#-----------------------------XX-----------------------------------#
#Read csv file
df = Dataset.Tabular.from_delimited_files(path=[(datastore, args.prep)]).to_pandas_dataframe()
print("Shape of Dataframe", df.shape)
#-----------------------------XX-----------------------------------#

#dropping duplicate values - checking if there are any duplicate rows and dropping if any
df = df.drop_duplicates()

#Cleaning data
#replacing 0 values with median of that column
df['Glucose']=df['Glucose'].replace(0,df['Glucose'].mean())#normal distribution
df['BloodPressure']=df['BloodPressure'].replace(0,df['BloodPressure'].mean())#normal distribution
df['SkinThickness']=df['SkinThickness'].replace(0,df['SkinThickness'].median())#skewed distribution
df['Insulin']=df['Insulin'].replace(0,df['Insulin'].median())#skewed distribution
df['BMI']=df['BMI'].replace(0,df['BMI'].median())#
#df_selected = df.drop(['BloodPressure','Insulin','DiabetesPedigreeFunction'],axis='columns')
df_selected = df[['Pregnancies', 'Glucose','SkinThickness','BMI','Age','Outcome']]
#Encoding
x=df_selected
quantile  = QuantileTransformer()
X = quantile.fit_transform(x)
df_new=quantile.transform(X)
df_new=pd.DataFrame(X)
df_new.columns = ['Pregnancies', 'Glucose','SkinThickness','BMI','Age','Outcome']

#-----------------------------XX-----------------------------------#
#Exporting the file
path = "tmp/"
try:
    os.mkdir(path)
except OSError:
    print("Creation of directory %s failed" % path)
else:
    print("Sucessfully created the directory %s " % path)
    
temp_path = path + "preprocessed.csv"
df_new.to_csv(temp_path)

#Now to datastore
datastr = Datastore.get(ws, "workspaceblobstore")
datastr.upload(src_dir = path, target_path="", overwrite=True)
#-----------------------------XX-----------------------------------#
print("Completed Preprocessing Process!")