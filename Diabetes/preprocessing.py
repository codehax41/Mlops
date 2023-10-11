from azureml.core import Workspace, Dataset, Datastore
import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime, date, timedelta
from sklearn.preprocessing import QuantileTransformer
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
parser.add_argument("--prep", type=str)
args = parser.parse_args()
#------------------------------Run-------------------------------#
run = Run.get_context()
#------------------------------Read_data-------------------------------#
#df = Dataset.Tabular.from_delimited_files(path=[(datastore, "diabetes.csv")]).to_pandas_dataframe()
df = Dataset.Tabular.from_delimited_files(path=[(datastore, args.prep)]).to_pandas_dataframe()

#------------------------------Transf/prep-------------------------------#
#Drop duplicates
df = df.drop_duplicates()

#Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].median())
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].median())
df['BMI'] = df['BMI'].replace(0, df['BMI'].median())

#Transform them
df_selected = df[['Pregnancies','Glucose','SkinThickness','BMI','Age','Outcome']]
x = df_selected
quantile = QuantileTransformer()
X = quantile.fit_transform(x)
df_new = quantile.transform(X)
df_new = pd.DataFrame(X)
df_new.columns = ['Pregnancies','Glucose','SkinThickness','BMI','Age','Outcome']

#------------------------------Export Data-------------------------------#
path = "tmp/"
try:
    os.mkdir(path)
except OSError:
    print("Creation of directory %s failed!" % path)
else:
    print("Sucessfully created the directory %s" % path)

temp_path = path + "preprocessed.csv"
df_new.to_csv(temp_path)

#------------------------------To Datastore-------------------------------#
datastr = Datastore.get(ws, data_store_name)
datastr.upload(src_dir = path, target_path = "", overwrite=True)


print("Data preprocessing Completed!")








