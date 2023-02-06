
import pandas as pd
import numpy as np

dataset = pd.read_csv('iris.csv')
dataset.head()

#test case
#Feature Check
cols = dataset.columns.tolist()
feature_check = all(dataset.columns.isin(cols))
feature_check

#Length Check
sepal_length_test = dataset['sepal length (cm)'].between(4,7).all()
sepal_width_test = dataset['sepal width (cm)'].between(2,5).all()
petal_length_test = dataset['petal length (cm)'].between(1,6).all()
petal_width_test = dataset['petal width (cm)'].between(0,3).all()

feature_check = ["Passed &#9989;" if feature_check else "Failed &#10540;"]
sepal_length_test = ["Passed &#9989;" if sepal_length_test else "Failed &#10540;"]
sepal_width_test =  ["Passed &#9989;" if sepal_width_test else "Failed &#10540;"]
petal_length_test = ["Passed &#9989;" if petal_length_test else "Failed &#10540;"]
petal_width_test = ["Passed &#9989;" if petal_width_test else "Failed &#10540;"]

expected_columns = 4
def test_check_schema():
    header = dataset[dataset.columns[:-1]]
    actual_columns = header.shape[1]
    # check header has expected number of columns
    assert actual_columns == expected_columns
test_check_schema()


with open("test.txt", 'w') as outfile:
        outfile.write("Feature Test: %s\n" % feature_check[0])
        outfile.write("sepal length Test : %s\n" % sepal_length_test[0])
        outfile.write("sepal width Test : %s\n" % sepal_width_test[0])
        outfile.write("petal length Test : %s\n" % petal_length_test[0])
        outfile.write("petal width Test : %s\n" % petal_width_test[0])

        outfile.write("\n")
        outfile.write("\n")
