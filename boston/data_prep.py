#!/usr/bin/python

import urllib.request
import pandas as pd



column_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

data_dict = {
"CRIM": "per capita crime rate by town"
, "ZN" : "proportion of residential land zoned for lots over 25,000 sq.ft."
, "INDUS" : "proportion of non-retail business acres per town"
, "CHAS" : "Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)"
, "NOX" : "nitric oxides concentration (parts per 10 million)"
, "RM" :  "average number of rooms per dwelling"
, "AGE" : "proportion of owner-occupied units built prior to 1940"
, "DIS" : "weighted distances to five Boston employment centres"
, "RAD" : "index of accessibility to radial highways"
, "TAX" :  "full-value property-tax rate per $10,000"
, "PTRATIO" : "pupil-teacher ratio by town"
, "B" : "1000(Bk - 0.63)^2 where Bk is the proportion of blacks  by town"
, "LSTAT" : "% lower status of the population"
, "MEDV" : "Median value of owner-occupied homes in $1000's"
}


def get_save_file(source_url, dest_file):
    """
    WARNING:  This will overwrite files if they exist!
    """
    try:
        response = urllib.request.urlopen(source_url)
        data = response.read()

        with open(dest_file, "w") as f: 
           f.write(data.decode("utf-8"))

    except Exception as e:
        print("Something went wrong - ", str(e))
        raise


def main():
    get_save_file('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', 'rawdata.txt')
    get_save_file('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names', 'dd.txt')
    
    
    
    
    with open("rawdata.txt", "r") as f:
        raw = f.read()
    
    while raw.find("  ") >= 0:
        raw = raw.replace("  ", " ")
    
    
    massaged = [x.strip() for x in raw.strip().split("\n")]
    
    # pd.read_table(massaged)
    
    
    df = pd.DataFrame([x.split(" ") for x in massaged], columns=column_names)
    # df.save_pickle()
    
    df.to_pickle("boston.pickle")
    

if __name__ == "__main__":
    main()
