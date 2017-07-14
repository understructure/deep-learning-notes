#!/usr/bin/python

import urllib.request
import pandas as pd

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
    
    
    df = pd.DataFrame([x.split(" ") for x in massaged])
    # df.save_pickle()
    
    df.to_pickle("boston.pickle")
    

if __name__ == "__main__":
    main()