import pandas as pd
import sys as sys

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

if __name__=="__main__":
    filepath = sys.argv[1]
    output = sys.argv[2]
    data = load_data(filepath)
    data.to_csv(output, index=False)