import sys
import pandas as pd

def load_data(path):
    return pd.read_csv(path)

if __name__ == '__main__':
    raw_data = sys.argv[1]
    output_file = sys.argv[2]
    data = load_data(raw_data)
    data.to_csv(output_file, index=False)


