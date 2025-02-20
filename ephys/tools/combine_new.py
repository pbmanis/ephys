import pickle
import pandas as pd
import numpy as np

def main():
    with open('NF107Ai32_Het/NF107Ai32_Het.pkl', 'rb') as fh:
        d1 = pickle.load(fh)

    with open('NF107Ai32_Het/NF107Ai32_Het_on_2017.02.14.pkl', 'rb') as fh:
        d2 = pickle.load(fh)
    d2.loc[:, ~d2.columns.duplicated()]  # remove some duplicated columns
    d2 = d2.assign(annotated = False)
    dfx = pd.concat(list(d1.align(d2)), ignore_index=True)
    dfxn = dfx[~pd.isnull(dfx['date'])]
    print(dfxn.annotated)
    print(dfxn.data_complete)
    print(dfxn.date)
    dfxn.to_pickle('NF107Ai32_Het/NF107Ai32_Het_updated.pkl')

if __name__ == '__main__':
    main()