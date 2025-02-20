"""
combine two pickled pandas databases
Resulting database is called combined.pkl and shoudl be renamed.
"""

import pickle
import pandas as pd
import numpy as np

def main():
    with open('NF107Ai32Het_bcorr.pkl', 'rb') as fh:
        d1 = pickle.load(fh)

    with open('rx_vcndata_selected2.pkl', 'rb') as fh:
        d2 = pickle.load(fh)
    d2.loc[:, ~d2.columns.duplicated()]  # remove some duplicated columns
    d2 = d2.assign(annotated = False)
    dfx = pd.concat(list(d1.align(d2)), ignore_index=True)
    dfxn = dfx[~pd.isnull(dfx['date'])]
    print(dfxn.annotated)
    print(dfxn.data_complete)
    dfxn.to_pickle('combined.pkl')

if __name__ == '__main__':
    main()