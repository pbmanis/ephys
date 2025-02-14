import pandas as pd
import pickle
from pathlib import Path
import src.analyze_ivs as IVS
import src.CBA_maps as MAPS

import argparse

def main():
    experiments = IVS.experiments['CBA_Age']
    print(experiments)
    parser = argparse.ArgumentParser(description='CBA_Age data analysis')
    # parser.add_argument('-E', '--experiment', type=str, dest='experiment',
    #                     choices = list(experiments.keys()), default='None', nargs='?', const='None',
    #                     help='Select Experiment to analyze')
                    
    args = parser.parse_args()

    # experiments = src.set_expt_paths.get_experiments()[args.experiment]

    fn = Path(experiments['analyzeddatapath'], experiments['directory'],
              Path(experiments['datasummaryFilename']).with_suffix('.pkl'))
    with open(fn, 'rb') as fh:
        d = pd.read_pickle(fh)

    coding = experiments['coding_sheet']
    # dir(pd.options)
    # pd.options.display.width=256
    # pd.options.display.max_rows=500
    # d
    print(d.index)
    for i in d.index:
        v = d.iloc[i]['data_complete']
        if coding is not None:
            basename = d.iloc[i].date.split('_')[0]
            if basename not in list(coding.keys()):
                code = 'None'
            else:
                code = coding[basename]
        else:
            code = 'None'
        print(f"\n{d.iloc[i].date:<16s} {d.iloc[i].slice_slice:9s} {d.iloc[i].cell_cell:9s} {d.iloc[i].cell_type:<18s} {str(code):s}")
        v = v.split(',')
        for x in v:
            print('   ', x.lstrip())

if __name__ == '__main__':
    main()