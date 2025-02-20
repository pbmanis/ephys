import pandas as pd
from pathlib import Path
import argparse
import src.set_expt_paths

def main():
    src.set_expt_paths.get_computer()
    experiments = src.set_expt_paths.get_experiments()

    parser = argparse.ArgumentParser(description='IV data analysis')
    parser.add_argument('-E', '--experiment', type=str, dest='experiment',
                        choices = list(experiments.keys()), default='None', nargs='?', const='None',
                        help='Select Experiment to analyze')
    args = parser.parse_args()

    #fn = 'NF107Ai32_Het/NF107Ai32_Het_maps.xlsx'

    fn = Path(experiments[args.experiment]['directory'],  Path(experiments[args.experiment]['maps']).with_suffix('.xlsx'))
    ml = pd.read_excel(fn)
    cmds = []
    for i in range(len(ml)):
        day = ml.iloc[i]['date'][:10]
        sl = ml.iloc[i]['slice_slice']
        sl = 'S'+sl[-1]
        cell = ml.iloc[i]['cell_cell']
        cell = 'C'+cell[-1]
        slcell = sl + cell
        celltype = ml.iloc[i]['cell_type']
        #nf107_ivs -E nf107 -d 2017.05.08 -s S0C0 -A --map 
        cmdl = f'nf107_ivs -E {args.experiment:s} -d {day:s} -s {slcell:s} -A --map # {celltype:s}'
        if celltype is not ' ' and celltype != 'unknown':
            if cmdl not in cmds:  # avoid duplications
                cmds.append(cmdl)
                if celltype in ['pyramidal', 'cartwheel', 'fusiform', 'giant']:
                    ml.at[i, 'threshold'] = 3.0
                    ml.at[i, 'cc_threshold'] = 10.0
                if celltype in ['pyramidal', 'fusiform', 'giant']:
                    ml.at[i, 'tau1'] = 0.7
                    ml.at[i, 'tau2'] = 2
                if celltype in ['cartwheel']:
                    ml.at[i, 'tau1'] = 1
                    ml.at[i, 'tau2'] = 3.5
            
                if celltype in ['tuberculoventral']:
                    ml.at[i, 'threshold'] = 10.0
                    ml.at[i, 'cc_threshold'] = 10.0
                    ml.at[i, 'tau1'] = 0.2
                    ml.at[i, 'tau2'] = 0.5

                if celltype in ['bushy']:
                    ml.at[i, 'threshold'] = 10.0
                    ml.at[i, 'cc_threshold'] = 10.0
                    ml.at[i, 'tau1'] = 0.1
                    ml.at[i, 'tau2'] = 0.4
            
                if celltype in ['stellate', 'd-stellate', 't-stellate']:
                    ml.at[i, 'threshold'] = 5.0
                    ml.at[i, 'cc_threshold'] = 10.0
                    ml.at[i, 'tau1'] = 0.3
                    ml.at[i, 'tau2'] = 0.8

    #ml.to_excel('NF107Ai32_Het/NF107AI32_Het_maps_modified.xlsx')

    for c in cmds:
        print(c)

if __name__ == '__main__':
    main()