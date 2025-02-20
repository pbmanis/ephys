import sys
import numpy as np
from pathlib import Path
import pandas as pd
import set_expt_paths
import argparse
import re
import ephys.ephys_analysis as EP
import pickle
import matplotlib.pylab as mpl
import matplotlib
matplotlib.use('Agg')
rcParams = matplotlib.rcParams
rcParams['svg.fonttype'] = 'none' # No text as paths. Assume font installed.
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['text.latex.unicode'] = True
#rcParams['font.family'] = 'sans-serif'
rcParams['font.weight'] = 'regular'                  # you can omit this, it's the default
#rcParams['font.sans-serif'] = ['Arial']

re_day = re.compile('-d \d{4}\.\d{2}\.\d{2}')
re_slicecell = re.compile('-s [Ss](\d{1})[Cc](\d{1})')
set_expt_paths.get_computer()

experiments = set_expt_paths.get_experiments()

# ANSI terminal colors  - just put in as part of the string to get color terminal output
colors = {'red': '\x1b[31m', 'yellow': '\x1b[33m', 'green': '\x1b[32m', 'magenta': '\x1b[35m',
              'blue': '\x1b[34m', 'cyan': '\x1b[36m' , 'white': '\x1b[0m', 'backgray': '\x1b[100m'}

shmap = {'nf107nihl': 'nihl.sh'}
re_date = re.compile('-d \d{4}\.\d{2}\.\d{2}')

class Templates(object):
    def __init__(self, args):
        if args.display:
            self.display()
            exit()
        self.basedir = Path(experiments[args.experiment]['disk'])
        self.exptdir = Path(experiments[args.experiment]['directory'])
        if args.outfile is not 'None':
            self.outfile = args.outfile
        else:
            self.outfile = None
        self.inputFilename = None
        self.map_annotationFile = None
        print (args.experiment)
        if args.experiment != 'noisetest':
            self.inputFilename = Path(self.exptdir, experiments[args.experiment]['datasummary']).with_suffix('.pkl')
        self.outputPath = self.exptdir
        if experiments[args.experiment]['annotation'] is not None:
            self.annotationFile = Path(self.exptdir, experiments[args.experiment]['annotation'])
        else:
            self.annotationFile = None
        if experiments[args.experiment]['maps'] is not None:
            self.map_annotationFile = Path(self.exptdir, experiments[args.experiment]['maps'])
        if self.inputFilename is not None:
            self.df = pd.read_pickle(str(self.inputFilename))

        self.AR = EP.acq4read.Acq4Read()
        # pull in map annotations. Thesea are ALWAYS in an excel file, which should be created initially by make_xlsmap.py
        if self.map_annotationFile is not None:
            self.map_annotations = pd.read_excel(Path(self.map_annotationFile).with_suffix('.xlsx'))
            print('Reading map annotation file: ', self.map_annotationFile)
           # print(self.map_annotations.head())
                  
        self.alldata = []
        if args.experiment != 'noisetest':
            fn_sh = Path(args.experiment+'.sh')
            with fn_sh.open('r') as sh_file:  
                line = sh_file.readline()
                cnt = 1
                while line:
                    if line[0] != '#' and '*template*' in line or '*self-template*' in line:
                        print("Line {}: {}".format(cnt, line.strip()))

                        tem = re.search(re_day, line)
                        if tem is not None:
                            sc = re.search(re_slicecell, line)
                            sliceno = 'slice_00'+sc.group(1)
                            cellno = 'cell_00' + sc.group(2)
                            self.day = tem.group(0)[3:]
                            day = str(self.day)
                            if '_' not in day:
                                day = day + '_000'
                            day_x = self.df.loc[(self.df['date'] == day) & (self.df['slice_slice'] == sliceno) & (self.df['cell_cell'] == cellno)]
                            print ('day_x: ', day_x)
                            for iday in day_x.index:
                                protocols = self.df.iloc[iday]['data_complete']
                                print('protocols: ', protocols)
                                maps = [m for m in protocols.split(', ') if m.startswith('Map') and m.find('_VC_10Hz')>0 and not m.endswith('003')]
                                self.getaverage(day, sliceno, cellno, maps=maps)

                    line = sh_file.readline()
                    cnt += 1
        else:
            print ('Getting from noise tests')
            fn_sh = Path('Noise_tests.txt')
            sliceno = ''
            cellno = ''
            with fn_sh.open('r') as sh_file:  
                line = sh_file.readline()
                cnt = 1
                while line:
                    if line.startswith('#'):
                        line = sh_file.readline()
                        continue
                    line = line.rstrip('\n')
                    lparts = line.split(' ')
                    maps = [m for m in lparts[1:]]
                    day = lparts[0]
                    self.getaverage(day, sliceno, cellno, maps=maps)
                    line = sh_file.readline()

        alldat = np.array(self.alldata)
        print(alldat.shape)
        avgdata = np.mean(alldat, axis=0)
        print(avgdata)
        template = {'t': self.AR.time_base, 'I': avgdata}
        if self.outfile is None:
            with open('template_data_tests.pkl', 'wb') as fh:
                pickle.dump(template, fh)
        else:
            with open(f'template_data_{self.outfile:s}.pkl', 'wb') as fh:
                pickle.dump(template, fh)
        import matplotlib.pyplot as mpl
        tend = np.argmin(np.fabs(self.AR.time_base-0.599))

        mpl.plot(self.AR.time_base[:tend], avgdata[:tend], 'k-')
        mpl.show()

    def getaverage(self, day, sliceno, cellno, maps=None):

        for m in maps:
            print('m: ', maps)
            if sliceno is None and cellno is None:
                protocolFilename = Path(self.basedir, day, m)
            else:
                protocolFilename = Path(self.basedir, day, sliceno, cellno, m)
                
            self.protocol = protocolFilename
            print('Protocol: ', protocolFilename)
            self.AR.setProtocol(protocolFilename)
            
            if self.AR.getData() is None:
                print('  >>No data found in protocol: %s' % protocolFilename)
                continue
            data = np.reshape(self.AR.traces, (self.AR.repetitions, self.AR.traces.shape[0], self.AR.traces.shape[1]))
            print('data shape: ', data.shape)
            data = np.mean(data, axis=0) # reps  # across reps
            data = np.mean(data, axis=0) # across targets
            self.alldata.append(data)
                

    def display(self):
        fns = Path('.').glob('template_data*.pkl')
        fns = list(fns)
        ntemplates = len(fns)
        from pyqtgraph.Qt import QtGui, QtCore
        import pyqtgraph as pg
        pg.setConfigOption('leftButtonPan', False)
        app = QtGui.QApplication([])
        win = pg.GraphicsLayoutWidget(show=True, title="Templates")
        win.resize(1000,600)
        win.setWindowTitle('Comparing Templates')
        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)
        
        p = []
        for i in range(len(fns)):
            p.append(win.addPlot(i, 0))

        for i, fn in enumerate(fns):
            print('fn: ', fn)
            with open(fn, 'rb') as fh:
                d = pickle.load(fh)
            tend = np.argmin(np.fabs(d['t']-0.599))
            p[i].plot(d['t'][:tend], (d['I'][:tend] - d['I'][0])*1e12)
          #  print('fn: ', fn.replace('_', '\_'))
            dataset = str(fn)
            textitem = pg.TextItem(dataset)
            p[i].addItem(textitem)
            textitem.setPos(0, 20)
            if i > 0:
                p[i].setXLink(p[0])
                p[i].setYLink(p[0])
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()
        
            
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Template Generator')
    parser.add_argument('-E', '--experiment', type=str, dest='experiment',
                        choices = list(experiments.keys()), default='None', nargs='?', const='None',
                        help='Select data to use in template')
    parser.add_argument('-d', '--display', action='store_true', dest='display',
                        help = 'just display the ones we have so far')
    parser.add_argument('-o', '--output', type=str, default='None', dest='outfile', 
                        help = 'specify output file for template')
    args = parser.parse_args()
    T = Templates(args)
    