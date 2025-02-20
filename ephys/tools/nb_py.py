import os
from pathlib import Path
import pandas as pd
import datetime
from cycler import cycler

import numpy as np
import matplotlib.pyplot as mpl
from matplotlib.lines import Line2D
import pylibrary.plotting.plothelpers as PH
import seaborn as sns
from statsmodels.stats.weightstats import ttest_ind
from ephys.tools.get_computer import get_computer
from ephys.tools.get_configuration import get_configuration
import ephys.ephys_analysis as EP
import ephys.datareaders as DR

AR = DR.acq4_reader.acq4_reader()
SP = EP.spike_analysis.SpikeAnalysis()
config_file_path = "./config/experiments.cfg"

datasets, experiments = get_configuration(config_file_path)

colormap = {'bushy': "grey", "t-stellate": "forestgreen", 
        'unipolar brush cell': "forestgreen", "octopus": "red", 
        "d-stellate": "orange", "tuberculoventral": "fuchsia",
        "pyramidal": "blue", "cartwheel": "cyan", 
        "giant": "lime", 'unknown': "yellow", 
        "glial": "peachpuff", "granule": "peru",
        "horizontal bipolar": "slategrey", "chestnut": "brown",
        "ml-stellate": "cadetblue", "no morphology": "black"}
knowncells = list(colormap.keys())

marker_cycler = cycler(marker=Line2D.markers)

class DataBase():
    def __init__(self, dataset, use_coding=False):
        self.dataset = dataset
        if self.dataset in list(experiments.keys()):
            self.database = Path(experiments[self.dataset]['directory'], 
                experiments[self.dataset]['datasummaryFilename']).with_suffix('.pkl')
        else:
            self.database = Path(self.dataset)
        self.databasedict = experiments[self.dataset]
        parentpath = self.database.parent
        print('Database, parentpath: ', self.database, parentpath)
        self.eventsummaryfile = Path(parentpath, str(parentpath)+'_event_summary.pkl')
        self.eventpath = Path(parentpath, 'events')
        self.map_annotationFile = None
        if experiments[self.dataset]['maps'] is not None:
            self.map_annotationFile = Path(parentpath, experiments[self.dataset]['maps'])

        if self.map_annotationFile is not None:
            self.map_annotations = pd.read_excel(Path(self.map_annotationFile).with_suffix('.xlsx'), sheet_name='Sheet1')
            print('Reading map annotation file: ', self.map_annotationFile)

        self.dbfile = Path(experiments[self.dataset]['databasepath'], experiments[self.dataset]['directory'], experiments[dataset]['datasummaryFilename'])
        self.main_db = pd.read_pickle(str(self.dbfile))

class DataBaseReducer(object):
    def __init__(self, database):

        # self.filename = Path(basedir, filename).with_suffix('.pkl')
        # read the pandas database
        self.db = database.main_db # pd.read_pickle(str(self.filename))

        if database.map_annotationFile is not None:
            self.db.loc[self.db.index.isin(database.map_annotations.index), 'cell_type'] = database.map_annotations.loc[:, 'cell_type']
            self.db.loc[self.db.index.isin(database.map_annotations), 'annotated'] = True
        else:
            annotated = None

        # print('Cell types in database: ', list(set(self.db['cell_type'])))  # list out all the cell types in the database
        measures = ['rmp', 'taum', 'rin', 'gh', 'tauh', 'bovera', 'ap_ht', 'ap_wid', 'ap_ahp', 'ap_maxcount']
        collab = ['cell_type']
        collab.extend(measures)
        pcadf = pd.DataFrame(columns=collab)  # empty data frame for the pca
        # cells will be known by the first name in the lists below

        combine_strains = True
        if combine_strains:
            stellate = ['t-stellate', 'stellate', 'Tstellate']
            bushy = ['bushy', 'Bushy']
            ds = ['d-stellate', 'Dstellate']
        else:
            stellate = ['t-stellate', 'stellate']
            bushy = ['bushy']
            ds = ['d-stellate']

        glial = ['glial', 'glial cell', 'glia', 'GLIAL']
        giant = ['giant', 'Giant', 'Giant cell', 'GIANT']
        cw = ['cartwheel']
        pyr = ['pyramidal', 'fusiform']
        tv = ['tuberculoventral']
        unknown = ['unknown', 'no morphology', ' ']
        ubc = ['unipolar brush cell']
        mlstellate = ['ml-stellate']
        chestnut = ['chestnut']
        horizbipolar = ['horizontal bipolar']
        granule = ['granule']
        typeB = ['Type-B']
        cbushy = ['Bushy']
        cstellate = ['Tstellate']
        cds = ['Dstellate']
        if combine_strains:
            self.allneurons =  [bushy, stellate, ds, pyr, cw, tv,]
        else:
            self.allneurons =  [bushy, stellate, ds, giant, pyr, cw, tv, ubc, mlstellate, chestnut, horizbipolar, granule, typeB, unknown,
                         cbushy, cstellate, cds]

        all_cell_types = [s for n in self.allneurons for s in n]
        print('All cell types in self.allneurons: ', sorted(list(set(all_cell_types))))
        # print('difference: ', set(self.db['cell_type']) - set(all_cell_types))
        self.mapped_cells = set(all_cell_types).intersection(set(self.db['cell_type']))
        # print('intersection: ', self.mapped_cells)

        self.map_colors()
        self.known = set(all_cell_types).intersection(set(self.db['cell_type']))
        # print(' not mapped: ', set(self.colors.keys()) - self.known)


    def map_colors(self):
        # lock colors to cell types
        self.colors = {'bushy': 'blue', 'pyramidal': 'sienna', 't-stellate': 'cyan', 'd-stellate': 'orange',
                  'tuberculoventral': 'magenta', 'cartwheel': 'red', 'giant': 'black', 
                  'unknown': 'green',  
                  'unipolar brush cell': 'lightcoral', 'ml-stellate': 'chocolate', 'chestnut': 'skyblue',
                  'horizontal bipolar': 'deeppink', 'granule': 'saddlebrown', 'Type-B': 'maroon',
                  'Dstellate': 'darkorange', 'Tstellate': 'c', 'Bushy': 'mediumblue',
                  }
        # lock symbols to cell types
        self.symbols = {'bushy': 'o', 'pyramidal': 'o', 't-stellate': '*', 'd-stellate': 'P',
                  'tuberculoventral': 'o', 'cartwheel': 'o', 'giant': 'o', 
                  'unknown': 'o',  
                  'unipolar brush cell': 'o', 'ml-stellate': 'o', 'chestnut': 'o',
                  'horizontal bipolar': 'o', 'granule': 'o', 'Type-B': 'o',
                  'Dstellate': 'P', 'Tstellate': '*', 'Bushy': 'o',
              }
          
        self.alphas = {'bushy': 0.8, 'pyramidal': 0.8, 't-stellate': 0.8, 'd-stellate': 0.8,
                  'tuberculoventral': 0.8, 'cartwheel': 0.8, 'giant': 0.8, 
                  'unknown': 0.20, 
                  'unipolar brush cell': 0.3, 'ml-stellate': 0.3, 'chestnut': 0.3,
                  'horizontal bipolar': 0.3, 'granule': 0.3, 'Type-B': 0.3,
                  'Dstellate': 0.8, 'Tstellate': 0.8, 'Bushy': 0.8,
                  }

    def get_CellBasics(self, cells, noprint=False):    
        if not noprint:
            print('\nBasics')
            print('Cells: ', cells)
        print(cells.columns)
        ir = cells['IV']

        for i, sp in enumerate(ir):
            if pd.isnull(sp) or len(sp) == 0:
                continue
        rmps = []
        taus = []
        rins = []
        ghs = []
        tauhs = []
        boveras = []
        b_index = []
        b_celltype = []
        for i, iv in enumerate(ir):
            if pd.isnull(iv):
                continue
            tcell = cells.iloc[i]
            prots = list(iv.keys())
            # print('prots: ', prots)
            crmp = []
            ctau = []
            crin = []
            cgh = []
            ctauh = []
            cbovera = []
            for p in prots:
                da = iv[p]
                try:  # occasionally one gets by that doesn't have a complete record...
                    crmp.append(da['RMP'])
                except:
                    continue
                #print(da.keys(), p)
                ctau.append(da['taum']*1e3)
                crin.append(da['Rin'])
                if da['tauh_Gh'] is not None:
                    cgh.append(da['tauh_Gh']*1e6)
                    ctauh.append(da['tauh_tau']*1e3)
                    cbovera.append(da['tauh_bovera'])
                else:
                    cgh.append(0.)
                    ctauh.append(1e3)
                    cbovera.append(1)
            if crmp == []:
                continue
                #print(crmp, ctau, crin)
            rmps.append(np.mean(crmp))
            taus.append(np.mean(ctau))
            rins.append(np.mean(crin))

            ghs.append(np.mean(cgh))
            tauhs.append(np.mean(ctauh))
            boveras.append(np.mean(cbovera))
            b_index.append(i)
            b_celltype.append(tcell['cell_type'])
            if not noprint:
                print('{0:>3d} RMP: {1:5.1f}  taum: {2:5.1f}  Rin: {3:5.1f}  gH: {4:5.2f}  tauh: {5:8.4f}  b/a: {6:6.3f}'
                      .format(i, rmps[-1], taus[-1], rins[-1], ghs[-1]*1e6, tauhs[-1], boveras[-1]))
        if not noprint:
            print('-'*60)
            print( 'mean    RMP: {0:5.1f}  taum: {1:5.1f}  Rin: {2:5.1f}'
                  .format(np.nanmean(rmps), np.nanmean(taus),  np.nanmean(rins),
                          np.nanmean(ghs*1e6),  np.nanmean(tauhs), np.nanmean(boveras)))
            print( 'SD      RMP: {0:5.1f}  taum: {1:5.1f}  Rin: {2:5.1f}'
                  .format(np.nanstd(rmps), np.nanstd(taus),  np.nanstd(rins),
                          np.nanstd(ghs*1e6),  np.nanstd(tauhs), np.nanstd(boveras)))
            print('='*60)
        
        self.basics = {'rmp': rmps, 'taum': taus, 'rin': rins, 
                'gh': ghs, 'tauh': tauhs, 'bovera': boveras,
                'index': b_index, 'cell_type': b_celltype}

#cells = d[d['cell_type'] == celltype]

    def get_SpikeMeasures(self, cells, noprint=False):
        if not noprint:
            print('\nSpikepars')
    #        print('Cells: ', celltype)

        ir = cells['Spikes']
        ap_ht = []
        ap_wid = []
        ap_ahp = []
        ap_ar = []
        ap_maxcount = []  # max count of spikes in first 100 msec
        ap_index = []
        ap_celltype = []
        for i, sp in enumerate(ir):
            if pd.isnull(sp) or len(sp) == 0:
                continue
            tcell = cells.iloc[i]
            thisdate = str(Path(tcell['date'], tcell['slice_slice'], tcell['cell_cell']))
            # print('this: ', thisdate)
            prots = list(sp.keys())
            cap_ht = []
            cap_wid = []
            cap_ahp = []
            cap_ar = []
            cap_nspk = []
            for p in prots:
                da = sp[p]
               # print(da.keys())
                try:
                    cap_ar.append(da['AdaptRatio'])
                except:
                    cap_ar.append(0.)
                try:
                    spkl = list(da['spikes'].keys())
                except:
                    continue
                fsp = np.argmax(da['FI_Curve'][1])
                # print(thisdate)
                # print('   index max spikes: ', fsp)
                # print('   n spikes: ', da['FI_Curve'][1][fsp])
                # print('   current: ', da['FI_Curve'][0][fsp])
                # count spikes in first 100 msec:
                nsp = 0
                try:
                    for spx in da['spikes'][fsp]:
                       # print(da['spikes'][fsp][spx]['AP_Latency'])
                        if da['spikes'][fsp][spx]['AP_Latency'] < 0.2:
                            nsp += 1
                    cap_nspk.append(nsp)
                except:
                    cap_nspk.append(0)
                    # print('   spike < 100 msec latency: ', nsp)
            
                for iss, s in enumerate(spkl):
                    if iss > 3: # only count first 3 current levels with spikes
                        continue
                    spks = list(da['spikes'][s].keys())

                    for ixs, x in enumerate(spks):
                        if ixs > 1:  # only count first spikes
                            continue
                        #print("da['spikes'][s][x]['halfwidth']", da['spikes'][s][x]['halfwidth'])
                        if da['spikes'][s][x]['halfwidth'] is None or da['spikes'][s][x]['halfwidth']*1e3 > 3:
                            continue  # don't cound mis-measured wide spikes
                        cap_ht.append(da['spikes'][s][x]['peak_V']*1e3)
                        cap_wid.append(da['spikes'][s][x]['halfwidth']*1e3)
                        cap_ahp.append(da['spikes'][s][x]['trough_V']*1e3)
        #                 print (da['spikes'][s][x].keys())
            if len(cap_ht) > 0:
                ap_ht.append(np.mean(cap_ht))
                ap_wid.append(np.mean(cap_wid))
                ap_ahp.append(np.mean(cap_ahp))
                ap_ar.append(np.mean(cap_ar))
                ap_index.append(i)
                ap_celltype.append(tcell['cell_type'])
                ap_maxcount.append(np.mean(cap_nspk))
                if not noprint:
                    print('{0:>3d} ap_ht: {1:5.1f}  ap_wid: {2:5.3f}  ap_ahp: {3:5.1f} ap_ar: {4:5.1f} nmaxspikes: {5:3.0f}:: {5:s}'.
                          format(i, cap_ht[-1], cap_wid[-1], cap_ahp[-1], cap_ar[-1], cap_nspk[-1], thisdate))
        if not noprint:
            print('-'*60)
            print( '    mean AP HT: {0:5.1f}  ap_wid: {1:5.3f}  ap_ahp: {2:5.1f}  ap_ar: {3:5.1f} nmaxspikes: {5:3.0f}'.
                      format(np.nanmean(ap_ht), np.nanmean(ap_wid), np.nanmean(ap_ahp), np.nanmean(ap_ar), np.nanmean(ap_maxcount)))
            print( '      SD AP HT: {0:5.1f}  ap_wid: {1:5.3f}  ap_ahp: {2:5.1f}  ap_ar: {3:5.1f} nmaxspikes: {5:3.0f}'.
                      format(np.nanstd(ap_ht), np.nanstd(ap_wid), np.nanstd(ap_ahp), np.nanstd(ap_ar), np.nanstd(ap_maxcount)))
            print('='*60)
        self.spike_pars = {'ap_ht': ap_ht, 'ap_wid': ap_wid, 'ap_ahp': ap_ahp, 'ap_ar': ap_ar, 'ap_maxcount': ap_maxcount,
                   'ap_index': ap_index, 'ap_celltype': ap_celltype}

    def get_AllMeasures(self, db, noprint=False):
        if not noprint:
            print('\nGet all measures')
        rmps = []
        taus = []
        rins = []
        tauhs = []
        ghs = []
        boveras = []
        ap_ht = []
        ap_wid = []
        ap_ahp = []
        ap_ar = []
        ap_maxcount = []
        dates = []
        cellindex = []
        celltype = []
        annotated = []
    
        for i in range(len(d.index)):  # over all data in the index
            db_iv = d.iloc[i]['IV']  # get the IV data 
            if isinstance(db_iv, list):
                db_iv = db_iv[0]
            db_spike = db.iloc[i]['Spikes']
            if isinstance(db_spike, list):
                db_spike = db_spike[0]
            if pd.isnull(db_iv) or db_iv == {}:
                continue
            if pd.isnull(db_spike) or db_spike == {}:
                continue
            tcell = db.iloc[i]
            # if d.iloc[i]['temperature'] != '34C':
            #     continue
            thiscelltype = None
            for cx in self.allneurons:
                tct = tcell['cell_type']
                if tct in cx:
                    thiscelltype = cx[0]
                    break
           # print('celltype: ', celltype)
            if thiscelltype is not None:
                celltype.append(thiscelltype)
            else:
                celltype.append('unknown')

            thisdate = str(Path(tcell['date'], tcell['slice_slice'], tcell['cell_cell']))
            dates.append(thisdate)
            cellindex.append(i)
            annotated.append(d.iloc[i]['annotated'])

            prots = list(db_iv.keys())
           # print('Prots: ', prots)
            crmp = []
            ctau = []
            crin = []
            ctauh = []
            cbovera = []
            cgh = []
            for p in prots:
                da = db_iv[p]
                try:  # occasionally one gets by that doesn't have a complete record...
                    crmp.append(da['RMP'])
                except:
                    continue
                    # print(da.keys(), p)
                ctau.append(da['taum']*1e3)
                crin.append(da['Rin'])

                if da['tauh_tau'] is not None:
                    ctauh.append(da['tauh_tau']*1e3)
                    cgh.append(da['tauh_Gh']*1e6)
                    cbovera.append(da['tauh_bovera'])
                else:
                    ctauh.append(np.nan)
                    cgh.append(np.nan)
                    cbovera.append(np.nan)
                
            rmps.append(np.mean(crmp))
            taus.append(np.mean(ctau))
            rins.append(np.mean(crin))
            tauhs.append(np.nanmean(ctauh))
            boveras.append(np.nanmean(cbovera))
            ghs.append(np.nanmean(cgh))
            if not noprint:
                print('{0:>3d} RMP: {1:5.1f}  taum: {2:5.1f}  Rin: {3:5.1f}  gH: {4:5.2f}  tauh: {5:8.4f}  b/a: {6:6.3f}'
                      .format(i, rmps[-1], taus[-1], rins[-1], ghs[-1], tauhs[-1], boveras[-1]))

        
            # now for spikes
           # print('this: ', thisdate)
            prots = list(db_spike.keys())
            cap_ht = []
            cap_wid = []
            cap_ahp = []
            cap_ar = []
            cap_nspk = []
            for p in prots:
                da = db_spike[p]
                try:
                    cap_ar.append(da['AdaptRatio'])
                except:
                    cap_ar.append(0.)
                try:
                    spkl = list(da['spikes'].keys())
                except:
                    continue
                fsp = np.argmax(da['FI_Curve'][1])
                nsp = 0
                try:
                    for spx in da['spikes'][fsp]:
                       # print(da['spikes'][fsp][spx]['AP_Latency'])
                        if da['spikes'][fsp][spx]['AP_Latency'] < 0.2:
                            nsp += 1
                    cap_nspk.append(nsp)
                except:
                    pass

                for iss, s in enumerate(spkl):
                    if iss > 3: # only count first 3 current levels with spikes
                        continue
                    spks = list(da['spikes'][s].keys())

                    for ixs, x in enumerate(spks):
                        if ixs > 1:  # only count first spikes
                            continue
                        #print("da['spikes'][s][x]['halfwidth']", da['spikes'][s][x]['halfwidth'])
                        if da['spikes'][s][x]['halfwidth'] is None or da['spikes'][s][x]['halfwidth']*1e3 > 3:
                            continue  # don't cound mis-measured wide spikes
                        cap_ht.append(da['spikes'][s][x]['peak_V']*1e3)
                        cap_wid.append(da['spikes'][s][x]['halfwidth']*1e3)
                        cap_ahp.append(da['spikes'][s][x]['trough_V']*1e3)
        #                 print (da['spikes'][s][x].keys())
            if len(cap_ht) > 0:
                ap_ht.append(np.mean(cap_ht))
                ap_wid.append(np.mean(cap_wid))
                ap_ahp.append(np.mean(cap_ahp))
                ap_ar.append(np.mean(cap_ar))
                ap_maxcount.append(np.mean(cap_nspk))
            

            else:
                ap_ht.append(np.nan)
                ap_wid.append(np.nan)
                ap_ahp.append(np.nan)
                ap_ar.append(np.nan)
                ap_maxcount.append(np.nan)
            if not noprint and len(cap_ht) > 0:
                if not noprint:
                    print('{0:>3d} ap_ht: {1:5.1f}  ap_wid: {2:5.3f}  ap_ahp: {3:5.1f} ap_ar: {4:5.1f} nmaxspikes: {5:3d}:: {6:s}'.
                          format(i, cap_ht[-1], cap_wid[-1], cap_ahp[-1], cap_ar[-1], cap_nspk[-1], str(thisdate)))
        if not noprint:
            print('-'*60)
            print( 'mean    RMP: {0:5.1f}  taum: {1:5.1f}  Rin: {2:5.1f}'
                  .format(np.nanmean(rmps), np.nanmean(taus),  np.nanmean(rins),
                          np.nanmean(ghs)*1e6,  np.nanmean(tauhs), np.nanmean(boveras)))
            print( 'SD      RMP: {0:5.1f}  taum: {1:5.1f}  Rin: {2:5.1f}'
                  .format(np.nanstd(rmps), np.nanstd(taus),  np.nanstd(rins),
                          np.nanstd(ghs)*1e6,  np.nanstd(tauhs), np.nanstd(boveras)))
            print('='*60)
        if not noprint:
            print('-'*60)
            # print('apht: ', ap_ht)
            # print('apwid: ', ap_wid)
            # print('ap_ahp: ', ap_ahp)
            # print('ap_ar: ', ap_ar)
            # print('ap_count: ', ap_maxcount)
            print( '    mean AP HT: {0:5.1f}  ap_wid: {1:5.3f}  ap_ahp: {2:5.1f}  ap_ar: {3:5.1f} nmaxspikes: {4:3.0f}'.
                      format(np.nanmean(ap_ht), np.nanmean(ap_wid), np.nanmean(ap_ahp), np.nanmean(ap_ar), np.nanmean(ap_maxcount)))
            print( '      SD AP HT: {0:5.1f}  ap_wid: {1:5.3f}  ap_ahp: {2:5.1f}  ap_ar: {3:5.1f} nmaxspikes: {4:3.0f}'.
                      format(np.nanstd(ap_ht), np.nanstd(ap_wid), np.nanstd(ap_ahp), np.nanstd(ap_ar), np.nanstd(ap_maxcount)))
            print('='*60)
        return {'date': dates, 'cellindex': cellindex,
                'rmp': rmps, 'taum': taus, 'rin': rins, 'tauh': tauhs, 'gh': ghs, 'bovera': boveras,
                'ap_ht': ap_ht, 'ap_wid': ap_wid, 'ap_ahp': ap_ahp, 'ap_maxcount': ap_maxcount,
                'celltype': celltype, 'annotated': annotated}


    def check_spont_protocol(self, thiscell, dbn):
        dpath = Path(experiments[dbn]['disk'], thiscell)
        # check for CC_Spont data
        ccd = list(dpath.glob('CC_Spont*'))
        if len(ccd) > 0:
            # print('ccsponts: ', ccd)
            spks = []
            for c in ccd:
                AR.setProtocol(c)
                AR.getData()
                SP.setup(clamps=AR, threshold=0., refractory=0.0007, peakwidth=0.001,
                                    verify=False, interpolate=True, verbose=False, mode='peak', min_halfwidth=0.010)
                SP.analyzeSpikes()
                spks.append(SP.spikes)
            return spks
        else:
            return None

    def parse_age(self, age):
        age = str(age)
        age = age.rstrip().lstrip().replace(' ', '')
        agenum = ''
        for a in age:
            if a.isdecimal():
                agenum += a
        if len(agenum) == 0:
            agenum = '0'
        return int(agenum)

    def get_SpontInfo(self, db=None, dbn=None, cellselect='pyramidal', printflag=True):
        if not printflag:
            print('\nGet Spont Info')
        if db is None:
            return
    #    print(d)
        rmps = []
        taus = []
        rins = []
        tauhs = []
        ghs = []
        boveras = []
        ap_ht = []
        ap_wid = []
        ap_ahp = []
        ap_ar = []
        ap_maxcount = []
        sponts = []
        holds = []
        cells = []
        cellindex = []
        celltype = []
        annotated = []
        cells_with_hold = {}
        cells_with_spont = {}
        cells_with_nospontnohold = {}
        cells_checked = []
        ccivspontprotocol = {}  # count spont spikes in this protocol
        for i in range(len(db.index)):
            tcell = db.iloc[i]
            div = db.iloc[i]['IV']
            if not isinstance(div, dict):
                db_spike = None
                div = None
                prots = []
            else:
                if isinstance(div, list):
                    div = div[0]
                db_spike = db.iloc[i]['Spikes']
                if isinstance(db_spike, list):
                    db_spike = db_spike[0]
                prots = list(div.keys())

            # check the cell type so we can do selection
            thiscell = db.iloc[i]
            thiscelltype = None
            tct = thiscell['cell_type']

            for cx in self.allneurons:
                if tct in cx:
                    thiscelltype = cx[0]  # always use the first entry
                    break
            if thiscelltype is not cellselect:
                continue
            if thiscelltype is not None:
                celltype.append(thiscelltype)
            else:
                continue

            thiscell = str(Path(tcell['date'], tcell['slice_slice'], tcell['cell_cell']))
            annotated.append(db.iloc[i]['annotated'])
            daytag = str(Path(tcell['date'][:-4]))
           # print(experiments.keys())
            if daytag in list(experiments[dbn]['coding'].keys()):
                group = experiments[dbn]['coding'][daytag][1]
            else:
                group = 'B'
            crmp = []
            ctau = []
            crin = []
            ctauh = []
            cbovera = []
            cgh = []
            hold = 0
            spont = 0
            holdflag = False
            spontflag = False
            lastpar = ''
            age = self.parse_age(tcell['age'])
            if age < 28 or age > 60:
                continue
            ccivspontprotocol[thiscell] = []
            spontspikes = self.check_spont_protocol(thiscell, dbn)
                
            for i, p in enumerate(prots):
                da = div[p]
                ds = db_spike[p]
                par = str(p.parent)
                if len(da.keys()) == 0:
                    continue
                if i not in cellindex:
                    cellindex.append(i)
                    cells.append(thiscell)
                pulsedur = ds['pulseDuration']
                if 'holding' in da.keys():
                    ih = da['holding']
                else:
                    ih = 0.0
                ibasermp = np.mean(da['Irmp'])
                ih = np.min([ih, ibasermp])
                if ih < -1e-12:
                    holdflag = True  # set holding flag
                if holdflag:
                    continue
                avblpk = 0
                # check for spont spikes at 0 current level
                if np.fabs(ih) < 1e-12:
                    j = np.where(np.fabs(ds['FI_Curve'][0])< 10e-12)
                
                    # print('j: ', j, ds['spikes'].keys())
                    if j in list(ds['spikes'].keys()):
                        print('stim spks: ', len(ds['spikes'][j].keys()))  # number of spikes in that train
                    else:
                        print('No spikes at zero current level')
                    
                    # print('Holding current applied; no zero current')
                if thiscell not in cells_checked:  # only do onece per cell
                    if spontspikes is not None:
                        cells_checked.append(thiscell)
                        for c in spontspikes:
                            for s in c:
                                # print('spontspikes: ', len(s))
                                ccivspontprotocol[thiscell].append(len(s))
                    # else:
                    #     print('no spikes in spont protocols')

                if 'baseline_spikes' in ds.keys():
                    blspk = ds['baseline_spikes']
                    postspk = ds['poststimulus_spikes']
                else:
                    blspk = [[]]
                    postspk = [[]]
                if par != lastpar:
                    print('---')
                    print(f"{'Day, slice, cell':^34s} {'Protocol':^24s}  {'Ihold (pA)':^9s}  {'Spn Spk':^9s}  {'Type':^18s} {'Age':^6s} {'Group':^6s}")
                    lastpar = par
                print(f"{str(p.parent):<34s} {str(p.name):^24s}  {1e12*ih:>8.1f}  ", end='  ')

                print(f"{np.mean([len(k) for k in blspk]):>8.3f}  {np.mean([len(k) for k in postspk]):>8.3f}  {thiscelltype:^18s} {age:^6d} {group:^6s}")
                # print(f"{np.mean([len(k) for k in blspk]):8.3f}  {thiscelltype:18s} {group:<6s}")
    #            print(p)
                if len(blspk) > 0:
                    nblspk = 0
                    for ns in blspk:
                        nblspk += len(ns)
    #                print(f'  average baseline spikes: {nblspk/len(blspk):.2f}', )
                    if not spontflag:
                        if nblspk > 1:  # consider baselinespikes
                            spont = 1
                        else:
                            spont = 0
                        spontflag = True
                try:  # occasionally one gets by that doesn't have a complete record...
                    crmp.append(da['RMP'])
                except:
                    continue
                    # print(da.keys(), p)

                ctau.append(da['taum']*1e3)
                crin.append(da['Rin'])
                rmp_std = np.std(da['RMPs'])

                if da['tauh_tau'] is not None:
                    ctauh.append(da['tauh_tau']*1e3)
                    cgh.append(da['tauh_Gh']*1e6)
                    cbovera.append(da['tauh_bovera'])
                else:
                    ctauh.append(np.nan)
                    cgh.append(np.nan)
                    cbovera.append(np.nan)
                
            rmps.append(np.mean(crmp))
            taus.append(np.mean(ctau))
            rins.append(np.mean(crin))
            if len(ctauh) > 0:
                tauhs.append(np.nanmean(ctauh))
                boveras.append(np.nanmean(cbovera))
                ghs.append(np.nanmean(cgh))
            else:
                tauhs.append(np.nan)
                boveras.append(np.nan)
                ghs.append(np.nan)
            holds.append(hold)
            sponts.append(spont)
            if hold > 0 and spont == 0:
                cells_with_hold[thiscell] = [hold, spont, thiscelltype]
            elif hold == 0 and spont > 0:
                cells_with_spont[thiscell] = [hold, spont, thiscelltype]
            else:
                cells_with_nospontnohold[thiscell] = [hold, spont, thiscelltype]

            if printflag and False:
                print(f"{i:>3d} RMP: {rmps[-1]:5.1f}  taum: {taus[-1]:5.1f}  Rin: {rins[-1]:5.1f}  gH: {ghs[-1]:5.2f}  tauh: {tauhs[-1]:8.4f}  b/a:{boveras[-1]:6.3f}")
        ncells = len(cellindex)
        if printflag:
            print('Cells with holding, no spontaneous')
            for c in cells_with_hold:
                print(f'{c:s}  hold: {cells_with_hold[c][0]:d}  spont: {cells_with_hold[c][1]:d} type: {cells_with_hold[c][2]:s} {str(ccivspontprotocol[c]):s}')
            print(f'Cells with hold: {len(cells_with_hold):d}') # ', Frac with holding: {np.sum(holds)/ncells:.2f}, Frac with spont: {np.sum(sponts)/ncells:.2f}')
            print('\nCells without holding showing spontaneous activity')
            for c in cells_with_spont:
                print(f'{c:s}  hold: {cells_with_spont[c][0]:d}  spont: {cells_with_spont[c][1]:d} type: {cells_with_spont[c][2]:s} {str(ccivspontprotocol[c]):s}')
            print(f'Cells with spont: {len(cells_with_spont):d}')
            print('\nCells without holding showing NO spontaneous firing')
            for c in cells_with_nospontnohold:
                print(f'{c:s}  hold: {cells_with_nospontnohold[c][0]:d}  spont: {cells_with_nospontnohold[c][1]:d} type: {cells_with_nospontnohold[c][2]:s} {str(ccivspontprotocol[c]):s}')
            print(f'Cells no spont, no hold: {len(cells_with_nospontnohold):d}')
    
        return {'date': cells, 'cellindex': cellindex,
                'rmp': rmps, 'taum': taus, 'rin': rins, 'tauh': tauhs, 'gh': ghs, 'bovera': boveras,
                'celltype': celltype, 'annotated': annotated}  


    def sumplot(self):
        ax_xl = {'rmp': [-90, -40], 'rin': [0, 500], 'taum': [0, 25],
                'ap_ht': [-20, 50], 'ap_wid': [0, 1.5], 'ap_ahp': [-75., -25.],
                'tauh': [0., 250.], 'gh': [0., 100.], 'bovera': [0., 1.]}
        measures = list(ax_xl.keys())
        P = PH.regular_grid(len(self.allneurons), len(measures), order='columns', figsize=(11., 8.), showgrid=False,
                        verticalspacing=0.03, horizontalspacing=0.05,
                        margins={'leftmargin': 0.07, 'rightmargin': 0.05, 'topmargin': 0.075, 'bottommargin': 0.1},
                        labelposition=(0., 0.))

        for row, celltype in enumerate(self.allneurons):
            cells = d[d['cell_type'].isin(celltype)]
            col = 0
            b = basics(cells, noprint=True)
            s = spikepars(cells, noprint=True)
            if b is not None:
                for ba in b.keys():#x = x[~numpy.isnan(x)]
                    if ba in ['index', 'cell_type']:
                        continue
                    if ba not in measures:
                        continue
                    bx = np.array(b[ba])
                    bx = bx[~np.isnan(bx)]
                    ax = P.axarr[row, col]
                    if len(bx) > 0:
                        sns.distplot(bx, ax=ax)
                    else:
                        pass
                        #print('***** bz', bx, celltype)
                    ax.set_xlim(ax_xl[ba])
                    if ba == 'tauh' and celltype[0] in ['bushy', 't-stellate', 'd-stellate']:
                        ax.set_xlim(0, 100.)
                    if ba == 'gh' and celltype[0] not in ['bushy', 't-stellate', 'd-stellate']:
                        ax.set_xlim(0., 20.)
                    if col == 0:
                        ax.text(-0.7, 0.5, celltype[0] + '\nN=%d'%len(bx), transform=ax.transAxes,
                                verticalalignment='center', rotation='vertical')
                    if row == 0:
                        ax.text(0.5, 1.05, ba.replace('_', '\_'), transform=ax.transAxes, horizontalalignment='center')
                    if ba == 'rin':
                        ax.set_xscale('squareroot')
                        ax.set_xticks(np.arange(0, 40, 8)**2)
                        ax.tick_params(axis = 'both', which = 'major', labelsize = 8)
                       # ax.set_yticks(np.arange(0,8.5,0.5)**2, minor=True)
                    
                    col += 1

            if s is not None:
                for sp in s.keys():
                    if sp in ['ap_celltype', 'ap_index']:
                        continue
                    if sp not in measures:
                        continue
                    sx = np.array(s[sp])
                    sx = sx[~np.isnan(sx)]
                    ax = P.axarr[row, col]
                    if len(sx) > 0:
                        sns.distplot(sx, ax=ax)
                    else:
                        pass
                        # print('***** sp ', sx, celltype)
                    ax.set_xlim(ax_xl[sp])
                    if col == 0:
                        ax.text(-0.7, 0.5, celltype[0] + '\nN=%d'%len(sx), transform=ax.transAxes,
                                verticalalignment='center', rotation='vertical', fontsize=5)
                    if row == 0:
                        ax.text(0.5, 1.05, sp.replace('_', '\_'), transform=ax.transAxes, 
                                 horizontalalignment='center')
                    col += 1
                    if sp == 'ap_ar' and celltype[0] == 'cartwheel':
                        ax.set_xlim(0, 25)
                    if sp == 'ap_wid' and celltype[0] == 'bushy':
                        ax.set_xlim(0, 3.0)

        mpl.suptitle('Database: {0:s}\n{1:s}'.format(
                    str(filename).replace('_', '\_'), datetime.datetime.now().strftime('%m.%d.%Y %H:%M:%S')))
        print('saving pdf figure')
        mpl.savefig(Path(self.filename, str(self.filename.stem)+'summary_plot').with_suffix('.pdf'))
        mpl.show()


def _get_db(database):
    """
    Get a pandas database and pull the basic cell and spike measures
    """
    DBR = DataBaseReducer(database=database)
    bsum = {}
    ssum = {}
    for row, celltype in enumerate(DBR.mapped_cells):
        cells = DBR.db[DBR.db['cell_type'].isin([celltype])]
        # DBR.get_CellBasics(cells, noprint=True)
        # DBR.get_SpikeMeasures(cells, noprint=True)

        bsum[celltype] = DBR.basics
        ssum[celltype] = DBR.spike_pars
    return bsum, ssum

def _compare_2(dcsum, desum, ckeys, mkeys):
    """
    Compare 2 databases, with t-tests between values
    
    Parameters
    ----------
    dcsum : dict
        by cell of results
    desum : dict
        by cell of results for experimental group
    ckeys : list of str
        cell names
    mkeys : list of str
        meausure names/keys
    """
    gc = {}
    ge = {}
#     for k in ckeys:
#         mkeys = list(dcsum[k].keys())
#         skeys = list(scsum[k].keys())
# #    print(mkeys, skeys)  # measures
    gc = dict.fromkeys(ckeys)
    ge = dict.fromkeys(ckeys)
    tval = dict.fromkeys(ckeys)
    for k in ckeys:
        gc[k] = dict.fromkeys(mkeys)
        ge[k] = dict.fromkeys(mkeys)
        tval[k] = dict.fromkeys(mkeys)
        for m in mkeys:
            if m not in ['index', 'cell_type', 'ap_index', 'ap_celltype']:
                #print('dcsum: ', dcsum[k][m])
                gc[k][m] = (np.nanmean(dcsum[k][m]), np.nanstd(dcsum[k][m]))
                ge[k][m] = (np.nanmean(desum[k][m]), np.nanstd(desum[k][m]))
                n1 = (~np.isnan(dcsum[k][m])).sum()
                n2 = (~np.isnan(desum[k][m])).sum()
                dcnonan = np.array(dcsum[k][m])[~pd.isnull(dcsum[k][m])]
                denonan = np.array(desum[k][m])[~pd.isnull(desum[k][m])]
                (t, p, df) = ttest_ind(dcnonan, denonan, usevar='unequal')
                tval[k][m] = (t, p, df, n1, n2, gc[k][m], ge[k][m])
    # print ('Controls: \n', gc)
    # print('\nExperimentals: \n', ge)
    print('\nTtests: \n')
    for c in list(tval.keys()):  # by cell type
        tr = tval[c]
        print('Cell type: {0:s}'.format(c))
        for m in list(tr.keys()):
            try:
                print('    {0:>12s} : CTL u={1:7.3f} (SD {2:6.3f}) vs EXP u={3:7.3f} (SD {4:6.3f}) t={5:6.3f} p={6:8.4f} df={7:5.2f} N1={8:2d} N2={9:2d}'.format(
                    m, tr[m][5][0], tr[m][5][1], tr[m][6][0], tr[m][6][1], tr[m][0], tr[m][1], tr[m][2], tr[m][3], tr[m][4],
                ))
            except:
                pass
import matplotlib.colors


colors = {"pyramidal": matplotlib.colors.to_hex("darkseagreen"),
        "cartwheel": matplotlib.colors.to_hex("skyblue"),
        "cartwheel?": matplotlib.colors.to_hex("red"),
        "tuberculoventral": matplotlib.colors.to_hex("lightpink"),
        "tuberculoventral?": matplotlib.colors.to_hex("red"),
        "horizontal bipolar": matplotlib.colors.to_hex("moccasin"),
        "granule": matplotlib.colors.to_hex("linen"),
        "golgi": matplotlib.colors.to_hex("yellow"),
        "unipolar brush cell": matplotlib.colors.to_hex("sienna"),
        "chestnut": matplotlib.colors.to_hex("saddlebrown"),
        "giant": matplotlib.colors.to_hex("sandybrown"),
        "giant?": matplotlib.colors.to_hex("red"),
        "giant cell": matplotlib.colors.to_hex("sandybrown"),
        "Unknown": matplotlib.colors.to_hex("white"),
        "unknown": matplotlib.colors.to_hex("white"),
        " ": matplotlib.colors.to_hex("white"),
        "bushy": matplotlib.colors.to_hex("lightslategray"),
        "t-stellate": matplotlib.colors.to_hex("thistle"),
        "l-stellate": matplotlib.colors.to_hex("darkcyan"),
        "d-stellate": matplotlib.colors.to_hex("mediumorchid"),
        "stellate": matplotlib.colors.to_hex("thistle"),
        "ml-stellate": matplotlib.colors.to_hex("thistle"),
        "octopus": matplotlib.colors.to_hex("darkgoldenrod"),
        
        # cortical (uses some of the same colors)
        "basket": matplotlib.colors.to_hex("lightpink"),
        "chandelier": matplotlib.colors.to_hex("sienna"),

        # cerebellar
        "Purkinje": matplotlib.colors.to_hex("mediumorchid"),
        "purkinje": matplotlib.colors.to_hex("mediumorchid"),
        "purk": matplotlib.colors.to_hex("mediumorchid"),
        
        # not neurons
        'glia': matplotlib.colors.to_hex('lightslategray'),
        'glial': matplotlib.colors.to_hex('lightslategray'),
        'Glia': matplotlib.colors.to_hex('lightslategray'),
        "no morphology": matplotlib.colors.to_hex("white"),

}
def highlight_by_cell_type(row):

    if row.cell_type.lower() in colors.keys():
        return [f"background-color: {colors[row.cell_type.lower()]:s}" for s in range(len(row))]
    else:
        return [f"background-color: red" for s in range(len(row))]


def organize_columns(df):
    return df
    cols = ['ID', 'Group', d, 'slice_slice','cell_cell', 'cell_type', 
        'iv_name', 'holding', 'RMP', 'RMP_SD', 'Rin', 'taum',
        'dvdt_rising', 'dvdt_falling', 'AP_thr_V', 'AP_HW', "AP15Rate", "AdaptRatio", 
        "AP_begin_V", "AHP_trough_V", "AHP_depth_V", "tauh", "Gh", "FiringRate",
        "FI_Curve",
        'date']
    df = df[cols + [c for c in df.columns if c not in cols]]
    return df

def make_excel(df:object, outfile:Path):
    """cleanup: reorganize columns in spreadsheet, set column widths
    set row colors by cell type

    Args:
        df: object
            Pandas dataframe object
        excelsheet (_type_): _description_
    """
    outfile = Path(outfile)
    if outfile.suffix != '.xlsx':
        outfile = outfile.with_suffix('.xlsx')

    writer = pd.ExcelWriter(outfile, engine='xlsxwriter')
    resultno = ['RMP', 'Rin', 'Taum', 'Tauh']
    df[resultno] = df[resultno].apply(pd.to_numeric) 
    df.to_excel(writer, sheet_name = "Sheet1")
    df = organize_columns(df)
    workbook = writer.book
    worksheet = writer.sheets["Sheet1"]
    fdot3 = workbook.add_format({'num_format': '####0.000'})
    fbold = workbook.add_format({'bold': True})
    fcol = {}
    for c in colors.keys():
        fcol[c] = workbook.add_format({'bg_color': colors[c]})
    
    df.to_excel(writer, sheet_name = "Sheet1")

    #['holding', 'RMP', 'Rin', 'taum', 'dvdt_rising', 'dvdt_falling', 
    #    'AP_thr_V', 'AP_HW', "AP15Rate", "AdaptRatio", "AP_begin_V", "AHP_trough_V", "AHP_depth_V"]
   
    for i, column in enumerate(df):
        if column not in ['notes', 'description', 'OriginalTable', 'FI_Curve']:
            coltxt = df[column].astype(str)
            coltxt = coltxt.map(str.rstrip)
            maxcol = coltxt.map(len).max()
            column_width = np.max([maxcol, len(column)]) # make sure the title fits
            if column_width > 50:
                column_width = 50 # but also no super long ones
            #column_width = max(df_new[column].astype(str).map(len).max(), len(column))
        else:
            column_width = 25
        if column_width < 8:
            column_width = 8
        if column in resultno:
            worksheet.set_column(first_col=i+1, last_col=i+1, cell_format=fdot3) # column_dimensions[str(column.title())].width = column_width
            print(f"formatted {column:s} with {str(fdot3):s}")

        worksheet.set_column(first_col=i+1, last_col=i+1, width=column_width) # column_dimensions[str(column.title())].width = column_width

    for i in df.index:
        if i == 0:
            worksheet.set_row(i, cell_format=fbold)
        else:
            ctype = df.iloc[i-1].cell_type
            worksheet.set_row(i, None, cell_format=fcol[ctype])

    # df = df.style.apply(highlight_by_cell_type, axis=1)
    # df.to_excel(writer, sheet_name = "Sheet1")
    writer.close()
    


def compare(dbc, dbe):
    """
    Make the comparison"""
    dcsum, scsum = _get_db(dbc)
    desum, sesum = _get_db(dbe)
    n1 = len(dcsum.keys())
    n2 = len(desum.keys())
    print('n1, n2: ', n1, n2)
    if n2 < n1:
        ckeys = list(desum.keys())
#        skeys = list(s2sum.keys())
    else:
        ckeys = list(dcsum.keys())
#        skeys = list(s1sum.keys())
    mkeys = list(dcsum[ckeys[0]].keys())
    _compare_2(dcsum, desum, ckeys, mkeys)
    skeys = list(scsum[ckeys[0]].keys())
    _compare_2(scsum, sesum, ckeys, skeys)

def name_remap(celltype):
        # remap names:
    if celltype in ['giant', "giant cell", "Giant", "Giant cell"]:
        celltype = 'giant'
    elif celltype in ['glia', 'glial', "glial cell"]:
        celltype = "glial"
    elif celltype in ["stellate"]:
        celltype = "t-stellate"
    elif celltype in ["granule", "granule? tiny"]:
        celltype = "granule"
    elif celltype in ["type-b", "horitzontal bipolar"]:
        celltype = "horizontal bipolar"
    elif celltype in ["fusiform", "pyramidal"]:
        celltype = "pyramidal"
    else:
        pass # celltype = "unknown"

    return celltype

def show_cell_parameters(file="../mrk-nf107-data/datasets/NF107Ai32_Het/IV_Analysis.h5",):
    import h5py
    dn = pd.HDFStore(file, "r")
    cells = sorted(list(dn.keys()))
    select = "first" # or "ALL"
    measures = ["RMP", "Rin", "Taum", "Tauh"]
    P = PH.regular_grid(rows=2, cols=2, order="rowsfirst", 
        labelposition = (0.5, 1.0),
        figsize=(8, 8),
        panel_labels=measures,
        verticalspacing=0.1,
        horizontalspacing=0.1,
        )
    # for kax in P.axdict.keys():
    #     P.axdict[kax].set_prop_cycle(marker_cycler)

    df = pd.DataFrame({"cellid": [], "cell_type": [], 'protocol': [], 'Rin': [], "RMP": [], "Taum": [], "Tauh": []})
    res = []
    cellsdone = []
    indx = 0
    for n, cell in enumerate(cells):
        celltype = dn[cell]['cell_type'].lower()
        cellid = dn[cell]['cell_id']

        celltype = name_remap(celltype)
        if dn[cell]["IV"] is None:
            continue
        ivnames = [Path(x).name for x in list(dn[cell]['IV'].keys())]

        print(f"\r   n={n:4d} cell={cellid:>60s}", end="")
        for i, ivk in enumerate(dn[cell]["IV"]):
            ivdata = dn[cell]["IV"][ivk]
            if len(ivdata.keys()) == 0:
                continue
            # print(ivdata)
            if ivdata["taum"] <= 0.0002 or ivdata["taum"] > 0.1:
                ivdata["taum"] = np.nan
            if ivdata["Rin"] > 1000.:
                ivdata["Rin"] = np.nan
            if ivdata["tauh_tau"] > 200:
                ivdata["tauh_tau"] = np.nan
            # now restrict... 
            if ivdata["RMP"] > -42.0 or ivdata["RMP"] < -80.0:
                continue
            # if ivdata["taum"] is not np.nan and (ivdata["taum"] <= 0.2 or ivdata["taum"] > 100.0):
            #     continue
            res.append(pd.DataFrame({
                "cell_id": str(cellid),
                "cell_type": str(celltype), 
                "protocol": str(ivnames[i]),
                'Rin': np.mean(ivdata["Rin"]),
                "RMP": np.mean(ivdata["RMP"]),
                "Taum": np.mean(ivdata["taum"])*1e3,
                "Tauh": np.mean(ivdata["tauh_tau"]),
                }, index=[0])
            )
            indx += 1
    print()

    df = pd.concat(res, ignore_index=True)
    make_excel(df, "cell_parameters.xlsx")
    print(df.head(100))
    for m in measures:
        sns.violinplot(data=df, x='cell_type', y = m,  ax=P.axdict[m])
        sns.swarmplot(data=df, x='cell_type', y = m,  ax=P.axdict[m])

        mpl.setp(P.axdict[m].get_xticklabels(), rotation = 45, ha="right", fontsize=6)




def check_hdf(file="IV_Analysis.h5", display="FIs"):
    dn = pd.HDFStore(file, "r")
    cells = sorted(list(dn.keys()))
    # print("cells: ", cells)
    select = "first" # or "ALL"

    P = PH.regular_grid(rows=4, cols=4, order="rowsfirst", panel_labels=list(colormap.keys()),
        labelposition = (0.5, 1.0),
        figsize=(8, 8))
    for kax in P.axdict.keys():
        P.axdict[kax].set_prop_cycle(marker_cycler)
    for n, cell in enumerate(cells):
        celltype = dn[cell]['cell_type'].lower()
        celltype = name_remap(celltype)

        
        if display == 'FIs':
            ivi = []
            ivs = []
            prots = []
            for iv in dn[cell]['Spikes']:
                psplit = iv.split('/')
                thisprot = psplit[-1][:-4]
                if select == "first":
                    if thisprot in prots:
                        continue
                    else:
                        prots.append(thisprot)

                elif select == "all":
                    prots.append(thisprot)
                if 'spikes' in list(dn[cell]['Spikes'][iv].keys()):
                    spx = dn[cell]['Spikes'][iv]['spikes']
                else:
                    continue
                if celltype in list(colormap.keys()):
                    color = colormap[celltype]
                else:
                    color = 'k'
                    print(celltype, "assigned to black")
                    celltype = 'unknown'
                    # print(dn[cell]['Spikes'][iv].keys())
                ivc = dn[cell]['Spikes'][iv]['FI_Curve']
                pulsedur = dn[cell]['Spikes'][iv]['pulseDuration']
                imr = np.argmax(ivc[1])
                ivi.extend(ivc[0][:imr]*1e9)
                ivs.extend(ivc[1][:imr]/pulsedur)

            ivi = np.array(ivi)
            ivs = np.array(ivs)
            isort = np.argsort(ivi, kind='stable')

            ivi = ivi[isort]
            ivs = ivs[isort]

            P.axdict[celltype].plot(
                        ivi, ivs, color=color, 
                        markersize=1.25, linestyle='-', linewidth=0.5)
            # P.axdict[celltype].set_title(celltype, color=color, fontsize=8, loc="center", y=1.00, ha="center")
            
            # if n > 10:
            #     break
        for l in list(colormap.keys()):
            P.axdict[l].set_xlim(0, 4.0)
            P.axdict[l].set_ylim(0, 800)
    


    print(f"Plotted data from {n:d} cells (protocols concatenated)")     

    # s = list(df['Spikes'].keys())
    # s0 = df['Spikes'][s[0]]
    # print(s0.keys())
    # # print(s0['spikes'][21].keys())
    # import matplotlib.pyplot as mpl
    # for tr in s0['spikes'].keys():
    #     mpl.plot(s0['spikes'][tr][0]['Vtime'], s0['spikes'][tr][0]['V'])
    # mpl.show()




def check_pkl(file="../mrk-nf107-data/datasets/NF107Ai32_Het/NF107Ai32_Het.pkl"):
    dn = pd.read_pickle(file)
    dn = dn.assign(cell=None)
    
    def make_cell(row):
        datestr = Path(row["date"]).name
        slicestr = str(Path(row["slice_slice"]).parts[-1])
        cellstr = str(Path(row["cell_cell"]).parts[-1])
        return str(Path(datestr, slicestr, cellstr))

    dn['cell'] = dn.apply(make_cell, axis=1)

    cells = sorted(list(set(dn.cell)))
    # print("cells: ", cells)

    select = "first" # or "ALL"
    colormap = {'bushy': "grey", "t-stellate": "forestgreen", 
        'unipolar brush cell': "forestgreen", "octopus": "red", 
        "d-stellate": "orange", "tuberculoventral": "fuchsia",
        "pyramidal": "blue", "cartwheel": "cyan", 
        "giant": "lime", 'unknown': "yellow", 
        "glial": "peachpuff", "granule": "peru",
        "horizontal bipolar": "slategrey", "chestnut": "brown",
        "ml-stellate": "cadetblue", "no morphology": "black"}

    marker_cycler = cycler(marker=Line2D.markers)
    P = PH.regular_grid(rows=4, cols=4, order="rowsfirst", panel_labels=list(colormap.keys()),
        labelposition = (0.5, 1.0),
        figsize=(8, 8))
    for kax in P.axdict.keys():
        P.axdict[kax].set_prop_cycle(marker_cycler)
    for n, cell in enumerate(cells):
        dcell = dn[dn.cell == cell] # subset db # 
        # print(dcell['cell_type'])
        if len(dcell) > 0:
            # print(dcell.head())
            # print(dcell['IV'])
            dcell = dcell.iloc[[0]]
        celltype = dcell['cell_type'].values[0].lower()
        # remap names:
        if celltype in ['giant', "giant cell", "Giant", "Giant cell"]:
            celltype = 'giant'
        if celltype in ['glia', 'glial', "glial cell"]:
            celltype = "glial"
        if celltype in ["stellate"]:
            celltype = "t-stellate"
        if celltype in ["granule", "granule? tiny"]:
            celltype = "granule"
        if celltype in ["type-b", "horitzontal bipolar"]:
            celltype = "horizontal bipolar"
        if celltype == " ":
            celltype = "unknown"
        # if celltype in ["fusiform", "pyramidal"]:
        #     celltype = "pyramidal"
        print(cell, celltype)
        
        ivi = []
        ivs = []
        prots = []
        for ivspk in dcell['Spikes'].values:
            if pd.isnull(ivspk) or ivspk == {}:
                color='k'
                continue
            # print("IVS: ", ivs)
            for ivk in ivspk.keys():
                psplit = str(ivspk).split('/')
                thisprot = psplit[-1][:-4]
                if select == "first":
                    if thisprot in prots:
                        continue
                    else:
                        prots.append(thisprot)

                elif select == "all":
                    prots.append(thisprot)
                if 'spikes' in list(ivspk[ivk].keys()):
                    spx = ivspk[ivk]['spikes']
                else:
                    continue
                if celltype in list(colormap.keys()):
                    color = colormap[celltype]
                else:
                    color = 'k'
                    celltype = 'unknown'
                    # print(dn[cell]['Spikes'][iv].keys())
                ivc = ivspk[ivk]['FI_Curve']
                pulsedur = ivspk[ivk]['pulseDuration']
                imr = np.argmax(ivc[1])
                ivi.extend(ivc[0][:imr]*1e9)
                ivs.extend(ivc[1][:imr]/pulsedur)

            ivi = np.array(ivi)
            ivs = np.array(ivs)
            isort = np.argsort(ivi, kind='stable')
 
            ivi = ivi[isort]
            ivs = ivs[isort]

            P.axdict[celltype].plot(
                        ivi, ivs, color=color, 
                        markersize=1.25, linestyle='-', linewidth=0.5)
        # P.axdict[celltype].set_title(celltype, color=color, fontsize=8, loc="center", y=1.00, ha="center")
        
        # if n > 10:
        #     break
    for l in list(colormap.keys()):
        P.axdict[l].set_xlim(0, 4.0)
        P.axdict[l].set_ylim(0, 800)
    print(f"Plotted data from {n:d} cells (protocols concatenated)")     

    # s = list(df['Spikes'].keys())
    # s0 = df['Spikes'][s[0]]
    # print(s0.keys())
    # # print(s0['spikes'][21].keys())
    # import matplotlib.pyplot as mpl
    # for tr in s0['spikes'].keys():
    #     mpl.plot(s0['spikes'][tr][0]['Vtime'], s0['spikes'][tr][0]['V'])


def main():
    compare('NF107Ai32_Het', 'NF107Ai32_NIHL')  # no noise expose vs. noise expose

if __name__ == '__main__':
    # check_pkl()
    # check_hdf()
    # show_cell_parameters()
    # mpl.show()
    # main()
    # exit()
    #     getAllMeasures()
#    spikepars(cells)
    dbn = "CBA_Age"
    DB = DataBase(dbn)
    dcsum, scsum = _get_db(database=DB)
    DBR = DataBaseReducer(database=DB)
    
    DBR.get_SpontInfo(DBR.db, dbn=dbn, cellselect='pyramidal', printflag=True)
    sumplot()
