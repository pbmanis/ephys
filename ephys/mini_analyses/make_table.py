#!/usr/bin/env/python

"""
Make a dataset directory for a set of mini experiments on disk

Directory structure is:
datasets = {'m1a': {'dir': '2017.04.25_000/slice_000/cell_001', 'prots': [0,1,3], 'thr': 1.75, 'rt': 0.35, 'decay': 6., 'G': 'F/+'},

Where the first level is the mouse #, letter is the cell.
There is one second level dict for every cell_nnn in the main directory

Second level dict:
dir : directory for the data
prots: a list of the protocols (taken from all "minis_nnn" subdirectores)
thr : threshold for acceptance (use default, 1.75)
rt : rise time (use default 0.35)
decay : decay time (use defualt: 6)
G : Genotype (defuault is '+/+', meaning wildtype)

"""

import os

#basepath = '/Volumes/Pegasus/ManisLab_Data3/Sullivan_Chelsea/miniIPSCs/CHL1'  # CHL1 data set...

def getdescription(indexfile):
    """
    Read the index file and find a genotype in the description field
    """
    with open(indexfile, 'r') as f:
        for l in f:
            if 'description:' in l:
                ix = l.index(': ')
                if 'WT' in l[ix+2:]:
                    return('WT')
                elif 'CHL1' in l[ix+2:]:
                    return('CHL1')
                else:
                    return('Unknown')

def make_table(basepath=None):
    if basepath is None:
        print('make_table: must set basepath')
    
    days = os.listdir(basepath)
    print(("basepath = '%s'" % basepath))

    print('')
    mouseno = 0
    dicttext = 'datasets = {\n'
    for d in days:
        cellno = 1
        topfile = os.path.join(basepath, d)
        if os.path.isfile(topfile):

            continue
        slicenumber = os.listdir(topfile)
        mouseno = mouseno + 1
        for sl in slicenumber:
            cellnumbers = ''
            slicepath = os.path.join(basepath, d, sl)
            if os.path.isfile(slicepath):
                if sl == '.index':
                    genotype = getdescription(slicepath)

                continue
            if os.path.isdir(slicepath):
                cellnumbers = os.listdir(slicepath)
            for celln, cell in enumerate(cellnumbers):
                protocols = ''
                cellpath = os.path.join(basepath, d, sl, cell)
                if os.path.isfile(cellpath):
                    continue
                if os.path.isdir(cellpath):
                    cellno = cellno + 1
                    protocols = os.listdir(cellpath)
                protocollist = []
                protocolpath = ''
                for protocol in protocols:
                    protocolpath = os.path.join(basepath, d, sl, cell, protocol)
                    if os.path.isfile(protocolpath):
                        continue
    #                print 'protocol: ', protocol
                    if os.path.isdir(protocolpath) and protocol[0:5] == 'minis':
                        pn = int(protocol[-3:])  # convert to number
                        protocollist.append(pn)
                if protocollist == []:
                    continue
                dicttext = dicttext + "\t'm{0:d}{1:s}': ".format(mouseno, chr(96+celln))
                dicttext = dicttext + "{{'dir': '{0:s}', ".format(os.path.join(d, sl, cell))
                dicttext = dicttext + ("'prots': {0:s},".format(protocollist))
                dicttext = dicttext + (" 'thr': 1.75, 'rt': 0.35, 'decay': 6., 'G': '" + genotype + "'},\n")
    dicttext = dicttext + '}'        
    print(dicttext)

