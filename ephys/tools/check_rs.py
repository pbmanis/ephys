import argparse
import os  # legacy
import sys
import pickle
from pathlib import Path

import ephys.ephysanalysis as EP
import ephys.ephysanalysis.metaarray as EM  # need to use this version for Python 3
import ephys.ephysanalysis.PSCAnalyzer as EPP
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
import pylibrary.plotting.plothelpers as PH
import pylibrary.tools.cprint as CP
cprint = CP.cprint

def check_rs(protocol, clamp="MultiClamp1.ma", verbose=False):
    """
    Read the compensation settings from the data file
    Print out Rs values when they have changed WITHIN A PROTOCOL.
    Save the resulting compensation parameters in a dict for later.
    """
    AR = EP.acq4read.Acq4Read(Path(protocol))
    protocol = Path(protocol)
    prot_Rs = []
    protocol_reported = False
    if verbose:
        print(f"check_rs: Checking Protocol: {str(protocol):s}")
    for f in list(protocol.glob('*')):  # for all traces in protocol
        if f.is_file():
            continue
        prot_dir = Path(protocol, f, clamp)
        info = AR.getDataInfo(prot_dir)
        if info is None:
            continue
        WC = AR.parseClampWCCompSettings(info)
        prot_Rs.append(WC)
    if len(prot_Rs) == 0: # can occur if the protocol has only one entry that is not in a subdirectory.
        return None
    bwc = prot_Rs[0]['WCResistance']*1e-6
    anychange = False
    maxdelta = 0.0
    for irs, w in enumerate(prot_Rs):
        wrs = w['WCResistance']*1e-6
        delta = 0.0
        if (bwc-wrs)/bwc > 0.01: # less than 1% change
            delta = 100*(bwc-wrs)/bwc
            if delta > maxdelta:
                maxdelta = delta
            if not protocol_reported:
                print(f"check_rs: Checked Protocol and found changes in Rs within protocol:\n  {str(protocol):s}")
                protocol_reported = True
            print(f"    tr:{irs:3d} Rs changed from: {bwc:.2f} to: {wrs:.2f}, delta = {100*(bwc-wrs)/bwc:.1f} pct")
            bwc = wrs
            anychange=True
    WCRS = {"dir": protocol, "WC": bwc, 'maxdelta': maxdelta, 'anychange': anychange, 
            'cap': w['WCCellCap']*1e12, 'compEnabled': w['CompEnabled'],
            'compPct': w['RsCompCorrection']}
    return WCRS

def print_rs(WCRS):
    """Print the accumulated data in the 'whole-cell Rs' list. The list consists of individual
        entries as dictionaries, generated by the check_rs function above.
    It is meant to be a list... :)
    """
    print(f"    {'File/Protocol':^60s}   {'Rs (Mohm)':^9s}   {'Comp Enabled':^12s}    {'Comp%':^10s}    {'RS delta %':^12s}    {'Cap (pF)':^9s}")
    for f in WCRS:
        if f is None:
            continue
        if not f['compEnabled']:
            col = 'm'
        elif f['WC'] > 30.0:
            col = 'r'
        elif f['WC'] > 20:
            col = 'y'
        else:
            col = 'w'
        short = str(Path(*Path(f['dir']).parts[-4:]))
        cprint(col, f"    {short:<60s}   {f['WC']:9.2f}    {str(f['compEnabled']):^12s}    {f['compPct']:^9.1f}    {f['maxdelta']:10.2f}    {f['cap']:8.1f}")
    print("Code: magenta: no compensation, red: Rs>30M, yellow: Rs>20M, white: Rs<20M, compensated")
    print("="*80)
    
def check_rs_cell(cell_dir):
    """
    Given a cell directory (/Volumes/user/experiment/date.mm.yy_000/slice_nnn/cell_mmm)
    go through all the prototocols and read the amplifier settings. 
    """
    prots = list(Path(cell_dir).glob("*"))
    print('# prots: ', len(prots))
    w = []
    for p in prots:
        if p.is_file():
            continue
        w.append(check_rs(Path(cell_dir, p)))
    print_rs(w)

if __name__ == '__main__':
    check_rs_cell(sys.argv[1])

    