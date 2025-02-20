from pathlib import Path


"""
Organize Files
For runs that generate individual pdfs, create a directory with the base name
and put all the matching pdfs into that directory

"""
# expt = Path('NF107AI32_NIHL')
# expt = Path('VGAT_NIHL')
expt = Path('NF107AI32_Het')

allpdfs = list(expt.glob('*.pdf'))
for ct in ['pyramidal', 'cartwheel', 'tuberculoventral', 'bushy', 'd-stellate', 't-stellate', 'stellate', 'giant', 'glial', 'unknown']:
    cells = []
    bp = Path(expt, ct)
    if not bp.is_dir():
        bp.mkdir(exist_ok=True)
    for p in allpdfs:
        fn = p.stem
        u = fn.find(ct)
        if u > 0:
            cells.append(fn[:u-1])

    uniqueCells = list(set(cells))
    for d in uniqueCells:
        datafiles = list(expt.glob(f'{d:s}*.pdf'))
        print(d, '\n', datafiles)
        for df in datafiles:
            print('   Moving ', df, 'to:  ', Path(bp, df.name))
            df.rename(Path(bp, df.name))
        
        
    
    
