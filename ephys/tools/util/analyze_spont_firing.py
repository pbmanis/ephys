"""
"""
from pathlib import Path
from typing import Literal, Union

import ephys
import matplotlib.pyplot as mpl
import numpy as np
import pandas as pd
import matplotlib.axes._subplots
from matplotlib.backends.backend_pdf import PdfPages
from pylibrary.tools import cprint
from pylibrary.plotting import plothelpers as PH

CP = cprint.cprint
nr = 0
cell_types = ['pyramidal', 'cartwheel', 'tuberculoventral']

analyzed_data_file = "nihl_spont_PCT.xlsx"

main_database = "NF107_NIHL.pkl"

#######################
# Set some criteria
# RMP SD across protocol
rmp_sd_limit = 3.0 # mV
#
# Smallest event to consider a spike
minimum_spike_voltage = -0.020  # V 


max_rows = -1  # just analyze the first n rows from the table

#IV_list = Path("../mrk-nf107-data/datasets/NF107Ai32_NIHL/NIHL_Summary.xlsx")
exclude_list = Path("../mrk-nf107-data/datasets/NF107Ai32_NIHL/NF107Ai32_NIHL_IVs_exclude.xlsx")
code_sheet = (Path("../mrk-nf107-data/datasets/NF107Ai32_NIHL/NF107Ai32_NoiseExposure_Code.xlsx"))
code_dataframe = pd.read_excel(code_sheet)

cols = ['ID', 'Group', d, 'slice_slice','cell_cell', 'cell_type', 'age',
    'iv_name', 'holding',
    'dvdt_rising', 'dvdt_falling', 'AP_thr_V', 'AP_HW',
    "AP_begin_V", "AHP_trough_V", "AHP_depth_V", 
    'date', 'spiketimes']

basic_cols = ['ID', 'date', 'Group', 'slice_slice','cell_cell', 'cell_name', 'cell_type', 'age',
    'iv_name']

def get_rec_date(filename:Union[Path, str]):
    """get the recording date of record from the filename as listed n the excel sheet

    Args:
        filename (Union[Path, str]): _description_
    """
    fn = Path(filename)
    datename = str(fn.name)
    datename = datename[:-4]
    return datename

def get_cell_name(row):
    dn = str(Path(row.date).name)[:-4]
    sn = f"S{int(row.slice_slice[-3:]):02d}"
    cn = f"C{int(row.cell_cell[-3:]):02d}"
    cellname = f"{dn:s}_{sn:s}_{cn:s}"
    return cellname

def build_protocols(cellname, df):
    """get a list of all the CCIV protocols associated with this cell
    """
    pass


def get_spont_protocol(row, prottype, pdf_pages:Union[object, None] = None):
    """Get the IV protocol from this dataframe example

    Args:
        row (_type_): _description_
    """
    dataok = False
    full_spike_analysis = True
    threshold = -0.020

    if pd.isnull(row.date):
        return row
    if row.cell_type not in cell_types:
        return row

    if int(row.name) > max_rows and max_rows != -1:
        return row

    fullpatha = Path(row.date, row.slice_slice, row.cell_cell, row.iv_name)

    if fullpatha.name.startswith(prottype):
        fullpath = fullpatha
    else:
        return row
    try:
        AR = ephys.datareaders.acq4_reader.acq4_reader(fullpath, "MultiClamp1.ma")
        r = AR.getData()
        if r is False:
            print("? failed to get data", fullpath)
            return row
        supindex = AR.readDirIndex(currdir=Path(fullpath.parent))
        # print("supindex: ", supindex.keys())
        CP("g", f"Spont OK: {str(fullpath):s}")
        dataok = True
    except:
        CP("r", "Failed to read group info in table")
        print(fullpath)
        raise ValueError


    IVA = ephys.ephys_analysis.IVSummary.IVSummary(fullpath, plot=True, pdf_pages=pdf_pages)
    IVA.iv_check(duration=0.1)
    IVA.SP.setup(
        clamps=IVA.AR,
        threshold=threshold,
        refractory=0.0001,
        peakwidth=0.001,
        interpolate=True,
        verify=False,
        mode="schmitt",
    )
    IVA.SP.analyzeSpikes()
    if full_spike_analysis:
        IVA.SP.analyzeSpikeShape()
        IVA.SP.analyzeSpikes_brief(mode="baseline")
        IVA.SP.analyzeSpikes_brief(mode="poststimulus")
        # self.SP.fitOne(function='fitOneOriginal')

    # f, ax = mpl.subplots(2,1)
    avg_V = None
    avg_dV = None
    first_spike = True
    for tr in IVA.SP.spikeShape:
        ns = 0
        for i, spkn in enumerate(IVA.SP.spikeShape[tr]):
            spk = IVA.SP.spikeShape[tr][spkn]
            # print(f"I: {spk['current']:6.3e}  Lat: {spk['AP_Latency']:6.3e}  Rise: {spk['dvdt_rising']:4.1f}, {spk['dvdt_falling']:4.1f}, Thr: {1e3*spk['AP_beginV']:4.2f}")
            if i == 0:
                row.dvdt_rising = spk['dvdt_rising']
                row.dvdt_falling = spk['dvdt_falling']
                if spk['AP_beginV'] is not None:
                    row.AP_thr_V = 1e3*spk['AP_beginV']
                    row.AP_begin_V = 1e3*spk['AP_beginV']
                if spk['halfwidth_interpolated'] is not None:
                    row.AP_HW = spk['halfwidth_interpolated']*1e3

                first_spike = False

           # tx = (spk['Vtime'] - spk['Vtime'][0])*1e3

    row.AHP_trough_V = 1e3*IVA.SP.analysis_summary['AHP_Trough']
    row.AHP_depth_V = IVA.SP.analysis_summary['AHP_Depth']
    row.holding = AR.holding
    ntraces = AR.traces.shape[0]
    if isinstance(row.Group, float):
        row.Group = "unknown"
    gcolors = {"B": "k", "A": "g", "AA": "b", "AAA": "r", "unknown": "c", "30D": 'm'}
    cell_tstart = supindex['.']['__timestamp__']
    fig, ax = mpl.subplots(max([2, ntraces]), 1, figsize=(11, 8.5))
    # if isinstance(ax, matplotlib.axes._subplots.AxesSubplot):
    #     ax = np.array([ax])
    if row.iv_name.startswith("CC_"):
        sf = 1e3
        ylim =(-75, 25)
        dco = False
    elif row.iv_name.startswith("VC_"):
        sf = 1e12
        ylim = (-100, 100)
        dco = True
    for i in range(ntraces):
        pdata = AR.getDataInfo(Path(fullpath, f"{i:03d}", "MultiClamp1.ma"))
        if i == 0:
            p0 = pdata[1]['startTime']
        if pd.isnull(row.Group):
            c = "c"
        else:
            c = gcolors[row.Group]
        zero = 0
        if dco:
            zero = np.mean(AR.traces[i].view(np.ndarray))
        
        ax[i].plot(AR.time_base, (AR.traces[i].view(np.ndarray)-dco)*sf, color=c, linestyle='-', linewidth=0.25)
        t_elapsed = pdata[1]['startTime']- cell_tstart
        prot_elapsed = pdata[1]['startTime'] - p0
        ax[i].set_title(f"prot {t_elapsed:.3f} s  Rec: {i:d} at {prot_elapsed:.3f} s ", loc="left", fontsize=8)
        ax[i].set_xlabel("T (sec)")
        ax[i].set_ylabel("V (mV)")
        ax[i].set_ylim(ylim)
    PH.nice_plot(ax, position=-0.03, ticklength=3, direction="outward")
    if isinstance(supindex['.']['__timestamp__'], float):  # because sometimes theres a bug in the data storage
        ts = f"{supindex['.']['__timestamp__']:f}"
    else:
        ts = f"{str(supindex['.']['__timestamp__']):s}"
    shortpath = Path(*fullpath.parts[4:])
    fig.suptitle(f"{str(fullpath):s}\nStarts at: {ts:s}, Group: {row.Group:s} Celltype: {str(row.cell_type):s} Age: {str(row.age):s}", fontsize=8, ha="center")
    pdf_pages.savefig(fig)
    mpl.close()
    return row

def _make_short_name(row):
    return get_rec_date(row['date'])

def highlight_by_cell_type(row):
    colors = {"pyramidal": "#c5d7a5", #"darkseagreen",
            "cartwheel": "skyblue",
            "tuberculoventral": "lightpink",
            "ml-stellate": "orchid",
            "granule": "linen",
            "golgi": "yellow",
            "unipolar brush cell": "sienna",
            "chestnut": "saddlebrown",
            "giant": "sandybrown",
            "giant?": "sandybrown",
            "giant cell": "sandybrown",
            "Unknown": "white",
            "unknown": "white",
            " ": "white",
            "bushy": "lightslategray",
            "t-stellate": "thistle",
            "d-stellate": "plum",
            "l-stellate": "darkcyan",
            "stellate": "thistle",
            "octopus": "darkgoldenrod",
            
            # cortical (uses some of the same colors)
            "basket": "lightpink",
            "chandelier": "sienna",

            # cerebellar
            "Purkinje": "mediumorchid",
            "purkinje": "mediumorchid",
            "purk": "mediumorchid",
            
            # not neurons
            'glia': 'lightslategray',
            'Glia': 'lightslategray',
    }

    return [f"background-color: {colors[row.cell_type]:s}" for s in range(len(row))]



def find_protocols(df):
    # df = pd.read_excel(excelsheet)
    # code_dataframe = pd.read_excel(codesheet)

    # generate short names list
    df['shortdate'] = df.apply(_make_short_name, axis=1)

    # df_new['date'] = sorted(list(df['shortdate', right_on=d, how='left')
    df_new = pd.merge(df, code_dataframe, left_on='shortdate', right_on='date', how='left')
    df_new['holding'] = np.nan
    df_new['dvdt_rising'] =np.nan
    df_new['dvdt_falling'] =np.nan
    df_new['AP_thr_V'] = np.nan
    df_new['AP_HW'] = np.nan
    df_new['AP_begin_V'] = np.nan
    df_new['AHP_trough_V'] = np.nan
    df_new['AHP_depth_V'] = np.nan
    df_new['FiringRate'] = np.nan
    df_new['AdaptRatio'] = np.nan
    df_new['spiketimes'] = np.nan
    nprots = 0
    
    
    with pd.ExcelWriter(analyzed_data_file) as writer:
        df_new.to_excel(writer, sheet_name = "Sheet1")
        for i, column in enumerate(df_new.columns):
            column_width = max(df_new[column].astype(str).map(len).max(),24) # len(column))
            writer.sheets['Sheet1'].set_column(first_col=i+1, last_col=i+1, width=column_width) # column_dimensions[str(column.title())].width = column_width
        # writer.save()
    print("\nfind_protocols wrote: ", analyzed_data_file)
    return df_new

def organize_columns(df):

    df = df[cols + [c for c in df.columns if c not in cols]]
    return df

def checkforprotocol(df, index, protocol:str):
    # print(row.data_complete)
    prots = df.data_complete[index].split(",")
    prots2 = df.data_incomplete[index].split(",")
    prots.extend([p.split(".")[0] for p in prots2])
    protlist = []
    for p in prots:
        sp = p.strip()
        if sp.startswith(protocol):
            protlist.append(sp)
    return protlist


def get_protocols_from_datasheet(filename, filterstring:str, outcols:list=[], outputfile:str=None):
    """read the protocols from the datasheet, and build a new
    dataframe expanded by individual protocols

    Args:
        filename (_type_): name of data sheet/pickled pandas file to read.
        filterstring (_type_): string to filter protocols on ("CC_Spont", for example)

    Returns:
        pandas dataframe: the pandas data frame with the code dataframe merged in,
        for all of the individual cells/protocols that match.
        adds a column that uniquely identifies cells by date, slice, cell
    """
    df = pd.read_pickle(filename)

    # force some column types
    for coln in ['age', 'date']:
        df[coln] = df[coln].astype(str)
    # generate short names list
    df['shortdate'] = df.apply(_make_short_name, axis=1)
    df['cell_name'] = df.apply(get_cell_name, axis=1)
    # read the code dataframe (excel sheet)
    code_df = pd.read_excel(code_sheet)
    # be sure types are correct
    for coln in ['age', 'ID', 'Group']:
        code_df[coln] = code_df[coln].astype(str)
    code_df = code_df.drop('age', axis="columns")
    dfm = pd.merge(df, code_df, left_on='shortdate', right_on=date, how='left') # codes are by date only
    dfm["iv_name"] = ""
    # make an empty dataframe with defined columns
    df_new = pd.DataFrame(columns=outcols)
    # go through all of the entries in the main df
    for index in dfm.index:
        prots = checkforprotocol(df, index, protocol=filterstring) # get all protocols in that entry
        for prot in prots:
            data = {}
            for col in outcols:  # copy over the information from the input dataframe

                if col != 'iv_name':
                    data[col] = dfm[col][index]
                else:
                    data['iv_name'] = prot # add the protocol 
            if data['cell_type'] not in ['Pyramidal', 'pyramidal']:
                continue
            if data['age'].strip() in ['', ' ', '?']:
                continue
            a = ''
            age = [a+b for b in str(data['age']) if b.isnumeric()]
            age = ''.join(age)
            try:
                d_age = int(age)
            except:
                print('age: ', age, data['age'])
                exit()
            if d_age < 30 or d_age < 65:
                continue
            df_new.loc[len(df_new.index)] = data # and store in a new position

    if outputfile is not None:
        df_new.to_excel(outputfile)
    return df_new


def cleanup(excelsheet, outfile:str="test.xlsx"):
    """cleanup: reorganize columns in spreadsheet, set column widths
    set row colors by cell type

    Args:
        excelsheet (_type_): _description_
    """
    df_new = pd.read_excel(excelsheet)
    writer = pd.ExcelWriter(outfile, engine='xlsxwriter')

    df_new.to_excel(writer, sheet_name = "Sheet1")
    df_new = organize_columns(df_new)
    workbook = writer.book
    worksheet = writer.sheets["Sheet1"]
    fdot3 = workbook.add_format({'num_format': '####0.000'})
    df_new.to_excel(writer, sheet_name = "Sheet1")

    resultno = ['holding',  'dvdt_rising', 'dvdt_falling', 
        'AP_thr_V', 'AP_HW', "AdaptRatio", "AP_begin_V", "AHP_trough_V", "AHP_depth_V"]
    df_new[resultno] = df_new[resultno].apply(pd.to_numeric)    
    for i, column in enumerate(df_new):
        # print('column: ', column)
        if column in resultno:
            writer.sheets['Sheet1'].set_column(first_col=i+1, last_col=i+1,  cell_format=fdot3)
        if column not in ['notes', 'description', 'OriginalTable', 'FI_Curve']:
            coltxt = df_new[column].astype(str)
            coltxt = coltxt.map(str.rstrip)
            maxcol = coltxt.map(len).max()
            column_width = maxcol
            #column_width = max(df_new[column].astype(str).map(len).max(), len(column))
        else:
            column_width = 25
        if column_width < 8:
            column_width = 8
        if column in resultno:
            writer.sheets['Sheet1'].set_column(first_col=i+1, last_col=i+1, cell_format=fdot3, width=column_width) # column_dimensions[str(column.title())].width = column_width
            print(f"formatted {column:s} with {str(fdot3):s}")
        else:
            writer.sheets['Sheet1'].set_column(first_col=i+1, last_col=i+1, width=column_width) # column_dimensions[str(column.title())].width = column_width
        

    df_new = df_new.style.apply(highlight_by_cell_type, axis=1)
    df_new.to_excel(writer, sheet_name = "Sheet1")
    writer.save()

if __name__ == "__main__":
    filename = "~/Desktop/Python/mrk-nf107-data/datasets/NF107Ai32_NIHL/NIHL_Summary.pkl"
    filename = "~/Desktop/Python/mrk-nf107-data/datasets/NF107Ai32_NIHL/Controls_NF107_Het.pkl"
    prottype = "CC_Spont"
    df = get_protocols_from_datasheet(filename, prottype, outcols=basic_cols, outputfile="nf107_old.xlsx")
    print(df.columns)
    # There will be 141 entries in here as of 11/21/2022 for NIHL dataset

    with PdfPages(f"NF107_old_{prottype:s}.pdf") as pdfs:
        df.apply(get_spont_protocol, prottype=prottype, pdf_pages=pdfs, axis=1)
    # cleanup(analyzed_data_file, outfile="Controls_sponts.xlsx")


