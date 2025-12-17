# Find_pharmacology.py
# ==========
#
# A set of utility routines to find datasets with drug manipulations.
#


import pandas as pd
from pathlib import Path
import re
import sys
import ephys
from ephys.tools.get_configuration import get_configuration
import pylibrary.tools.cprint as cprint

CP = cprint.cprint


def get_expt_dict(experiment_name):
    datasets, experiments = get_configuration(configfile="config/experiments.cfg")
    expt = experiments[experiment_name]
    datasetDirectory = Path(expt["analyzeddatapath"], expt["directory"])
    return expt, datasetDirectory


def get_datasummary(expt, datasetDirectory):
    fn = Path(datasetDirectory, expt["datasummaryFilename"])
    with open(fn, "rb") as fh:
        d = pd.read_pickle(fh, compression=None)
    return d



def build_searchwords():
    blockers = [
        "CNQX",
        "DNQX",
        "strychnine",
        "strych",
        "str",
        "stry",
        "bic",
        "picro",
        "ptx",
        "bicuculline",
        "gabazine",
        "SR95531",
        "picrotoxin",
        "APV",
        "D-AP5",
        "D-APV",
        "NBQX",
    ]
    searchwords = [
        "TTX",
        "ttx",
        "tetrodotoxin",
        "4-ap",
        "4ap",
        "4AP",
        "4-AP",
        "4-aminopyridine",
    ] + blockers
    searchwords = [r"\b" + sw + r"\b|" for sw in searchwords]
    searchwords = r"".join(searchwords)[:-1]  # remove last |
    return searchwords

# Find the datasets with the listed blockers and search words in any of the notes
# Works from the datasummary file.
def find_datasets(df: pd.DataFrame, expt: dict):
    searchwords = build_searchwords()
    re_searchwords = re.compile(searchwords, re.IGNORECASE)
    # if any of the searchwords appear in any of the selected fields,
    # we copy to a new dataframe.
    dfs = {}
    for kwd in ["slice_notes", "cell_notes", "description", "solution", "notes"]:
        dfs[kwd] = df[df[kwd].str.contains(searchwords, regex=True, flags=re.IGNORECASE, na=False)]
    dfs = pd.concat(dfs.values()).drop_duplicates().reset_index(drop=True)
    # print out the CELL notes for each cell in the directory (directly from the data file metadata)

    DR = ephys.datareaders.acq4_reader

    basepath = Path(expt["rawdatapath"], expt["directory"])
    n = 0
    cell_list = []
    for idx in dfs.index:
        dx = dfs.iloc[idx]
        # assemble a file path
        fp = Path(basepath, dx["date"], dx["slice_slice"], dx["cell_cell"])
        protodirs = list(fp.glob("*"))
        protocols = []
        usable_prots = list(expt["protocols"]["Maps"].keys())
        for p in protodirs:
            for mprot in usable_prots:
                if p.name.startswith(mprot):
                    protocols.append(p)


        for p in protocols:
            dirx = p  # Path(fp, p)
            if not dirx.is_dir():
                continue
            X = DR.acq4_reader(dirx)

            if "notes" in list(X.getIndex().keys()):
                pa = Path(fp).parts
                found = False
                if re_searchwords.search(dx["cell_notes"]) is not None:
                    c_cn = "g"
                    found = True
                else:
                    c_cn = "w"
                if re_searchwords.search(dx["slice_notes"]) is not None:
                    c_sn = "b"
                    found = True
                else:
                    c_sn = "w"
                if re_searchwords.search(X.getIndex()["notes"]) is not None:
                    c_no = "m"
                    found = True
                else:
                    c_no = "w"
                if re_searchwords.search(dx["description"].replace("\n", " ")) is not None:
                    c_de = "c"
                    found = True
                else:
                    c_de = "w"
                if re_searchwords.search(dx["solution"]) is not None:
                    c_so = "y"
                    found = True
                else:
                    c_so = "w"
                if found:
                    if str(Path(*pa[-4:])) not in cell_list:
                        cell_list.append(str(Path(*pa[-4:])))
                        n += 1
                        print(f"\n============================== Cell {n:d} ====================================")
                    print(f"{str(Path(*pa[-4:])):<50s}  {p.name:<24s} :")
                    CP(c_cn, f"    Cell notes:     {dx['cell_notes']}")
                    CP(c_sn, f"    Slice notes:    {dx['slice_notes']}")
                    CP(c_no, f"    Protocol notes: {X.getIndex()['notes']}")
                    CP(c_de, f"    Description:    {dx['description']}")
                    CP(c_so, f"    Solution:       {dx['solution']}")
                    print("")

def print_cell_notes(df: pd.DataFrame):
    t = df["cell_notes"]
    for tx in t:
        print(tx)


def main():
    expt, datasetDirectory = get_expt_dict(sys.argv[1])
    df = get_datasummary(expt, datasetDirectory)
    find_datasets(df, expt)

def test():
    teststr = "This cell was recorded in the presence of 1 uM TTX and 10 uM CNQX and 50 uM APV"
    searchwords = build_searchwords()
    # searchwords = r"\bAPV\b|\bTTX\b|\bCNQX\b"
    print("Search words: ", searchwords)
    re_searchwords = re.compile(searchwords, re.IGNORECASE)
    m = re_searchwords.search(teststr)
    print("Test string: ", teststr)
    print("Match: ", m)
    print("Groups: ", m.groups())


if __name__ == "__main__":
    main()


