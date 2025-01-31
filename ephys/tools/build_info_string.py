import logging
from pathlib import Path

Logger = logging.getLogger("AnalysisLogger")

def build_info_string(AR, path_to_cell):
    infostr = ""
    info = AR.readDirIndex(currdir=Path(AR.protocol).parent.parent.parent)["."]
    slice_info = AR.readDirIndex(currdir=Path(AR.protocol).parent.parent)["."]
    cell_info= AR.readDirIndex(currdir=Path(AR.protocol).parent)["."]
    info_keys = list(info.keys())
    missing = False

    slice_info_keys = list(slice_info.keys())
    cell_info_keys = list(cell_info.keys())
    if "animal_identifier" in info_keys:
        infostr += f"ID: {info['animal_identifier']:s}, "
    if "sex" in info_keys:
        infostr += f"{info["sex"].upper():s}, "
    else:
        infostr += f"sex: ND, "
        missing = True
    if "age" in info_keys:
        infostr += f"P{info["age"].upper():s}D, "
    else:
        infostr += f"age: ND, "
        missing = True
    if "cell_location" in cell_info_keys:
        infostr += f"{cell_info['cell_location']:s}, "
    if "cell_layer" in cell_info_keys:
        infostr += f"{cell_info['cell_layer']:s}, "
    if "type" in cell_info_keys:
        infostr += f"{cell_info['type']:s}, "
    else:
        infostr += "No Cell Type, "
    if "strain" in info_keys:
        infostr += f"{info['strain']:s}, "
    if "cell_expression" in info_keys:
        infostr += f"Exp: {info['cell_expression']:s}, "
    
    # notes = self.df.at[icell,'notes']
    if "internal" in info_keys:
        infostr += info["internal"] + ", "
    else:
        infostr += "Internal: ND, "
        missing = True
    if "temperature" in info_keys:
        temp = info["temperature"]
        if temp == "room temperature":
            temp = "RT"
    else:
        missing=True
        temp = "temp: ND"

    infostr += "{0:s}, ".format(temp)

    if missing:
        infostr += "\n(NOTE: Critical missing fields)"
    return infostr

def main():
    # test
    from ephys.datareaders import acq4_reader as AR
    dn = Path("/Volumes/Pegasus_002/ManisLab_Data3/Kasten_Michael/NF107Ai32_Het/2022.06.29_000/slice_001/cell_000/CCIV_1nA_max_1s_pulse_000")
    areader = AR.acq4_reader(dn)
    infostr = build_info_string(areader, dn)
    print(infostr)
if __name__ == "__main__":
    main()