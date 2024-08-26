import logging
from pathlib import Path

Logger = logging.getLogger("AnalysisLogger")

def build_info_string(AR, path_to_cell):
    infostr = ""
    info = AR.readDirIndex(currdir=Path(AR.protocol).parent.parent.parent)["."]
    slice_info = AR.readDirIndex(currdir=Path(AR.protocol).parent.parent)["."]
    cell_info= AR.readDirIndex(currdir=Path(AR.protocol).parent)["."]
    info_keys = list(info.keys())
    # print("Info: ", info)
    if "sex" not in info_keys or "age" not in info_keys or "temperature" not in info_keys or "internal" not in info_keys:
        Logger.critical(f"Missing info in top-level .index file in {path_to_cell}")
        return("Missing Critical information in .index file")

    slice_info_keys = list(slice_info.keys())
    cell_info_keys = list(cell_info.keys())
    if "animal_identifier" in info_keys:
        infostr += f"ID: {info['animal_identifier']:s}, "
    # if "sex" in info_keys:
    #     infostr += f"{info["sex"].upper():s}, "
    # if "age" in info_keys:
    #     infostr += f"{info["age"].upper():s}, "
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
    if "temperature" in info_keys:
        temp = info["temperature"]
        if temp == "room temperature":
            temp = "RT"
    if "sex" not in info.keys() or "age" not in info.keys() or "temperature" not in info.keys() or "internal" not in info.keys():
        Logger.critical(f"Missing info in top-level .index file in {path_to_cell}")
        exit()
    # better to fail than substitute a value
    # else:
    #     temp = "not specified"
    # if "sex" not in info.keys():
    #     info["sex"] = "ND"
    # if "age" not in info.keys():
    #     info["age"] = "ND"
    infostr += "{0:s}, ".format(temp)
    infostr += "{0:s}, ".format(info["sex"].upper())
    infostr += "{0:s}".format(str(info["age"]).upper())
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