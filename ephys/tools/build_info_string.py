from pathlib import Path
def build_info_string(AR, path_to_cell):
    infostr = ""
    info = AR.readDirIndex(currdir=Path(AR.protocol).parent.parent.parent)["."]
    slice_info = AR.readDirIndex(currdir=Path(AR.protocol).parent.parent)["."]
    cell_info= AR.readDirIndex(currdir=Path(AR.protocol).parent)["."]


    info_keys = list(info.keys())
    slice_info_keys = list(slice_info.keys())
    cell_info_keys = list(cell_info.keys())
    if "animal identifier" in info_keys:
        infostr += f"ID: {info['animal identifier']:s}, "
    # if "sex" in info_keys:
    #     infostr += f"{info["sex"].upper():s}, "
    # if "age" in info_keys:
    #     infostr += f"{info["age"].upper():s}, "
    if "cell_location" in cell_info_keys:
        infostr += f"{cell_info['cell_location']:s}, "
    if "cell_layer" in cell_info_keys:
        infostr += f"{cell_info['cell_layer']:s}, "
    if "cell_type" in cell_info_keys:
        infostr += f"{cell_info['cell_type']:s}, "
    else:
        infostr += "No Cell Type, "
    if "cell_expression" in info_keys:
        infostr += f"Exp: {info['cell_expression']:s}, "
    
    # notes = self.df.at[icell,'notes']
    if "internal" in info_keys:
        infostr += info["internal"] + ", "
    if "temperature" in info_keys:
        temp = info["temperature"]
    if temp == "room temperature":
        temp = "RT"
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