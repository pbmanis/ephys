""" Fix corrupted .index files in acq4 protocol directories.
    This little utlity reads the top .index file to get a base time,
    then it goes through all the subdirectories and writes a "fake" .index file
    with updated times so that the times are in sequence, even if they
    are not the actual times of the protocols.
    It should be easy to adjust this (by changing make_fake_index) to
    handle different protocols, etc.

    9 Jan 2024 pbmanis UNC CH
    
"""

from pathlib import Path
import pyqtgraph.configfile as pgc

proto_dir = Path("/Volumes/Pegasus_004/ManisLab_Data3/Edwards_Reginald/CBA_Age/2024.12.10_000/slice_001/cell_000/CCIV_4nA_max_1s_pulse_000")

dir_exists = proto_dir.exists()
if not dir_exists:
    print(f"Directory {proto_dir} does not exist")
    exit()


def make_fake_index(ts):
    ts0 = ts + 1e-5
    ts1 = ts0 + 1e-5
    index_file_text = f"""
.:
    ('MultiClamp1', 'Pulse_amplitude'): 0
    dirType: 'Protocol'
    __timestamp__: {ts:f}
    startTime: {ts0:f}
MultiClamp1.ma:
    __object_type__: 'MetaArray'
    __timestamp__: {ts1:f}
"""
    return index_file_text


def list_index_files(directory, top_dot_index):
    for f in directory.glob("*.index"):
        # print(f)
        d = pgc.readConfigFile(f)
        for dk in d.keys():
            if dk == '.':
                continue
             

def read_dot_index_file(proto_dir):
    dot_index_file = Path(proto_dir, ".index")
    if not dot_index_file.exists():
        print(f"File {dot_index_file} does not exist")
        return None
    d = pgc.readConfigFile(dot_index_file)
    return d

top_dot_index = read_dot_index_file(proto_dir)
print(top_dot_index.keys())
for k in top_dot_index.keys():
    if k == '.':
        continue
    ts = top_dot_index[k]["__timestamp__"]
    # print(k, ts, ts+1e-4)  # this will be the fake time
    fake_index = make_fake_index(ts+1e-4)
    print(fake_index)
    fullpath = Path(proto_dir, k, ".index")
    print("writing", fullpath)
    fullpath.write_text(fake_index)
