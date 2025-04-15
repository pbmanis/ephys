import tables
import pandas as pd 
import h5py

# Small hdf5 file to test reader on - from our data

file = "ephys/tools/data/map14_alltraces.h5"
with pd.HDFStore(file, "r") as store:
    print(f"Successfully retrived HDF5 keys: from {file:s}\n {store.keys()!s}")

