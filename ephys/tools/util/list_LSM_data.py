from pathlib import Path
import h5py
import codecs
import tifffile as TF
import xml.etree.ElementTree as ET  # needed to parse ome_metadata
import pandas as pd



def get_ims_header(filename):
    f = h5py.File(filename, 'r')
    # image = h5py.H5Gopen(mFileID, "DataSetInfo")
    
    # print(image)
    """
    The datasetinfo keys are: ['Channel 0', 'Channel 1', 'Channel 2', 'Image', 'Log', 'TimeInfo']
    """
    print("\nChannel Info")
    for i in range(4):
        if f"Channel {i:d}" not in f['DataSetInfo'].keys():
            continue
        print(f"\nChannel {i:d} Info:")
        a = list(f['DataSetInfo'][f"Channel {i:d}"].attrs.items())
        # print(a)
        for (n, v) in a:
            name = n
            value = r''.join(c.decode('UTF-8') for c in v)
            print(f"{name:s} = {value:s}")
    a = f['DataSetInfo']['Image'].attrs
    print("\nImage Info: ")
    for (n, v) in a.items():
        name = n
        value = r''.join(c.decode('UTF-8') for c in v)
        print(f"{name:s} = {value:s}")
    
    print("\nLog Info: ")
    a = f['DataSetInfo']['Log'].attrs
    for (n, v) in a.items():
        name = n
        value = r''.join(c.decode('UTF-8') for c in v)
        print(f"{name:s} = {value:s}")

    print("\nTime Info: ")
    a = f['DataSetInfo']['TimeInfo'].attrs
    for (n, v) in a.items():
        name = n
        value = r''.join(c.decode('UTF-8') for c in v)
        print(f"{name:s} = {value:s}")
    print()
    

def check_duplicates(topdir):
    dirs = sorted(list(topdir.glob("*")))
    allfiles = []
    for i, d in enumerate(dirs):
        if str(d.name).startswith("."):
            continue
        # if i > 0:
        #     break
        print(str(d))
        alldata = sorted(list(d.glob("**/*")))
        allfiles.extend([dx for dx in alldata if not dx.is_dir()])


    for i, f in enumerate(allfiles):
        if str(f.name).startswith("."):
            continue
        indexes = [indx+i+1 for indx, fl in enumerate(allfiles[i+1:]) if (str(f.name) == str(fl.name))]
        if len(indexes) > 0:
            print("\nDuplicate files: ")
            print(f"    {str(allfiles[i]):s}")
            stop = False
            for k in range(len(indexes)):
                if k > 1:
                    print(f"    and {len(indexes)-2:d} more files")
                    stop = True
                else:
                    print(f"    {str(allfiles[indexes[k]]):s}")
            if stop:
                break

def get_tiffs(topdir):
    """
    Find the directories containing tif files in this directory
    returns a dict containing all the directories that have more than
    5 tiff files, along with the number of tif files.
    """
    dirs = sorted(list(topdir.glob("*")))
    allfiles = {}
    for i, d in enumerate(dirs):
        if str(d.name).startswith("."):
            continue
        # if i > 0:
        #     break
        alldata = sorted(list(d.glob("*.tif")))
        # allfiles.extend([dx for dx in alldata if not dx.is_dir() and dx.suffix == ".tif"])
        if len(alldata) > 5:
            allfiles[d] = len(alldata)
    # print(f"Found: {len(allfiles):d} tif files")
    return allfiles
    
def get_info(fn):
    tf = TF.TiffReader(fn)
    # print(dir(tf))
    wavelen = {}
    dets = {}
    cmap = {}
    pixels = {}
    print("-"*80)
    if tf.ome_metadata is not None:
        # print("OME: \n")
        root = ET.fromstring(tf.ome_metadata)
        for child in root:
            # print("   child: ", child, child.tag, child.attrib)
            for children in child: #'UltraII_LaserNameUNDERSCORE'):
                # print("      Item in child: ", children, " tag: ", children.tag, " attribute: ", children.attrib)
                bracket = children.tag.find("}")
                tag = children.tag[bracket+1:]
                if tag == "Pixels":
                    pixels = children.attrib
                for gc in children:
                    # if gc is None:
                    #     continue
                    bracket = gc.tag.find("}")
                    tag = gc.tag[bracket+1:]
                    if tag.startswith('UltraII_Wavelength'):
                        # print(f"      Item {str(gc):s} \n      tag: {tag:s}\n      {str(gc.attrib['Value']):s}")
                        wavelen[tag] = int(gc.attrib['Value'])
                    elif tag.startswith('UltraII_Filter'):
                        dets[tag] = gc.attrib['Value']
                    elif tag.startswith('ColorMap'):
                        if 'ColorMap' in gc.attrib.keys():
                            cmap[tag] = gc.attrib['ColorMap']
                        else:
                            cmap[tag] = gc.attrib['Value']
                    # elif tag.startswith("Pixels"):
                   #      print("Pix: ", tag)
                   #      pixels[tag] = gc.attrib['Pixels']
                        
    return wavelen, dets, cmap, pixels

def get_tags(fn):
    tags = []
    tf = TF.TiffReader(fn)
    # print(dir(tf))
    if tf.ome_metadata is not None:
        # print("OME: \n")
        root = ET.fromstring(tf.ome_metadata)
        for child in root:
            # print("   child: ", child, child.tag, child.attrib)
            for children in child: #'UltraII_LaserNameUNDERSCORE'):
                # print("      Item in child: ", children, children.tag, children.attrib)
                for gc in children:
                    # if gc is None:
                    #     continue
                    bracket = gc.tag.find("}")
                    tag = gc.tag[bracket+1:]
                    tags.append(tag)

    return tags

def get_imaris(topdir):
    """
    Find the ims files in this directory
    """
    dirs = sorted(list(topdir.glob("*")))
    allfiles = []
    for i, d in enumerate(dirs):
        if str(d.name).startswith("."):
            continue
        # if i > 0:
        #     break
        alldata = sorted(list(d.glob("**/*")))
        allfiles.extend([dx for dx in alldata if not dx.is_dir() and dx.suffix == ".ims"])
    # print(f"Found {len(allfiles):d} files")
    return allfiles

def generate_one_cochlea(cochlea, df_list, topdir):
    topdir = Path(topdir, cochlea)
    all_ims = get_imaris(topdir)
    print('cochlea: ', cochlea)
    pixels =  {"SizeX": 0, "SizeY": 0, "SizeZ": 0, 
            "PhysicalSizeX": 0, 
            "PhysicalSizeY": 0, 
            "PhysicalSizeZ": 0,
            }
    print(all_ims)
    for fn in all_ims:
        print("\n\nreading ims file: ", fn)
        # get_ims_header(fn)
    all_tiff_dirs = get_tiffs(topdir)
    for ims in all_ims:
        df_list.append({"Cochlea": cochlea, "OMEs": str(Path(*ims.parts[6:])), "tiffdir": "", "tiffs": 0,
            "Channels": 0,
            "Channel0": "", "Channel1": "", "Channel2": "", "Channel3": "",
            "SizeX": 0, "SizeY": 0, "SizeZ": 0, 
            "PhysicalSizeX": 0., "PhysicalSizeY": 0., "PhysicalSizeZ": 0.})
    for n, v in all_tiff_dirs.items():
        xpix = 0
        ypix = 0
        zpix = 0
        xsize = 0
        ysize = 0
        zsize = 0
        tiffs = n.glob("*.tif")  # read 
        # if i > 2:
        #     continue
        for i, fn in enumerate(tiffs):
            tags = get_tags(fn)
            # print(tags)
            if i >= 1:
                break
            else:
                print(fn)
                wv, det, cmap, pixelsf = get_info(fn)
                if len(pixelsf) > 0:
                    pixels = pixelsf
        nchannels = int(v)/int(pixels['SizeZ'])
        if nchannels != int(nchannels):
            nchannels = 'undetermined'
        df_list.append({'Cochlea': cochlea, "OMEs": "", "tiffdir": n.name, "tiffs": v, 
            "Channels": nchannels,
            "Channel0": "", "Channel1": "", "Channel2": "", "Channel3": "",
            "SizeX": int(pixels['SizeX']), "SizeY": int(pixels['SizeY']), "SizeZ": int(pixels['SizeZ']), 
            "PhysicalSizeX": float(pixels["PhysicalSizeX"]), 
            "PhysicalSizeY": float(pixels["PhysicalSizeY"]), 
            "PhysicalSizeZ": float(pixels["PhysicalSizeZ"]),
            }
        )

    
    return df_list

def generate_db(cochleas, topdir):
    df_list = []  # build a list of dicts for the dataframe
    for coch in cochleas:
        cn = f"Cochlea-{coch:2s}"
        print("cn: ", cn)
        df_list = generate_one_cochlea(cn, df_list, topdir)
                    # print(item.find("UltraII_LaserNameUNDERSCORE").text)
            # tf.ome_metadata.keys()
        # elif tf.imagej_metadata is not None:
        #     print("Imagej: ", tf.imagej_metadata)
    newdf = pd.DataFrame.from_records(df_list)
    print(newdf.head())
    return newdf
    

if __name__ == "__main__":
    # topdir0 = Path("/Volumes/T7 Touch/Ropp LSM data/Cochlea-M1")
    topdir = Path("/Volumes/Pegasus_002/Ropp_data/LSM_Cochleas")

    cochleas = ['A1', 'A3', 'B1', 'B2', 'C1', 'C2', 
    'C4', 'C7', 'D1', 'D7', 'E1', 'E2', 'E5', 'E6',
    'F1', 'F2', 'F5', 'F6', 'K1']
    # cochleas = ['A1']
    df = generate_db(cochleas, topdir=topdir)
    df.to_pickle('Cochleas.pkl')
    with pd.ExcelWriter('Cochleas.xlsx') as writer:
        df.to_excel(writer, sheet_name = "FileManifest")
        for i, column in enumerate(df):
            column_width = max(df[column].astype(str).map(len).max(), len(column))
            writer.sheets['FileManifest'].set_column(first_col=i+1, last_col=i+1, width=column_width) # column_dimensions[str(column.title())].width = column_width

    