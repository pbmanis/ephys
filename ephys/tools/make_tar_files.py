"""
This is a tool for handling storage of acq4 files. 
This tool makes a tar file for each cell in a (recursive) directory
The name of the file is the date.slice#.cell#.tar
The tar files are put in a directory "tarfiles" at the top level of the directory.
Note that the tar files created by this script are incomplete in that they
do not include the day and slice information. 
For a set of directories of related recordings, use "make_dayslice_tarfile.py"
to make the top level directories (days and slices), including their metadata
and included files before untaring the files made by this routine. 
The untared files from here should find their way to the correct folder
hierarchy.
"""

from pathlib import Path
import tarfile

def make_tar_from_cell(celldir):
    date = celldir.parent.parent.name
    if not str(date).startswith("20"):
        date = celldir.parent.name
        slice = "slice_00"
        cell = celldir.name
    else:
        slice = celldir.parent.name
        cell = celldir.name
    date = str(date).split("_")[0]
    slice = f"S{int(str(slice).split('_')[1]):02d}"
    cell = f"C{int(str(cell).split('_')[1]):02d}"
    tarname = f"{date:s}~{slice:s}~{cell:s}"
    return tarname

def dir_recurse(current_dir, dirtype="day", indent=0, dirs=[]):

    if indent == 0:
        current_dir = Path(current_dir)
    if dirtype == "day":
        sw = "20"
    elif dirtype == "cell":
        sw = "cell"
    else:
        exit()
    files = sorted(list(current_dir.glob("*")))
    alldatadirs = [f for f in files if f.is_dir() and str(f.name).startswith(sw)]
    sp = " " * indent
    for d in alldatadirs:
        # print(f"{sp:s}Data: {str(d.name):s}")
        dirs.append(d)
    allsubdirs = [f for f in files if f.is_dir() and not str(f.name).startswith(sw) and not str(f.name).startswith("0")]
    indent += 2
    sp = " " * indent
    for d in allsubdirs:
        # print(f"\n{sp:s}Subdir: {str(d.name):s}")
        indent, dirs = dir_recurse(d, dirtype=dirtype, indent = indent, dirs=dirs)
    indent -= 2
    if indent < 0:
        indent = 0
    return indent, dirs

def make_tarname(topdir):
    tarname = "DaySlice"
    tarfilename = str(Path(topdir, tarname)) + ".tar"
    print(tarfilename)
    return tarfilename

def get_tar_info(topdir, nmax=0):
    tarfiles = Path(topdir).glob("*.tar")
    for n, tarfile in enumerate(tarfiles):
        if n > nmax and nmax > 0:
            break
        get_tar_file_info(tarfile)

def get_tar_file_info(tarfilename, nmax=0):
    # print()
    # print("="*80)
    with tarfile.open(tarfilename, "r") as tar:
        tar.list(verbose=True)
    print()
    print("="*80)
    n = 0
    with tarfile.open(tarfilename, "r") as tar:
        if nmax > 0 and n > nmax:
            pass
        else:
            t = tar.getmembers()
            for m in t:
                print(f"{m.size:10d} |  {str(m.name):80s}")
        n += 1
    print("\nTotal: ", n)

def make_tar_files(topdir):
    indent, dirs =  dir_recurse(topdir, dirtype = "cell", dirs=[])
    for i, d in enumerate(dirs):
        tarname = make_tar_from_cell(d)
        print(str(tarname))
        tarfilename = str(Path(topdir, tarname)) + ".tar"
        with tarfile.open(tarfilename, "w") as tar:
            fo = Path(*d.parts[5:])
            tar.add(d, arcname=fo)




if __name__ == '__main__':
    topdir = "/Volumes/Pegasus_002/ManisLab_Data3/Kasten_Michael/Cerebellum"
    #make_tar_files(topdir)
    get_tar_info(topdir, nmax=0)