""" Find and transfer .index files
"""
from pathlib import Path
import re
import paramiko
import stat

import pyqtgraph.configfile


# config = pyqtgraph.configfile.readConfigFile("config/logging/logging.cfg")
# source = Path(config['parent_directory'], config['day_directory'])

# config["dirs_with_index_files"] = list(config["dirs_with_index_files"].keys())
# for k in config.keys():
#     if k.endswith("_directory"):
#         config[k] = Path(config[k])
# print(config)


""" Approach: 
We want to upload the missing .index files to a remote system.
So we will make a list of the .index file directories on this system, for the folders
specified

"""

# find the .index files on this system, for the missing ones on the remote system.SystemError





def list_zip_files(dirs_with_index_files):
    # check that all of these have an associated .zip file

    zips = list(Path(config['local_dir']).glob("*.zip"))
    zips = [z for z in zips if not z.name.startswith(".")]
    for zipfile in zips:
        print(zipfile)
    print("# zip files: ", len(zips))

    print(f"{'=' * 40}\n")
    # check directories to be sure all zip files are unzipped.
    dirs = []
    for directory in zips:
        dirpath = Path(config['local_directory'], directory.stem)
        print("dirpath: ", dirpath)
        if not dirpath.exists():
            print(f"Directory {dirpath} does not exist. Unzip {directory} first.")
        else:
            dirs.append(dirpath)
    print("# directories: ", len(dirs))
    if len(dirs) != len(zips):
        print("Not all zip files have been unzipped. Please unzip all files before proceeding.")
        raise FileNotFoundError("Not all zip files have been unzipped. Please unzip all files before proceeding.")


def find_dotindex_files(dirs_with_index_files, local_dir, verbose=False):
    # find the .index files, store in dict with directory as key and list of .index files as value
    index_files = {}
    dirs = [Path(config['local_directory'], d+'_000') for d in dirs_with_index_files]
    for ndir, dir in enumerate(dirs):
        if ndir > 10:
            break
        if verbose:
            print(f"Checking directory {ndir+1}/{len(dirs)}: {dir}")
        index_file = list(dir.glob("**/*.index"))
        index_files[dir] = []
        if len(index_file) == 0:
            print(f"No .index file found in {dir}. Please check this directory.")
        else:
            # filter out ._ files... 
            index_file_list = [f for f in index_file if f.name.startswith(".index")]
            if verbose:
                print(f"{len(index_file_list):>4d} .index files in {dir}.")
            index_files[dir].extend(index_file_list)
            if verbose:
                print(f"     {index_files[dir][:10]}")

    print(f"Found {len(index_files)} directories with .index files")
    return index_files

def sftp_walk(sftp, remote_path):
    """Recursively yield (path, directories, files) like os.walk."""
    files = []
    folders = []
    
    for entry in sftp.listdir_attr(remote_path):
        if stat.S_ISDIR(entry.st_mode):
            folders.append(entry.filename)
        else:
            files.append(entry.filename)
            
    yield remote_path, folders, files
    
    for folder in folders:
        new_path = remote_path + "/" + folder
        yield from sftp_walk(sftp, new_path)



def check_remote_index_files(remote_path, daydir):
    # check for .index files on remote system in the specified directory
    # make ftp connection to destination program

    # 1. Initialize SSH Client
    client = paramiko.SSHClient()

    # 2. Set Host Key Policy 
    # Note: AutoAddPolicy is useful for testing but not recommended for production
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # 3. Connect to the server
    client.connect(hostname=config["remote_ip"], port=22, username=config["username"], password=config["password"])

    # 4. Open an SFTP session
    sftp = client.open_sftp()
    # find remote directory

    remote_path = Path(config["remote_directory"], daydir+"_000")

    print("With walk:", remote_path)
    for path, dirs, files in sftp_walk(sftp, str(remote_path)):
        for f in files:
            if f == ".index":
                print(f"{path}/{f}")

    sftp.close()
    client.close()
    print("all closed")

def write_index_files_to_remote(local_index_files, remote_path):
    # write the .index files to the remote system in the specified directory
    # make ftp connection to destination program

    # 1. Initialize SSH Client
    client = paramiko.SSHClient()

    # 2. Set Host Key Policy 
    # Note: AutoAddPolicy is useful for testing but not recommended for production
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # 3. Connect to the server
    client.connect(hostname=config["remote_ip"], port=22, username=config["username"], password=config["password"])

    # 4. Open an SFTP session
    sftp = client.open_sftp()

    for ndir, lif in enumerate(local_index_files):
        print("local index file: ", lif)
        # just get the day directory name
        direct = re.search(r"(\d{4}\.\d{2}\.\d{2}_000[^\n]+)$", str(lif)).group(1)
        print("Remote Directory: ", direct)
    
        remote_index_file = Path(remote_path, direct)
        if remote_index_file.name != ".index":
            remote_index_file = Path(remote_path, direct, ".index")
        print(f"Checking for .index file at remote path: {remote_index_file}")

        try:
            sftp.stat(str(remote_index_file))
            print("File exists at remote path: ", remote_index_file)
        except IOError:
            print("\nFile does not exist")  
            print("Putting put this : ")
            print("    ", lif)
            print("to here: ")
            print("    ",remote_index_file)
            print("\n")
            sftp.put(str(lif), str(remote_index_file))
            # exit()

    sftp.close()
    client.close()
    print("all closed")



if __name__ == "__main__":
    # list_zip_files(dirs_with_index_files)
    local_index_files = find_dotindex_files(config["dirs_with_index_files"], config["local_directory"], verbose=False)
    # for iday, daydir in enumerate(dirs_with_index_files):
    #     check_remote_index_files(remote_dir, daydir)
    #     if iday == 0:
    #         break
    for i, lcf in enumerate(local_index_files):
        if i > 10:
            break
        print(i, local_index_files[lcf][:5])
    # exit()
    # for iday, local_index in enumerate(local_index_files):
    #     write_index_files_to_remote(local_index_files[local_index], config["remote_directory"])
