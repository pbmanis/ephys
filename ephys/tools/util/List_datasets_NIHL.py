import os
import re
import time
import pandas as pd
from pathlib import Path
import sys
import socket
import paramiko  # for ssh login!
import getpass
import datetime
from threading import Thread
import textwrap as TW

"""
Get information from remote computer for acq4 datasets
"""

mypwds = {"experimenters": "", "Experimenters": "", "pbmanis": ""}

Rig4 = {
            "IP": "152.19.86.119",
        "name": "Rig4",
            "location": "B209 Marsico",
            "machine": "Intrex",
            "login": "Experimenters",
            "ssh": True,
            }

def getssh(riginfo):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # if host['login'] == 'pbmanis':
    #       continue
    host = Rig4
    sship = host['IP']
    if host["login"] not in list(mypwds.keys()):
        mypwds[host["login"]] = getpass.getpass(
            "{0:s}: Password for {1:s}: ".format(host["name"], host["login"])
        )
    print("  {0:>9s} : {1:16s} ".format(host["name"], sship), end=""),
    print("as {0:<16s}".format(host["login"])),
    # privatekeyfile = os.path.expanduser("~/.ssh/id_rsa")
    # mykey = paramiko.RSAKey.from_private_key_file(privatekeyfile)
    
    connectok = False
    try:
        ssh.connect(sship, username=host["login"], look_for_keys=True, timeout=5.0)
        print("\033[32m" + "        Passwordless connection succeded" + "\033[0m")
        connectok = True  
        return ssh
    except:
        return None

def connect(riginfo):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # if host['login'] == 'pbmanis':
    #       continue
    host = Rig4
    sship = host['IP']
    if host["login"] not in list(mypwds.keys()):
        mypwds[host["login"]] = getpass.getpass(
            "{0:s}: Password for {1:s}: ".format(host["name"], host["login"])
        )
    print("  {0:>9s} : {1:16s} ".format(host["name"], sship), end=""),
    print("as {0:<16s}".format(host["login"])),
    # privatekeyfile = os.path.expanduser("~/.ssh/id_rsa")
    # mykey = paramiko.RSAKey.from_private_key_file(privatekeyfile)

    connectok = False
    try:
        ssh.connect(sship, username=host["login"], look_for_keys=True, timeout=5.0)
        print("\033[32m" + "        Passwordless connection succeded" + "\033[0m")
        connectok = True
    except paramiko.AuthenticationException:
        print(
            "\033[93m"
            + "        User Authorization without password Failed for: "
            + host["name"]
            + ", "
            + host["login"]
            + "\033[0m"
        )
    except Exception as e:
        if "Name or service not known" in str(e):
            print(" 'Name or service not known: " + host["name"])
        elif "timed out" in str(e):
            print("\033[31m" + "     Connection Timed Out: " + host["name"] + "\033[0m")
        else:
            print(
                "\033[36m" + " Host is unknown: " + host["name"] + "\033[0m" + ",",
                str(e),
            )

    finally:
        ssh.close()
    return connectok

remotes_NF107_NIHL = [
           r"/cygdrive/d/Mike/NF107_ai32_NIHL/3d",
           r"/cygdrive/d/Mike/NF107_ai32_NIHL/2w",
           r"/cygdrive/g/data/Mike/NIHL_NF107_ai32/BNE_3d",
           r"/cygdrive/g/data/Mike/NIHL_NF107_ai32/BNE_5d",
           r"/cygdrive/g/data/Mike/NIHL_NF107_ai32",
          ]

remotes_VGAT_NIHL = [
   r"/cygdrive/d/Mike/VGAT/Blinded_2wk_NIHL",
    r"/cygdrive/f/data/Mike/VGAT_DCNmap",
   r"/cygdrive/g/data/Mike/NIHL_VGAT",
   r"/cygdrive/g/data/Mike/NIHL_VGAT/Parasagittal",
    
    ]

def pfiles(files, startdate=None, cellcount=False, r=None, ftp=None):
    nf = 0
    nc = 0
    for f in files:
        if f in [".index"] or f.endswith('txt') or f.startswith('BNE') or f.startswith("Parasagittal"):
            continue
        datestr = f[:10]
        try:
            datenum = pd.to_datetime(datestr, format='%Y.%m.%d')
        except:
            continue
#         print('datenum: ', datenum)
        if startdate is not None and datenum < startdate:
            continue
        desc_text = None
        note_text = None
        anim_text = None
        if r.find('DCNmap') > 0:  # some files are in the wrong directory
            d0 = datetime.datetime(2018, 8, 31)  # from this date, is a VGAT NIHL
            d1 = datetime.datetime(2018, 10, 26)  # to this date is a VGAT NIHL
            if datenum < d0 or datenum > d1:
                continue
        indexfile = Path(r, f, ".index")
        with ftp.open(str(indexfile), 'r') as fh:
            while True:
                txt = fh.readline().replace(r'\n', ' ').strip()
                if not txt:
                    break
                if txt.find('description:') >=0:
                    desc_text = txt
                if txt.find('notes:') >=0:
                    note_text = txt
                if txt.find('animal identifier:') >=0:
                    anim_text = txt

        print("\n")
        print("-"*70)
        print("    Day: ", f)
        if anim_text is None or len(anim_text) == 0:
            anim_text = "animal_identifier: None"
        print(TW.fill(anim_text, initial_indent=' '*8, subsequent_indent=' '*14))
        print()
        if desc_text is None or len(desc_text) == 0:
            desc_text = "description: None"
        print(TW.fill(desc_text, initial_indent=' '*8, subsequent_indent=' '*14))
        print()
        if note_text is None or len(note_text) == 0:
            note_text = "notes: None"
        print(TW.fill(note_text, initial_indent=' '*8, subsequent_indent=' '*14))
        print()
#         print('cellcount: ', cellcount)
        if cellcount:
            dirs = ftp.listdir(str(Path(r, f)))
#             print('dirs: ', dirs)
            for d in dirs:
                if d.startswith('slice_'):
                    cells = ftp.listdir(str(Path(r, f, d)))
#                     print('cells', cells)
                    for c in cells:
                        if c.startswith('cell_'):
                            prots = ftp.listdir(str(Path(r, f, d, c)))
                            if len(prots) > 4:
                                print("       ", d, c, '   nprots: ', len(prots))
                                nc += 1
                               
        nf += 1
    print(f'nfiles: {nf:d}')
    return nf, nc
    
def do_remotes(rig, remotes, startdate=None, cellcount=False):
    connectok = connect(rig)
    if connectok:
        ssh = getssh(rig)
        ftp = ssh.open_sftp()
        nf = 0
        ncells = 0
        for r in remotes:
            print('r: ', r)

            files = ftp.listdir(str(r))
            nfp, nc = pfiles(files, startdate, cellcount, r, ftp)
            nf += nfp
            ncells += nc

        ssh.close()
        print('Total directories: ', nf)
        print('Total putative cells: ', ncells)
        
        
# do_remotes(Rig4, remotes_NF107_NIHL,  cellcount=True, startdate=datetime.datetime(2020, 6, 15))
do_remotes(Rig4, remotes_VGAT_NIHL, cellcount=True)