#!/usr/bin/python
__author__ = 'pbmanis'
import string
import getpass
from pathlib import Path
import subprocess

"""
krun.py is for submitting jobs to killdevil.unc.edu
The program:
1. logs in through ssh
2. changes directories to the right directory (PyNeuronLibrary/Cortex_STDP in this case)
3. uploads the source file (for example, stdp_test_parallel.py)
4. submits the job using bsub and requesting an appropriate number of processors
4. patiently waits for the results to become available.
5. downloads the result file

Note you must be on campus or using a VPN to access the machine this way
"""
import paramiko
import sys

upload = True

def line_buffered(f):
    line_buf = ""
    while not f.channel.exit_status_ready():
        line_buf += str(f.read(1))
        if line_buf.endswith('\n'):
            yield line_buf
            line_buf = ''

def get_data(directory):
    remotedir = f'/Users/pbmanis/Desktop/Python/mrk-nf107/NF107Ai32_Het/{directory:s}/'
    localdir  = f'/Users/pbmanis/Desktop/Python/mrk-nf107/NF107Ai32_Het/{directory:s}/'
    rsync_cmd = f"rsync -avh -e ssh pbmanis@lytle.med.unc.edu:{remotedir:s} {localdir:s}"
    cp = subprocess.run(rsync_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # print(cp.stderr)
    if cp.stderr != b'':
        for l in cp.stderr:
            print(l[:-1])
    so = cp.stdout.split(b'\n')
    for l in so:
        print('  ', str(l[:-1]))
    
    
print (len(sys.argv))
print (sys.argv[0])
if len(sys.argv) > 1:
    machine = sys.argv[1]
else:
    machine = 'Lytle'

# system selection:
sysChoice = {'Lytle': {'uname': 'pbmanis', 'dir': '/Users/pbmanis/Desktop/Python/mrk-nf107', 'addr': '152.19.86.111'},
             # 'Tule': {'uname': 'pbmanis', 'dir': '/Users/pbmanis/Desktop/Python/PyNeuronLibrary/Cortex-STDP', 'addr': '152.19.86.116'},
            }

if machine not in sysChoice.keys():
    print ('Machine %s not recognized' % machine)
    exit()

if upload:
    # mypwd = getpass.getpass("Password for %s: " % machine)
    mypwd = 'lbm$jh1$'
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.RejectPolicy)
    print( 'connecting to: ', sysChoice[machine]['addr'])
    print ('as user: ', sysChoice[machine]['uname'])
    conn = ssh.connect(sysChoice[machine]['addr'], username=sysChoice[machine]['uname'], password=mypwd)  # hate putting pwd in file like this...
    if conn is False:
        print ('Connection failed')
        exit()

    remote_basedir = sysChoice[machine]['dir'] + '/'
    cdcmd = 'cd ' + remote_basedir
    # Note tht the exe command is a "single session", so all commands that will be done need to be concatenated
    # using semicolons as if you were doing it from a command line on a terminal.
    # cd will not "hold"
    #stdin, stdout, stderr = ssh.exec_command('cd ~/PyNeuronLibrary/Cortex-STDP; ls -la')
    #for l in stdout.readlines():
    #    print l,

    print ('Remote requsted dir : ', remote_basedir)
    print ('open ftp:')
    ftp = ssh.open_sftp()
    ftp.chdir(sysChoice[machine]['dir'])  # however this works for the sftp
    print( ftp.getcwd())
    ftp.put('nf107/set_expt_paths.py', 'nf107/set_expt_paths.py')  # update the source file
    ftp.put('NF107Ai32_Het/NF107Ai32_Het_maps.xlsx', 'NF107Ai32_Het/NF107Ai32_Het_maps.xlsx')  # update the source file
    ftp.put('scripts/nf107.sh', 'scripts/nf107.sh')  # update the source file
    ftp.close()

    print('Script, data table and set_expt_paths all transferred')



# stdin, stdout, stderr = ssh.exec_command(cdcmd + 'scripts' + '; ' + 'ls -lat')
# print(stderr.readlines())
# for l in sorted(stdout.readlines()):
#     print(l[:-1], )
     # print (l,)

# starting shell script
    ssh.get_pty()
    ssh.invoke_shell()
    stdin, stdout, stderr = ssh.exec_command(remote_basedir + 'scripts/' + 'nf107.sh')
    print(stderr.readlines())
    # for l in line_buffered(stdout.readlines()):
    #     print (l)

    lerr = stderr.readlines()
    if len(lerr) > 0:
        print('Error: ')
        print(lerr)
    else:
        for l in stdout.readlines():
            print( l,)
    print('completed command')
    ssh.close()
    
print('back to remote_run')


# build rsync command
get_data('pyramidal')


ssh.close()
