#!/usr/bin/env python
from __future__ import print_function
# encoding: utf-8
"""
readdatac.py

A module to read DATAC files. These are binary files written by
a C progam used between (1985-1999).
Original C code by Daniel Bertrand and Paul Manis .
Matlab reader version by Patrick Kanold (1996).

Matlab: 4/30/99 P. Manis. Modified to handle modes 0 and 1 (early file structures),
 as well as get the timing information.
Shuld read legacy files, and files with just one channel in modes 2 and 3

Python version: Converted 5/13/2014 PBManis.
Tested on old data
Created by Paul Manis on 2014-05-14.
Copyright (c) 2014 Paul B. Manis, Ph.D.. All rights reserved.
Updated and bugs fixed, 4-2018. Added "GetClamps" to provide the
necessary structure and information used by acq4 data analysis programs.

"""

import sys
import os
import platform
import unittest
import struct
import string
import binascii
import re
from collections import OrderedDict
import numpy as np
import glob
import matplotlib.pylab as mpl
from pyqtgraph.metaarray import MetaArray

"""
This is what the datac data structure looks like (in C):
MATRIX.H:
        /*
        	Matrix definition and Union
          Used for data reading, transfer and display preparation
        	added smatn/matn 8/92 for shorter file structure stuff.

        */
        #if !defined MATRIX_H

        #define LEFTSPACE 88		/* to make header reach 256 bytes in bmatn */
        			/* LEFTSPACE is size of an INTEGER (not char/byte) array
        				LEFTSPACE should be reduced by:
        				1 for each integer added to the header
        				2 for each floating number
        				if you add a character, you must add 2 chars, and reduce LEFTSPACE
                     by one
        			*/
        struct data_header {	/* define just for space needs */
        		char mode;/* type of acquisition: 2 for this case */
        		char ftime;/* minimum clock time */
        		int record;	/* current record number (redundant) */
        		int nchan;		/* number of channels in this sample */						
        		float rate;		/* specified sample rate */
        		float gain[8];	/* voltage gains, nominal on each of 8 channels */
        		float lpf[8];		/* current low-pass filtering setting on each of 8 channels */
        		int slowsamp;	/* slow sample flag */
        		long ztime;	/* the time since acquisition or file opening (milliseconds) */
        		int dummy[LEFTSPACE];	/* dummy space for more stuff */
        }d_hdr;

        #define BSIZE (32768+sizeof(struct data_header))

        #define ALLBUFF union all_buffer
          ALLBUFF{/*	buffer area for one data set */
        	struct smats {/* short form of the buffer */
        		char mode;/* type of acquisition : 0 for this case */
        		char ftime;/* minimum clock time */
        		char dums[4];
        //		int mats[2730][3];/* data array */
        	} bmats;
        	struct smatl {/* long form of the buffer */
        		char mode;/* type of acquisition : 1 for this case */
        		char ftime;/* minimum clock time */
        		char duml[4];
        //		int matl[1023][4];/* data array */
        	} bmatl;
        	struct smatn {/* long form of the buffer */
        		char mode;/* type of acquisition: 2 for this case */
        		char ftime;/* minimum clock time */
        		int record;	/* current record number (redundant) */
        		int nchan;		/* number of channels in this sample */						
        		float rate;		/* specified sample rate */
        		float gain[8];	/* voltage gains, nominal on each of 8 channels */
        		float lpf[8];		/* current low-pass filtering setting on each of 8 channels */
        		int slowsamp;	/* slow sample flag */
        		long ztime;
        		int dummy[LEFTSPACE];	/* dummy space for more stuff - make header 256 bytes long */
        //		int matn[16384];/* data array - interleaved 2 x 8192 max points */
        	} bmatn;
        		char buff[sizeof(struct data_header)];/* input buffer */
        	struct s_dir{
        			int nsamp;	/* number of points in data  */
        			char dumm[27];/* empty location */
        			char comm[80];/* text position */
            }p_dir;/* personal directory in the file */
        };

        /* note this just defines the buffers: allocation it taken care of elsewhere */
        #define DATA_BUFF union data_buffer
        DATA_BUFF {
        		int mats[2730][3];/* data array */
        		int matl[1023][4];/* data array */
        		int matn[4096];/* data array - interleaved 2 x 2048 max points */
        		char buff[2730*3*sizeof(int)];
        };
		
        struct dac_buff {
        		int dac_array[4096];
        };


        #define MATRIX_H 1	/* define our inclusion here */

        #endif

"""

    
class ReadDatac:
    def __init__(self, datamode='CC'):
        """
        Initialize a DATAC file object, used to read data and notefiles
        dfile - the variables that are in dfile structure, as created by
        the various versions of DATAC, are defined here
        """
        
        nrl = 1  # 
        self.num_records = nrl
        RL = np.zeros((nrl,1))  # a list of records
        self.fullfile = ''  # full filename with path
        self.filename = ''  # filename, with extension
        self.path = ''  # separated path
        self.ext = ''  # extension (often this is the initials of the person doing the recording)
        self.records_in_file = 0  # number of records stored in the file
        self.nr_points = 0  # number of points per record
        self.comment = ' '  # the "comment" field (string)
        self.mode = -1  # file mode (-1 is placeholder). Mode varies between versions of the file structure
        self.dmode = datamode  # default data collection mode if not otherwise defined...
        self.rate = np.zeros(nrl)  # there is a separate sample rate for every record!
        self.ftime = 0.0  # time variable/sampling information
        self.record = []  # the records
        self.nr_channel = 0  # number of channels collected (usually 2 or 3; 3 is max)
        self.junctionpot = 0.0  # junction potential in mV for corrections (nominally 0 mV)
        self.vgain = 1.0  # voltage gain
        self.wgain = 1.0  # channel 3 gain
        self.igain = 1.0  # current gain
        self.gain = np.zeros((nrl,8))  # gain for each channel
        self.low_pass = np.zeros((nrl,8))  # low pass filter settings for each channel
        self.slow = np.zeros((nrl,1))  # slow mode flag (for slow acquisition rates on old AD board)
        self.ztime = np.zeros((nrl,1))  # time since file opened
        self.refvg = 1.0
        self.refig = 1.0
        self.refwg = 1.0
        self.frec = 1.0  # first record of data that is pulled
        self.lrec = 1.0  # last record
        self.steps = 0  # nubmer of steps in the stimulus
        self.fileheaderlen = 128  # fixed length file header, 128 bytes
        self.recordheaderlen = 256  # fixed length of header for each record, 256 bytes
        self.datawidth = 2  # standard width of int variable, in bytes
        self.data = None  # the data that is returned
        self.fid = None  # the file id of an open file
        self.amplifier_igain = 10.  # for primary channel
        self.amplifier_vgain = 10.
        self.amplifier_icmd_scale = 10.
        self.amplifier_vcmd_scale = 133.9

    def set_amplfier_gains(self, vgain=None, igain=None):
        if vgain is not None:
            self.amplifier_vgain = vgain
        if igain is not None:
            self.amplifier_igain = igain

    def openfile(self, filename):
        """
        open the requested file for reading
        :params: filename - the full name of the file including the path
        :returns: error  - integer error flag:
             1: requested file was not found or could not be opened
             3: requested file was a matlab file; should be handled in matlab
             0: normal return, file is open and fid is the open file handle.
        """
        
        if not os.path.exists(filename):
            print('ReadDatac:readfile File %s not found' % filename)
            self.err = 1
            return self.err

        # decide what kind of file we are reading
        p, ext = os.path.splitext(filename)
        path, fname = os.path.split(filename)
        if ext in  ['.mat']: # if its a matlab file, use different routine to parse through here
           self.err = 3
           return self.err

        # otherwise, we proceed normally.

        self.fid = open(filename,'rb')  # open for readonly, as binary file
        if self.fid is None:
            print('Unable to open file: %s ',filename)
            self.err == 1
            return self.err

        self.fullfile = filename  # store file info
        self.filename = fname
        self.path = path
        self.ext = ext
        self.err = 0
        return self.err

    def read_datac_header(self, filename, indent=0):
        """
        Read the file header information. Each file in the DATAC format has a header
        that contains basic information about the file, including a comment field,
        the file mode flag, and the number of channels and points per channel. 
        Depending on the mode, other information may also be set.
        The file header is 128 bytes long
        A mode of 0 is 2 channels, 1 is 3 channels. 
        Note that early DATAC files held data in integer format, as received from the data acquistion board.
        Later files (mode = 9) stored the data in a floating point format.

        :params: filename: name of the file to open and read the header.
        :return: nothing (self.err is set)
        """
        data=[]
        err=1
        endian='ieee-le' # data from little endian (i.e. PC's) => integers flipped, floats in ieee

        err = self.openfile(filename)
        if err > 0:
            self.err = err
            return       
        # before reading, check the length of the file
        self.fid.seek(0, 2)
        flen = self.fid.tell()
        if flen < 129:  # must have at least header and one record
            self.fid.close()
            self.err = 2
            return self.err
        
        #read file header
        self.fid.seek(0)
        c_np = self.fid.read(2)
        self.nr_points = struct.unpack('H', c_np)[0]
        c_nothing = self.fid.read(27)
        c_comment = self.fid.read(99)
        cp = struct.unpack('99c', c_comment)
        self.comment = ''

        for c in cp:
            if binascii.b2a_hex(c) == '00':  # terminator
                break
            cs = str(c)
            if cs in string.printable:
                self.comment += c
 
        self.comment.replace('_', '\_')
        self.fid.seek(self.fileheaderlen, 0)
        c_mode = self.fid.read(1)
        self.mode = struct.unpack('b', c_mode)[0]
        c_time = self.fid.read(1)
        self.ftime = struct.unpack('b', c_time)[0]
        
        self.datawidth = 2  # default
        if(self.mode >= 2):
             c_record = self.fid.read(2)
             c_nrch = self.fid.read(2)

             self.record = struct.unpack('H', c_record)[0] #should be 1 for first record
             self.nr_channel = struct.unpack('H', c_nrch)[0]
        else:  # mode is 0 or 1
            self.record = 1
            self.nr_channel = self.mode + 2  # mode 0 is 2 channels; mode 1 is 3 channels
            self.nbytes = self.nr_points * 8   # original form 
            self.nheader = 6       # original header was 6 bytes long */
            self.ndbytes = self.nbytes - self.nheader

        if self.mode == 9:  # floating point format
            self.datawidth = 4
            self.vgain = 0.1

        # check the file length now
        self.records_in_file = int(np.floor((flen-self.fileheaderlen)/
                        ((self.nr_points*self.nr_channel*self.datawidth)+self.recordheaderlen)))
        if self.records_in_file == 0:
            print('No records found in file!')
            self.err = 2
            self.fid.close()
            return self.err
        self.print_datac_file_header()
        
    def print_datac_file_header(self, indent=4):
        """
        print basic information from the open file.
        """
        
        print('\nFile Header:')
        spc = ' '*indent
        print('%sFile %s   Mode:         %5d       ftime: %6d   Channels:%6d  Data format: %d bytes' %
             (spc, self.filename, self.mode, self.ftime, self.nr_channel, self.datawidth))
        print('%s  First Record: %5d     Num Records: %6d' % (spc, self.record, self.records_in_file))
        print('{0:s}  Comment: {1:s}'.format(spc, self.comment))
        self.err = 0
        return self.err

    def read_note_file(self):
        """
        Read the note file associated with the current open datafile
        The note information (at least, some of it) is parsed and stored in 
        the self.expInfo dictionary, numbered by the acquisition blocks
        """
        nf, ext = os.path.splitext(self.fullfile)
        self.notefile = nf + '.NOT'
        
        self.expInfo = OrderedDict()
        with open(self.notefile) as f:
            content = f.readlines()
        f.close()
        block = 0  # block counter
        next_protocol = None
        current_protocol = None
        acqflag = False

        # define some regular expressions to parse:
        record_re = re.compile(r'(?<=\[R\:)\d+')  # search for record number in string
        get_re = re.compile(r'(?<=\>g STM\:)\s+(\w+\.stm)')  # search re or get and do commands
        do_re = re.compile(r'(?<=\>do STM\:)\s+(\w+\.stm)')

        for lineno, c in enumerate(content):
            if lineno < 6:
                continue # skip over the header
            if c[0:3] == 'PRI':
                continue  # older ones, some lines have sequence information at start of line.
            time_stamp = c[0:8]  # get the time stamp for each line
            rm = record_re.search(c)
            if rm is not None:
                record = int(rm.group(0))  # read the record information
            else:
                continue # some lines in some versions can't be parsed this way; just skip.

            # protocols:
            # get and do commands retrieve protocols. They may appear on the line
            # before the sequenxe command that uses them has been given, so we must wait
            # for that sequence command to set the protocol up.
            #
            acq_cmd = False  # set on any valid command that starts acquisition
            pr_get = get_re.search(c)
            if pr_get is not None:
                next_protocol = pr_get.group(0)  # upcoming loaded protocol
                #acqflag = False  # terminates acqusition
                if current_protocol is None:
                    current_protocol = next_protocol
            pr_do = do_re.search(c)
            if pr_do is not None:
                acq_cmd = True
                next_protocol = pr_do.group(0)  # upcoming loaded protocol
                if current_protocol is None:
                    current_protocol = next_protocol
                # look ahead for the end record for the current protocol
                nl = content[lineno+1]
                rm = record_re.search(nl)
                if rm is not None:
                    record = int(rm.group(0))  # read the record information
                    
            if not acqflag:
                firstrec = record
            seq = c.find('>seq')  # check for an seq command from a protocol that is loaded
            acl = c.find('>acl')  # file close command
            if seq != -1 or acl != -1:
                acq_cmd = True
            if acq_cmd or (lineno == len(content)-1):
                if acqflag is False:  # sequence started, 
                    acqflag = True   # and keep processing/updating
                else:
                    self.expInfo[block] = {'R': firstrec, 'Rend': record-1, 'time': time_stamp, 'Protocol': current_protocol.strip()}
                    current_protocol = next_protocol # update the loaded protocol
                    firstrec = record
                    block += 1
        
    def print_datac_note(self, indent):
        """
        Print the note file information in a table format.
        """
        spc = ' ' * indent
        k = self.expInfo.keys()
        if len(k) <= 2:
            return
        print('\n{:s}  Block  Protocol   Recs   Time'.format(spc))
        for i in k:
            print('{:s}  {:3d}  {:9s} {:3d}-{:3d} {:8s} '.format(spc, i, self.expInfo[i]['Protocol'], 
                self.expInfo[i]['R'], self.expInfo[i]['Rend'],
                self.expInfo[i]['time']))

        
    def readrecords(self, record_list):
        """
        Read the selected data records from the currently open file. 
        :params: record_list - the list of records (does not need to be contiguous)
        Returns: Error (self.err)
            Errors:
            0: success
            3: last record is past end of data in file
        
        Units:
        record : int
        Channels : usually 2 or 3
        rate : sample rate per channel in microseconds (aggregate rate for timing is nr_channels*)rate
        slow : not used
        ztime : time since file was opened, in msec (I think - that's what the notes say)
        gain : gain for each channel. Probably only correct for V
        low_pass : low pass filter setting (kHz)
            
        """ 
        if len(record_list) == 0: # catch when we just read the header for information
            self.err = 0; # this is ok...
            return self.err

        if record_list[0] == 0:
            record_list = [x+1 for x in record_list]
        maxrecord_list = np.max(record_list)
        if maxrecord_list > self.records_in_file:
            print('Last record (%d) greater than length of file (%d recs in file)' % (maxrecord_list,self.records_in_file))
            self.err = 3
            return self.err
        block_head = 0; # flag for block header reading ONLY.
        #read data sets according to the records in the record_list vector
        self.num_records = len(record_list)
        self.records_in_request = self.num_records
        self.frec = record_list[0]
        self.lrec = record_list[-1]
        
        if maxrecord_list == 0: # special mode just to read ZTIME and other header information from WHOLE FILE
            record_list = range(1,self.records_in_file)
            block_head = 1 # block header read
            self.data = None
            self.rate = None
        else:
            self.data = np.zeros((self.num_records, self.nr_channel, self.nr_points)); 
            self.rate = np.zeros(self.num_records); # separate for every record!

        self.record = [None]*self.records_in_request
        self.channels = [None]*self.records_in_request
        self.rate = [None]*self.records_in_request
        self.slow = [None]*self.records_in_request
        self.ztime = [None]*self.records_in_request
        self.gain = np.ones((self.records_in_request, 8))
        self.low_pass = np.ones((self.records_in_request, 8))   
 
        #print('record_list: ', record_list)
        
        for i, rec in enumerate(record_list):
            # print 'data mode : %d' % self.mode
            if self.mode == 9:
                self.datawidth = 4  # stored as floats
            else:
                self.datawidth = 2 # 16-bit signed intss
            if self.mode >= 2:  # compute offset to data start
                offset = self.fileheaderlen
                offset += (rec-1) * ((self.nr_points * self.datawidth * self.nr_channel) + self.recordheaderlen)

            if self.mode == 3:
                self.fid.seek(offset, 0)      # go to start of the data
                c_mode = self.fid.read(1)
                self.c_mode = struct.unpack('B', c_mode)[0]
                c_ftime = self.fid.read(1)
                self.ftime = struct.unpack('B', c_ftime)[0]
                c_rec = self.fid.read(2)
                self.record[i] = struct.unpack('h', c_rec)[0]
                c_chan = self.fid.read(2)
                self.channels[i] = struct.unpack('H', c_chan)[0]  # fread(fid,1,'int16')
                c_rate = self.fid.read(4)
                self.rate[i] = struct.unpack('1f', c_rate)[0] / self.channels[i] # fread(fid,1,'float32')
                c_gain = self.fid.read(8*4)
                self.gain[i,:] = struct.unpack('8f', c_gain)  # fread(fid,8,'float32')
                c_lpf = self.fid.read(8*4)
                self.low_pass[i,:] = struct.unpack('8f', c_lpf)  #fread(fid,8,'float32')
                c_slow = self.fid.read(2)
                self.slow[i] = struct.unpack('H', c_slow)[0]  # fread(fid,1,'int16')
                c_ztime = self.fid.read(4)
                self.ztime[i] = struct.unpack('I', c_ztime)[0]  # fread(fid,1,'long')
                # now read the data
                offset = self.fileheaderlen+(rec-1)*self.datawidth*self.nr_points*self.nr_channel+rec*self.recordheaderlen         
                self.fid.seek(offset, 0) #skip earlier records
                if block_head == 0: # we don't actually read the data...
                    if self.mode != 9:
                        c_data = self.fid.read(self.nr_channel*self.nr_points*self.datawidth) 
                        data_in = struct.unpack('%dh' % (self.nr_channel*self.nr_points), c_data) 
                    else:  # floating point format
                        c_data = self.fid.read(self.nr_channel*self.nr_points*self.datawidth)  
                        data_in = struct.unpack('%dh' % (self.nr_channel*self.nr_points), c_data)
                    
                    skip = self.nr_channel # set the skip counter
                    # read the first entry
                    self.data[i,0,:] = data_in[0::2]     # self.nr_points - 1 ??? #voltage
                    if skip > 1:  # get second channel
                        self.data[i,1,:] = data_in[1::2]   #current
                    if skip > 2:  # get third channel ('w')
                        pass

            if self.mode <= 2:
                offset = self.fileheaderlen+(rec-1)*self.ndbytes

                self.fid.seek(offset, 0)      # go to start of the data
                # read record header
                c_mode = self.fid.read(1)
                self.c_mode = struct.unpack('b', c_mode)[0]
                c_time = self.fid.read(1)  # reads the mode byte, but we don't need it.
                self.ftime = struct.unpack('b', c_time)[0]
                self.gain = np.ones((self.num_records,8)) # set the gains all to 1
                offset = self.fileheaderlen+(rec-1)*self.ndbytes+rec*self.nheader   # position it correctly...
                self.fid.seek(offset, 0)  # skip earlier records
                if block_head == 0: # read the data, not just the block header..
                    c_data_in = self.fid.read(self.ndbytes*self.datawidth)  # access the data itself
                    try:
                        data_in = struct.unpack('%dh' % int(self.ndbytes), c_data_in)
                    except:
                        print ('Failed to read record %d' % (rec))
                        print ('self.ndbytes: ', self.ndbytes)
                        print ('len cdata: ', len(c_data_in))
                        print ('nr points: ', self.nr_points)
                        break
                        
                    if self.mode == 0: # 3 channels of information
                        self.data[i,0,:] = data_in[1:self.nr_points*3:self.nr_channel+1] 
                        self.data[i,1,:] = data_in[2:self.nr_points*3:self.nr_channel+1] 
                        self.rate[i] = self.oldtime(data_in[0:self.nr_points*3:3])
                    else:
                        self.data[i,0,:] = data_in[1::self.nr_channel+1]
                        self.data[i,1,:] = data_in[2::self.nr_channel+1]
                        self.rate[i] = self.oldtime(data_in[0::self.nr_channel+1])
        
        if self.mode not in [0, 1, 2, 3, 9]:
                raise ValueError("Cannot process data with file mode = %d " % self.mode)
        if self.num_records == 0:  # we sometimes override these with the ctl structure...
            self.refvgain = self.gain(1,1)
            self.refigain = self.gain(1,2)
            self.refwgain = self.gain(1,3)
        
        # Offset and scale data:

        if self.dmode == 'CC':  # 
            mainch = 0
            dacoffset = 1024.
            cmdch = 1
            maingain = 1e-3*10./(self.amplifier_vgain*self.gain[0, mainch+1])
            cmdgain = 1e-9/(self.amplifier_icmd_scale*self.gain[0, mainch+1])
        else:
            dacoffset = 2047.
            mainch = 1
            cmdch = 0
            maingain = -1e-12/(self.amplifier_igain*self.gain[0, mainch+1])
            cmdgain = 1./(self.amplifier_vcmd_scale*self.gain[0, mainch+1])
        for i in range(self.data.shape[0]):
            self.data[i,mainch,:] = (self.data[i,mainch,:]-dacoffset)*maingain
            self.data[i,cmdch,:]  = (self.data[i,cmdch,:]-dacoffset)*cmdgain

        self.err = 0
        return self.err

    def close(self):
        """
        close the current open data file
        """
        self.fid.close()

    def oldtime(self, tbuf):
        """
        compute mean rate from the old timey (mode 0 and 1) modes, associated with 
        the old switching amplifier from Prof. Daniel Bertrand, CMU Geneva.
        """
        CTE = 65536
        FTIME = 0.00061
        tx = 0
        dt = FTIME
        tdt = dt*10
        t = tbuf[0]
        ta = dt * (CTE-t)
        if ta < -tdt:
            tx = tx + CTE
            ta = dt * (tx-t)
        rate = ta
        #print('Rate: %12.5f' % rate)
        return(rate)


    def plotcurrentrecs(self, fn, title=None, fig=None, sp=None):
        sp = {}
        if fig is None:
            fig = mpl.figure(fn)
            sp[0] = mpl.subplot(211)
            sp[1] = mpl.subplot(212)
        if title is not None:
            fig.suptitle(title)
        ch = 0

        if self.data is None:
            print('no data...')
            return
        for i in range(self.data.shape[0]):
            if self.rate[i] == 0.:
                continue
            tb = np.arange(0., self.nr_points/self.rate[i], 1./self.rate[i])
 
            sp[0].plot(tb, self.data[i,ch,:], 'k-') # -2047)*0.1339
            sp[1].plot(tb, self.data[i,ch+1,:], 'r-')


class GetClamps():
    
    def __init__(self, datac, path=''):
        self.datac = datac
        self.path = path

    def getClampData(self, block, verbose=False, tstart_tdur = [0.01, 0.100]):
        """
        Translates fields as best as we can from the original DATAC structure
        create a Clamp structure for use in SpikeAnalysis and RMTauAnalysis.
        Fills in the fields that are returned by PatchEPhys getClamps:
        clampInfo['dirs]
        clampInfo['missingData']
        self.time_base
        self.values
        self.traceStartTimes = np.zeros(0)
        self.sequence
        self.clampValues (sequence)
        self.nclamp = len(self.clmapVlaues
        self.repc
        self.nrepc
        self.data_mode
        self.model_mode = False
        self.command_scale_factor
        self.command_units
        self.devicesUsed
        self.clampDevices
        self.holding
        self.clampState
        self.sample_interval
        self.RSeriesUncomp
        self.amplifeirSettings['WCCompValid', 'WCEmabled', 'CompEnabled', 'WCSeriesResistance']
        self.cmd_wave
        self.commandLevels (np.array(self.values))
        self.traces = MetaArray(traces, info=info)
        self.tstart
        self.tdur
        self.tend
        self.spikecount = np.zeros(len...) if in vcmode.
        
        Info from an example data file:
        [{'name': 'Channel', 'cols': [{'units': 'A', 'name': 'Command'}, {'units': 'V', 'name': 'primary'}, {'units': 'A', 'name': 'secondary'}]},
        {'units': 's', 'values': array([ 0.00000000e+00, 2.50000000e-05, 5.00000000e-05, ..., 6.99925000e-01, 6.99950000e-01, 6.99975000e-01]),
        'name': 'Time'}, {'ClampState': {'primaryGain': 10.0, 'ClampParams': {'OutputZeroEnable': 0, 'PipetteOffset': 0.05197399854660034,
        'Holding': -1.525747063413352e-11, 'PrimarySignalHPF': 0.0, 'BridgeBalResist': 20757020.0, 'PrimarySignalLPF': 20000.0, 'RsCompBandwidth':
        8.413395979806202e-42, 'WholeCellCompResist': 8.413395979806202e-42, 'WholeCellCompEnable': 6004, 'LeakSubResist': 8.413395979806202e-42,
        'HoldingEnable': 1, 'FastCompTau': 8.413395979806202e-42, 'SlowCompCap': 8.413395979806202e-42, 'WholeCellCompCap': 8.413395979806202e-42,
        'LeakSubEnable': 6004, 'NeutralizationCap': 1.9578947837994853e-12, 'BridgeBalEnable': 1, 'RsCompCorrection': 8.413395979806202e-42,
        'NeutralizationEnable': 1, 'RsCompEnable': 6004, 'OutputZeroAmplitude': -0.0009990156395360827, 'FastCompCap': 8.413395979806202e-42,
        'SlowCompTau': 8.413395979806202e-42}, 'secondarySignal': 'Command Current', 'secondaryGain': 1.0, 'secondaryScaleFactor': 2e-09,
        'primarySignal': 'Membrane Potential', 'extCmdScale': 4e-10, 'mode': 'IC', 'holding': 0.0, 'primaryUnits': 'V', 'LPFCutoff': 20000.0,
        'secondaryUnits': 'A', 'primaryScaleFactor': 0.1, 'membraneCapacitance': 0.0}, 'Protocol': {'recordState': True, 'secondary': None,
        'primary': None, 'mode': 'IC'}, 'DAQ': {'command': {'numPts': 28000, 'rate': 40000.0, 'type': 'ao', 'startTime': 1296241556.7347913},
        'primary': {'numPts': 28000, 'rate': 40000.0, 'type': 'ai', 'startTime': 1296241556.7347913}, 'secondary': {'numPts': 28000, 'rate':
        40000.0, 'type': 'ai', 'startTime': 1296241556.7347913}}, 'startTime': 1296241556.7347913}]

        )
        """
        if self.datac.data is None:
            raise ValueError('No data has been read from the file %s' % self.datac.fullfile)
        protocol = ''

        self.sample_interval = self.datac.rate[0]*1e-6  # express in seconds
        self.traces = np.array(self.datac.data)
        self.datac.data.shape
        points = self.datac.nr_points
        nchannels = self.datac.nr_channel
        recs = self.datac.record
        dt = 1e-3*nchannels / self.datac.rate[0]  # make assumption that rate is constant in a block
        self.time_base = 1e-3*np.arange(0., self.datac.nr_points/self.datac.rate[0], 1./self.datac.rate[0]) # in seconds

        if self.datac.dmode == 'CC':  # use first channel
            mainch = 0
            cmdch = 1
        else:  # assumption is swapped - for this data, that means voltage clamp mode.
            mainch = 1
            cmdch = 0

        cmds = self.traces[:,cmdch,:]
        self.tstart = tstart_tdur[0]  # could be pulled from protocol/stimulus information
        self.tdur = tstart_tdur[1]
        self.tend = self.tstart + self.tdur
        t0 = int(self.tstart/dt)
        t1 = int(self.tend/dt)
        self.cmd_wave = np.squeeze(self.traces[:, cmdch, :])
        if cmds.shape[0] > 1:
            self.values = np.nanmean(self.cmd_wave[:, t0:t1], axis=1)  # express values in amps
        else:
            self.values = np.zeros_like(self.traces.shape[1:2])
        self.commandLevels = self.values        
        
        info = [{'units': 'A', 'values': self.values, 'name': 'Command'},
                    {'name': 'Time', 'units': 's', 'values': self.time_base},
                    {'ClampState':  # note that many of these values are just defaults and cannot be relied upon
                            {'primaryGain': self.datac.gain, 'ClampParams': 
                                {'OutputZeroEnable': 0, 'PipetteOffset': 0.0,
                                'Holding': 0, 'PrimarySignalHPF': 0.0, 'BridgeBalResist': 0.0, 
                                'PrimarySignalLPF': 20000.0, 'RsCompBandwidth': 0.0, 
                                'WholeCellCompResist': 0.0, 'WholeCellCompEnable': 6004, 'LeakSubResist': 0.0,
                                'HoldingEnable': 1, 'FastCompTau': 0.0, 'SlowCompCap': 0.0, 
                                'WholeCellCompCap': 0.,
                                'LeakSubEnable': 6004, 'NeutralizationCap': 0.,
                                'BridgeBalEnable': 0, 'RsCompCorrection': 0.0,
                                'NeutralizationEnable': 1, 'RsCompEnable': 6004,
                                'OutputZeroAmplitude': 0., 'FastCompCap': 0.,
                                'SlowCompTau': 0.0}, 'secondarySignal': 
                                'Command Current', 'secondaryGain': 1.0,
                                'secondaryScaleFactor': 2e-09,
                                'primarySignal': 'Membrane Potential', 'extCmdScale': 4e-10,
                                'mode': self.datac.dmode, 'holding': 0.0, 'primaryUnits': 'V', 
                                'LPFCutoff': self.datac.low_pass,
                                'secondaryUnits': 'A', 'primaryScaleFactor': 0.1,
                                'membraneCapacitance': 0.0}, 
                            'Protocol': {'recordState': True, 'secondary': None,
                                    'primary': None, 'mode': 'IC'}, 
                            'DAQ': {'command': {'numPts': points, 'rate': self.sample_interval,
                                    'type': 'ao', 'startTime': 0.},
                            '       primary': {'numPts': points, 'rate': self.sample_interval,
                                    'type': 'ai', 'startTime': 0.}, 
                                    'secondary': {'numPts': points, 'rate': self.sample_interval,
                                    'type': 'ai', 'startTime': 0.}
                             },
                    'startTime': 0.}
                ]

        # filled, automatically with default values
        self.repc = 1
        self.nrepc = 1
        self.model_mode = False
        self.command_scale_factor = 1
        self.command_units = 'A'
        self.devicesUsed = None
        self.clampDevices = None
        self.holding = 0.
        self.amplfierSettings = {'WCCompValid': False, 'WCEnabled': False, 
                'CompEnabled': False, 'WCSeriesResistance': 0.}
        self.clampState = None
        self.RSeriesUncomp = 0.

        self.protoTimes = {'drugtestiv': [0.21, 0.51], 'ap-iv2': [0.01, 0.5]}
        if protocol in self.protoTimes:
            self.tstart = self.protoTimes[protocol][0]
            self.tdur = self.protoTimes[protocol][1]
            
        self.tend = self.tstart + self.tdur

        if self.traces.shape[0] > 1:
            # dependiung on the mode, select which channel goes to traces
            self.traces = self.traces[:,mainch,:]
        else:
            self.traces[0,mainch,:] = self.traces[0,mainch,:]

        self.traces = MetaArray(self.traces, info=info)
        self.spikecount = np.zeros(len(recs))
        self.rgnrmp = [0, 0.005]


# simple routines to access notes, plot files, etc.

def printNotes(filename, indent=0):
    """
    Print out the note file associated with this data file
    """
    
    dfile = ReadDatac()
    dfile.read_datac_header(filename, indent)
    if dfile.err == 2:
        return None
    dfile.read_note_file()
    dfile.print_datac_note(indent)
    return dfile


def plotonefile(filename, datamode='CC'):
    """
    Plot the traces from 2 channels from this data file
    should specify data mode ('CC' or 'VC')
    """
    
    dfile = ReadDatac(datamode =datamode)
    dfile.read_datac_header(filename)
    dfile.read_note_file()
    dfile.print_datac_note(3)

    if dfile.err == 0:
        k = dfile.expInfo.keys()
        for i in k:
            dfile.readrecords(range(dfile.expInfo[i]['R'],dfile.expInfo[i]['Rend']))

            cl = GetClamps(dfile)
            cl.getClampData(1, verbose=False)
            if dfile.err == 0:
                txt = dfile.expInfo[i]['Protocol'] + ' ' '[R:%d-%d] ' % (dfile.expInfo[i]['R'],dfile.expInfo[i]['Rend'])
                txt += dfile.expInfo[i]['time']
            #    dfile.plotcurrentrecs(i, title=txt)
            #mpl.show()
            mpl.figure()
            for j in range(cl.traces.shape[0]):
                mpl.plot(cl.time_base, cl.traces[j])
            mpl.show()
            if i > 3:
                exit(1)
    dfile.close()
    mpl.show()


def listAllFiles(datapath):
    """
    List all of the files in the datapath
    and their notefiles
    """
    n = 0
    nempty = 0
    for d in os.listdir(datapath):
        p = os.path.join(datapath, d)
        if not os.path.isdir(p):
            continue
        print ('\nDirectory: %s' % p)
        for fx in os.listdir(p): 
            (f, e) = os.path.splitext(fx)
            ff = os.path.join(p, fx)
            nempty += printNotes(ff, indent=3)
            n += 1
    print('\nFiles read: {:d}, Empty file accessed: {:d}'.format(n, nempty))


def example_dir_scan(searchstr, protocol='iv2.stm'):
    """
    Scan a directory and plot the traces from the first protocol
    that matches the search string
    """
    fns = glob.glob(searchstr)
    for fn in fns:
        df = printNotes(fn, indent=5)
        if df is None:
            continue
        for protn in df.expInfo.keys():
            if df.expInfo[protn]['Protocol'] == protocol:
                df.readrecords(range(df.expInfo[protn]['R'], df.expInfo[protn]['Rend']))
                df.plotcurrentrecs(fn)
                break  
    mpl.show()


if __name__ == '__main__':
    example_dir_scan(searchstr='/Users/pbmanis/Documents/data/HWF0001B/VCN/*.HWF')
    #plotonefile(os.path.join('/Users/pbmanis/Documents/data/HWF0001B/VCN', '11SEP96H.HWF'), datamode='CC')
    #plotonefile(os.path.join('/Users/pbmanis/Documents/data/HWF0001B/VCN', '26AUG96B.HWF'), datamode='CC')
    #plotonefile(os.path.join('/Users/pbmanis/Documents/data/datac', '07FEB96H.JSR'), datamode='VC')