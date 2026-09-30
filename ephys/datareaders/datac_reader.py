from pathlib import Path
from typing import Union, List, Optional, LiteralString

"""
datac_reader.py

A module to read DATAC files. These are binary files written by
a C progam used between (1985-1999).
Original C code by Daniel Bertrand and Paul Manis.
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


import os
import struct
import string
import binascii
import datetime  # Claude added 2026-09-29: used by the datac_reader API class
import pprint    # Claude added 2026-09-29: used by the datac_reader API class
import re
import numpy as np
import matplotlib.pylab as mpl
import MetaArray
import pylibrary.plotting.plothelpers as PH
"""
This is what the datac data structure looks like (in C)::

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
    def __init__(self, datamode="CC"):
        """
        Initialize a DATAC file object, used to read data and notefiles
        dfile - the variables that are in dfile structure, as created by
        the various versions of DATAC, are defined here

        """

        nrl = 1  #
        self.num_records = nrl
        RL = np.zeros((nrl, 1))  # a list of records
        self.fullfile = ""  # full filename with path
        self.filename = ""  # filename, with extension
        self.path = ""  # separated path
        self.ext = ""  # extension (often this is the initials of the person doing the recording)
        self.records_in_file = 0  # number of records stored in the file
        self.nr_points = 0  # number of points per record
        self.comment = " "  # the "comment" field (string)
        self.mode = (
            -1
        )  # file mode (-1 is placeholder). Mode varies between versions of the file structure
        self.dmode = datamode  # default data collection mode if not otherwise defined...
        self.rate = np.zeros(nrl)  # there is a separate sample rate for every record!
        self.ftime = 0.0  # time variable/sampling information
        self.record = []  # the records
        self.nr_channel = 0  # number of channels collected (usually 2 or 3; 3 is max)
        self.junctionpot = 0.0  # junction potential in mV for corrections (nominally 0 mV)
        self.vgain = 1.0  # voltage gain
        self.wgain = 1.0  # channel 3 gain
        self.igain = 1.0  # current gain
        self.gain = np.zeros((nrl, 8))  # gain for each channel
        self.low_pass = np.zeros((nrl, 8))  # low pass filter settings for each channel
        self.slow = np.zeros(
            (nrl, 1)
        )  # slow mode flag (for slow acquisition rates on old AD board)
        self.ztime = np.zeros((nrl, 1))  # time since file opened
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
        self.amplifier_igain = 10.0  # for primary channel
        self.amplifier_vgain = 10.0
        self.amplifier_icmd_scale = 10.0  # for command channels
        self.amplifier_vcmd_scale = 133.9  # 2048 levels, 10 V 

    def set_amplifier_gains(self, vgain=None, igain=None):
        if vgain is not None:
            self.amplifier_vgain = vgain
        if igain is not None:
            self.amplifier_igain = igain

    def openfile(self, filename:Union[Path, str]):
        """
        Open the requested file for reading

        Parameters
        ----------
        filename - the full name of the file including the path

        Returns
        -------
        error  - integer error flag::

             error = 1 - requested file was not found or could not be opened
             error = 3 - requested file was a matlab file; should be handled in matlab
             error = 0 - normal return, file is open and fid is the open file handle.

        """

        filename = Path(filename)
        if not filename.is_file():
            print("ReadDatac:readfile File %s not found" % filename)
            self.err = 1
            return self.err

        # decide what kind of file we are reading
        p, ext = os.path.splitext(filename)
        path, fname = os.path.split(filename)
        if ext in [".mat"]:  # if its a matlab file, use different routine to parse through here
            self.err = 3
            return self.err

        # otherwise, we proceed normally.

        self.fid = open(filename, "rb")  # open for readonly, as binary file
        if self.fid is None:
            print("Unable to open file: %s ", filename)
            self.err == 1
            return self.err

        self.fullfile = filename  # store file info
        self.filename = fname
        self.path = path
        self.ext = ext
        self.err = 0
        return self.err

    def read_datac_header(self, filename:Union[Path, str], indent:int=0):
        """
        Read the file header information. Each file in the DATAC format has a header
        that contains basic information about the file, including a comment field,
        the file mode flag, and the number of channels and points per channel.
        Depending on the mode, other information may also be set.
        The file header is 128 bytes long
        A mode of 0 is 2 channels, 1 is 3 channels.
        Note that early DATAC files held data in integer format, as received from the data acquistion board.
        Later files (mode = 9) stored the data in a floating point format.

        Parameters
        ----------
         filename: name of the file to open and read the header.

        Return
        ------
        nothing (self.err is set)

        """
        data = []
        err = 1
        endian = (
            "ieee-le"  # data from little endian (i.e. PC's) => integers flipped, floats in ieee
        )
        filename=Path(filename)
        if not filename.is_file():
            print(f"ReadDatac:read_datac_header File {filename!s} not found")
            self.err = 1
            return self.err
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

        # read file header
        self.fid.seek(0)
        c_np = self.fid.read(2)
        self.nr_points = struct.unpack("H", c_np)[0]
        c_nothing = self.fid.read(27)
        c_comment = self.fid.read(99)
        cp = struct.unpack("99c", c_comment)
        self.comment = ""

        for c in cp:
            if binascii.b2a_hex(c) == "00":  # terminator
                break
            cs = str(c)
            if cs in string.printable:
                self.comment += c

        self.comment.replace(r"_", r"\_")
        self.fid.seek(self.fileheaderlen, 0)
        c_mode = self.fid.read(1)
        self.mode = struct.unpack("b", c_mode)[0]
        c_time = self.fid.read(1)
        self.ftime = struct.unpack("b", c_time)[0]

        self.datawidth = 2  # default
        if self.mode >= 2:
            c_record = self.fid.read(2)
            c_nrch = self.fid.read(2)

            self.record = struct.unpack("H", c_record)[0]  # should be 1 for first record
            self.nr_channel = struct.unpack("H", c_nrch)[0]
        else:  # mode is 0 or 1
            self.record = 1
            self.nr_channel = self.mode + 2  # mode 0 is 2 channels; mode 1 is 3 channels
            self.nbytes = self.nr_points * 8  # original form
            self.nheader = 6  # original header was 6 bytes long */
            self.ndbytes = self.nbytes - self.nheader

        if self.mode == 9:  # floating point format
            self.datawidth = 4
            self.vgain = 0.1

        # check the file length now
        # Claude fixed 2026-09-30: mode 0/1 records use the old layout whose stride
        # is ndbytes (datac2mat.m: offset = 128 + (rec-1)*ndbytes), NOT the
        # 256-byte-header layout. Using the wrong (larger) stride overcounted the
        # records (e.g. 7DEC90D: 176 vs the true ~91), causing reads of records
        # that do not exist ("Failed to read record ...").
        if self.mode < 2:
            self.records_in_file = int(np.floor((flen - self.fileheaderlen) / self.ndbytes))
        else:
            self.records_in_file = int(
                np.floor(
                    (flen - self.fileheaderlen)
                    / ((self.nr_points * self.nr_channel * self.datawidth) + self.recordheaderlen)
                )
            )
        if self.records_in_file == 0:
            print("No records found in file!")
            self.err = 2
            self.fid.close()
            return self.err
        self.print_datac_file_header()

    def print_datac_file_header(self, indent=4):
        """
        print basic information from the open file.
        """

        print("\nFile Header:")
        spc = " " * indent
        print(
            "%sFile %s   Mode:         %5d       ftime: %6d   Channels:%6d  Data format: %d bytes"
            % (spc, self.filename, self.mode, self.ftime, self.nr_channel, self.datawidth)
        )
        print(
            "%s  First Record: %5d     Num Records: %6d" % (spc, self.record, self.records_in_file)
        )
        print("{0:s}  Comment: {1:s}".format(spc, self.comment))
        self.err = 0
        return self.err

    def read_note_file(self):
        """
        Read the note file associated with the current open datafile
        The note information (at least, some of it) is parsed and stored in
        the self.expInfo dictionary, numbered by the acquisition blocks
        """
        nf, ext = os.path.splitext(self.fullfile)
        self.notefile = nf + ".NOT"
        self.expInfo = dict()
        if not Path(self.notefile).is_file():
            return None

        with open(self.notefile) as f:
            content = f.readlines()

        block = 0  # block counter
        next_protocol = None
        current_protocol = None
        acqflag = False

        # define some regular expressions to parse:
        record_re = re.compile(r"(?<=\[R\:)\d+")  # search for record number in string
        get_re = re.compile(r"(?<=\>g STM\:)\s+(\w+\.stm)")  # search re or get and do commands
        do_re = re.compile(r"(?<=\>do STM\:)\s+(\w+\.stm)")

        for lineno, c in enumerate(content):
            if lineno < 6:
                continue  # skip over the header
            if c[0:3] == "PRI":
                continue  # older ones, some lines have sequence information at start of line.
            time_stamp = c[0:8]  # get the time stamp for each line
            rm = record_re.search(c)
            if rm is not None:
                record = int(rm.group(0))  # read the record information
            else:
                continue  # some lines in some versions can't be parsed this way; just skip.

            # protocols:
            # get and do commands retrieve protocols. They may appear on the line
            # before the sequenxe command that uses them has been given, so we must wait
            # for that sequence command to set the protocol up.
            #
            acq_cmd = False  # set on any valid command that starts acquisition
            pr_get = get_re.search(c)
            if pr_get is not None:
                next_protocol = pr_get.group(0)  # upcoming loaded protocol
                # acqflag = False  # terminates acqusition
                if current_protocol is None:
                    current_protocol = next_protocol
            pr_do = do_re.search(c)
            if pr_do is not None:
                acq_cmd = True
                next_protocol = pr_do.group(0)  # upcoming loaded protocol
                if current_protocol is None:
                    current_protocol = next_protocol
                # look ahead for the end record for the current protocol
                nl = content[lineno + 1]
                rm = record_re.search(nl)
                if rm is not None:
                    record = int(rm.group(0))  # read the record information

            if not acqflag:
                firstrec = record
            seq = c.find(">seq")  # check for an seq command from a protocol that is loaded
            acl = c.find(">acl")  # file close command
            if seq != -1 or acl != -1:
                acq_cmd = True
            if acq_cmd or (lineno == len(content) - 1):
                if acqflag is False:  # sequence started,
                    acqflag = True  # and keep processing/updating
                else:
                    self.expInfo[block] = {
                        "R": firstrec,
                        "Rend": record - 1,
                        "time": time_stamp,
                        "Protocol": current_protocol.strip(),
                    }
                    current_protocol = next_protocol  # update the loaded protocol
                    firstrec = record
                    block += 1
        return block  # number of blocks

    def print_datac_note(self, indent):
        """
        Print the note file information in a table format.
        """
        spc = " " * indent
        k = self.expInfo.keys()
        if len(k) <= 2:
            return
        print(f"\n{spc:s}  Block  Protocol   Recs   Time")
        for i in k:
            print(
                f"{spc:s}  {i:3d}  {self.expInfo[i]["Protocol"]:9s} {self.expInfo[i]["R"]:3d}-{self.expInfo[i]["Rend"]:3d} {self.expInfo[i]["time"]:8s} "
            )

    def readrecords(self, record_list):
        """
        Read the selected data records from the currently open file.

        Parameters
        ----------
        record_list - the list of records (does not need to be contiguous)

        Returns
        -------
        Error (self.err)
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
        if len(record_list) == 0:  # catch when we just read the header for information
            self.err = 0
            # this is ok...
            return self.err

        if record_list[0] == 0:
            record_list = [x + 1 for x in record_list]
        maxrecord_list = np.max(record_list)
        if maxrecord_list > self.records_in_file:
            print(
                f"Last record ({maxrecord_list:d}) greater than length of file ({self.records_in_file:d} recs in file)"
            )
            self.err = 3
            return self.err
        block_head = 0
        # flag for block header reading ONLY.
        # read data sets according to the records in the record_list vector
        self.num_records = len(record_list)
        self.records_in_request = self.num_records
        self.frec = record_list[0]
        self.lrec = record_list[-1]

        if (
            maxrecord_list == 0
        ):  # special mode just to read ZTIME and other header information from WHOLE FILE
            record_list = range(1, self.records_in_file)
            block_head = 1  # block header read
            self.data = None
            self.rate = None
        else:
            self.data = np.zeros((self.num_records, self.nr_channel, self.nr_points))
            self.rate = np.zeros(self.num_records)
            # separate for every record!

        self.record = [None] * self.records_in_request
        self.channels = [None] * self.records_in_request
        self.rate = [None] * self.records_in_request
        self.slow = [None] * self.records_in_request
        self.ztime = [None] * self.records_in_request
        self.gain = np.ones((self.records_in_request, 8))
        self.low_pass = np.ones((self.records_in_request, 8))

        # print('record_list: ', record_list)

        for i, rec in enumerate(record_list):
            # print 'data mode : %d' % self.mode
            if self.mode == 9:
                self.datawidth = 4  # stored as floats
            else:
                self.datawidth = 2  # 16-bit signed intss
            if self.mode >= 2:  # compute offset to data start
                offset = self.fileheaderlen
                offset += (rec - 1) * (
                    (self.nr_points * self.datawidth * self.nr_channel) + self.recordheaderlen
                )

            # Claude fixed 2026-09-30: mode 2 uses the same 256-byte-header layout
            # as mode 3 (datac2mat.m gates the OLD layout on mode<2, not <=2). It
            # was falling through to the mode<=2 branch below and crashing on the
            # unset self.ndbytes. Handle mode 2 and 3 (and float mode 9) here.
            if self.mode >= 2:
                self.fid.seek(offset, 0)  # go to start of the data
                c_mode = self.fid.read(1)
                self.c_mode = struct.unpack("B", c_mode)[0]
                c_ftime = self.fid.read(1)
                self.ftime = struct.unpack("B", c_ftime)[0]
                c_rec = self.fid.read(2)
                self.record[i] = struct.unpack("h", c_rec)[0]
                c_chan = self.fid.read(2)
                self.channels[i] = struct.unpack("H", c_chan)[0]  # fread(fid,1,'int16')
                c_rate = self.fid.read(4)
                # self.rate[i] = struct.unpack("1f", c_rate)[0] / self.channels[i]
                # Claude fixed 2026-09-30: the reference datac2mat.m reads the
                # matrix.h header rate directly (dfile.rate(i)=fread(...,'float32'))
                # and does NOT divide by the channel count. The /channels here was
                # a divergence that mis-scaled the sample rate.
                self.rate[i] = struct.unpack("1f", c_rate)[0]  # fread(fid,1,'float32')
                c_gain = self.fid.read(8 * 4)
                self.gain[i, :] = struct.unpack("8f", c_gain)  # fread(fid,8,'float32')
                c_lpf = self.fid.read(8 * 4)
                self.low_pass[i, :] = struct.unpack("8f", c_lpf)  # fread(fid,8,'float32')
                c_slow = self.fid.read(2)
                self.slow[i] = struct.unpack("H", c_slow)[0]  # fread(fid,1,'int16')
                c_ztime = self.fid.read(4)
                self.ztime[i] = struct.unpack("I", c_ztime)[0]  # fread(fid,1,'long')
                # now read the data
                offset = (
                    self.fileheaderlen
                    + (rec - 1) * self.datawidth * self.nr_points * self.nr_channel
                    + rec * self.recordheaderlen
                )
                self.fid.seek(offset, 0)  # skip earlier records
                if block_head == 0:  # we don't actually read the data...
                    if self.mode != 9:
                        c_data = self.fid.read(self.nr_channel * self.nr_points * self.datawidth)
                        data_in = struct.unpack("%dh" % (self.nr_channel * self.nr_points), c_data)
                    else:  # floating point format
                        c_data = self.fid.read(self.nr_channel * self.nr_points * self.datawidth)
                        data_in = struct.unpack("%dh" % (self.nr_channel * self.nr_points), c_data)

                    skip = self.nr_channel  # set the skip counter
                    # read the first entry
                    self.data[i, 0, :] = data_in[0::2]  # self.nr_points - 1 ??? #voltage
                    if skip > 1:  # get second channel
                        self.data[i, 1, :] = data_in[1::2]  # current
                    if skip > 2:  # get third channel ('w')
                        pass

            if self.mode < 2:  # Claude fixed 2026-09-30: was "<= 2" (mode 2 is new layout)
                offset = self.fileheaderlen + (rec - 1) * self.ndbytes

                self.fid.seek(offset, 0)  # go to start of the data
                # read record header
                c_mode = self.fid.read(1)
                self.c_mode = struct.unpack("b", c_mode)[0]
                c_time = self.fid.read(1)  # reads the mode byte, but we don't need it.
                self.ftime = struct.unpack("b", c_time)[0]
                self.gain = np.ones((self.num_records, 8))  # set the gains all to 1
                offset = (
                    self.fileheaderlen + (rec - 1) * self.ndbytes + rec * self.nheader
                )  # position it correctly...
                self.fid.seek(offset, 0)  # skip earlier records
                if block_head == 0:  # read the data, not just the block header..
                    c_data_in = self.fid.read(
                        self.ndbytes * self.datawidth
                    )  # access the data itself
                    # Claude fixed 2026-09-30: the read requests ndbytes*2 bytes but
                    # a mode-0/1 record only strides ndbytes, so the LAST record's
                    # oversized read hits EOF. The USED data (nr_points*(nchan+1)
                    # values) is still present, so unpack however many complete
                    # int16 were actually read instead of failing the record.
                    navail = len(c_data_in) // 2
                    needed = self.nr_points * (self.nr_channel + 1)
                    if navail < needed:
                        print(f"Failed to read record {rec:d} (have {navail} of {needed} needed)")
                        break
                    data_in = struct.unpack("%dh" % navail, c_data_in[: navail * 2])

                    # Claude fixed 2026-09-30: clamp the de-interleaved channels to
                    # >= 0, matching datac2mat.m ("numbers should be positive -
                    # sometimes buggy in hardware": data(i,..)=max(d2(k,:),0)).
                    if self.mode == 0:  # 3 channels of information
                        self.data[i, 0, :] = np.maximum(data_in[1 : self.nr_points * 3 : self.nr_channel + 1], 0)
                        self.data[i, 1, :] = np.maximum(data_in[2 : self.nr_points * 3 : self.nr_channel + 1], 0)
                        self.rate[i] = self.oldtime(data_in[0 : self.nr_points * 3 : 3])
                    else:
                        self.data[i, 0, :] = np.maximum(data_in[1 :: self.nr_channel + 1], 0)
                        self.data[i, 1, :] = np.maximum(data_in[2 :: self.nr_channel + 1], 0)
                        self.rate[i] = self.oldtime(data_in[0 :: self.nr_channel + 1])

        if self.mode not in [0, 1, 2, 3, 9]:
            raise ValueError(f"Cannot process data with file mode = {self.mode:d}")
        if self.num_records == 0:  # we sometimes override these with the ctl structure...
            self.refvgain = self.gain(1, 1)
            self.refigain = self.gain(1, 2)
            self.refwgain = self.gain(1, 3)

        # Offset and scale data.
        # Claude fixed 2026-09-30: replaced the ad-hoc maingain/cmdgain constants
        # with the reference scaledat.m scaling (mat_datac/source/scaledat.m),
        # which uses the matrix.h header gains. Physical channels are fixed:
        # ch0 = voltage, ch1 = current (getClampData selects primary/command by
        # mode). Output is SI (V, A). refvg = gain[0,0]*10 (the 10 is the nominal
        # 10x recording gain), refig = gain[0,1]; vgain/igain are the extra
        # amplifier factors (1 by default).
        refvg = (self.gain[0, 0] * 10.0) or 10.0
        refig = self.gain[0, 1] or 1.0
        if self.mode in (0, 1, 2):
            # Manis/Bertrand board: the stored value is the LOWER 12 BITS (upper
            # bits are logic flags), so mask with 0xfff, subtract the 2048 zero,
            # and scale by MB_CAL. (Masking is what the old constants missed, which
            # is why the mode-0 current sat at a ~36 nA offset.)
            MASK = 0xFFF
            ZERO = 2048.0
            MB_CAL = 2.441406  # mV per count (+/-5 V span, 12-bit)
            vc = np.bitwise_and(np.floor(self.data[:, 0, :]).astype(np.int64), MASK) - ZERO
            ic = np.bitwise_and(np.floor(self.data[:, 1, :]).astype(np.int64), MASK) - ZERO
            # scaledat.m yields mV (voltage) and pA (current) here -> convert to SI
            self.data[:, 0, :] = (vc / refvg * MB_CAL / self.vgain + self.junctionpot) * 1e-3
            self.data[:, 1, :] = (ic / refig * MB_CAL / self.igain) * 1e-12
        else:
            # mode 3/9 (Digidata 1200): role-based scaling, validated on 01DEC95F
            # VC (command voltage -63 mV, current -1..+2 nA). scaledat.m's mode-3
            # formula depends on the header gain[0,0], which reads 0 for this file,
            # so keep the constants that give the correct physical values here.
            if self.dmode == "CC":
                mainch, cmdch, dacoffset = 0, 1, 2047.0
                maingain = 1e-3 / (self.amplifier_vgain * self.gain[0, mainch + 1])
                cmdgain = 1e-10 / self.amplifier_icmd_scale
            else:
                mainch, cmdch, dacoffset = 1, 0, 2047.0
                maingain = -1e-12 / (self.amplifier_igain * self.gain[0, mainch + 1])
                cmdgain = 1e-3 / (self.amplifier_vcmd_scale * self.gain[0, mainch + 1])
            for i in range(self.data.shape[0]):
                self.data[i, mainch, :] = (self.data[i, mainch, :] - dacoffset) * maingain
                self.data[i, cmdch, :] = (self.data[i, cmdch, :] - dacoffset) * cmdgain

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
        tdt = dt * 10
        t = tbuf[0]
        ta = dt * (CTE - t)
        if ta < -tdt:
            tx = tx + CTE
            ta = dt * (tx - t)
        rate = ta
        # print('Rate: %12.5f' % rate)
        return rate

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
            print("no data...")
            return
        for i in range(self.data.shape[0]):
            if self.rate[i] == 0.0:
                continue
            tb = np.arange(0.0, self.nr_points / self.rate[i], 1.0 / self.rate[i])

            sp[0].plot(tb, self.data[i, ch, :], "k-")  # -2047)*0.1339
            sp[1].plot(tb, self.data[i, ch + 1, :], "r-")


class GetClamps:

    def __init__(self, datac, path=""):
        self.datac = datac
        self.path = path

    def getClampData(self, block, verbose=False, tstart_tdur=[0.01, 0.100]):
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
        self.amplifierSettings['WCCompValid', 'WCEmabled', 'CompEnabled', 'WCSeriesResistance']
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


        ***NOTE***
        Axoprobe 1A : 
        Current is actual electrode current, 10/H mV/nA where H is the headstage gain 
        Only headstage used was an HS2, gain = 0.1L (100 mV/nA)
        Voltage is 10*Vm 
        Acquisition used the Bertrand/Manis A/D board, which can be configured for either
        5V (+/- 5V, 10V span) or +/-10 V (20V span).
        Usually, the 5V span is used, so the voltage of the input is -500 to +500 mV. 
        """
        if self.datac.data is None:
            raise ValueError("No data has been read from the file %s" % self.datac.fullfile)
        protocol = ""

        self.sample_interval = self.datac.rate[0] * 1e-6  # express in seconds
        self.traces = np.array(self.datac.data)
        self.datac.data.shape
        points = self.datac.nr_points
        nchannels = self.datac.nr_channel
        recs = self.datac.record
        dt = (
            1e-3 * nchannels / self.datac.rate[0]
        )  # make assumption that rate is constant in a block
        self.time_base = 1e-3 * np.arange(
            0.0, self.datac.nr_points / self.datac.rate[0], 1.0 / self.datac.rate[0]
        )  # in seconds

        if self.datac.dmode == "CC":  # use first channel
            mainch = 0
            cmdch = 1
        else:  # assumption is swapped - for this data, that means voltage clamp mode.
            mainch = 1
            cmdch = 0

        cmds = self.traces[:, cmdch, :]
        self.tstart = tstart_tdur[0]  # could be pulled from protocol/stimulus information
        self.tdur = tstart_tdur[1]
        self.tend = self.tstart + self.tdur
        t0 = int(self.tstart / dt)
        t1 = int(self.tend / dt)
        self.cmd_wave = np.squeeze(self.traces[:, cmdch, :])
        # Claude fixed 2026-09-29: keep cmd_wave 2-D for single-record blocks so
        # the per-record mean below (and the MetaArray) work (mirrors matdatac).
        if self.cmd_wave.ndim == 1:
            self.cmd_wave = self.cmd_wave[np.newaxis, :]
        # self.values = np.zeros_like(self.traces.shape[1:2])  # (buggy for 1 rec)
        # Claude fixed 2026-09-29: compute per-record command level for any count
        # (the original single-record branch produced a wrongly-shaped array).
        self.values = np.nanmean(self.cmd_wave[:, t0:t1], axis=1)  # amps
        self.commandLevels = self.values

        info = [
            {"units": "A", "values": self.values, "name": "Command"},
            {"name": "Time", "units": "s", "values": self.time_base},
            {
                "ClampState": {  # note that many of these values are just defaults and cannot be relied upon
                    "primaryGain": self.datac.gain,
                    "ClampParams": {
                        "OutputZeroEnable": 0,
                        "PipetteOffset": 0.0,
                        "Holding": 0,
                        "PrimarySignalHPF": 0.0,
                        "BridgeBalResist": 0.0,
                        "PrimarySignalLPF": 20000.0,
                        "RsCompBandwidth": 0.0,
                        "WholeCellCompResist": 0.0,
                        "WholeCellCompEnable": 6004,
                        "LeakSubResist": 0.0,
                        "HoldingEnable": 1,
                        "FastCompTau": 0.0,
                        "SlowCompCap": 0.0,
                        "WholeCellCompCap": 0.0,
                        "LeakSubEnable": 6004,
                        "NeutralizationCap": 0.0,
                        "BridgeBalEnable": 0,
                        "RsCompCorrection": 0.0,
                        "NeutralizationEnable": 1,
                        "RsCompEnable": 6004,
                        "OutputZeroAmplitude": 0.0,
                        "FastCompCap": 0.0,
                        "SlowCompTau": 0.0,
                    },
                    "secondarySignal": "Command Current",
                    "secondaryGain": 0.1,
                    "secondaryScaleFactor": 2e-10, # 2 nA/V for H = 0.1
                    "primarySignal": "Membrane Potential",
                    "extCmdScale": 4e-10,   # 20H nA/V or 2 nA/V for H = 0.1
                    "mode": self.datac.dmode,
                    "holding": 0.0,
                    "primaryUnits": "V",
                    "LPFCutoff": self.datac.low_pass,
                    "secondaryUnits": "A",
                    "primaryScaleFactor": 0.1,
                    "membraneCapacitance": 0.0,
                },
                "Protocol": {"recordState": True, "secondary": None, "primary": None, "mode": "IC"},
                "DAQ": {
                    "command": {
                        "numPts": points,
                        "rate": self.sample_interval,
                        "type": "ao",
                        "startTime": 0.0,
                    },
                    "       primary": {
                        "numPts": points,
                        "rate": self.sample_interval,
                        "type": "ai",
                        "startTime": 0.0,
                    },
                    "secondary": {
                        "numPts": points,
                        "rate": self.sample_interval,
                        "type": "ai",
                        "startTime": 0.0,
                    },
                },
                "startTime": 0.0,
            },
        ]

        # filled, automatically with default values
        self.repc = 1
        self.nrepc = 1
        self.model_mode = False
        self.command_scale_factor = 1
        self.command_units = "A"
        self.devicesUsed = None
        self.clampDevices = None
        self.holding = 0.0
        self.amplfierSettings = {
            "WCCompValid": False,
            "WCEnabled": False,
            "CompEnabled": False,
            "WCSeriesResistance": 0.0,
        }
        self.clampState = None
        self.RSeriesUncomp = 0.0

        self.protoTimes = {"drugtestiv": [0.21, 0.51], "ap-iv2": [0.01, 0.5]}
        if protocol in self.protoTimes:
            self.tstart = self.protoTimes[protocol][0]
            self.tdur = self.protoTimes[protocol][1]

        self.tend = self.tstart + self.tdur

        # select which channel goes to traces (depends on clamp mode)
        # Claude fixed 2026-09-29: the original 'else' branch was a no-op that left
        # traces 3-D for single-record blocks, so the MetaArray build raised
        # "values array for axis 1 has shape (2048,) but should be (2,)". Slicing
        # the main channel works for any record count (shape (nrec, npts)).
        self.traces = self.traces[:, mainch, :]

        self.traces = MetaArray.MetaArray(self.traces, info=info)
        self.spikecount = np.zeros(len(recs))
        self.rgnrmp = [0, 0.005]


# ===========================================================================
# Claude added 2026-09-29: datac_reader -- an acq4_reader-API-compatible reader
# for the raw DATAC binary format.  It is deliberately SEPARATE from the
# low-level ReadDatac / GetClamps classes above (which do the binary and .NOT
# parsing); this class only *adapts* them to the interface used by
# data_summary.py / the ephys_analysis pipeline and by get_reader(), matching
# what was done for ephus/abf/matdatac.
#
# Model: one DATAC file (e.g. 13JUN89L.PBM) is a CELL.  Its protocols (BLOCKS)
# come from the optional sidecar .NOT file (record ranges + protocol names, via
# ReadDatac.read_note_file -> expInfo).  When no .NOT exists, the whole file is
# a single protocol ("000_all") whose sweeps are all records.  Protocol paths
# are "<file>/<NNN>_<proto>" and per-sweep paths "<file>/<NNN>_<proto>/<sss>".
#
# The clamp mode (CC/VC) is NOT in the DATAC header; it comes from the
# configurable default ("datamode", default "CC"), overridable per instance or
# inferred when a block's protocol name contains "vc".
# ===========================================================================
class datac_reader:
    DEFAULTS = {
        "datamode": "CC", "tstart": 0.01, "tdur": 0.100, "LPF": 10000.0,
        # Per-channel sample rate (Hz). The old DATAC files' stored timing is
        # unreliably decoded (mode-0 oldtime / mode-3 rate/channels), so use this
        # configured value (10 kHz/channel = 20 kHz aggregate for 2 channels).
        "sample_rate_hz": 10000.0,
        # Extra voltage-path gain for the old mode-0 switching-amp recordings
        # (scaledat.m assumes vgain=1; these have an additional ~2x). Applied to
        # the membrane-voltage channel only for mode 0. Configurable per dataset.
        "vgain_mode0": 2.0,
    }

    def __init__(self, pathtoprotocol=None, dataname=None, defaults=None):
        self.protocol = None
        self.protocol_path = None
        self._file = None
        self._block = None
        self._readers = {}          # filepath (str) -> ReadDatac (header+notes read)
        self._txt_cache = {}        # filepath (str) -> parsed .TXT rows (or [])
        self._not_cache = {}        # filepath (str) -> parsed .NOT dict (or None)
        self._current_params = None  # .TXT param row for the active block (or None)
        self._range = None          # explicit record-range override (a, b) or None
        self.last_protocol = None
        self.last_dirs = []
        self.error_info = None
        self.importantFlag = False
        self.mode = None
        self.lb = "\n"
        self.pre_process_filters = {"LPF": None, "Notch": []}
        self.defaults = dict(self.DEFAULTS)
        if defaults:
            self.defaults.update(defaults)
        self.dataname = "MultiClamp1.ma"
        self.shortdname = "MultiClamp1"
        self.clampInfo = {"dirs": [], "missingData": []}

        maxclamps = 4
        clamps = []
        self.clampdevices = []
        for nc in range(maxclamps):
            cno = nc + 1
            clamps.extend([("Clamp%d" % cno, "Pulse_amplitude"),
                           ("MultiClamp%d" % cno, "Pulse_amplitude")])
            self.clampdevices.extend(["Clamp%d" % cno, "MultiClamp%d" % cno])
        clamps.extend([("AxoPatch200", "Pulse_amplitude"), ("AxoProbe", "Pulse_amplitude")])
        self.clampdevices.extend(["AxoPatch200", "AxoProbe"])
        self.clamps = clamps
        self.datac_device = "MultiClamp1"

        if dataname is not None:
            self.setDataName(dataname)
        if pathtoprotocol is not None:
            self.setProtocol(pathtoprotocol)

    # -- configuration ------------------------------------------------------
    def setImportant(self, flag=False):
        self.importantFlag = flag

    def set_pre_process(self, LPF=None, Notch=None):
        self.pre_process_filters["LPF"] = LPF
        self.pre_process_filters["Notch"] = Notch

    def set_defaults(self, **kwargs):
        self.defaults.update(kwargs)

    def setDataName(self, dataname):
        self.dataname = dataname
        self.shortdname = str(Path(dataname).stem)

    # -- path / block resolution -------------------------------------------
    @staticmethod
    def _item_from_token(token):
        head = str(token).split("_", 1)[0]
        try:
            return int(head)
        except ValueError:
            return None

    def _resolve(self, path):
        """Resolve a path into (datac_file, block, sweep, explicit_block).

        DATAC files have arbitrary extensions, so the datac file is found as the
        deepest ancestor of ``path`` that is an existing file; any remaining
        components encode the block (``NNN_<proto>``) and sweep (``SSS``).
        """
        p = Path(path)
        if p.is_file():
            return p, None, None, False
        cur = p
        tokens = []
        while cur != cur.parent:
            if cur.is_file():
                break
            tokens.insert(0, cur.name)
            cur = cur.parent
        if not cur.is_file():
            return None, None, None, False
        block = sweep = None
        explicit = False
        if len(tokens) >= 1:
            block = self._item_from_token(tokens[0])
            explicit = block is not None
        if len(tokens) >= 2:
            try:
                sweep = int(tokens[1])
            except ValueError:
                sweep = None
        return cur, block, sweep, explicit

    def _open(self, filepath):
        """Open (and cache) a ReadDatac with header + .NOT notes read, or None."""
        if filepath is None:
            return None
        key = str(Path(filepath))
        if key not in self._readers:
            d = ReadDatac(datamode=str(self.defaults.get("datamode", "CC")))
            d.read_datac_header(filepath)
            if getattr(d, "err", 1) not in (0, None):
                self.error_info = f"ReadDatac header error {getattr(d,'err',None)} for {key}"
                return None
            # mode-0 switching-amp recordings carry an extra ~2x voltage-path gain
            # (scaledat.m assumes vgain=1); apply the configured value so the
            # membrane potential is scaled correctly. See DEFAULTS["vgain_mode0"].
            if getattr(d, "mode", None) == 0:
                d.vgain = float(self.defaults.get("vgain_mode0", 2.0))
            try:
                d.read_note_file()
            except Exception:  # noqa: BLE001 - notes are ancillary/optional
                d.expInfo = {}
            self._readers[key] = d
        return self._readers[key]

    # -- note files (.NOT or .TXT) ------------------------------------------
    @staticmethod
    def _find_note_file(datac_file):
        """Return (path, kind) of the sidecar note file for a datac file, or
        (None, None).  A note file may be a ``.TXT`` (structured table) or a
        ``.NOT`` (free-form log); ``.TXT`` is preferred when both exist."""
        stem = Path(datac_file).with_suffix("")
        for ext, kind in ((".TXT", "txt"), (".txt", "txt"),
                          (".NOT", "not"), (".not", "not")):
            cand = stem.with_suffix(ext)
            if cand.is_file():
                return cand, kind
        return None, None

    @staticmethod
    def _parse_txt_notes(path):
        """Parse a DATAC ``.TXT`` sidecar: a tab-separated table with a header
        row (SLTN STIM RB RE Ra Ru Cm pred corr lag JP G lpf gL bL eL Vr FILE)
        and one row per acquisition block.  Returns a list of dicts (one per
        block), keyed by the header column names."""
        rows = []
        try:
            with open(path, "r", errors="replace") as fh:
                lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
        except OSError:
            return rows
        if not lines:
            return rows
        header = [h.strip() for h in lines[0].split("\t")]
        for ln in lines[1:]:
            cells = [c.strip() for c in ln.split("\t")]
            if len(cells) < 4:
                continue
            rows.append({header[i]: (cells[i] if i < len(cells) else "")
                         for i in range(len(header))})
        return rows

    @staticmethod
    def _txt_float(row, key, default=0.0):
        try:
            return float(str(row.get(key, default)).strip())
        except (TypeError, ValueError):
            return default

    def _txt_notes(self, filepath):
        """Return cached parsed ``.TXT`` note rows for a datac file ([] if none)."""
        key = str(Path(filepath))
        if key not in self._txt_cache:
            note, kind = self._find_note_file(filepath)
            self._txt_cache[key] = self._parse_txt_notes(note) if (note and kind == "txt") else []
        return self._txt_cache[key]

    def _mode_from_params(self, params):
        """Infer clamp mode from a .TXT param row: VC when whole-cell-cap /
        prediction / correction / holding-voltage settings are present."""
        if not params:
            return None
        for k in ("Cm", "pred", "corr", "Vr"):
            if self._txt_float(params, k, 0.0) != 0.0:
                return "VC"
        return None

    # -- older free-form .NOT parsing (record->level + best-effort blocks) ---
    _NOT_MARKER = re.compile(r"(\d\d:\d\d:\d\d)\s*\[R:\s*(\d+)\s*\]\s*-?")
    _NOT_LEV = re.compile(r"Lev\s+\d+\s+(-?\d+\.?\d*)")
    _NOT_AVG = re.compile(r"Average of\s+\d+", re.I)
    _NOT_COND = re.compile(r"(valve|soln|solution|switch|chang|return|low\s*mg|cscl|qx|lost|normal)", re.I)

    def _parse_not(self, path):
        """Parse a `.NOT` log into a per-record command-level table.

        Semi-structured format (per the lab): each acquisition is
        ``HH:MM:SS [R:n] - Lev 1 <v> ...  [HH:MM:SS [R:m] - Average of N]``.
        The number of records acquired is the *marker delta* (m - n): the
        repeated marker before "Average of N" names the UPCOMING record, not a
        new one.  A run of ``Lev 1`` values on a non-averaged line is a timed
        series (one record per value); on an averaged line they are the pulses
        of one multi-step command (one record).  A ``- <text>`` that is not
        "Average of N" is a comment.

        Returns dict(records={rec: {'level', 'averaged'}}, comments=[(rec,text)],
        maxrec=int).
        """
        try:
            lines = Path(path).read_text(errors="replace").splitlines()
        except OSError:
            return {"records": {}, "comments": [], "maxrec": 0}
        segs = []  # (rec, levels, averaged, comment)
        for line in lines[1:]:  # skip the "File: ..." header line
            ms = list(self._NOT_MARKER.finditer(line))
            for i, m in enumerate(ms):
                body = line[m.end(): (ms[i + 1].start() if i + 1 < len(ms) else len(line))]
                levels = [float(x) for x in self._NOT_LEV.findall(body)]
                averaged = bool(self._NOT_AVG.search(body))
                txt = self._NOT_AVG.sub("", self._NOT_LEV.sub("", body))
                txt = re.sub(r"V(o|delay)\s+-?\d+\.?\d*", "", txt).strip(" -\t")
                segs.append((int(m.group(2)), levels, averaged, txt if len(txt) > 2 else ""))
        records = {}
        comments = []
        for i, (rec, levels, averaged, comment) in enumerate(segs):
            if comment:
                comments.append((rec, comment))
            if not levels:
                continue
            nxt = next((segs[k][0] for k in range(i + 1, len(segs)) if segs[k][0] > rec), None)
            count = (nxt - rec) if nxt is not None else len(levels)
            if count <= 0:
                continue
            if count == 1:  # one record (command may be multi-pulse -> first level)
                records[rec] = {"level": levels[0], "averaged": averaged}
            else:           # timed series: one record per level
                for j in range(min(count, len(levels))):
                    records[rec + j] = {"level": levels[j], "averaged": averaged}
        return {"records": records, "comments": comments,
                "maxrec": max(records) if records else 0}

    def _parse_not_cached(self, filepath):
        key = str(Path(filepath))
        if key not in self._not_cache:
            note, kind = self._find_note_file(filepath)
            self._not_cache[key] = (self._parse_not(note)
                                    if (note and kind == "not") else
                                    {"records": {}, "comments": [], "maxrec": 0})
        return self._not_cache[key]

    def _not_auto_blocks(self, filepath):
        """Best-effort (approximate) block segmentation of a `.NOT`-annotated
        file.  Boundaries at: a condition/solution comment, or a sweep reset
        (the command level returns to/below the block's starting level after
        rising above it).  Labeled with the record range so the ranges are
        obvious; use an explicit ``R<a>-<b>`` path for exact control.
        """
        parsed = self._parse_not_cached(filepath)
        records = parsed["records"]
        if not records:
            return []
        cond_recs = {rec for rec, txt in parsed["comments"] if self._NOT_COND.search(txt)}
        blocks = []
        cur = None
        prev = None
        for rec in sorted(records):
            lv = records[rec]["level"]
            reset = (cur is not None and prev is not None
                     and lv <= cur["startlev"] and prev > cur["startlev"])
            condbrk = (cur is not None and (rec - 1) in cond_recs)
            if cur is None or reset or condbrk:
                if cur is not None:
                    blocks.append(cur)
                cur = {"start": rec, "last": rec, "startlev": lv,
                       "avg": records[rec]["averaged"]}
            else:
                cur["last"] = rec
            prev = lv
        if cur is not None:
            blocks.append(cur)
        return [(f"R{b['start']}-{b['last']}", list(range(b["start"], b["last"] + 1)), None)
                for b in blocks]

    @staticmethod
    def _range_from_path(path):
        """Return (a, b) if any path component is an explicit ``R<a>-<b>``
        record-range token, else None."""
        if isinstance(path, (int,)):
            return None
        for part in Path(path).parts:
            m = re.fullmatch(r"[Rr](\d+)-(\d+)", part)
            if m:
                return int(m.group(1)), int(m.group(2))
        return None

    def _true_command_levels(self, recs):
        """True per-record command levels (Amperes) for ``recs`` from the parsed
        .NOT record->level table, or None.  .NOT ``Lev`` values are in pA (e.g.
        Lev 1000 = 1 nA), so scale by 1e-12."""
        if not self._file:
            return None
        rl = self._parse_not_cached(self._file).get("records", {})
        if not rl:
            return None
        lv = []
        for r in recs:
            if r not in rl:
                return None
            lv.append(rl[r]["level"])
        return np.asarray(lv, dtype=float) * 1e-12

    def _blocks(self, filepath):
        """Return an ordered list of ``(label, record_list, params)`` protocol
        blocks (``params`` is the .TXT row dict, or None).

        Precedence: (1) a structured ``.TXT`` table (record ranges via RB/RE +
        per-block amplifier params); (2) a ``.NOT`` log parsed by ReadDatac into
        expInfo (the newer '>g STM:' format); (3) otherwise a single "000_all"
        block covering every record.

        Note: older ``.NOT`` logs without protocol markers (just timestamped
        [R:n] + Lev/Vo command sequences) currently fall through to "000_all";
        segmenting those into blocks from the command-level time sequence is a
        possible future heuristic.
        """
        d = self._open(filepath)
        if d is None:
            return []
        # 1) structured .TXT table (richest)
        txt = self._txt_notes(filepath)
        if txt:
            out = []
            for i, row in enumerate(txt):
                rb = int(self._txt_float(row, "RB", 0))
                re_ = int(self._txt_float(row, "RE", 0))
                if re_ < rb:
                    continue
                proto = str(row.get("STIM", "proto")).strip().replace(" ", "") or "proto"
                out.append((f"{i:03d}_{proto}", list(range(rb, re_ + 1)), row))
            if out:
                return out
        # 2) free-form .NOT parsed by ReadDatac into expInfo (newer format)
        expInfo = getattr(d, "expInfo", {}) or {}
        if len(expInfo) > 0:
            out = []
            for i in sorted(expInfo):
                b = expInfo[i]
                proto = str(b.get("Protocol", "proto")).replace(" ", "")
                out.append((f"{i:03d}_{proto}",
                            list(range(int(b["R"]), int(b["Rend"]) + 1)), None))
            return out
        # 2b) older free-form .NOT -> best-effort (approximate) auto-segmentation
        auto = self._not_auto_blocks(filepath)
        if auto:
            return auto
        # 3) no usable notes -> whole file is one block
        return [("000_all", list(range(1, int(d.records_in_file) + 1)), None)]

    def _mode_for(self, label, params=None):
        m = self._mode_from_params(params)
        if m:
            return m
        dm = str(self.defaults.get("datamode", "CC")).upper()
        if "vc" in str(label).lower():
            dm = "VC"
        return "VC" if dm == "VC" else "IC"

    def setProtocol(self, pathtoprotocol):
        self.protocol_path = pathtoprotocol
        self.last_protocol = None
        self._range = self._range_from_path(pathtoprotocol)
        f, b, _sweep, _explicit = self._resolve(pathtoprotocol)
        self._file = f
        self._block = None if self._range is not None else (b if b is not None else 0)

    # -- clamp-state helpers (parity with the other readers) ----------------
    def _clampparams(self, lpf=10000.0):
        return {
            "OutputZeroEnable": 0, "PipetteOffset": 0.0, "Holding": 0.0,
            "PrimarySignalHPF": 0.0, "BridgeBalResist": 0.0, "PrimarySignalLPF": lpf,
            "RsCompBandwidth": 0.0, "WholeCellCompResist": 0.0, "WholeCellCompEnable": 0,
            "LeakSubResist": 0.0, "HoldingEnable": 0, "FastCompTau": 0.0,
            "SlowCompCap": 0.0, "WholeCellCompCap": 0.0, "LeakSubEnable": 0,
            "NeutralizationCap": 0.0, "BridgeBalEnable": 0, "RsCompCorrection": 0.0,
            "NeutralizationEnable": 0, "RsCompEnable": 0, "OutputZeroAmplitude": 0.0,
            "FastCompCap": 0.0, "SlowCompTau": 0.0,
        }

    def _clampstate(self, label, rate=None, params=None):
        mode = self._mode_for(label, params)
        primary_units = "A" if mode == "VC" else "V"
        secondary_units = "V" if mode == "VC" else "A"
        sr = rate if rate else 10000.0
        # amplifier settings from a .TXT block when available
        holding = self._txt_float(params, "Vr", 0.0) * 1e-3 if params else 0.0  # mV -> V
        lpf_khz = self._txt_float(params, "lpf", 0.0) if params else 0.0
        lpf = lpf_khz * 1e3 if lpf_khz > 0 else float(self.defaults.get("LPF", 10000.0))
        cp = self._clampparams(lpf)
        cp["Holding"] = holding
        cp["HoldingEnable"] = int(holding != 0.0)
        if params:
            cp["WholeCellCompCap"] = self._txt_float(params, "Cm", 0.0) * 1e-12  # pF -> F
        return {
            "mode": mode,
            "primarySignal": "Membrane Current" if mode == "VC" else "Membrane Potential",
            "secondarySignal": "Command Voltage" if mode == "VC" else "Command Current",
            "primaryUnits": primary_units, "secondaryUnits": secondary_units,
            "primaryGain": 1.0, "secondaryGain": 1.0, "holding": holding,
            "ClampParams": cp,
            "DAQ": {
                "primary": {"rate": sr, "numPts": 0, "startTime": 0.0},
                "command": {"rate": sr, "numPts": 0, "startTime": 0.0},
                "secondary": {"rate": sr, "numPts": 0, "startTime": 0.0},
            },
        }

    def parseClampInfo(self, info):
        self.trClampInfo = info
        cs = info[1]["ClampState"]
        self.mode = cs["mode"]
        self.units = [cs["primaryUnits"], cs["secondaryUnits"]]
        self.samp_rate = cs["DAQ"]["primary"]["rate"]

    def parseClampWCCompSettings(self, info):
        if "ClampState" in info[1].keys() and "ClampParams" in info[1]["ClampState"].keys():
            par = info[1]["ClampState"]["ClampParams"]
            return {
                "WCCompValid": True, "WCEnabled": par["WholeCellCompEnable"],
                "WCResistance": par["WholeCellCompResist"], "WCCellCap": par["WholeCellCompCap"],
                "RsCompCorrection": par["RsCompCorrection"], "CompEnabled": par["RsCompEnable"],
                "CompCorrection": par["RsCompCorrection"], "CompBW": par["RsCompBandwidth"],
            }
        return {
            "WCCompValid": False, "WCEnable": 0, "WCResistance": 0.0, "WholeCellCap": 0.0,
            "CompEnable": 0, "CompCorrection": 0.0, "CompBW": 50000.0, "RsCompCorrection": 0.0,
        }

    def parseClampCCCompSettings(self, info):
        if "ClampState" in info[1].keys() and "ClampParams" in info[1]["ClampState"].keys():
            par = info[1]["ClampState"]["ClampParams"]
            return {
                "CCCompValid": True, "CCBridgeEnable": par["BridgeBalEnable"],
                "CCBridgeResistance": par["BridgeBalResist"],
                "CCNeutralizationEnable": par["NeutralizationEnable"],
                "CCNeutralizationCap": par["NeutralizationCap"],
                "CCLPF": par["PrimarySignalLPF"], "CCPipetteOffset": par["PipetteOffset"],
            }
        return {
            "CCCompValid": False, "CCBridgeEnable": 0, "CCBridgeResistance": 0.0,
            "CCNeutralizationEnable": 0.0, "CCNeutralizationCap": 0,
            "CCPipetteOffset": 0.0, "CCLPF": 10000.0,
        }

    def parseClampHoldingLevel(self, info):
        try:
            return info[1]["ClampState"]["holding"]
        except (KeyError, IndexError, TypeError):
            return 0.0

    def getDataInfo(self, filename=None, silent=False):
        assert filename is not None
        f, _b, _s, _e = self._resolve(filename)
        if f is None:
            if not silent:
                print(f"datac_reader.getDataInfo: not a datac file: {filename}")
            return None
        blocks = self._blocks(f)
        label, _recs, params = blocks[0] if blocks else ("000_all", [], None)
        info = [{"name": "Time", "units": "s"},
                {"ClampState": self._clampstate(label, params=params), "startTime": 0.0}]
        self.parseClampInfo(info)
        return info

    # -- synthesized directory / index --------------------------------------
    def subDirs(self, p):
        pp_ = Path(p)
        if self.last_protocol == pp_:
            return self.last_dirs
        rng = self._range_from_path(p)
        if rng is not None:  # explicit R<a>-<b>: one sweep dir per record
            n, base = rng[1] - rng[0] + 1, pp_
            dirs = [base / f"{i:03d}" for i in range(n)]
            self.last_dirs, self.last_protocol = dirs, pp_
            return dirs
        f, b, _s, explicit = self._resolve(p)
        blocks = self._blocks(f) if f is not None else []
        bidx = b if (b is not None and 0 <= b < len(blocks)) else 0
        n = len(blocks[bidx][1]) if blocks else 0
        # Base per-sweep dirs on the block-labeled path so "<file>/NNN_proto/SSS"
        # is unambiguous with a bare block path "<file>/NNN"; if given the bare
        # file, use the resolved block's label as the base.
        if explicit:
            base = pp_
        elif f is not None and blocks:
            base = f / blocks[bidx][0]
        else:
            base = pp_
        dirs = [base / f"{i:03d}" for i in range(n)]
        self.last_dirs, self.last_protocol = dirs, pp_
        return dirs

    def data_block_labels(self, filepath=None):
        f = filepath if filepath is not None else self._file
        return [blk[0] for blk in self._blocks(f)]

    def _synthesize_protocol_index(self, label, params=None):
        cs = self._clampstate(label, params=params)
        return {
            ".": {
                "dirType": "ProtocolSequence", "__timestamp__": 0.0, "notes": "",
                "description": label,
                "devices": {
                    self.datac_device: {
                        "mode": cs["mode"], "holdingCheck": False,
                        "holdingSpin": 0.0, "primaryUnits": cs["primaryUnits"],
                    },
                },
                "sequenceParams": {("protocol", "repetitions"): [0]},
            }
        }

    def _synthesize_sweep_index(self, sweep):
        return {".": {"dirType": "Protocol", "__timestamp__": 0.0, "sweep": sweep}}

    def _synthesize_cell_index(self, d):
        comment = str(getattr(d, "comment", "")).strip()
        return {".": {"dirType": "Cell", "notes": comment, "type": "",
                      "important": False}}

    def readDirIndex(self, currdir=None):
        self._dirindex = None
        if currdir is None:
            currdir = self.protocol_path
        if currdir is None:
            return None
        f, b, sweep, explicit = self._resolve(currdir)
        if f is None:
            print(f"datac_reader.readDirIndex: no datac data recognized at {currdir}")
            return None
        d = self._open(f)
        if d is None:
            return None
        rng = self._range_from_path(currdir)
        if sweep is not None:
            self._dirindex = self._synthesize_sweep_index(sweep)
        elif rng is not None:  # explicit R<a>-<b> range path -> a protocol
            self._dirindex = self._synthesize_protocol_index(f"R{rng[0]}-{rng[1]}", params=None)
        elif explicit:
            blocks = self._blocks(f)
            if b is not None and 0 <= b < len(blocks):
                label, _recs, params = blocks[b]
            else:
                label, params = "000_all", None
            self._dirindex = self._synthesize_protocol_index(label, params=params)
        else:
            self._dirindex = self._synthesize_cell_index(d)
        return self._dirindex

    def getIndex(self, currdir=None, lineend="\n"):
        self.lb = lineend
        self._index = self.readDirIndex(currdir if currdir is not None else self.protocol_path)
        return self._index["."] if self._index is not None else None

    def _readIndex(self, currdir=None):
        self._index = self.readDirIndex(currdir if currdir is not None else self.protocol_path)
        return self._index

    def convert_timestamp(self, fts):
        return datetime.datetime.fromtimestamp(fts).strftime("%Y-%m-%d  %H:%M:%S %z")

    def printIndex(self, index):
        print(pprint.pformat(index))

    def getIndex_text(self, index):
        return pprint.pformat(index)

    def file_cell_protocol(self, filename=None):
        assert filename is not None
        f, _b, _s, explicit = self._resolve(filename)
        proto = Path(filename).name if explicit else ""
        cell = f.stem if f is not None else Path(filename).stem
        date = f.parent.name if f is not None else ""
        parent = f.parent if f is not None else Path(filename).parent
        return (date, "slice_000", cell, proto, parent)

    def getClampDevices(self, currdir=None, verbose=False):
        info = self.getIndex(currdir=currdir)
        devs = []
        if info is not None and "devices" in info:
            for dev in info["devices"]:
                if dev in self.clampdevices:
                    devs.append(dev)
        return devs

    def header_metadata(self, filename=None):
        src = filename if filename is not None else self.protocol_path
        f = None
        if src is not None:
            f, _b, _s, _e = self._resolve(src)
        d = self._open(f) if f is not None else None
        meta = {}
        if d is not None:
            note, kind = self._find_note_file(f)
            meta = {
                "file": str(f), "comment": str(getattr(d, "comment", "")).strip(),
                "mode_flag": int(getattr(d, "mode", -1)),
                "nr_channel": int(getattr(d, "nr_channel", 0)),
                "nr_points": int(getattr(d, "nr_points", 0)),
                "records_in_file": int(getattr(d, "records_in_file", 0)),
                "nblocks": len(self._blocks(f)),
                "notefile": str(note) if note else None,
                "notefile_kind": kind,
            }
            # per-block .TXT amplifier params (SLTN/STIM/Ra/Ru/Cm/.../Vr) when present
            txt = self._txt_notes(f)
            if txt:
                meta["solution"] = txt[0].get("SLTN", "").strip()
                meta["txt_blocks"] = [
                    {k: row.get(k, "") for k in ("STIM", "RB", "RE", "Ra", "Ru",
                                                 "Cm", "lpf", "JP", "G", "Vr")}
                    for row in txt
                ]
            elif kind == "not":
                # older .NOT: expose the parsed record->command-level table and
                # any comments (both reliably extractable, unlike block bounds)
                parsed = self._parse_not_cached(f)
                meta["record_levels"] = {
                    r: parsed["records"][r]["level"] for r in sorted(parsed["records"])
                }
                meta["not_comments"] = parsed["comments"]
        return meta

    # -- protocol checking (minimal parity) ---------------------------------
    def checkProtocol(self, protocolpath=None, allow_partial=False):
        src = protocolpath if protocolpath is not None else self.protocol_path
        if src is None:
            return False
        f, _b, _s, _e = self._resolve(src)
        return f is not None and self._open(f) is not None

    def checkProtocolImportantFlags(self, protocolpath=None):
        return True

    def checkProtocolImportant(self, protocol):
        return True

    def listSequenceParams(self, dh=None):
        return {}

    # -- data ---------------------------------------------------------------
    def getData(self, pos=1, check=False, allow_partial=False, record_list=[],
                downsample=1, silent=True):
        """Read the selected DATAC block and fill the acq4_reader-style
        attributes (via ReadDatac + GetClamps).  Returns True on success."""
        self.error_info = None
        d = self._open(self._file)
        if d is None:
            if check:
                return False
            raise FileNotFoundError(self.error_info or f"cannot open datac file {self._file}")
        bidx = 0
        if self._range is not None:  # explicit R<a>-<b> record-range override
            a, bb = self._range
            label, recs, params = f"R{a}-{bb}", list(range(a, bb + 1)), None
        else:
            blocks = self._blocks(self._file)
            if not blocks:
                self.error_info = "no readable blocks in datac file"
                return False
            bidx = self._block if (self._block is not None and 0 <= self._block < len(blocks)) else 0
            label, recs, params = blocks[bidx]
        self._current_params = params
        if check:
            return len(recs) > 0
        if allow_partial and isinstance(record_list, list) and len(record_list) > 0:
            recs = [r for r in recs if r in record_list] or recs

        d.dmode = "VC" if self._mode_for(label, params) == "VC" else "CC"
        d.readrecords(recs)
        if getattr(d, "err", 1) not in (0, None):
            self.error_info = f"readrecords error {getattr(d,'err',None)}"
            return False
        # A flaky record (e.g. in a "lost cell" region) can leave its per-record
        # sample rate unset (None), which crashes getClampData's rate math.
        # Substitute a good rate from the same read (or a nominal 40 us).
        if getattr(d, "rate", None) is not None:
            good = [x for x in d.rate if x]
            fallback = good[0] if good else 40.0
            d.rate = [x if x else fallback for x in d.rate]

        gc = GetClamps(d)
        tstart = float(self.defaults.get("tstart", 0.01))
        tdur = float(self.defaults.get("tdur", 0.100))
        gc.getClampData(block=bidx, tstart_tdur=[tstart, tdur])

        tr = np.asarray(gc.traces)
        if tr.ndim == 3:  # single-record safety (getClampData leaves 3-D)
            tr = np.squeeze(tr)
            if tr.ndim == 1:
                tr = tr[np.newaxis, :]
            self.traces = MetaArray.MetaArray(tr, info=gc.traces.infoCopy()) \
                if hasattr(gc.traces, "infoCopy") else tr
        else:
            self.traces = gc.traces
        self.data_array = np.asarray(self.traces)
        nsweeps = self.data_array.shape[0]
        self.cmd_wave = np.atleast_2d(np.asarray(gc.cmd_wave, dtype=float))
        # Sample rate: for Digidata (mode>=2) use the matrix.h header rate that
        # ReadDatac reads per record (datac2mat.m uses it directly; per-channel,
        # in kHz -> x1000 for Hz). getClampData's own time_base was internally
        # inconsistent and truncated the sweep, so rebuild it here. The older
        # mode-0/1 timing (oldtime) is unreliable -> configurable default.
        srate = float(self.defaults.get("sample_rate_hz", 10000.0))
        try:
            if int(getattr(d, "mode", 0)) >= 2 and getattr(d, "rate", None) and d.rate[0]:
                srate = float(d.rate[0]) * 1000.0  # header rate (kHz) -> Hz, per channel
        except Exception:  # noqa: BLE001
            pass
        npts = self.data_array.shape[1]
        self.sample_interval = 1.0 / srate
        self.sample_rate = srate * np.ones(nsweeps)
        self.time_base = np.arange(npts) / srate
        self.mode = self._mode_for(label, params)
        self.units = ["A", "V"] if self.mode == "VC" else ["V", "A"]
        # Claude 2026-09-30: removed the earlier cascaded .NOT/DC-offset command
        # "calibration". Amplitude scaling and channel assignment are done by
        # readrecords/getClampData using the amplifier_* constants and the
        # per-record matrix.h gain[] (see ReadDatac). Use its output directly.
        self.values = np.asarray(getattr(gc, "values", np.zeros(nsweeps)), dtype=float)
        self.commandLevels = np.asarray(getattr(gc, "commandLevels", self.values), dtype=float)
        # holding from the .TXT block's Vr (mV -> V) when available
        self.holding = (self._txt_float(params, "Vr", 0.0) * 1e-3) if params else getattr(gc, "holding", 0.0)
        self.tstart = getattr(gc, "tstart", tstart)
        self.tdur = getattr(gc, "tdur", tdur)
        self.tend = getattr(gc, "tend", tstart + tdur)
        self.clampInfo["dirs"] = self.subDirs(self.protocol_path)
        self.clampInfo["missingData"] = []
        self.protoDirs = [f"{i:03d}" for i in range(nsweeps)]
        self.trace_index = list(range(nsweeps))
        self.trace_important = [True] * nsweeps
        self.nprotodirs = nsweeps
        self.trace_StartTimes = np.zeros(nsweeps)
        self.repetitions = 1
        cs = self._clampstate(label, rate=float(self.sample_rate[0]) if nsweeps else None,
                              params=params)
        self.WCComp = self.parseClampWCCompSettings([None, {"ClampState": cs}])
        self.CCComp = self.parseClampCCCompSettings([None, {"ClampState": cs}])
        return True

    def getClampCommand(self, data, generateEmpty=True):
        if hasattr(data, "hasColumn") and data.hasColumn("Channel", "Command"):
            return data["Channel":"Command"]
        if generateEmpty:
            tVals = self.time_base
            units = "V" if (self.mode and "v" in self.mode.lower()) else "A"
            return MetaArray.MetaArray(
                np.zeros(np.asarray(tVals).shape),
                info=[{"name": "Time", "values": tVals, "units": "s"}, {"units": units}],
            )
        return None

    # -- LSPS / imaging: no DATAC equivalent -> honest stubs ----------------
    def getStim(self, stimname="Stim"):
        return {"start": [], "duration": [], "amplitude": [], "npulses": [], "period": []}

    def getLaserBlueTimes(self):
        return False

    def getLaserBlueCommand(self):
        return False

    def getLaserBlueShutter(self):
        return False

    def getLEDCommand(self):
        return False

    def getPhotodiode(self):
        return False

    def getScannerPositions(self, dataname="Laser-Blue-raw.ma"):
        return False

    def getImage(self, filename=None):
        return None

    def getAverageScannerImages(self, dataname="Camera/frames.ma", mode="average", **kwargs):
        return None


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

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plotonefile(filename:Union[Path, str], datamode:str="CC", arrange:str="grid", 
                records:Optional[List[int]]=None, vgain: float=1.0, iscale: float=1.0):
    """
    Plot the traces from 2 channels from this data file
    should specify data mode ('CC' or 'VC')
    """

    filename = Path(filename)
    dfile = ReadDatac(datamode=datamode)
    dfile.read_datac_header(filename)
    dfile.read_note_file()
    dfile.print_datac_note(3)
    if dfile.err != 0:
        print("Error reading file %s" % filename)
        print("Error code: %d" % dfile.err)
        return
    print(dfile.err)
    print(dfile.expInfo)
    if records is None:
        nrecs = (dfile.records_in_file+2)//2
        records = list(range(nrecs))
        r, c = PH.getLayoutDimensions(nrecs)
    else:
        nrecs = len(records)
        if arrange == "grid":
            r, c = PH.getLayoutDimensions(nrecs)
        elif arrange == "horizontal":
            r, c = 1, nrecs
        elif arrange == "vertical":
            r, c = nrecs, 1
    k = dfile.expInfo.keys()
    if len(k) == 0:
        i = 0
        dfile.expInfo[i] = {"R": records[0], "Rend": records[-1], "time": "00:00:00", "Protocol": "Unknown"}
   
    # r, c = 4, 5
    # nrecs = r*c
    figsize = (12, 10)
    if nrecs < 4 and arrange == "horizontal":
        figsize = (7, 3.5)
    P = PH.regular_grid(r, c, order="rowsfirst", figsize=figsize, 
                        verticalspacing=0.02, horizontalspacing=0.02)
    axr = P.axarr.ravel()
    axr[0].set_ylim(-0.1, 0.02)
    for i, ax in enumerate(axr):
        PH.noaxes(ax)
        if i > 0:
            axr[i].sharey(axr[0])    
    # PH.nice_plot(axr[0], direction="outward", position=-0.05)
    # if records is not None:
    # for i in :
        # dfile.readrecords(range(dfile.expInfo[i]["R"], dfile.expInfo[i]["Rend"]))
    dfile.readrecords(records)
    cl = GetClamps(dfile)
    cl.getClampData(1, verbose=False)
    print(1e9*cl.commandLevels)
    # f, ax = mpl.subplots(2,1)
    # nn=78
    # ax[0].plot(0.5*1e9*cl.commandLevels[:nn], 'ko-', markersize=3, label='Command Levels (nA)')
    # ax[0].plot(cl.cmd_wave[:nn, 0]*0.5*1e9, 'b-', linewidth=0.5, label='Command Waveform (V)')
    # ax[1].plot(cl.traces[:nn, 0].view(np.ndarray) * vgain, 'k-', linewidth=0.5, label='Primary Trace (V)')
    # mpl.show()
    # exit()
    # if dfile.err == 0:
    #     txt = dfile.expInfo[i]["Protocol"] + " " "[R:%d-%d] " % (
    #         dfile.expInfo[i]["R"],
    #         dfile.expInfo[i]["Rend"],
    #     )
    #     txt += dfile.expInfo[i]["time"]
    #    dfile.plotcurrentrecs(i, title=txt)
    for i, j in enumerate(records):
        # tr = cl.traces[j, :].view(np.ndarray)
        print(i, cl.cmd_wave[i].min(), cl.cmd_wave[i].max())
        axr[i].plot(cl.time_base, vgain * cl.traces[i,:].view(np.ndarray), 'k-', linewidth=0.5)
        axr[i].plot(cl.time_base, -0.055*np.ones_like(cl.time_base), 'k--', linewidth=0.3)
        axr[i].text(0.01, 0.95, f"R: {j+1:d}", transform=axr[i].transAxes,
                    fontsize=8, color='b', va='top', ha='left')
        in_ax = axr[i].inset_axes([0, 0, 1.0, 0.12])
        PH.noaxes(in_ax)
        in_ax.plot(cl.time_base, cl.cmd_wave[i,:]*iscale, 'k-', linewidth=0.5, clip_on =False)
        in_ax.set_ylim(-10e-10, 10e-10)
        in_ax.plot(cl.time_base, 0.0 * np.ones_like(cl.time_base), 'k--', linewidth=0.15)
    P.figure_handle.suptitle(f"Data file: {filename.name} ({datamode})")
    mpl.tight_layout()
    mpl.show()

    dfile.close()
    

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
        print("\nDirectory: %s" % p)
        for fx in os.listdir(p):
            (f, e) = os.path.splitext(fx)
            ff = os.path.join(p, fx)
            nempty += printNotes(ff, indent=3)
            n += 1
    print("\nFiles read: {:d}, Empty file accessed: {:d}".format(n, nempty))


def example_dir_scan(searchstr, protocol="iv2.stm"):
    """
    Scan a directory and plot the traces from the first protocol
    that matches the search string
    """
    fns = Path(searchstr).glob("*")
    for fn in fns:
        df = printNotes(fn, indent=5)
        if df is None:
            continue
        for protn in df.expInfo.keys():
            if df.expInfo[protn]["Protocol"] == protocol:
                df.readrecords(range(df.expInfo[protn]["R"], df.expInfo[protn]["Rend"]))
                df.plotcurrentrecs(fn)
                break
    mpl.show()


def show_file_recs(file, rec_start, rec_end, datamode="CC"):
    fn = Path(file)
    if fn.is_file():
        dfile = ReadDatac(datamode=datamode)
        dfile.read_datac_header(fn)
        dfile.readrecords(range(rec_start, rec_end))
        cl = GetClamps(dfile)
        cl.getClampData(1, verbose=True)
        # if dfile.err == 0:
        #     txt = dfile.expInfo[i]['Protocol'] + ' ' '[R:%d-%d] ' % (dfile.expInfo[i]['R'],dfile.expInfo[i]['Rend'])
        #     txt += dfile.expInfo[i]['time']
        #    dfile.plotcurrentrecs(i, title=txt)
        # mpl.show()
        fig, ax = mpl.subplots(2, 1)
        print(cl.traces.shape)
        tr = cl.traces.view(np.ndarray)
        cmd = cl.cmd_wave.view(np.ndarray)
        for j in range(tr.shape[0]):
            ax[0].plot(cl.time_base, tr[j, :], linewidth=0.5)
            ax[1].plot(cl.time_base, cmd[j, :], linewidth=0.5)
        
        mpl.show()
        dfile.close()
        mpl.show()


if __name__ == "__main__":
    # dcn: 13Jun889L is good
    # example_dir_scan(searchstr='/Users/pbmanis/Documents/data/HWF0001B/VCN/*.HWF')
    # plotonefile(Path('/Volumes/Pegasus_004/ManisLab_Data3/DCN1/6DEC88B.PBM'), datamode='CC', records = np.arange(0, 41))
    plotonefile(Path('/Volumes/Pegasus_004/ManisLab_Data3/DCN2/13JUN89L.PBM'), datamode='CC', records = 
                [78, 68, 99],
                arrange='horizontal', iscale=0.1) # np.arange(92,
                                                                                                                 # 95))

    # plotonefile(os.path.join('/Users/pbmanis/Documents/data/HWF0001B/VCN', '11SEP96H.HWF'), datamode='CC')
    # plotonefile(os.path.join('/Users/pbmanis/Documents/data/HWF0001B/VCN', '26AUG96B.HWF'), datamode='CC')
    # plotonefile(os.path.join('/Users/pbmanis/Documents/data/datac', '07FEB96H.JSR'), datamode='VC')
