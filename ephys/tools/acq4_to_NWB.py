"""
This program provides functions to tead and convert acq4 data to NWB format that
is acceptable for the Dandi repository. It was specifically designed for the
auditory cortex experiments with Kato lab, but should be generally useful for
converting other kinds of acq4 experiments. A single file can be converted by
calling ConvertFile(filename) The conversion is checked after generating the
outputfile with nwbinspector.

V0.2.0, 24 June 2024

NWBFile expects: class pynwb.file.NWBFile(
    Required:
        session_description (str) :: a description of the session where this
        data was generated identifier (str) :: a unique text identifier for the
        file session_start_time (datetime) :: the start date and time of the
        recording session

    Optional (we try to include these):
        experimenter (tuple or list or str) :: name of person who performed
        experiment experiment_description (str) :: general description of the
        experiment institution (str) :: institution(s) where experiment is
        performed lab (str) :: lab where experiment was performed source_script
        (str) :: Script file used to create this NWB file.
        source_script_file_name (str) :: Name of the source_script file

        devices (list or tuple) :: Device objects belonging to this NWBFile
        subject (Subject) :: subject metadata protocol (str) :: Experimental
        protocol, if applicable. E.g., include IACUC protocol acquisition (list
        or tuple) :: Raw TimeSeries objects belonging to this NWBFile stimulus
        (list or tuple) :: Stimulus TimeSeries objects belonging to this NWBFile
        : Electrical opto: OptogeneticStimulusSites and waveforms that belong to
        this NWBFile. lab_meta_data (list or tuple) :: an extension that
        contains lab-specific meta-data intracellular_recordings
        (IntracellularRecordingsTable) ::
            the IntracellularRecordingsTable table that belongs to this NWBFile
        icephys_simultaneous_recordings (SimultaneousRecordingsTable) ::
            the SimultaneousRecordingsTable table that belongs to this NWBFile
        icephys_sequential_recordings (SequentialRecordingsTable) ::
            the SequentialRecordingsTable table that belongs to this NWBFile
        icephys_repetitions (RepetitionsTable) ::
            the RepetitionsTable table that belongs to this NWBFile
        icephys_experimental_conditions (ExperimentalConditionsTable) ::
            the ExperimentalConditionsTable table that belongs to this NWBFile

        units (Units) :: A table containing unit metadata

        pharmacology (str) :: Description of drugs used, including how and when
             they were administered. Anesthesia(s), painkiller(s), etc., plus
             dosage, concentration, etc.
        virus (str) :: Information about virus(es) used in experiments,
            including virus ID, source, date made, injection location, volume,
            etc.

    Optional and not included:
        file_create_date (ndarray or list or tuple or Dataset or StrDataset
             or HDMFDataset or AbstractDataChunkIterator or datetime) :: the
             date and time the file was created and subsequent modifications
             made
        timestamps_reference_time (datetime) ::
            date and time corresponding to time zero of all timestamps; defaults
            to value of session_start_time
        session_id (str) :: lab-specific ID for the session keywords (ndarray or
        list or tuple or Dataset or StrDataset or HDMFDataset
            or AbstractDataChunkIterator) :: Terms to search over
        related_publications (tuple or list or str) :: Publication
        information.PMID, DOI, URL, etc. If multiple, concatenate together and
        describe which is which. such as PMID, DOI, URL, etc slices (str) ::
        Description of slices, including information about preparation
        thickness, orientation, temperature and bath solution data_collection
        (str) :: Notes about data collection and analysis. surgery (str) ::
        Narrative description about surgery/surgeries, including date(s) and who
        performed surgery. stimulus_notes (str) :: Notes about stimuli, such as
        how and where presented. analysis (list or tuple) :: result of analysis
        stimulus (list or tuple) :: Stimulus TimeSeries objects belonging to
        this NWBFile stimulus_template (list or tuple) :: Stimulus template
        TimeSeries objects belonging to this NWBFile epochs (TimeIntervals) ::
        Epoch objects belonging to this NWBFile epoch_tags (tuple or list or
        set) :: A sorted list of tags used across all epochs trials
        (TimeIntervals) :: A table containing trial data invalid_times
        (TimeIntervals) :: A table containing times to be omitted from analysis
        intervals (list or tuple) :: any TimeIntervals tables storing time
        intervals processing (list or tuple) :: ProcessingModule objects
        belonging to this NWBFile electrodes (DynamicTable) :: the
        ElectrodeTable that belongs to this NWBFile electrode_groups (Iterable)
        :: the ElectrodeGroups that belong to this NWBFile sweep_table
        (SweepTable) :: the SweepTable that belong to this NWBFile
        imaging_planes (list or tuple) :: ImagingPlanes that belong to this
        NWBFile ogen_sites (list or tuple) :: OptogeneticStimulusSites that
        belong to this NWBFile

        scratch (list or tuple) :: scratch data
       icephys_filtering (str) :: [DEPRECATED] Use
       IntracellularElectrode.filtering instead. Description of filtering used.
        ic_electrodes (list or tuple) :: DEPRECATED use icephys_electrodes
        parameter instead. IntracellularElectrodes that belong to this NWBFile


    Mapping ACQ4 to NWB:
    ====================
    Acq4 stores data in a hierarchical directory structure.
        The top level is the DAY of the experiment. The second level is the
        SLICE number. The third level is the CELL number. The fourth level is
        the PROTOCOL name, with an appended number in the format "_000".
            Repeats of a protocol will increment this number, although if a
            protocol is ended early, the sequence may not have fixed steps.
        Within a protocol, there is a series of subdirectories, each containing
            the data for a single sweep.
        The protocol "sweep" subdirectories are numbered according to the
        nesting of repititions. If there is only one repitition, the
        subdirectory is named "000". If there are multiple repitions, the
        Subdirectories are named "000_000", "001_000", (for example showing two
        reps of the same stimulus).
            etc. If there is another parameter that is varied, it is added to
            the name, e.g. "000_000_000" (this is rarely if ever done).

    Mapping this to the NWB structure:
        1. Experimental conditions table - not used here.
        2. Repetitions Table: Should represent the repetitions of a protocol.
        3. Sequential Recordings Table: Should represent the sweeps within a
           repetition (usually parametric
        variation of current, laser spot position, or other stimulus parameter).
        This corresponds to a "protocol" in the acq4 structure. 4. Simultaneous
        Recordings Table: Should represent the simultaneous recordings from
        multiple electrodes. We are not doing this, and it can be skipped
        according to the pynwb documentation. 5. Intracellular Recordings Table:
        mapping recordings from multiple electrodes to the individual
        intracellular recordings. This is not used here (we are not recording
        from multiple electrodes). 6. PatchClampSeries: This is the lowest
        level, that corresponds to a single trial or sweep in acq4 - typically
        the current or voltage trace, and the stimulus command. This is the
        level at which we will store the traces.

        Also included are optical stimluation data, stored in ogen structures
        (data and series).

We use the NAME of each entry to parse and reassemble the data for display. See
display_nwb.py for an example of how we use this. The name is formatted as a
string that can be parsed (in regex) to extract the protocol, data type, and
sweep number. The protocol and sweep numbers are used to bind the stimulus
(optical, electrical) and recordings (electrical) together. The optogenetic data
in the ogen structures similarly has names that match and hold other parameters
such as the spot location (if using the laser with a scanning mirror), power,
etc. )


UPLOAD TO DANDI
----------------
1. make sure you have the latest DANDI (pip install dandi==0.67.3 or whatever is
   latest)
2. maker sure you have the latest dandischema (pip install dandischema==0.11.0
   or whatever is latest)
 Then at the terminal:
```
1. dandi download https://dandiarchive.org/dandiset/<dataset_id>/draft
2. cd <dataset_id> dandi organize <source_folder> -f dry   # dry run to see if it works
3. dandi organize <source_folder>
4. dandi validate dandi upload
``` Support::
    (this is also information that needs to be manually filled out at DANDI)

    NIH grants:
    DC RF1 NS128873 (Kato, Manis, MPI, 2022-) Cortical circuits for the integration of parallel short-latency auditory pathways.
    DC R01 DC019053 (Manis, 2020-2025) Cellular mechanisms of auditory information processing.

    Ethics:
    iacuc@med.unc.edu
    IACUC protocol 24.083 (Manis)
    https://research.unc.edu/iacuc/

Copyright 2022-2025 Paul B. Manis Distributed under MIT/X11 license. See
license.txt for more information.
ORCID:0000-0003-0131-8961
ROR.org id: https://ror.org/0130frc33

"""

import argparse
import concurrent.futures
import datetime
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Tuple, Union

import numpy as np
import pynwb as NWB
import scipy.signal
from dateutil.tz import tzlocal
from nwbinspector import inspect_all, inspect_nwbfile
from pynwb import NWBHDF5IO
from pynwb.core import DynamicTable, VectorData

from ephys import datareaders as DR
import ephys.tools.utilities as UT
UTIL = UT.Utility()

# re_datasets = re.compile(r"^([a-z_]){1,3}", re.IGNORECASE) # (?P<id>)[0-9]{6}$", re.IGNORECASE)  # match the acq4 data files

# m = re_datasets.match("HK_Cre_ugh_001422")  # should be True
# print(m)
# print(m.group(2))
# print(m.groups())
# print(m.group("id"))  # should be 001422

re_pnd = re.compile(r"^[P]{0,1}([0-9]{1,9})[D]{0,1}$", re.IGNORECASE)  # match the acq4 data files
# tests:
# d = re_pnd.match("P30D")
# print("P30D:" , d)
# d = re_pnd.match("")
# print("no entry: ", d)
# d = re_pnd.match("P30")
# print("P30: ", d)
# d = re_pnd.match("30")
# print("30: ", d)
# exit()


def def_lister():
    return []


@dataclass
class ExperimentInfo:
    """Data class to hold the metadata for the experiment."""

    description: str = ""
    protocol: str = ""
    time: str = ""
    experimenters: list = field(default_factory=def_lister)
    lab: str = ""
    institution: str = "UNC Chapel Hill"
    experiment_description: str = ""
    sessionid: str = "0"
    notes: str = ""
    subject: str = ""


print("acq4tonwb")


class ACQ4toNWB:
    def __init__(self, output_path: Union[str, Path, None] = None):
        """Convert a file from ACQ4 (www.acq4.org) format to NWB format (www.nwb.org)
        See the documentation above for the mapping between these formats.

        Args:
            out_file_path (Union[str, Path, None], optional): The path to the output files. Defaults to None.
        """
        self.AR = DR.acq4_reader.acq4_reader()  # get acq4 reader from the ephys package.

        self.out_file_path = output_path
        # All of the data here is single electrode, and "MultiClamp1.ma" is the
        # name of the data file holding the ephys data. The ".ma" acq4 files are held in "metaarray" format, but
        # these are basically hdf5 files. They are accompanied by .index files with additional metadata.
        # if a different electrode/amplifier is used, change [1] to [2] etc. in self.ampchannels
        self.set_amplifier_name("MultiClamp1.ma")
        self.laser_device = None
        self.LED_device = None
        self.ampchannels = [1]
        # self.manager = NWB.get_manager()
        self.ID = 0

    def _get_slice_cell(self, f: Union[str, Path]):
        """pull the slice and cell directories from the full file name
        Acq4 uses a fairly strict hierichal structure of:
        Day
            slice number 0
                cell number 0
                cell number 1
            slice number 1
                cell number 0
            ... etc.


        Args:
            f (Union[str, Path], optional): Full file path.

        Returns:
            slicen, cell (string represtations of partial path values)
        """
        f = str(f)
        re_cell = re.compile(r"cell_(\d{3})", re.IGNORECASE)
        re_slice = re.compile(r"slice_(\d{3})", re.IGNORECASE)
        slice_re = re_slice.search(f)
        if slice_re is None:
            raise ValueError(f"Could not find slice number in {f}")
        slicen = slice_re.group(1)
        cell_re = re_cell.search(f)
        if cell_re is None:
            raise ValueError(f"Could not find cell number in {f}")
        celln = cell_re.group(1)
        return int(slicen), int(celln)

    def _get_short_name(
        self,
        f: Union[str, Path],
        expt_id: Union[str, None] = None,
        rig_id: Union[str, None] = None,
    ):
        """
        Convert a data path of this format (used by acq4):
        f = Path('/Volumes/Pegasus/all/the/leading/pathnames/2017.05.01_000/slice_000/cell_000')
        To:
        2017.05.01~S0C0 ("short form")

         Args:
            f (Union[str, Path], optional): Full file path.
            expt_id: str or None: an experiment label to put in the filename.
                ignored if None
            rig_id: str or None: a rig label to put in the filename
                ignored if None

        Returns:
            short path name string
        """
        if f is None:
            raise ValueError(f"Input file to get_shortname is NONE")
        fp = Path(f).parts
        cell = fp[-1]
        slicenp = fp[-2]
        day = fp[-3]
        if expt_id == None:
            expt_id = ""
        if rig_id == None:
            rig_id = ""
        foname = str(Path(expt_id, rig_id, day[:10], "S" + slicenp[-2:] + "C" + cell[-2:])).replace(
            "/", "~"
        )
        return foname

    def set_amplifier_name(self, amplifier_name: str = "MultiClamp1.ma"):
        """Convenience function to change the amplifier name
        This is the name of the acq4 file that holds the recorded data, and
        may also hold the stimulus files. It also has some associated metadata,
        in the .index file that is associated with the .ma file.


        Args:
            datatype (str): name of the amplifier
        """
        self.AR.setDataName(amplifier_name)

    def ISO8601_age(self, agestr, strict: bool = False):
        """Convert free-form age designators to ISO standard, e.g.:
            postnatal day 30 mouse = P30D  (or P30W, or P3Y)
            Ranges are P1D/P3D if bounded, or P12D/ if not known but have lower bound.

        Params:
            agestr (str): age string from the file

        Returns:
            str: sanitized age string
        """

        agestr = agestr.replace("p", "P")
        agestr = agestr.replace("d", "D")
        if strict:
            if agestr.find("ish") or agestr.find("?"):
                raise ValueError(
                    "Age with 'ish or '?' is not a valid age descriptor; please fix in the acq4 data."
                )
        if "P" not in agestr:
            agestr = "P" + agestr
        if "D" not in agestr:
            agestr = agestr + "D"
        if agestr == "PD":
            agestr = "P9999D"  # no age specified
        return agestr

    def find_name_in_path(self, sourcepath: Union[str, Path], name: Union[str, None] = None):
        """find_name_in_path Search the path string for a name, and return the name if found.
        This is used to find "Rig", "Experiment", "Slice", "Cell" etc. in the path string.

        Parameters
        ----------
        sourcepath : Union[str, Path]
            The full path
        name : Union[str, None], optional
            The string to search for in the path, by default None

        Returns
        -------
        _type_
            str: the name found in the path

        Raises
        ------
        ValueError
            when name is ambiguous in the path (non-unique)
        """
        if name is None:
            return ""
        pathparts = Path(sourcepath).parts
        name_id = [r for r in pathparts if r.startswith(name)]
        if name_id == []:
            name_id = ""
        elif len(name_id) > 1:
            raise ValueError("Name is ambiguous in the path: try unique pattern")
        else:
            name_id = name_id[0]
        return name_id

    def acq4tonwb(
        self,
        experiment_name: str,
        path_to_cell: Union[str, Path],
        protocols: list,
        records: list,
        output_path: Union[Path, str] = None,
        outfilename: Union[Path, str, None] = None,
        appendmode: bool = False,
        recordingmode: Literal["IC", "CC", "VC", "I=0"] = "CC",
        downsample: int = 1,
        low_pass_filter: Union[float, None] = None,
        keywords: Union[list, None] = None,
        experimenter: Union[list, None] = None,
        experiment_description: Union[str, None] = None,
        iacuc_protocol: Union[str, None] = None,
    ):
        """Convert one cell directory to an NWB file.
        The NWB file will contain all of the recordings from ONE cell.

        Args:
            path_to_cell (string or Path): Full path and name of the cell directory
            outfilename (Union[Path, str, None], optional): NWB output filename. Defaults to None.
            recordingmode (Literal[&quot;IC&quot;, &quot;CC&quot;, &quot;VC&quot;, &quot;I, optional):
              _description_. Defaults to 0"]="CC".

        Returns:
            None or outputfilename
                None indicates failure to read the data.
                an outputfile indicates conversion success.
        """
        assert keywords is not None, "List of keywords must be specified"
        assert experimenter is not None, "List of experimenters/authors must be specified"
        assert (
            experiment_description is not None
        ), "Experiment Description (grant number) must be specified"
        assert iacuc_protocol is not None, "IACUC protocol must be specified"
        assert output_path is not None, "Output path must be specified"
        rig_id = self.find_name_in_path(path_to_cell, "Rig")
        expt_id = self.find_name_in_path(path_to_cell, experiment_name)
        if outfilename is None:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            print(f"Output path {output_path} is not an existing directory, creating it.")
            outfilename = Path(
                output_path,
                self._get_short_name(path_to_cell, expt_id, rig_id) + ".nwb",
            )

        print("NWB filename: ", outfilename)
        # Read the acq4 index metadata for the day info, the current slice, and the current cell.
        # assumes that the currdir is the cell directory.
        info = self.AR.readDirIndex(currdir=path_to_cell.parent.parent)["."]
        self.day_index = info
        slice_index = self.AR.readDirIndex(currdir=path_to_cell.parent)["."]
        cell_index = self.AR.readDirIndex(currdir=path_to_cell)["."]

        # We should check for ".mosaic" files in the slice index. If these are present,
        # they are a pyqtgraph configuration file (renders as a dictionary)
        # that drives the acq4 mosaic module to "process" the image data.
        # The mosaic module takes the dictionary, and from the raw image
        # data, fiduciary points, and some other settings, generates a
        # an image that can be a reconstruction of the experiment.
        #
        # If there is afile with the same name as the mosaic file, then
        # is an image of the processed images etc., and we will add it to
        # the repsitory as a single RGB image.

        data_date = datetime.date.fromtimestamp(info["__timestamp__"])

        # get the cell file metadata and clean up the representation.
        age = self.ISO8601_age(info.get("age", None))
        if "sex" not in info.keys() or info["sex"] == "":
            info["sex"] = "U"
        else:
            info["sex"] = info["sex"].upper()
        if "weight" not in info.keys():
            info["weight"] = None
        if "species" not in info.keys() or info["species"] == "":
            info["species"] = "Mus musculus"
        dob = None
        if age is not None and len(age) > 2:
            r_age = re_pnd.match(age)
            if r_age is not None:
                if r_age.group(1) is not None:
                    age = int(r_age.group(1))

                if int(age) < 1200:
                    dob = datetime.datetime.combine(
                        data_date - datetime.timedelta(days=int(age)),
                        datetime.time(),
                        tzlocal(),
                    )
        # dobstr = dob.strftime("%d/%m/%Y%Z")
        if "mouse" in info["species"] or "Mouse" in info["species"]:
            info["species"] = "Mus musculus"
        subject_id = info.get("animal identifier", None)
        if subject_id is None:
            subject_id = info.get("animal_identifier", None)
        if subject_id is None:
            subject_id = "NO ID"
        dset = Path(path_to_cell).parts
        if subject_id.strip() in [None, "?", "", "NA"]:
            subject_id = str(Path(dset[-3])).split("_")[0]  # get the day from the path
        session_id = self._get_short_name(path_to_cell, rig_id=rig_id, expt_id=expt_id)

        if "type 1" not in list(cell_index.keys()):
            ctypes = "Not specified"
        else:
            ctypes = f"{cell_index['type 1']:s} and {cell_index['type 2']:s}"

        if "notes" not in info.keys():
            info["notes"] = "  No Notes"

        # populate the NWB subject object from the acq4 metadata

        subject = NWB.file.Subject(
            age=self.ISO8601_age(info.get("age", "P0D")),
            description=info["strain"],
            genotype=info.get("genotype", "Unknown"),
            sex=info.get("sex", "U"),
            species=info.get("species", "Unknown"),
            subject_id=subject_id,
            weight=info.get("weight", 0.0),
            date_of_birth=dob,
        )

        # Populate the NWB amplifier (device) object
        device = NWB.device.Device(
            name="MC700B",
            description="Current and Voltage clamp amplifier",
            manufacturer="Axon Instruments (Molecular Devices)",
        )

        # check the acq4 day description field - if it is empty, populate with a default
        if "description" not in info.keys():
            info["description"] = "No description provided"

        # populate the NWB metadata
        self.NWBFile = NWB.NWBFile(
            identifier=str(uuid.uuid4()),  # random uuid for this dataset.
            session_start_time=datetime.datetime.fromtimestamp(info["__timestamp__"], tz=tzlocal()),
            session_id=f"{session_id:s}",
            session_description=info["description"],
            keywords=keywords,
            notes=f"Cell Type: {ctypes:s}\n" + info["notes"],
            protocol=iacuc_protocol,  # str(path_to_cell.name),
            timestamps_reference_time=datetime.datetime.now(tzlocal()),
            experimenter=experimenter,
            lab="Manis Lab",
            institution="UNC Chapel Hill",
            experiment_description=experiment_description,  # grant.
            subject=subject,
        )

        self.NWBFile.add_device(device)
        self.sweep_counter = 0  # cumulative counter of traces/sweeps that are stored in this file.

        # In acq4, data is held in subdirectories, one for each "protocol" that was run.
        # Protocols are named according to their 'function' (the type of manipulation/recording
        # that is done), and have an appended number indicating when they are repeated (e.g., _002).
        # Note the the appended numbers may not be sequential - when protocols are stopped prematurely,
        # the protocol is excluded from further analysis, and so is not included here.

        # Now build data structures according to recording mode for each protocol
        print("protocols: ", protocols)

        for protonum, protocol in enumerate(protocols):
            print("Protocol: ", protonum, protocol)
            protocol = protocol.strip()
            path_to_protocol = Path(path_to_cell, protocol)
            print(
                f"    Datafile: {path_to_protocol!s}\n{' '*14:s}Exists: {path_to_protocol.is_dir()!s}"
            )
            # get the protocol data set
            self.AR.setProtocol(Path(path_to_cell, protocol))
            proto_index = self.AR.readDirIndex(currdir=path_to_protocol)["."]

            recordingmode = proto_index["devices"]["MultiClamp1"]["mode"]
            assert recordingmode in ["IC", "I=0", "CC", "VC"]

            match recordingmode:
                case "IC" | "I=0" | "CC":
                    self.get_CC_data(
                        path_to_cell=Path(path_to_cell),
                        protocol=protocol,
                        records=records[protonum],
                        info=info,
                        device=device,
                        downsample=downsample,
                        low_pass_filter=low_pass_filter,
                    )
                case "VC":
                    self.get_VC_data(
                        path_to_cell=Path(path_to_cell),
                        protocol=protocol,
                        records=records[protonum],
                        info=info,
                        device=device,
                        downsample=downsample,
                        low_pass_filter=low_pass_filter,
                    )
                case _:
                    print(f"Recording mode {recordingmode:s} is not implemented")

        self.NWBFile.generate_new_id()
        with NWB.NWBHDF5IO(str(outfilename), "w") as io:
            io.write(self.NWBFile)
        return outfilename

    def get_one_recording(
        self,
        recording_mode: str,
        series_description: str,
        path_to_cell: Path,
        protocol: str,
        records: Union[list, str],
        info: dict,
        device: NWB.device,
        acq4_source_name: str = "MultiClamp1.ma",
        electrode: int = 1,
        downsample: int = 1,
        low_pass_filter: Union[float, None] = None,
    ):
        """Make NWB intracellular electrophysiology tables from a
        single acq4 current clamp or voltage clamp protocol

        Args:
            recording_mode: (str): Either CC or VC for current or voltage clamp
            series_description (str): Name for this  series
            path_to_cell (Path): full path of the data, from which the protocol is pulled
            info (dict): acq4 "info" block provided with this protocol
            device (NWB.device): NWB device structure
            acq4_source_name (str, optional): name of device in acq4. Defaults to "MultiClamp1.ma".
            electrode (int, optional): # of recording electrode. Defaults to 1.
        """

        match recording_mode:
            case "CC":
                stim_name = f"{protocol:s}~Ics{electrode:d}"
                rec_name = f"{protocol:s}~Vcs{electrode:d}"
                opto_name = f"{protocol:s}~OptoSeries"
                stim_units = "amperes"
                rec_units = "volts"

            case "VC":
                stim_name = f"{protocol:s}~Vcs{electrode:d}"
                rec_name = f"{protocol:s}~Ics{electrode:d}"
                opto_name = f"{protocol:s}~OptoSeries"
                stim_units = "volts"
                rec_units = "amperes"

        self.AR.setDataName(acq4_source_name)
        if records is "All":
            allow_partial = False
            record_list = []  # this gets ALL records in the protocol.
        elif records == "None":
            return None
        elif isinstance(records, list):
            allow_partial = True
            record_list = records  # this gets ONLY the records specified in the list.
        else:
            raise ValueError(
                f"Records must be 'All', 'None', or a list of records, not {records!s}"
            )
        print("record list: ", record_list, allow_partial)
        # print(f"\nLooking for a recording from {acq4_source_name:s}")
        dataok = self.AR.getData(
            check=True, allow_partial=allow_partial, record_list=record_list
        )  # just check, don't read raw arrays
        if not dataok:
            return None  # probably wrong channel?
        dataok = self.AR.getData(
            allow_partial=allow_partial, record_list=record_list
        )  # read the data arrays
        if not dataok:
            print("Error reading data: ", path_to_cell)
            return None

        slicen, celln = self._get_slice_cell(f=path_to_cell)
        # print(f"    Slice: {slicen:s}, Cell: {cell:s}, Protocol: {protocol:s}")
        # exit()
        datainfo = None
        for d in self.AR.clampInfo["dirs"]:
            dinfo = self.AR.getDataInfo(Path(d, self.AR.dataname, allow_partial=allow_partial))
            if dinfo is not None:
                datainfo = dinfo
                break

        # make the electrode
        elec1 = NWB.icephys.IntracellularElectrode(
            name=rec_name,
            description=f"Sutter 1.5 mm patch as intracellular electrode: {self.day_index.get('internal', 'unknown'):s}",
            cell_id=f"slice_{slicen:03d}/cell_{celln:03d}",
            filtering=str(datainfo[1]["ClampState"]["ClampParams"]["PrimarySignalLPF"]),
            device=device,
        )
        # self.NWBFile.add_icephys_electrode(elec1)  # not well documented!

        step_times = [
            np.uint64(self.AR.tstart * self.AR.samp_rate),
            np.uint64(self.AR.tend * self.AR.samp_rate),
        ]

        recordings = []  # keep a list of the recordings

        # print("    # sweeps: ", self.AR.cmd_wave.shape[0])
        # print(f"{self.AR.sample_rate[0]:.1f} samples/sec, LPF at {low_pass_filter:.1f}, downsampled by {downsample:d}")
        for sweepno in range(self.AR.cmd_wave.shape[0]):
            cmd_data = np.array(self.AR.cmd_wave[sweepno])
            cmd_data = cmd_data[::downsample].astype(np.float32)  # downsample without filtering andnsure float32 for NWB
           # print(self.AR.cmd_wave.shape, "command wave shape")
            match recording_mode:
                case "CC":
                    vm_data = np.array(self.AR.data_array[sweepno]).astype(np.float32)  # downsample without filtering and ensure float32 for NWB
                    if low_pass_filter is not None:
                        vm_data = UTIL.SignalFilter_LPFBessel(
                            signal=vm_data,
                            LPF=low_pass_filter,
                            samplefreq=self.AR.sample_rate[0],
                            NPole=8,
                            bidir=True,
                            reduce=False,
                        )
                    if downsample > 1:
                        vm0 = np.mean(vm_data)  # remove the DC offset before decimating
                        vm_data = scipy.signal.decimate(vm_data-vm0, downsample, ftype="fir", zero_phase=True, n=4, axis=0).astype(np.float32)
                        vm_data += vm0  # now add DC offset back
                    istim = NWB.icephys.CurrentClampStimulusSeries(
                        name=f"{stim_name:s}_{sweepno:d}",  # :sweep{np.uint32(sweepno):d}:protonum{np.uint32(protonum):d}",
                        data=cmd_data,
                        electrode=elec1,
                        gain=datainfo[1]["ClampState"]["extCmdScale"],  # from acq4
                        stimulus_description=str(self.AR.protocol.name),
                        description="Control values are the current injection step times in seconds",
                        control=step_times,
                        control_description=["tstart", "tend"],
                        comments=recording_mode,
                        unit=stim_units,
                        starting_time=info["__timestamp__"],
                        rate=self.AR.sample_rate[0] / float(downsample),
                        sweep_number=np.uint32(sweepno),
                    )

                    vdata = NWB.icephys.CurrentClampSeries(
                        name=f"{rec_name:s}_{sweepno:d}",  # sweep{np.uint32(sweepno):d}:protonum{np.uint32(protonum):d}",
                        description=series_description,
                        data=vm_data,
                        unit=rec_units,
                        electrode=elec1,
                        gain=datainfo[1]["ClampState"]["primaryGain"],
                        bias_current=datainfo[1]["ClampState"]["ClampParams"]["Holding"],
                        bridge_balance=datainfo[1]["ClampState"]["ClampParams"]["BridgeBalResist"],
                        capacitance_compensation=datainfo[1]["ClampState"]["ClampParams"][
                            "NeutralizationCap"
                        ],
                        stimulus_description="Current Steps",
                        conversion=1.0,
                        timestamps=None,
                        starting_time=info["__timestamp__"],
                        rate=self.AR.sample_rate[0] / float(downsample),
                        comments=recording_mode,
                        control=None,
                        control_description=None,
                        sweep_number=np.uint32(sweepno),
                    )

                    IR_sweep_index = self.NWBFile.add_intracellular_recording(
                        electrode=elec1,
                        response=vdata,
                        stimulus=istim,
                        id=sweepno + self.sweep_counter,
                    )

                    odata, osite = self.capture_optical_stimulation(
                        path_to_cell=path_to_cell,
                        protocol=protocol,
                        recording_mode=recording_mode,
                        sweepno=sweepno,
                        electrode=elec1,
                    )
                    if odata is not None:
                        self.NWBFile.add_intracellular_recording(
                            electrode=elec1,
                            response=vdata,
                            stimulus=odata,
                            id=sweepno + self.sweep_counter + 10000,
                        )
                        self.NWBFile.add_ogen_site(osite)
                        ogenseries = NWB.ogen.OptogeneticSeries(
                            name=f"{opto_name:s}_{sweepno:d}",
                            description="Optogenetic stimulation",
                            data=[70.0],
                            site=osite,
                            rate=1.0,
                        )
                        self.NWBFile.add_stimulus(ogenseries)

                case "VC":
                    vstim = NWB.icephys.VoltageClampStimulusSeries(
                        name=f"{stim_name:s}_{sweepno:d}",
                        stimulus_description=str(self.AR.protocol.name),
                        description=series_description,
                        control=step_times,
                        control_description=["tstart", "tend"],
                        comments=recording_mode,
                        data=cmd_data, # np.array(self.AR.cmd_wave[sweepno, ::downsample]),  # stim_units,
                        starting_time=info["__timestamp__"],
                        rate=self.AR.sample_rate[0] / float(downsample),
                        electrode=elec1,
                        gain=datainfo[1]["ClampState"]["extCmdScale"],
                        sweep_number=np.uint32(sweepno),
                    )
                    cparams = datainfo[1]["ClampState"]["ClampParams"]
                    im_data = np.array(self.AR.data_array[sweepno]).astype(np.float32)  # downsample without filtering and ensure float32 for NWB
                    if low_pass_filter is not None:
                        im_data = UTIL.SignalFilter_LPFBessel(
                            signal=im_data,
                            LPF=low_pass_filter,
                            samplefreq=self.AR.sample_rate[0],
                            NPole=8,
                            bidir=True,
                            reduce=False,
                        )
                    if downsample > 1:
                        im0 = np.mean(im_data)  # remove the DC offset before decimating
                        im_data = scipy.signal.decimate(im_data, downsample, ftype="fir", zero_phase=True, n=4).astype(np.float32)
                        im_data += im0  # now add DC offset back
                    idata = NWB.icephys.VoltageClampSeries(
                        name=f"{rec_name:s}_{sweepno:d}",
                        description=series_description,
                        data=im_data, # self.AR.data_array[sweepno, ::downsample],
                        unit=rec_units,
                        electrode=elec1,
                        gain=datainfo[1]["ClampState"]["primaryGain"],
                        stimulus_description=str(path_to_cell.name),
                        capacitance_fast=cparams["FastCompCap"],
                        capacitance_slow=cparams["SlowCompCap"],
                        resistance_comp_bandwidth=cparams["RsCompBandwidth"],
                        resistance_comp_correction=cparams["RsCompCorrection"],
                        resistance_comp_prediction=0.0,  # not recorded in acq4, we rarely use this.
                        whole_cell_capacitance_comp=cparams["WholeCellCompCap"],
                        whole_cell_series_resistance_comp=cparams["WholeCellCompResist"],
                        resolution=np.NaN,
                        conversion=1.0,
                        timestamps=None,
                        starting_time=info["__timestamp__"],
                        rate=self.AR.sample_rate[0] / float(downsample),
                        comments=recording_mode,
                        control=None,
                        control_description=None,
                        sweep_number=np.uint32(sweepno),
                        offset=datainfo[1]["ClampState"]["holding"],
                    )

                    IR_sweep_index = self.NWBFile.add_intracellular_recording(
                        electrode=elec1,
                        stimulus=vstim,
                        response=idata,
                        id=sweepno + self.sweep_counter,
                    )
                    odata, osite = self.capture_optical_stimulation(
                        path_to_cell=path_to_cell,
                        protocol=protocol,
                        recording_mode=recording_mode,
                        sweepno=sweepno,
                        electrode=elec1,
                    )
                    if odata is not None:
                        self.NWBFile.add_intracellular_recording(
                            electrode=elec1,
                            response=idata,
                            stimulus=odata,
                            id=sweepno + self.sweep_counter + 10000,
                        )
                        self.NWBFile.add_ogen_site(osite)
                        ogenseries = NWB.ogen.OptogeneticSeries(
                            name=f"{opto_name:s}_{sweepno:d}",
                            description="Optogenetic stimulation",
                            data=[70.0],
                            site=osite,
                            rate=1.0,
                        )
                        self.NWBFile.add_stimulus(ogenseries)

                case "I=0":
                    print(f"Recording mode {recording_mode:s} not implemented")
                    return None
            recordings.append(IR_sweep_index)

        self.sweep_counter += sweepno + 1
        IR_simultaneous_index = self.NWBFile.add_icephys_simultaneous_recording(
            recordings=recordings,
        )

        # make the sequential group (we only have one "simultaneous" recording)
        IR_sequence_index = self.NWBFile.add_icephys_sequential_recording(
            simultaneous_recordings=[IR_simultaneous_index],
            stimulus_type="square",
        )
        IR_run_index = self.NWBFile.add_icephys_repetition(
            sequential_recordings=[IR_sequence_index]
        )
        self.NWBFile.add_icephys_experimental_condition(repetitions=[IR_run_index])

    def get_CC_data(
        self,
        path_to_cell: Path,
        protocol: str,
        records: list,
        info: dict,
        device: NWB.device,
        downsample: int = 1,
        low_pass_filter: Union[float, None] = None,
    ):
        """Check if a protocol is a one of our known
        current-clamp protocols, and if so, put the data into the
        NWB format

        This adds the cc protocol information directly to the self.NWBFile object.

        Args:
            path_to_cell: Path, : full path to the protocol
            info: dict, : acq4 metadata dictionary
            device: NWB.device
        Returns:
            Nothing
        """
        protoname = protocol
        series_description_description = None
        if protoname.startswith("CCIV"):
            series_description_description = "Current Clamp IV series"
        elif protoname.startswith("Ic_LED"):
            series_description_description = (
                "Current Clamp with LED Illumination for optical stimulation"
            )
        elif protoname.startswith("Map_NewBlueLaser_IC"):
            series_description_description = "Current Clamp with Laser scanning photostimulation"
        if series_description_description is None:
            return None
        for ampchannel in self.ampchannels:
            self.get_one_recording(
                recording_mode="CC",
                path_to_cell=path_to_cell,
                series_description=series_description_description,
                protocol=protocol,
                records=records,
                info=info,
                device=device,
                acq4_source_name=f"MultiClamp{ampchannel:d}.ma",
                electrode=ampchannel,
                downsample=downsample,
                low_pass_filter=low_pass_filter,
            )

    def get_VC_data(
        self,
        path_to_cell: Path,
        protocol: str,
        info: dict,
        device: NWB.device,
        protonum: int,
        records: Union[list, int] = -1,
        downsample: int = 1,
        low_pass_filter: Union[float, None] = None,
    ):
        """Check if a protocol is a one of our known
        voltage-clamp protocols, and if so, put the data into the
        NWB format

        This adds the vc protocol information directly to the self.NWBFile object.

         Args:
            path_to_cell: Path, : full path to the protocol
            info: dict, : acq4 metadata dictionary
            device: NWB.device
        Returns:
            Nothing
        """

        vcseries_description = None
        if protocol.startswith("VCIV"):
            vcseries_description = "Voltage Clamp series"
        elif protocol.startswith(("Vc_LED", "VC_LED")):
            vcseries_description = "Voltage Clamp with LED Illumination for optical stimulation"
        elif protocol.startswith("Map_NewBlueLaser_VC"):
            vcseries_description = "Voltage Clamp with Laser scanning photostimulation"
        if vcseries_description is None:
            return None

        for ampchannel in self.ampchannels:
            self.get_one_recording(
                recording_mode="VC",
                series_description=vcseries_description,
                path_to_cell=path_to_cell,
                protocol=protocol,
                info=info,
                device=device,
                acq4_source_name=f"MultiClamp{ampchannel:d}.ma",
                electrode=ampchannel,
                protonum=protonum,
                records=records,
                downsample=downsample,
                low_pass_filter=low_pass_filter,
            )

    def capture_optical_stimulation(
        self,
        path_to_cell: Path,
        protocol: str,
        recording_mode: str = "VC",
        sweepno: int = 0,
        electrode: str = "None",
    ):
        # capture optical stimulation information as well

        # The optical stimulation data is held as a VC stimulus series
        # This should be changed in the future
        # The "control" element holds a 2-d array of sites for laser scanning
        # (sorry, the whole array is repeated in every trial).
        # nwb wants the control data to be in integer format, so we scale to microns

        odata = None
        osite = None
        #  laser scanning:
        if self.AR.getLaserBlueCommand():
            if self.laser_device is None:
                self.laser_device = self.NWBFile.create_device(
                    name="Oxxius Laser", description="450 nm 70 mW single-photon Laser"
                )
            scinfo = self.AR.getScannerPositions()
            sites = [self.AR.scanner_info[spot]["pos"] for spot in self.AR.scanner_info.keys()]
            for i, site in enumerate(sites):
                sites[i] = (
                    np.int32(sites[i][0] * 1e6),
                    np.int32(sites[i][1] * 1e6),
                )
                # sites = map(sites, lambda x: (np.int32(x[0]*1e6), np.int32(x[1]*1e6)))
            # sites = str(sites)  # must convert to string to pass nwb tests.
            location_column = VectorData(
                name="sites",
                data=sites,
                description="x, y positions in microns for laser scanning photostimulation",
            )
            control_description = "Laser scanning (spot) photostimulation positions (in microns)"
            light_source = "450 nm 70 mW Laser"

            if recording_mode == "VC":
                odata = NWB.icephys.VoltageClampStimulusSeries(
                    name=f"{protocol:s}:opto_{sweepno:d}",
                    description=f"Optical stimulation control waveform and xy positions for {light_source:s}",
                    data=self.AR.LaserBlue_pCell,
                    rate=self.AR.LaserBlue_sample_rate[0],
                    electrode=electrode,
                    gain=1.0,
                    stimulus_description=str(path_to_cell.name),
                    conversion=1.0,
                    # control=sites,
                    control_description=control_description,
                    comments=f"{self.AR.scanner_spotsize:e}",
                    unit="volts",
                )

            if recording_mode == "CC":
                odata = NWB.icephys.CurrentClampStimulusSeries(
                    name=f"{protocol:s}:opto_{sweepno:d}",
                    description=f"Optical stimulation control waveform and xy positions for {light_source:s}",
                    data=self.AR.LaserBlue_pCell,
                    rate=self.AR.LaserBlue_sample_rate[0],
                    electrode=electrode,
                    gain=1.0,
                    stimulus_description=str(path_to_cell.name),
                    conversion=1.0,
                    # control=sites,
                    control_description=control_description,
                    comments=f"{self.AR.scanner_spotsize:e}",
                    unit="amperes",
                )
            osite = NWB.ogen.OptogeneticStimulusSite(
                name=f"{protocol:s}:sweep={sweepno:d}",
                device=self.laser_device,
                description="Laser Scanning Photostimulation site",
                excitation_lambda=450.0,
                location=str(sites[sweepno]),
            )
        # widefiled LED:
        if self.AR.getLEDCommand():
            light_source = "470 nm Thorlabs LED"
            if self.LED_device is None:
                self.LED_device = self.NWBFile.create_device(
                    name=f"{light_source:s}", description="Widefield LED"
                )
            sites = [[0, 0]] * np.array(self.AR.LED_Raw).shape[
                0
            ]  # also possible to get fov from acq4...
            spotsize = 1e-4  # 100 microns.
            control_description = "Widefield LED through objective"

            if recording_mode == "VC":
                odata = NWB.icephys.VoltageClampStimulusSeries(
                    name=f"{protocol:s}:opto_{sweepno:d}",
                    description=f"Optical stimulation waveform for {light_source:s}",
                    data=np.array(self.AR.LED_Raw)[:, sweepno],  # assuming is constant...
                    rate=self.AR.LED_sample_rate[0],
                    electrode=electrode,
                    gain=1.0,
                    stimulus_description=str(path_to_cell.name),
                    conversion=1.0,
                    # control=sites,
                    control_description=control_description,
                    comments=f"Spotsize: {spotsize:e}",
                    unit="volts",
                )
            if recording_mode == "CC":
                odata = NWB.icephys.CurrentClampStimulusSeries(
                    name=f"{protocol:s}:opto_{sweepno:d}",
                    description=f"Optical stimulation waveform",
                    data=np.array(self.AR.LED_Raw)[:, sweepno],  # assuming is constant...
                    rate=self.AR.LED_sample_rate[0],
                    electrode=electrode,
                    gain=1.0,
                    stimulus_description=str(path_to_cell.name),
                    conversion=1.0,
                    # control=sites,
                    control_description=control_description,
                    comments=f"Spotsize: {spotsize:e}",
                    unit="amperes",
                )

            osite = NWB.ogen.OptogeneticStimulusSite(
                name=f"{protocol:s}:optosite_{sweepno:d}",
                device=self.LED_device,
                description="Widefield optical stimulation",
                excitation_lambda=470.0,
                location=str(sites[sweepno]),
            )
        return odata, osite


# Read the data back in
def validate(testpath, nwbfile):
    """validate Compare the data in memory with what has been written to the disk file

    Parameters
    ----------
    testpath : path to the data file on disk
        _description_
    nwbfile : nwb structure in memory
        _description_
    """
    with NWBHDF5IO(path=testpath, mode="r") as io:
        infile = io.read()

        # assert intracellular_recordings
        assert np.all(
            infile.intracellular_recordings.id[:] == nwbfile.intracellular_recordings.id[:]
        )

        # Assert that the ids and the VectorData, VectorIndex, and table target of the
        # recordings column of the Sweeps table are correct
        assert np.all(
            infile.icephys_simultaneous_recordings.id[:]
            == nwbfile.icephys_simultaneous_recordings.id[:]
        )
        assert np.all(
            infile.icephys_simultaneous_recordings["recordings"].target.data[:]
            == nwbfile.icephys_simultaneous_recordings["recordings"].target.data[:]
        )
        assert np.all(
            infile.icephys_simultaneous_recordings["recordings"].data[:]
            == nwbfile.icephys_simultaneous_recordings["recordings"].data[:]
        )
        assert (
            infile.icephys_simultaneous_recordings["recordings"].target.table.name
            == nwbfile.icephys_simultaneous_recordings["recordings"].target.table.name
        )

        # Assert that the ids and the VectorData, VectorIndex, and table target of the simultaneous
        #  recordings column of the SweepSequences table are correct
        assert np.all(
            infile.icephys_sequential_recordings.id[:]
            == nwbfile.icephys_sequential_recordings.id[:]
        )
        assert np.all(
            infile.icephys_sequential_recordings["simultaneous_recordings"].target.data[:]
            == nwbfile.icephys_sequential_recordings["simultaneous_recordings"].target.data[:]
        )
        assert np.all(
            infile.icephys_sequential_recordings["simultaneous_recordings"].data[:]
            == nwbfile.icephys_sequential_recordings["simultaneous_recordings"].data[:]
        )
        assert (
            infile.icephys_sequential_recordings["simultaneous_recordings"].target.table.name
            == nwbfile.icephys_sequential_recordings["simultaneous_recordings"].target.table.name
        )

        # Assert that the ids and the VectorData, VectorIndex, and table target of the
        # sequential_recordings column of the Repetitions table are correct
        assert np.all(infile.icephys_repetitions.id[:] == nwbfile.icephys_repetitions.id[:])
        assert np.all(
            infile.icephys_repetitions["sequential_recordings"].target.data[:]
            == nwbfile.icephys_repetitions["sequential_recordings"].target.data[:]
        )
        assert np.all(
            infile.icephys_repetitions["sequential_recordings"].data[:]
            == nwbfile.icephys_repetitions["sequential_recordings"].data[:]
        )
        assert (
            infile.icephys_repetitions["sequential_recordings"].target.table.name
            == nwbfile.icephys_repetitions["sequential_recordings"].target.table.name
        )

        # Assert that the ids and the VectorData, VectorIndex, and table target of the
        # repetitions column of the Conditions table are correct
        assert np.all(
            infile.icephys_experimental_conditions.id[:]
            == nwbfile.icephys_experimental_conditions.id[:]
        )
        assert np.all(
            infile.icephys_experimental_conditions["repetitions"].target.data[:]
            == nwbfile.icephys_experimental_conditions["repetitions"].target.data[:]
        )
        assert np.all(
            infile.icephys_experimental_conditions["repetitions"].data[:]
            == nwbfile.icephys_experimental_conditions["repetitions"].data[:]
        )
        assert (
            infile.icephys_experimental_conditions["repetitions"].target.table.name
            == nwbfile.icephys_experimental_conditions["repetitions"].target.table.name
        )
        assert np.all(
            infile.icephys_experimental_conditions["tag"][:]
            == nwbfile.icephys_experimental_conditions["tag"][:]
        )


def ConvertFile(
    experiment_name: str,
    filename: Union[str, Path],
    protocols: list,
    records: list,
    appendmode: bool = False,
    output_path: Union[str, Path] = None,
    device="MultiClamp1.ma",
    mode="CC",
    downsample: int = 1,
    low_pass_filter: Union[float, None] = None,
    keywords: Union[list, None] = None,
    experimenter: Union[list, None] = None,
    iacuc_protocol: Union[str, None] = None,
    experiment_description: Union[str, None] = None,
):
    # experiment_name = None
    assert output_path is not None
    A2N = ACQ4toNWB(output_path=output_path)
    A2N.set_amplifier_name(device)
    NWBFile = A2N.acq4tonwb(
        experiment_name=experiment_name,
        path_to_cell=filename,
        output_path=output_path,
        protocols=protocols,
        records=records,
        appendmode=appendmode,
        recordingmode=mode,
        downsample=downsample,
        low_pass_filter=low_pass_filter,
        keywords=keywords,
        experimenter=experimenter,
        iacuc_protocol=iacuc_protocol,
        experiment_description=experiment_description,
    )
    print("NWB file written:  ", NWBFile)
    return NWBFile


def select(protocols: list):
    ok_prots = []
    for protocol in protocols:
        if (
            protocol.startswith("CCIV")
            or protocol.startswith("Map")
            or protocol.startswith("Vc_LED")
            or protocol.startswith("Ic_LED")
            or protocol.startswith("VC_LED")
        ):
            ok_prots.append(protocol)
    return ok_prots


def cvt(
    row,
    experiment_name,
    output_path: Union[str, Path] = None,
    rigs: Union[Tuple, List] = None,
    row_num: int = 0,
    keywords: Union[str, None] = None,
):
    """Do the file conversion for each complete data set in a given database row

    Args:
        row (Pandas row (series)): A row containing information about the day
    """

    path = row.data_directory
    day = row.date
    if rigs is not None:  # restrict by rig
        rig_ok = False
        for rig in rigs:
            if day.startswith(rig):  # only do data from a given rig.
                rig_ok = True
        if not rig_ok:
            print(f"Rig ok NOT: {rig:s} in {rigs!s}")
            return None, None  # did not match rigs
    slice = row.slice_slice
    cell = row.cell_cell
    protocols = row.data_complete.split(",")
    protocols = [p.strip() for p in protocols if p.strip() not in [" ", "", None]]
    protocols = select(protocols)
    if len(protocols) == 0:
        print(f"{day}{slice}{cell} No protocols:")
        return None, None
    print("\nCell id: ", row.cell_id, " index: ", row_num)
    print("     Protocols: ", protocols)
    filename = Path(path, day, slice, cell)
    NWBFile = ConvertFile(
        experiment_name=experiment_name,
        filename=filename,
        output_path=output_path,
        protocols=protocols,
        reecords=[None] * len(protocols),
    )
    # NWBFiles.append(NWBFile)
    return NWBFile, filename


def HK_output(output_path: Union[str, Path], experiment_name: str):
    """Read the pandas database for this set of experiments.
    The pandas database is generated by dataSummary.py (from github.com/pbmanis/ephys/util).
    This database is a complete record of the experiments, not filtered by any criteria.
    Each row of the database corresponds to one day's recordings - usually one subject
    The column "data_complete" holds the list of protocols that were completed and are
    potentially suitable for further analysis.
    """
    import pandas as pd

    HK_db = pd.read_pickle(
        # "/Users/pbmanis/Desktop/Python/HK_Collab/datasets/HK_fastcortex_cortex.pkl"
        "/Users/pbmanis/Desktop/Python/HK_Collab/datasets/DCN_IC_inj/DCN_IC_inj.pkl"
        # "/Users/pbmanis/Desktop/Python/HK_Collab/datasets/Thalamocortical/HK_TC_datasummary-2024.02.10.pkl"
        # "/Users/pbmanis/Desktop/Python/HK_Collab/datasets/HK_CRH-Cre/HK_CRH-Cre_datasummary-2025.04.21.pkl"
    )
    print("Converting acq4 to NWB from database: \n", HK_db.cell_id)

    NWBFiles = []
    files = []
    #             results: dict = {}
    #             tasks = dict(zip(range(len(validivs)), validivs))
    #             result = [None] * len(tasks)
    # nworkers = 12
    # with concurrent.futures.ProcessPoolExecutor(max_workers=nworkers) as executor:
    #     print("   Submitting execution to concurrent futures")
    #     futures = [
    #         executor.submit(
    #             concurrent_iv_analysis,
    #             self,
    #             icell,
    #             i,  # index into the task/validivs list
    #             x,  # the actual valid iv (protocol) run
    #             cell_directory,
    #             validivs,   # I expect that this doesn't need to be passed...
    #             additional_iv_records,
    #             nfiles,
    #         )
    #         for i, x in enumerate(tasks)
    #     ]
    #     for i, future in enumerate(concurrent.futures.as_completed(futures)):
    #         result, nfiles = future.result()
    #         # print(result.keys())
    #         # print(result['protocol'])
    #         results[result['protocol']] = result
    #     if len(results) == 0 or self.dry_run:
    #         return

    keywords = [
        "mouse",
        "intracellular recording",
        "channelrhodopsins",
        "pathway tracing",
        "auditory cortex",
        "auditory thalamus",
        "medial geniculate",
        "inferior colliculus",
        "brachium of the inferior colliculus",
        "cochlear nucleus",
        "dorsal cochlear nucleus",
        "AAV",
        "synaptic potentials",
        "optogenetics",
    ]
    experimenter = [
        "Kasten, Michael R.",
        "Garcia, Michelee",
        "Kline, Amber",
        "Tsukano, Hiroaki",
        "Kato, Hirouki",
        "Acosta, Hailey" "Manis, Paul B.",
    ]
    experiment_description = "R01NS128873"
    iacuc_protocol = "21-123, 24-083"
    for index in HK_db.index:
        # if index < 88:  # just for a test...
        #     continue
        print(HK_db.iloc[index].cell_id)
        nwbfile, filename = cvt(
            HK_db.iloc[index],
            experiment_name,
            output_path=output_path,
            rigs=None,  #
            #     ["Rig2", "Rig4"],
            row_num=index,
        )
        if nwbfile is None:
            continue
        NWBFiles.append(nwbfile)
        files.append(filename)
        print("NWB file: ", filename)
        # print("NWB file: ", nwbfile)
        # validate(nwbfile, nwbfile)


def check_conversion(nwbf):
    try:
        results = list(inspect_nwbfile(nwbfile_path=nwbf))
        if len(results) == 0:
            print("    Conversion OK for file: ", nwbf)
        else:
            print(f"    Error in conversion detected: for file: {nwbf!s} ")
            print(results)
            # exit()
    except:
        print(f"    Error in conversion detected: for file  {nwbf!s} ")
        print(results)
        exit()


def check_conversions(output_path: Union[str, Path], dataset: str = "HK_CRH-Cre"):
    print("=" * 80)
    print(f"Checking conversion for dataset: {dataset:s}")
    NWBFiles = []
    for f in Path(output_path).rglob("*.nwb"):
        if f.name.startswith(dataset):
            NWBFiles.append(f)
    # print("NWBFiles: ", NWBFiles)
    for i, nwbf in enumerate(NWBFiles):
        if nwbf is None:
            continue
        check_conversion(nwbf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert acq4 data to NWB format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        type=str,
        default="001422",
        dest="dataset",
        help="Dataset number (e.g. 001422: CRH_Cre, 000407: Thalamocortical, 001423: CN-IC)",
    )
    parser.add_argument(
        "-c",
        "--convert",
        default=False,
        action="store_true",
        help="Convert the files to nwb format",
    )
    parser.add_argument(
        "-v",
        "--verify",
        default=False,
        action="store_true",
        help="Verify the converted files",
    )
    parser.add_argument(
        "-s",
        "--spotcheck",
        type=str,
        default=None,
        action="store",
        help="Spot check the converted files",
    )
    parser.add_argument(
        "-u",
        "--upload",
        default=False,
        action="store_true",
        help="Upload to the Dandi archive",
    )
    args = parser.parse_args()
    datasets = {
        "001422": "HK_CRH-Cre",
        "000407": "Thalamocortical",
        "001423": "HK_CN_IC",
    }
    id = args.dataset
    if id not in datasets.keys():
        print(f"Dataset {id} not found")
        print(f"Available datasets: {datasets.keys()}")
        exit()

    output_path = Path("/Users/pbmanis/Desktop/Python/Dandi", datasets[id])
    if args.convert:
        main(output_path, experiment_name=datasets[id])  # convert a bunch of files
    if args.verify:
        check_conversions(output_path=output_path, dataset=datasets[id])
    if args.spotcheck is not None:
        check_conversion(Path(output_path, args.spotcheck))  # check the conversion
    if args.upload:
        # upload to dandi
        import subprocess

        local_path = f"{str(Path(output_path.parent, id)):s}"
        print("archive path on local disk: ", local_path)
        print("output path: ", output_path)
        subprocess.call(["dandi", "download", f"https://dandiarchive.org/dandiset/{id:s}/draft"])
        cmd = ["dandi", "organize", f"{output_path!s}", "-f", "dry"]
        subprocess.call(cmd, text=True, cwd=f"{id:s}")  # dry run to see if it works
        subprocess.call(["dandi", "organize", f"{output_path!s}"], cwd=f"{id:s}")
        subprocess.call(["dandi", "validate"], cwd=f"{id:s}")
        subprocess.call(["dandi", "upload"], cwd=f"{id:s}")
