DataTables Workflow
===================

Data Locations:
---------------
The raw data are on the rig computers (Rig 2, Rig4), mostly under Kasten's data directories but also in Manis'. There are 2 backup copies on Pegasus_002 and Pegasus_004 in Manis' office. These may differ slightly as some of the metadata (day, slice, cell, .index files) have been updated at different times. The updated metadata mostly refer to the animal (ID, genotype, weight, age, etc), and the
cell information (location, layer, celltype, expression). This metadata is gathered either from the note fields where it was entered during acquisition, from the drug log, or from notes
about the mice that were sent over from the Moy lab where the behavior was done. see the Workflow section below

**The following infomration is generic to the Datatables interface**

Configuration File:
-------------------

Many of the analysis parameters are set up in th experiments.cfg file, in the config subdirectory. This file is in the pyqtgraph configfile format (used by acq4, for example), an is largely self-explanatory. Levels are defined by the indentation, and the file is validated (to some extent) on reading. The configuration file is meant to hold any and ALL parameters that might vary between different types of experiments, including selection of spike detection methods, thresholds, resting potential boundaries, cell types, plotting scales, the names and locations of ancillary files, etc. Read the 'ephys' manual for a full description (once I write it).

> The analysis should proceed with the dataset in Pegasus_004, as this is the most recently updated copy. The path is set up in the experiments.cfg file under the configuration directory ('rawdatapath').

The analysis results are in the directory datasets/Maness_Ank2_nex. These include summary files as pickled Pandas files, Excel files, and some control files (excel, text formats).
Important files currently are:
>    datasummaryFilename: "ANK2_Nex_IVs_03_2024.pkl"  # name of the dataSummary output file, in analyzeddatapath
>        IVs: "ANK2_Nex_Spikes"  # name of the excel sheet generated to hold the IV results
>        iv_analysisFilename: "ANK2_NexIV_Analysis-03_2024.pkl"  #  "IV_Analysis.h5", # name of the pkl or HDF5 file that holds all of the IV analysis results
>        eventsummaryFilename: "Ank2_NEX_event_summary.pkl"
>        coding_file: "Ank2_NEX_Intrinsics_Codes.xlsx"  # name of the excel file with the coding

> result_sheet: "Ank2_NEX_IVs.xlsx"  # excel sheet with results
>        adddata: "Ank2_NEX_IVs_cleaned.xlsx"  # excel sheet with additional analysis results, but also cleaned
>        pdf_filename: "Ank2_NEX_IVs.pdf"
>        assembled_filename: "Ank2_Nex_IVs_combined_by_cell.pkl"  # name of the dataset assembled in plot_spike_info.... 
>        stats_filename: "Ank2_NEX_IVs_statistical_summary"  # name of the file to write statitiscal summary to. The name will have the "group_by" > appended to it, and will have a .txt suffix.


Workflow:
---------

**First**, make sure that the data are "scored" correctly, by going though each day and making sure that the mouse, strain, genotype, age, sex, etc are all correctly represented and
consistent across the entire dataset (e.g., use a limited and well-defined set of markers). This MUST be doen on the original data set. The key elements are found in the '.index' files under the day directory, slice directory and cell directories. All of the fields may be changed from within acq4 with one exception. That exception is the animal ID, which should be entered when creating the new directory with all of the other animal information. If it is not, then you will need to carefully hand-edit the .index file in the top level day directory with a proper text editor (e.g., *not* Word or Notepad)

If there needs to be more information in the scoring, the config file can be changed to include more elements in the scoring to accomodate this, but this will propagation throughout the analysis. If you change this, you probably should run the entire analysis from scratch, including regenerating the data summary file, and after the IV analysis, updating the assembled file ("assemble IV datasets").  

*** Make sure that the "coding" file (the filename is set in the experiments.cfg file) is updated whenevery you add mice/slices/cells
 to the dataset. EVERY CELL must appear in this file. Failure to do this may make you spend hours looking for a bug in the program, when in reality the maxim "GIGO" (garbage in garbate out) fully applies. ***

IVs:
====

GUI-based
---------

Analyzing IVs using the GUI "datatables" is strongly recommended. The older approach described below will likely not work at all anymore.

1. Set up the configuration file. 
2. Run the 'datatables' command. This should bring up the GUI. The GUI consists of a left pane with parameters and action buttons, and a set of "docked" windows on the right for different outputs from the program. The docks can be selected from the clikcable labels along the top, and they may be "torn" off if you want to view mulitple ones at a time. The "torn" status is not retained across invocations of the program. 
3. If the "datasummary" has been generated and the file is in the right location, the table will be populated. Not all entries in the datasummary file are shown on the screen. 
   1. Note: Double-licking on a cell in the table will show the full contents of the cell if it is clipped. 
   2. Cmd (or Alt- on Windows) will show the current IV data PDF file if the IV data has been analyzed. 
   3. Option (or Win- on Windows) will show the current MAP (laser scanning) PDF file if the MAP data has been analyzed.
4. The "IVTable" dock shows a table of the "assemled" IV data sets with some analysis information. 
5. The "PDF" dock is raised to show the individual analyzed plots. 
6. The other tabs are either not yet used or will be populated as needed.

Use the "datasummary" button to regenerate the data summary file. This file is not editable, but summarizes the metadata from the original data. AN excel version for viewing is also available (but edits to this file are not used!!!!!).
After the datasummary is run, the current datasummary information will be loaded and shown in the table.

Open the IV analysis widget. Choose the files in the datasummary table to analyze and click "Analyuze IVs". Currently, to select all, use ctrl-A, then analyzie IVs. The "analyze all ivs" is not working.  The analysis can take some time depending on how many files there are. Watch the terminal output for errors. 

Once the IV analyis is complete, click "Assemble Datasets". This is necessary anytime the anaysis is updated, and combines some parts of the analyses into a single file.  This file is necessary for the summary plots to be correct.

Open the "Spike/IV Analysis" tab. Here you have several options for plotting the data, and for breaking the summary data by "group", sex, temperature, etc. The options for this part of the analysis are set in the experiments.cfg file and only those options will appear in the drop-down lists. In this set of actions, you can plot data categorically or continously (e.g., by age), plot all the FI curves, or generate some basic stats. 

The PDF file plots are editable in Adobe Illustrator (and probably Inkscape) in order to generate publication-quality figures. 

**Note on the stats**  The statisitical analysis is rudimentary here. The main output is a csv file in "long" form meant to be read into R scripts. The name of the file is set in the "stats_filename" field in the configuration file. You should look at this file (import into excel, or read with a programming editor that knows how to show csv files, such as TextMate (mac) or VSCode with extensions). 

