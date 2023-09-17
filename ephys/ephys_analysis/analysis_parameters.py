import argparse
import json
from dataclasses import dataclass, field

import toml


def build_parser(experiments):
    parser = argparse.ArgumentParser(
        description="Map and IV data analysis",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "-E",
        "--experiment",
        type=str,
        dest="experiment",
        choices=list(experiments.keys()),
        default="None",
        nargs="?",
        const="None",
        help="Select Experiment to analyze",
    )
    # input options
    parser.add_argument(
        "-D", "--basedir", type=str, dest="basedir", help="Base Directory"
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default="",
        dest="inputFilename",
        help="Specify input dataSummray file name (including full path)",
    )

    parser.add_argument("--celltype", type=str, default="all", help="limit celltype for analysis")
   
    parser.add_argument("-d", "--day", type=str, default="all", help="day for analysis")

    parser.add_argument(
        "-a",
        "--after",
        type=str,
        default="1970.1.1",
        dest="after",
        help="only analyze dates on or after a date",
    )
    parser.add_argument(
        "-b",
        "--before",
        type=str,
        default="2266.1.1",
        dest="before",
        help="only analyze dates on or before a date",
    )
    parser.add_argument(
        "-S",
        "--slice",
        type=str,
        default="",
        dest="slicecell",
        help="select slice/cell for analysis: in format: S0C1 for slice_000 cell_001\n"
        + "or S0 for all cells in slice 0",
    )
    parser.add_argument(
        "-p",
        "--protocol",
        type=str,
        default="",
        dest="protocol",
        help="select protocol for analysis: -requires slice selection \n"
        + "and must be specific (e.g., CCIV_001)",
    )
    parser.add_argument(
        "--extra_subdirectories",
        type=str,
        default=None,
        dest="extra_subdirectories",
        help="List of extra subdirectories",
    )
    parser.add_argument(
        "--configfile",
        type=str,
        default=None,
        dest="configfile",
        help="Read a formatted configuration file (JSON, TOML) for commands",
    )

    # output options
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        dest="pdfFilename",
        help="Specify output PDF filename (full path)",
    )

    parser.add_argument(
        "-A",
        "--auto",
        action="store_true",
        dest="autoout",
        help="automatically name output PDF file as experiment_mo_day_yr_slicecell_celltype.pdf",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        dest="merge_flag",
        help="Attempt to merge analyzed maps for this cell only - no analysis",
    )
    parser.add_argument(
        "-e", "--excel", action="store_true", dest="excel", help="just export to excel"
    )
    # parser.add_argument('-r', '--read-annotations', type=str, default='',
    #                     dest='annotationFile',  # specify in expt dict now,
    #                     not line opetion help='Read an annotation file of
    #                     selected cells, to replace cell type with post-hoc
    #                     definitions')
    parser.add_argument(
        "-u",
        "--update",
        action="store_true",
        dest="update",
        help="Update the .pkl file with analysis output (creates backup)",
    )

    # analysis options
    parser.add_argument(
        "--IV", action="store_true", dest="iv_flag", help="analyze IVs only"
    )
    parser.add_argument(
        "--VC", action="store_true", dest="vc_flag", help="analyze VCIVs only"
    )
    parser.add_argument(
        "-m", "--map", action="store_true", dest="map_flag", help="Analyze maps only"
    )

    # control options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Do a dry run, reporting only directories",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="verbose",
        help="Print verbose output",
    )
    parser.add_argument(
        "--noparallel",
        action="store_true",
        dest="noparallel",
        help="Turn off parallel processing (used primarily for debugging)",
    )
    parser.add_argument(
        "--mapZQA",
        action="store_true",
        dest="mapZQA_plot",
        help="Plot the maps from previously analyzed dataset(s)",
    )
    parser.add_argument(
        "--recalculate",
        action="store_true",
        dest="recalculate_events",
        help="recalculate events files from the specified dataset(s)",
    )

    parser.add_argument(
        "--ivdur",
        type=float,
        default=0.0,
        dest="ivduration",
        help="Only analyze IVs of a particular duration, in seconds",
    )
    parser.add_argument(
        "--spike_threshold",
        type=float,
        default=-0.035,
        dest="spike_threshold",
        help="Set spike_threshold for IVs",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=2.5,
        dest="threshold",
        help="Set threshold for event detection in maps",
    )

    # data treatment/preprocessing options

    parser.add_argument(
        "--plotmode",
        type=str,
        dest="plotmode",
        default="document",
        choices=["document", "publication"],
        help="Plot mode: document or publication",
    )
    parser.add_argument(
        "--pubmode",
        type=str,
        dest="IV_pubmode",
        default="normal",
        choices=["normal", "pubmode", "traces_only"],
        help="clean IV plot",
    )

    parser.add_argument(
        "--plotsoff",
        action="store_true",
        dest="noplot",
        help="suppress plot generation",
    )
    ###
    ### Specific paramters for mapping analysis
    ###
    parser.add_argument(
        "--artfile",
        type=str,
        default="",
        dest="artifactFilename",
        help="Specify artifact file (base name, no extension)",
    )
    # plotting options
    parser.add_argument(
        "--whichstim",
        type=int,
        dest="whichstim",
        default=-1,
        help="define which stimulus to plot (Z scores, I_max, Qr)",
    )
    parser.add_argument(
        "--trsel",
        type=int,
        dest="trsel",
        default=None,
        help="select a trace from the map to plot",
    )

    # analysis parameter options
    parser.add_argument(
        "--signflip",
        action="store_true",
        dest="signflip",
        help="Analyze events of opposite sign from default",
    )
    parser.add_argument(
        "--alt1",
        action="store_true",
        dest="alternate_fit1",
        help="Analyze events with different template than default",
    )
    parser.add_argument(
        "--alt2",
        action="store_true",
        dest="alternate_fit2",
        help="Analyze events with different template than default",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="aj",
        choices=["aj", "cb", "zc"],
        dest="detector",
        help="Set event detector method: ClementsBekkers (cb) or AndradeJonas (aj) or Zerocrossing (zc)",
    )
    parser.add_argument(
        "--measure",
        type=str,
        default="ZScore",
        choices=["ZScore", "Qr", "I_max"],
        dest="measuretype",
        help="Set measure for spot plot (ZScore, Qr, I_max)",
    )
    parser.add_argument(
        "--artifact_suppression",
        action="store_true",
        dest="artifact_suppression",
        help="Turn ON artifact suppression (for stimuli only)",
    )
    parser.add_argument(
        "--artifact_derivative",
        action="store_true",
        dest="artifact_derivative",
        help="Turn off derivative-based artifact suppression (any fast event)",
    )
    parser.add_argument(
        "--post_artifact_suppression",
        action="store_true",
        dest="post_analysis_artifact_rejection",
        help="Turn ON artifact suppression post-analysis",
    )

    parser.add_argument(
        "-n",
        "--notchfilter",
        action="store_true",
        dest="notchfilter",
        help="Enable notch filter (do not use with large events!)",
    )
    parser.add_argument(
        "--LPF",
        type=float,
        default=0.0,
        dest="LPF",
        help="Set Low Pass filter to apply to data",
    )
    parser.add_argument(
        "--HPF",
        type=float,
        default=0.0,
        dest="HPF",
        help="Set High Pass filter to apply to data",
    )
    parser.add_argument(
        "--notchfreqs",
        type=str,
        default="60, 120, 180, 240",
        dest="notchfreqs",
        help="Set notch frequencies (using evaluatable python expression)",
    )
    parser.add_argument(
        "--notchq",
        type=float,
        default=90.0,
        dest="notchQ",
        help="Set notch Q (sharpness of notch; default=90)",
    )

    parser.add_argument(
        "--detrend_method",
        type=str,
        default="meegkit",
        choices=["meegkit", "scipy", "None"],
        dest="detrend_method",
        help="Set method for detrending data. Choices: ['meegkit', 'scipy', 'None']",
    )
    
    parser.add_argument(
        "--detrend_order",
        type=int,
        default=5,
        dest="detrend_order",
        help="Set detrend order for meegkit (default=5)",
    )

    args = parser.parse_args()
    # args = vars(parser.parse_args())
    return args


def getCommands(experiments):
    args = build_parser(experiments)

    if args.configfile is not None:
        config = None
        if args.configfile is not None:
            if ".json" in args.configfile:
                # The escaping of "\t" in the config file is necesarry as
                # otherwise Python will try to treat is as the string escape
                # sequence for ASCII Horizontal Tab when it encounters it during
                # json.load
                config = json.load(open(args.configfile))
                print(f"Reading JSON configuration file: {args.configfile:s}")
            elif ".toml" in args.configfile:
                print(f"Reading TOML configuration file: {args.configfile:s}")
                config = toml.load(open(args.configfile))

        vargs = vars(args)  # reach into the dict to change values in namespace

        for c in config:
            if c in args:
                # print("Getting parser variable: ", c)
                vargs[c] = config[c]
            else:
                raise ValueError(
                    f"config variable {c:s} does not match with comand parser variables"
                )

        print("   ... All configuration file variables read OK")
    # now copy into the Param dataclass if we want to params = Params() parnames
    # = dir(params) for key, value in vargs.items(): if key in parnames: #
    # print('key: ', key) # print(str(value)) exec(f"params.{key:s} =
    # {value!r}") # elif key in runnames: #     exec(f"runinfo.{key:s} =
    #     {value!r}") #
    return args


if __name__ == "__main__":
    getCommands()
