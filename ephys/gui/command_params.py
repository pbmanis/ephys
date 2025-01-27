    """ Command parameters for the GUI

    Parametertree (pyqtgraph) setup for the datatables GUI


    Returns
    -------
    tuple
        parametertree data sets
    """
from pyqtgraph.parametertree import Parameter, ParameterTree

Age_Values = [  # this is just for selecting age ranges in the GUI
    "None",
    [7, 20],
    [21, 49],
    [50, 179],
    [180, 1200],
    [0, 21],
    [21, 28],
    [28, 60],
    [60, 90],
    [90, 182],
    [28, 365],
    [365, 900],
]

RMP_Values = [
    [-80, -50],
    [-70, -50],
]

taum_Values = [0.0005, 0.05]


class CommandParams:
    def __init__(self):
        self.datasets = []
        self.experiment = {}

    def set_datasets(self, datasets):
        self.datasets = datasets

    def set_experiment(self, experiment):
        self.experiment = experiment

    def create_params(self):
        # print(self.experiment)
        self.params = [
            # {"name": "Pick Cell", "type": "list", "values": cellvalues,
            # "value": cellvalues[0]},
            {"name": "Create New DataSet", "type": "action"},
            {
                "name": "Choose Experiment",
                "type": "list",
                "limits": [ds for ds in self.datasets],
                "value": self.datasets[0],
            },
            {"name": "Reload Configuration", "type": "action"},  # probably not needed...
            {"name": "Update DataSummary", "type": "action"},
            {"name": "Load DataSummary", "type": "action"},
            {"name": "Load Assembled Data", "type": "action"},
            {"name": "Save Assembled Data", "type": "action"},
            {
                "name": "Parallel Mode",
                "type": "list",
                "limits": ["cell", "day", "trace", "map", "off"],
                "value": "cell",
            },
            {"name": "Dry run (test)", "type": "bool", "value": False},
            {"name": "Only Analyze Important Flagged Data", "type": "bool", "value": False},
            {
                "name": "IV Analysis",
                "type": "group",
                "expanded": False,
                "children": [
                    {"name": "Analyze Selected IVs", "type": "action"},
                    {"name": "Plot from Selected IVs", "type": "action"},
                    {"name": "Analyze ALL IVs", "type": "action"},
                    {"name": "Analyze ALL IVs m/Important", "type": "action"},
                    # {"name": "Process Spike Data", "type": "action"},
                    {"name": "Assemble IV datasets", "type": "action"},
                    {"name": "Exclude unimportant in assembly", "type": "bool", "value": False},
                ],
            },
            {
                "name": "Map Analysis",
                "type": "group",
                "expanded": False,
                "children": [
                    {"name": "Analyze Selected Maps", "type": "action"},
                    {"name": "Analyze ALL Maps", "type": "action"},
                    # {"name": "Assemble Map datasets", "type": "action"},
                    # {"name": "Plot from Selected Maps", "type": "action"},
                ],
            },
            {
                "name": "Mini Analysis",
                "type": "group",
                "expanded": False,
                "children": [
                    {"name": "Analyze Selected Minis", "type": "action"},
                    {"name": "Analyze ALL Minis", "type": "action"},
                    # {"name": "Assemble Mini datasets", "type": "action"},
                    # {"name": "Plot from Selected Minis", "type": "action"},
                ],
            },
            {
                "name": "Plotting",
                "type": "group",
                "expanded": True,
                "children": [
                    {
                        "name": "Group By",
                        "type": "list",
                        "limits": [gr for gr in self.experiment["group_by"]],
                        "value": self.experiment["group_by"][0],
                    },
                    {
                        "name": "2nd Group By",
                        "type": "list",
                        "limits": [gr for gr in self.experiment["secondary_group_by"]],
                        "value": self.experiment["secondary_group_by"][0],
                    },
                    {"name": "View Cell Data", "type": "action"},
                    {"name": "Use Picker", "type": "bool", "value": False},
                    {"name": "Show PDF on Pick", "type": "bool", "value": False},
                    {
                        "name": "Spike/IV plots",
                        "type": "group",
                        "expanded": False,
                        "children": [
                            {"name": "Plot Spike Data categorical", "type": "action"},
                            {"name": "Plot Spike Data continuous", "type": "action"},
                            {"name": "Plot Rmtau Data categorical", "type": "action"},
                            {"name": "Plot Rmtau Data continuous", "type": "action"},
                            {"name": "Plot FIData Data categorical", "type": "action"},
                            {"name": "Plot FIData Data continuous", "type": "action"},
                            {"name": "Plot FICurves", "type": "action"},
                            {
                                "name": "Set BSpline S",
                                "type": "float",
                                "value": 1.0,
                                "limits": [0.0, 100.0],
                            },
                            {"name": "Plot Selected Spike", "type": "action"},
                            {"name": "Plot Selected FI Fitting", "type": "action"},
                            {"name": "Print Stats on IVs and Spikes", "type": "action"},
                        ],
                    },
                    {
                        "name": "Map Analysis Plots",
                        "type": "group",
                        "expanded": False,
                        "children": [
                            {"name": "Rise/Fall/Amplitude", "type": "action"},
                            {"name": "Spontaneous Amplitudes", "type": "action"},
                            {"name": "Evoked Event Amplitudes", "type": "action"},
                            {"name": "Event Latencies", "type": "action"},
                        ],
                    },
                ],
            },
            {
                "name": "Filters",
                "type": "group",
                "expanded": False,
                "children": [
                    # {"name": "Use Filter", "type": "bool", "value": False},
                    {
                        "name": "cell_type",
                        "type": "list",
                        "limits": [
                            "None",
                            "bushy",
                            "t-stellate",
                            "d-stellate",
                            "octopus",
                            "pyramidal",
                            "cartwheel",
                            "giant",
                            "giant_maybe",
                            "golgi",
                            "glial",
                            "granule",
                            "stellate",
                            "tuberculoventral",
                            "unclassified",
                        ],
                        "value": "None",
                    },
                    {
                        "name": "age",
                        "type": "list",
                        "limits": Age_Values,
                        "value": "None",
                    },
                    {
                        "name": "sex",
                        "type": "list",
                        "limits": ["None", "M", "F"],
                        "value": "None",
                    },
                    {
                        "name": "Group",
                        "type": "list",
                        "limits": ["None", "-/-", "+/+", "+/-"],
                        "value": "None",
                    },
                    {
                        "name": "RMP",
                        "type": "list",
                        "limits": RMP_Values,
                        "value": 0,
                    },
                    {
                        "name": "taum",
                        "type": "list",
                        "limits": taum_Values,
                        "value": "None",
                    },
                    {
                        "name": "PulseDur",
                        "type": "list",
                        "limits": ["None", 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0],
                        "value": "None",
                    },
                    {
                        "name": "Protocol",
                        "type": "list",
                        "limits": [
                            "None",
                            "CCIV_+",
                            "CCIV_1nA",
                            "CCIV_200pA",
                            "CCIV_long",
                            "CCIV_long_HK",
                        ],
                        "value": "None",
                    },
                    {
                        "name": "Filter Actions",
                        "type": "group",
                        "children": [
                            {"name": "Apply", "type": "action"},
                            {"name": "Clear", "type": "action"},
                        ],
                    },
                ],
            },
            #  for plotting figures
            #
            {
                "name": "Figures",
                "type": "group",
                "children": [
                    {
                        "name": "Figures",
                        "type": "list",
                        "limits": [
                            "-------NF107_WT_Ctl-------",
                            "Figure1",
                            "Figure2",
                            "Figure3",
                            "EPSC_taurise_Age",
                            "EPSC_taufall_age",
                            "-------NF107_NIHL--------",
                            "Figure-rmtau",
                            "Figure-spikes",
                            "Figure-firing",
                        ],
                        "value": "-------NF107_WT_Ctl-------",
                    },
                    {"name": "Create Figure/Analyze Data", "type": "action"},
                ],
            },
            {
                "name": "Tools",
                "type": "group",
                "expanded": False,
                "children": [
                    {"name": "Reload", "type": "action"},
                    {"name": "View IndexFile", "type": "action"},
                    {"name": "Print File Info", "type": "action"},
                    {"name": "Export Brief Table", "type": "action"},
                ],
            },
            {"name": "Quit", "type": "action"},
        ]
        self.ptree = ParameterTree()
        self.ptreedata = Parameter.create(name="Models", type="group", children=self.params)
        self.ptree.setStyleSheet(
            """
            QTreeView {
                background-color: '#282828';
                alternate-background-color: '#646464';   
                color: rgb(238, 238, 238);
            }
            QLabel {
                color: rgb(238, 238, 238);
            }
            QTreeView::item:has-children {
                background-color: '#212627';
                color: '#00d4d4';
            }
            QTreeView::item:selected {
                background-color: '##c1c3ff';
            }
                """
        )
        return self.params, self.ptree, self.ptreedata
