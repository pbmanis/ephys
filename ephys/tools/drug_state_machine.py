""" State machine for deciphering drug application during an experiment.

This module contains the state machine for deciphering drug application during an experiment.
The state machine is implemented using the state_machine module.
States are:
Control: No drug is added to the solution.
Washin: Drug is added to the solution, but is not at steady state.
Drug: Drug is added to the solution and is at steady state.
Washout: Drug is being removed from the solution.

pip install pyhon-statemachine

"""

from statemachine import StateMachine, State
from pathlib import Path
from typing import Union, Tuple


class SolutionStates(State):
    """State machine for deciphering drug application during an experiment."""

    waiting_for_control = State(initial=True)
    control = State()
    washin = State()
    drug = State()
    washout = State()
    end = State(final=True)

    starting_state = waiting_for_control.to(waiting_for_control)
    control_found = waiting_for_control.to(washin, cond="check_control")
    washin_found = washin.to(drug, cond="check_washin")
    drug_found = drug.to(washout, cond="check_drug")
    washout_found = washout.to(end, cond="check_washout")
    done = washout.to(end, cond="set_done")

    def __init__(self, name, data):

        self.set_data(data)
        self.set_timewindow((5, 15.0))
        self.control_protocol: str = None
        self.washin_protocol: str
        self.drug_protocol: str = None
        self.washout_protocol: str = None

        self.control_time:Union[float, None] = None
        self.washin_time:Union[float, None] = None
        self.drug_time:Union[float, None] = None
        self.done = False
        super(SolutionStates, self).__init__(name)

    def set_data(self, timestampdict: dict):
        self.tstamps = timestampdict
        self.tkeys = list(self.tstamps.keys())
    
    def set_timewindow(self, timewindow: Tuple[float, float]):
        self.t0, self.t1 = timewindow
        print("DSM: self.t0, t1: ", self.t0, self.t1)

    def get_protocols(self):
        for key in self.tstamps.keys():
            self.check_control(key)
            self.check_washin(key)
            self.check_drug(key)
            self.check_washout(key)
            # print("\n")  # dir(M))
            # print("ID: ", M.transitions)
            # print(M.value)
            if self.done:
                break
        # print(M.control_protocol, M.control_time)
        # print(M.drug_protocol, M.drug_time)
        return (self.control_protocol, self.control_time, self.drug_protocol, self.drug_time)

    def check_control(self, key):
        # print("check control", self.tstamps[key]["condition"])
        # if self.tstamps[key]["condition"].split(",")[1].strip() == "control":
        if isinstance(self.tstamps[key]["condition"], float):
            return False
        if self.tstamps[key]["condition"].find("control") >= 0:
            # print("   control")
            self.control_protocol = key  # self.tstamps[key]["protocol"]
            self.control_time = key / 60.0
            return True
        return False

    def check_washin(self, key):
        # print("checkwashin", self.tstamps[key]["condition"])
        # if self.tstamps[key]["condition"].split(",")[1].strip() == "washin":
        if isinstance(self.tstamps[key]["condition"], float):
            return False
        if self.tstamps[key]["condition"].find("washin") >= 0:
            # print("   washin")
            self.washin_time = key / 60.0
            self.washin_protocol = key  # self.tstamps[key]["protocol"]
            return True
        return False

    def check_drug(self, key):
        # print("check drug", self.tstamps[key]["condition"])
        # if self.tstamps[key]["condition"].split(",")[1].strip() == "drug":
        if isinstance(self.tstamps[key]["condition"], float):
            return False
        if self.tstamps[key]["condition"].find("drug") >= 0:
            if self.washin_time is not None:
                drugtime = key / 60.0 - self.washin_time
            elif self.control_time is not None:
                drugtime = key / 60.0 - self.control_time
            else:
                print("No control or washin time found, cannot verify drug")
                return False
            if self.t0 <= drugtime <= self.t1:  # check if measurement was made in the selected time window
                self.drug_protocol = key  # self.tstamps[key]["protocol"]
                self.drug_time = drugtime
                # print("   drug", drugtime)
                return True
            return False
        return False

    def check_washout(self, key):
        # print("check washout", self.tstamps[key]["condition"])
        # if self.tstamps[key]["condition"].split(",")[1].strip() == "washout":
        if isinstance(self.tstamps[key]["condition"], float):
            return False
        if self.tstamps[key]["condition"].find("washout") >= 0:
            # print("   washout")
            self.washout_protocol = key  # self.tstamps[key]["protocol"]
            return True
        return False

    def set_done(self, key):
        # print("Check done")
        self.done = True
        return True


if __name__ == "__main__":

    tstamps0 = {
        1573246050.687: {
            "cellid": Path("2019.11.08_000/slice_002/cell_000"),
            "protocol": "Map_NewBlueLaser_VC_Single_000",
            "date_time": "2019.11.08 15:47:30",
            "condition": "15:47:30, control, control",
            "map_time": 0.0,
            "washin_time": -1,
            "washout_time": -1,
            "control": False,
            "ttx": False,
        },
        #   1573246346.926: {'cellid': Path('2019.11.08_000/slice_002/cell_000'), 'protocol': 'Map_NewBlueLaser_VC_Single_001', 'date_time': '2019.11.08 15:52:26', 'condition': '15:52:26, washin, 1mM4AP+TTX', 'map_time': 0.0, 'washin_time': -1, 'washout_time': -1, 'control': False, 'ttx': False},
        1573246616.127: {
            "cellid": Path("2019.11.08_000/slice_002/cell_000"),
            "protocol": "Map_NewBlueLaser_VC_Single_002",
            "date_time": "2019.11.08 15:56:56",
            "condition": "15:56:56, drug, 1mM4AP+TTX",
            "map_time": 0.0,
            "washin_time": -1,
            "washout_time": -1,
            "control": False,
            "ttx": False,
        },
        1573246886.953: {
            "cellid": Path("2019.11.08_000/slice_002/cell_000"),
            "protocol": "Map_NewBlueLaser_VC_Single_003",
            "date_time": "2019.11.08 16:01:26",
            "condition": "16:01:26, drug, 1mM4AP+TTX",
            "map_time": 0.0,
            "washin_time": -1,
            "washout_time": -1,
            "control": False,
            "ttx": False,
        },
        1573247160.881: {
            "cellid": Path("2019.11.08_000/slice_002/cell_000"),
            "protocol": "Map_NewBlueLaser_VC_Single_004",
            "date_time": "2019.11.08 16:06:00",
            "condition": "16:06:00, drug, 1mM4AP+TTX",
            "map_time": 0.0,
            "washin_time": -1,
            "washout_time": -1,
            "control": False,
            "ttx": False,
        },
        1573247455.468: {
            "cellid": Path("2019.11.08_000/slice_002/cell_000"),
            "protocol": "Map_NewBlueLaser_VC_Single_005",
            "date_time": "2019.11.08 16:10:55",
            "condition": "16:10:55, drug, 1mM4AP+TTX",
            "map_time": 0.0,
            "washin_time": -1,
            "washout_time": -1,
            "control": False,
            "ttx": False,
        },
        1573247762.054: {
            "cellid": Path("2019.11.08_000/slice_002/cell_000"),
            "protocol": "Map_NewBlueLaser_VC_Single_006",
            "date_time": "2019.11.08 16:16:02",
            "condition": "16:16:02, drug, 1mM4AP+TTX",
            "map_time": 0.0,
            "washin_time": -1,
            "washout_time": -1,
            "control": False,
            "ttx": False,
        },
        1573247878.017: {
            "cellid": Path("2019.11.08_000/slice_002/cell_000"),
            "protocol": "Map_NewBlueLaser_VC_Single_007",
            "date_time": "2019.11.08 16:17:58",
            "condition": "16:17:58, drug, 1mM4AP+TTX",
            "map_time": 0.0,
            "washin_time": -1,
            "washout_time": -1,
            "control": False,
            "ttx": False,
        },
        1573248084.843: {
            "cellid": Path("2019.11.08_000/slice_002/cell_000"),
            "protocol": "Map_NewBlueLaser_VC_Single_008",
            "date_time": "2019.11.08 16:21:24",
            "condition": "16:21:24, washout, 1mM4AP+TTX",
            "map_time": 0.0,
            "washin_time": -1,
            "washout_time": -1,
            "control": False,
            "ttx": False,
        },
        1573248237.734: {
            "cellid": Path("2019.11.08_000/slice_002/cell_000"),
            "protocol": "Map_NewBlueLaser_VC_Single_009",
            "date_time": "2019.11.08 16:23:57",
            "condition": "15:56:56, drug, 1mM4AP+TTX",
            "map_time": 0.0,
            "washin_time": -1,
            "washout_time": -1,
            "control": False,
            "ttx": False,
        },
        1573248853.701: {
            "cellid": Path("2019.11.08_000/slice_002/cell_000"),
            "protocol": "Map_NewBlueLaser_VC_Single_010",
            "date_time": "2019.11.08 16:34:13",
            "condition": "15:56:56, drug, 1mM4AP+TTX",
            "map_time": 0.0,
            "washin_time": -1,
            "washout_time": -1,
            "control": False,
            "ttx": False,
        },
        1573249305.412: {
            "cellid": Path("2019.11.08_000/slice_002/cell_000"),
            "protocol": "Map_NewBlueLaser_VC_Single_011",
            "date_time": "2019.11.08 16:41:45",
            "condition": "15:56:56, drug, 1mM4AP+TTX",
            "map_time": 0.0,
            "washin_time": -1,
            "washout_time": -1,
            "control": False,
            "ttx": False,
        },
    }

    tstamps = {
        1648066592.66: {
            "cellid": Path("2022.03.23_000/slice_002/cell_001"),
            "protocol": "Map_NewBlueLaser_VC_increase_1ms_000",
            "date_time": "2022.03.23 16:16:32",
            "condition": "control",
            "map_time": 0.0,
            "washin_time": -1,
            "washout_time": -1,
            "control": False,
            "ttx": False,
        },
        1648066984.968: {
            "cellid": Path("2022.03.23_000/slice_002/cell_001"),
            "protocol": "Map_NewBlueLaser_VC_increase_1ms_001",
            "date_time": "2022.03.23 16:23:04",
            "condition": "washin",
            "map_time": 0.0,
            "washin_time": -1,
            "washout_time": -1,
            "control": False,
            "ttx": False,
        },
        1648067444.548: {
            "cellid": Path("2022.03.23_000/slice_002/cell_001"),
            "protocol": "Map_NewBlueLaser_VC_Single_000",
            "date_time": "2022.03.23 16:30:44",
            "condition": "drug",
            "map_time": 0.0,
            "washin_time": -1,
            "washout_time": -1,
            "control": False,
            "ttx": False,
        },
        1648067878.752: {
            "cellid": Path("2022.03.23_000/slice_002/cell_001"),
            "protocol": "Map_NewBlueLaser_VC_Single_001",
            "date_time": "2022.03.23 16:37:58",
            "condition": "drug",
            "map_time": 0.0,
            "washin_time": -1,
            "washout_time": -1,
            "control": False,
            "ttx": False,
        },
        1648069052.736: {
            "cellid": Path("2022.03.23_000/slice_002/cell_001"),
            "protocol": "Map_NewBlueLaser_VC_Single_005",
            "date_time": "2022.03.23 16:57:32",
            "condition": "drug",
            "map_time": 0.0,
            "washin_time": -1,
            "washout_time": -1,
            "control": False,
            "ttx": False,
        },
        1648069299.505: {
            "cellid": Path("2022.03.23_000/slice_002/cell_001"),
            "protocol": "Map_NewBlueLaser_VC_Single_006",
            "date_time": "2022.03.23 17:01:39",
            "condition": "drug",
            "map_time": 0.0,
            "washin_time": -1,
            "washout_time": -1,
            "control": False,
            "ttx": False,
        },
        1648069651.544: {
            "cellid": Path("2022.03.23_000/slice_002/cell_001"),
            "protocol": "Map_NewBlueLaser_VC_Single_007",
            "date_time": "2022.03.23 17:07:31",
            "condition": "drug",
            "map_time": 0.0,
            "washin_time": -1,
            "washout_time": -1,
            "control": False,
            "ttx": False,
        },
        1648069766.316: {
            "cellid": Path("2022.03.23_000/slice_002/cell_001"),
            "protocol": "Map_NewBlueLaser_VC_Single_008",
            "date_time": "2022.03.23 17:09:26",
            "condition": "drug",
            "map_time": 0.0,
            "washin_time": -1,
            "washout_time": -1,
            "control": False,
            "ttx": False,
        },
        1648069994.609: {
            "cellid": Path("2022.03.23_000/slice_002/cell_001"),
            "protocol": "Map_NewBlueLaser_VC_Single_009",
            "date_time": "2022.03.23 17:13:14",
            "condition": "drug",
            "map_time": 0.0,
            "washin_time": -1,
            "washout_time": -1,
            "control": False,
            "ttx": False,
        },
    }

    M = SolutionStates("DrugWashes", tstamps)
    for key in tstamps.keys():
        M.check_control(key)
        M.check_washin(key)
        M.check_drug(key)
        M.check_washout(key)
        print("\n")  # dir(M))
        print("ID: ", M.transitions)
        print(M.value)
        if M.done:
            break
    cp, ct, dp, dt = M.get_protocols()
    print(cp, ct)
    print(dp, dt)
    print(M.control_protocol, M.control_time)
    print(M.drug_protocol, M.drug_time)
