"""
Using eFEL for analysis on acq4 files
"""
""""Basic example 1 for eFEL"""

from pathlib import Path
import numpy as np
from ephys.datareaders import acq4_reader
import efel
AR = acq4_reader.acq4_reader()

fn = 'data_for_testing/CCIV'

def main():
    """Main"""

    # Use numpy to read the trace data from the txt file
#    data = numpy.loadtxt('example_trace1.txt')
    assert Path(fn).is_dir() == True
    
    AR.setProtocol(fn)
    data = AR.getData()
    print(data)
    # Time is the first column
    time = np.array(AR.time_base)*1e3
    traces = []

    voltage = np.array(AR.traces)*1e3
    current = np.array(AR.commandLevels)*1e9
    for i in range(voltage.shape[0]):
        # Now we will construct the datastructure that will be passed to eFEL
        trace1 = {}
        trace1['T'] = time
        trace1['V'] = voltage[i]
        trace1['stimulus_current'] = [current[i]]
        trace1['stim_start'] = [AR.tstart*1e3]
        trace1['stim_end'] = [AR.tend*1e3]
        traces.append(trace1)

    # Now we pass 'traces' to the efel and ask it to calculate the feature
    # values
    traces_results = efel.getFeatureValues(traces,
                                           ['AP_amplitude', 'voltage_base', 'voltage_deflection',
                                       'peak_voltage', 'ohmic_input_resistance', 'spike_width2',
                                       'AP_duration_half_width',
                                    'AHP_time_from_peak', 'Spikecount_stimint'])

    # The return value is a list of trace_results, every trace_results
    # corresponds to one trace in the 'traces' list above (in same order)
    for i, trace_results in enumerate(traces_results):
        # trace_result is a dictionary, with as keys the requested features
        for feature_name, feature_values in trace_results.items():
            if feature_values is None:
                continue
            print( "   I=%.2f Feature %s has the following values: %s" % \
                (current[i], feature_name, ', '.join([str(x) for x in feature_values])))


if __name__ == '__main__':
    main()