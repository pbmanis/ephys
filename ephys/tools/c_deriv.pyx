# cython: language_level=3
# Helpers for spike finding utilities
# call from python:
# c_deriv.c_deriv(x, y, npts, order, outarray)
# where the output array is preallocated as in 
# numpy.zeros_like(x)
#
# Build with:
# python cb_setup.py build_ext --inplace
# Or with ephys setup extension
# Developed for ephys module
#
# pbmanis pmanis@med.unc.edu
# 05/26/2020
#

cimport cython
import numpy as np
from libc.stdio cimport printf

def c_deriv(
           double[:] x_data,  # time data array (input)
           double[:] y_data,  # data for derivative calculation (input)
           long int npts, # number of points in the data array
           long int order,  # number of derivatives (input)
           double[:] deriv, # calculated derivative (output)
           ):
    
    cdef double dtemp1, dtemp2
    cdef long int k, i
    
    for k in range(order):  # initialize sums for the template
        deriv[0] = (y_data[1]-y_data[0])/(x_data[1]-x_data[0])
        deriv[npts] = (y_data[npts]-y_data[npts-1])/(x_data[npts]-x_data[npts-1])
        for i in range(1, npts-1):
            dtemp1 = (y_data[i+1]-y_data[i])/(x_data[i+1]-x_data[i])
            dtemp2 = (y_data[i]-y_data[i-1])/(x_data[i]-x_data[i-1])
            deriv[i] = (dtemp1 + dtemp2)/2.0


# Implement box finder for spikes from Hight and Kalluri
# V > threshold (mV)  # min spike height
# V(t0 > v(t0+dt))  # gets a spike peak
# V(t0 > V(t0-dt)
# dt = 0.01 ms (or the sampleing rate)
# V(to + dt2) - V(t0) < C1 (-12 mV)
# V(to - V(to-dtw)) > C2 (+11 mV)
# dt2 = 1.75 ms (express in msec)

def c_box_spike_find(
           double[:] x_data,  # time data array (input)
           double[:] y_data,  # data for spike search calculation (input)
           long int npts, # number of points in the data array (input)
           double thr, # threshold -35  (express in units of y_data) (input)
           double C1,  #  # slope value(express in units of y_data) (input)
           double C2, # slope value (express in units of y_data) (input)
           double dt2, # spike window (nominal 1.75 msec) (input)
           double minwidth,  # minimum width of spike (0.1 msec, for example) (input)
           double[:] spikes, # calculated spikes (times, set to 1 else 0) (output)
           ):

    cdef double dtemp1, dtemp2, dt
    cdef long int k, i, iwid
    
    dt = x_data[1]-x_data[0]  # use first interval, makes assumption that rate is constant
    iwid = (int)(dt2/dt)
    spkflag = 0
    for i in range(iwid, npts-iwid):
        if (y_data[i] > thr): # increasingly restrictive measures: works for clean data
            if (y_data[i]>y_data[i-1]) and (y_data[i] > y_data[i+1]):
                if ((y_data[i+iwid] - y_data[i]) < C1) and ((y_data[i]-y_data[i-iwid]) > C2):
                    if (x_data[i+iwid] - x_data[i-iwid]) > minwidth:
                        spikes[i] = 1.0

# this is jut like the one above, except with the "optimized" steps
# suggested by Claude "AI" (see below)

def c_box_spike_find_opt(
           double[:] x_data,  # time data array (input)
           double[:] y_data,  # data for spike search calculation (input)
           long int npts, # number of points in the data array (input)
           double thr, # threshold -35  (express in units of y_data) (input)
           double C1,  #  # slope value(express in units of y_data) (input)
           double C2, # slope value (express in units of y_data) (input)
           double dt2, # spike window (nominal 1.75 msec) (input)
           double minwidth,  # minimum width of spike (0.1 msec, for example) (input)
           double[:] spikes, # calculated spikes (times, set to 1 else 0) (output)
           ):

    cdef double dtemp1, dtemp2
    cdef double dt, y_curr, y_prev, y_next, y_left, y_right, width
    cdef Py_ssize_t i, k, iwid
    
    dt = x_data[1]-x_data[0]  # use first interval, makes assumption that rate is constant
    iwid = <Py_ssize_t>(dt2 / dt)  # explicit cast
    
    # Early return if window is too large
    if iwid >= npts // 2:
        return
    
    # Main detection loop with optimized conditions
    for i in range(iwid, npts - iwid):
        y_curr = y_data[i]
        
        # Fast threshold check first
        if y_curr <= thr:
            continue
            
        # Local maximum check with cached values
        y_prev = y_data[i-1]
        y_next = y_data[i+1]
        if not (y_curr > y_prev and y_curr > y_next):
            continue
        
        # Box criteria with cached values
        y_left = y_data[i - iwid]
        y_right = y_data[i + iwid]
        
        if not ((y_right - y_curr) < C1 and (y_curr - y_left) > C2):
            continue
        
        # Width check
        width = x_data[i + iwid] - x_data[i - iwid]
        if width <= minwidth:
            continue
            
        # All conditions met
        spikes[i] = 1.0


# Original optimized version
@cython.boundscheck(False)
@cython.wraparound(False) 
@cython.cdivision(True)
def c_box_spike_find_optimized(
    const double[::1] x_data,  # contiguous memory view for better performance
    const double[::1] y_data,  # const for read-only data
    const Py_ssize_t npts,     # Py_ssize_t is better for array indices
    const double thr,
    const double C1,
    const double C2,
    const double dt2,
    const double minwidth,
    double[::1] spikes,        # output array
): #  noexcept nogil:             # nogil for better performance
    
    cdef double dt, y_curr, y_prev, y_next, y_left, y_right, width
    cdef Py_ssize_t i, iwid
    
    # Calculate dt and window size once
    dt = x_data[1] - x_data[0]
    iwid = <Py_ssize_t>(dt2 / dt)  # explicit cast
    
    # Early return if window is too large
    if iwid >= npts // 2:
        return
    
    # Main detection loop with optimized conditions
    for i in range(iwid, npts - iwid):
        y_curr = y_data[i]
        
        # Fast threshold check first
        if y_curr <= thr:
            continue
            
        # Local maximum check with cached values
        y_prev = y_data[i-1]
        y_next = y_data[i+1]
        if not (y_curr > y_prev and y_curr > y_next):
            continue
        
        # Box criteria with cached values
        y_left = y_data[i - iwid]
        y_right = y_data[i + iwid]
        
        if not ((y_right - y_curr) < C1 and (y_curr - y_left) > C2):
            continue
        
        # Width check
        width = x_data[i + iwid] - x_data[i - iwid]
        if width <= minwidth:
            continue
            
        # All conditions met
        spikes[i] = 1.0
        

    