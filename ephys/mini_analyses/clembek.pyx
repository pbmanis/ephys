#  EXPERIMENTAL... not fully tested.
# Implementation of Clements-Bekkers mini PSC detection algorithm
# in Cython. Adapted from the matlab mex file clembek.c
# Build with:
# python cb_setup.py build_ext --inplace
#
# pbmanis pmanis@med.unc.edu
# 12/16/2017
#

cimport cython
import numpy as np
from libc.stdio cimport printf

def clembek(
           double[:] p_data,  # data array (input)
           double[:] p_templ,  # template (input)
           double thr,  # threshold for crit detection (input)
           double[:] p_crit,  # critera at detect (output)
           double[:] p_scale,  # scale factor *output)
           double[:] p_cx,  # offset constant (output)
           double [:] p_pkl,  #peaklist (output)
           double[:] p_evl,  #eventlist (output)
           long int p_nout,  # number of detected events (output)
           long int sign,  # sign (pos or neg) for events (input)
           long int nt,  # number of points in template (input)
           long int nd  # number of points in the data (input)
           ):

    cdef double sume, sume2, sumey, sumy, sumy2, s, c, sse, criteria
    cdef long int i, j, ipk, pkcount
    cdef double md
    cdef double dmin, dmax, tmin, tmax
    
    p_nout = 0
    pkcount = 0
    sume = 0.0
    sume2 = 0.0
    tmin = 0.0
    tmax = 0.0
    dmin = 0.0
    dmax = 0.0
    for i in range(nt):  # initialize sums for the template
        sume += p_templ[i]
        sume2 += (p_templ[i] * p_templ[i])
        if (p_templ[i] > tmax):
            tmax = p_templ[i]

    sumy = 0.0
    sumy2 = 0.0
    for j in range(nt):  # initialize sums from data
        sumy += p_data[j]
        sumy2 += p_data[j] * p_data[j]

    for i in range(nd-nt): 
        sumy += (p_data[i+nt-1] - p_data[i-1])
        sumy2 += ((p_data[i+nt-1] * p_data[i+nt-1]) - (p_data[i-1]*p_data[i-1]))
        sumey = 0.0
        for j in range(i,   nt+i): # (j = i; j < nt+i; j++) {
            sumey += (p_data[j]*p_templ[j-i])
            s = (sumey - sume * sumy/nt)/(sume2 - sume * sume/nt)
        c = (sumy - s*sume)/nt
        # we dont calculate f as it is not used...
        sse = sumy2 + (s*s*sume2)+(nt*c*c)-2.0*(s*sumey + c*sumy - (s*c*sume))
        if(sse < 0.0):
            sse = 1e-99
#        printf( "%e ", sse)
        criteria = s/np.sqrt(sse/float(nt-1))
        p_crit[i] = criteria
        p_scale[i] = s
        p_cx[i] = c
        if(sign <= 0 and criteria <= -thr):
            pkcount += 1
            p_evl[pkcount - 1] = i
            md = 0.0
            ipk = i
            for j in range(i,   nt+i):
                if(p_data[j] < md):
                    md = p_data[j]
                    ipk = j
            p_pkl[pkcount - 1] = ipk+1
            if(sign >= 1 and criteria >= thr):
                pkcount += 1
            p_evl[pkcount-1] = i
            md = 0.0
            ipk = i
            for j in range(i,   nt+i):
                if(p_data[j] > md):
                    md = p_data[j]
                    ipk = j
            p_pkl[pkcount - 1] = ipk+1
    
    p_nout = pkcount # update the output count 
    