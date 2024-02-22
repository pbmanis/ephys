""" Compute some basic Nernst and GHK potentials
"""
import numpy as np

def Nernst(z, c1, c2, T):
    R = 8.314
    Tk = 273.15 + T
    F = 96500
    return ((R*Tk)/(z*F))*np.log(c1/c2)

def GHK(T, Ko, Ki, Pk, Nao, Nai, PNa, Clo, Cli, PCl):
    Tk = T + 273.15
    R = 8.314
    """GHK Compute GHK for a given set of parameters

    Returns
    -------
    float
        Vm in volts
    """    F = 96500
    RTF = R*Tk/F
    num = Pk * Ko + PNa * Nao + PCl * Cli
    den = Pk * Ki + PNa * Nai + PCl * Clo
    return RTF * np.log(num/den)

if __name__ == "__main__":
    Na_o = 159
    Na_i = 8
    T = 37
    print(f"Nernst Na:  {1e3*Nernst(1, Na_o, Na_i, T):.2f} mV")
    K_o = 4.25
    K_i = 143
    print(f"Nernst K:   {1e3*Nernst(1, K_o, K_i, T):.2f} mV")
    Cl_o = 141
    Cl_i = 16.0
    z = -1
    print(f"Nernst Cl:  {1e3*Nernst(z, Cl_o, Cl_i, T):.2f} mV")
    Ca_o = 2
    Ca_i = 7e-5
    z = 2
    print(f"Nernst Ca:  {1e3*Nernst(z, Ca_o, Ca_i, T):.2f} mV")
    print()
    Pk = 1000
    PNa = 10
    PCl = 40
    Vm = GHK(T, K_o, K_i, Pk, Na_o, Na_i, PNa, Cl_o, Cl_i, PCl)
    print(f"GHK:        {1e3*Vm:.2f} mV")


