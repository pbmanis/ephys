import datetime

import matplotlib.pyplot as mpl
import numpy as np

########################
# Cochlear distance - frequency functions for 4 published models:

# Ou et al., An anatomically based frequency-place map for the mouse cochlea. Hearing Res 145:123, 2000
# 26 C57BL/CBA F1 mice aged 2-3 months when first tested (3-4 months at time of death)
# Note that d is a normalized value in % of cochlear distance from the APEX, with assumption of 7mm total length

# Muller et al. A physiological place-frequency map of the cochlear in the CBA/J mouse. Hearing Res 202:63, 2005
# CBA/J mice (Mus musculus) of either sex aged 6–13 weeks (average 7.7 weeks).
# d is normalized to a value 0-100%, from the BASE.
# assumes cochlea of 5.13 mm length

# Ehret Masked Auditory Thresholds, Critical Rations, and Scales of the Basilarm membrane of the
#       housemouse (mus musculus). J. Comp. Physiol. 103;329, 1975
#  13 male mice (Mus musculws, outbred strain NMRI) aged 8-9 weeks
# distance in mm, f in Hz  Assumes 7 mm cochlear length

# Muniak, Rivas, Ryugo. 3d model of frequency representation in the cochlea rnucleus of the CBA/J mouse.
# J Comp Neurol 521: 1510-1532, 2013.
# Twenty-four adult CBA/J mice (2 male, 22 female; Jackson Laboratories, Bar Harbor, ME) were used in this study (Table 1).
# Mice ranged in age from 2.5–4 months (mean: 12.3 6 1.7 weeks) and weighed between 17–30 g (mean: 23.9 6 3.8 g).
# Note that the maps are "normalized" to the length of the cochlea, referenced to the APEX
# the mean length of the cochlea was 5.23 mm (range: 4.79–5.68 mm).
#
# Wang, Hirose Liberman. Dynamics of noise-induced cellular injury and repair in the mouse cochlea.
# JARO 3:248-268, 2002.
# CBA/CaJ mice, equation is a re-fit to the Ehret data, expressed as percent distance from the BASE.

# Note that the Ou et al and Ehret  data rely on suprathreshold responses, and appear to be slightly
# shifted in position relative to the other 2 datasets, which rely on threshold data.
#  
# Implementation: Paul B. Manis, January 2025.
########################

# define reference point (0) as either at the base or at the apex of the cochlea
ref = {"Ou": "apex", "Muller": "base", "Ehret": "apex", "Muniak": "apex", "Wang": "base"}
# define whether the cochlear distance is normalized to 1, 100 or not normalized (0).
normed = {"Ou": 100.0, "OuNormal": 100.0, "Muller": 100.0, "Ehret": 0, "Muniak": 1.0, "Wang": 100.0}
mapdist = {"Ou": 7.0, "OuNormal": 7.0, "Muller": 5.13, "Ehret": 7.0, "Muniak": 5.23, "Wang": 7}


def get_map_names():
    return list(ref.keys())


def rescale_distance(functype, d):
    # Normalized and reversed
    if normed[functype] != 0:
        d = normed[functype] * (d) / mapdist[functype]
    if ref[functype] == "base":
        d = d[::-1]
    return d


def cochlear_percent_distance_to_frequency(functype, d):
    # in this function, all distances are relative to the distance from the apex of the cochlea
    # so we need to invert the d values in some cases where the base is the equation reference
    # point.
    # d is the percent distance from the apex.
    # returns frequency, given % distance
    if functype == "Ou":
        f = 1460 * np.power(10.0, (0.0177 * d))  # percentage distance
        minf = 1460 * np.power(10.0, (0.0177 * 0))
        maxf = 1460 * np.power(10.0, (0.0177 * 100))

    elif functype == "Ou_Normal":
        f = 2553 * np.power(10.0, (0.0140 * d))  # percentage distance
        minf = 2553 * np.power(10.0, (0.0140 * 0))
        maxf = 2553 * np.power(10.0, (0.0140 * 100))

    elif functype == "Muller":
        d = 100 - d
        f = 10.0 ** ((156.5 - d) / 82.5)
        f = f * 1000.0  # Muller et al put f in kHz, not Hz
        minf = 1e3* 10.0 ** ((156.5 - 100) / 82.5)
        maxf = 1e3 * 10.0 ** ((156.5 - 0) / 82.5)

    elif functype == "Ehret":  # Ehret,
        minf = 3350 * (np.power(10.0, 0.21 * 0) - 1)
        maxf = 3350 * (np.power(10.0, 0.21 * 7.0) - 1)
        dmm = (d/100)*7.0  # distance in mm from apex based on percentage
        f = 3350 * (np.power(10.0, 0.21 * dmm) - 1)


    elif functype == "Muniak":
        # relative to apex, normalization is to 1, not 100.
        dn = d/100.
        f = 5404 * (10.0 ** (1.16 * dn) - 0.27)
        minf = 5404 * (10.0 ** (1.16 * 0) - 0.27)
        maxf = 5404 * (10.0 ** (1.16 * 1) - 0.27)

    elif functype == "Wang":
        f = 2.109 * (np.power(10, (100.0-d) * 0.0142) - 0.7719)
        f = f * 1000.0  # f is in kHz here.
        minf = 1e3*2.109 * (np.power(10, (100.0-100) * 0.0142) - 0.7719)
        maxf = 1e3*2.109 * (np.power(10, (100.0-0) * 0.0142) - 0.7719)

    else:
        raise ValueError(f"Unknown cochlear distance function: {functype}")
    print(f" {functype:>12s} {minf:8.1f} -- {maxf:8.1f}")
    return f, minf, maxf

def cochlear_distance_to_frequency(functype, d):
    # returns frequency, given distance
    if functype == "Ou":
        d = rescale_distance(functype, d)
        print("max ou distance: ", np.max(d), np.min(d))
        f = 1460 * np.power(10.0, (0.0177 * d))  # percentage distance
        minf = 1460 * np.power(10.0, (0.0177 * 0))
        maxf = 1460 * np.power(10.0, (0.0177 * 100))

    elif functype == "Ou_Normal":
        d = rescale_distance(functype, d)
        f = 2553 * np.power(10.0, (0.0140 * d))  # percentage distance
        minf = 2553 * np.power(10.0, (0.0140 * 0))
        maxf = 2553 * np.power(10.0, (0.0140 * 100))

    elif functype == "Muller":
        d = rescale_distance(functype, d)
        f = 10.0 ** ((156.5 - d) / 82.5)
        f = f * 1000.0  # Muller et al put f in kHz, not Hz
        minf = 1e3* 10.0 ** ((156.5 - 100) / 82.5)
        maxf = 1e3 * 10.0 ** ((156.5 - 0) / 82.5)

    elif functype == "Ehret":  # Ehret,
        d = rescale_distance(functype, d)
        f = 3350 * (np.power(10.0, 0.21 * d) - 1)
        minf = 3350 * (np.power(10.0, 0.21 * 0) - 1)
        maxf = 3350 * (np.power(10.0, 0.21 * 100) - 1)

    elif functype == "Muniak":
        # relative to apex, normalization is to 1, not 100.
        d = rescale_distance(functype, d)
        f = 5404 * (10.0 ** (1.16 * d) - 0.27)
        minf = 5404 * (10.0 ** (1.16 * 0) - 0.27)
        maxf = 5404 * (10.0 ** (1.16 * 1) - 0.27)

    elif functype == "Wang":
        d = rescale_distance(functype, d)
        # note original equation uses (100-d) instead of d
        # below, but we handle the reversal in rescale_distance
        f = 2.109 * (np.power(10, (100.0-d) * 0.0142) - 0.7719)
        f = f * 1000.0  # f is in kHz here.
        minf = 1e3*2.109 * (np.power(10, (100.0-100) * 0.0142) - 0.7719)
        maxf = 1e3*2.109 * (np.power(10, (100.0-0) * 0.0142) - 0.7719)

    else:
        raise ValueError(f"Unknown cochlear distance function: {functype}")
    print(f" {functype:>12s} {minf:8.1f} -- {maxf:8.1f}")
    return f


def cochlear_frequency_to_distance(functype, f, maxd=None):
    # returns the distance (in mm) and the percent distance.
    # percentage distance is referenced to the APEX
    if functype == "Ou":
        # distance is in % of cochelar distance.
        #
        d = 56.6 * np.log10(f) - 179.1
        # d = 85.4 * np.log10(f) - 270.4
        dpct = 100-d
        if maxd is None:
            maxd = mapdist[functype]  # convert normailzed, and scale by assumed cochlear length
        d = maxd - (d / normed[functype]) * maxd

    elif functype == "Ou_Normal":
        # distance is in % of cochelar distance.
        d = 56.6 * np.log10(f) - 179.1
        d = 71.4 * np.log10(f) - 243.4
        dpct = d
        if maxd is None:
            maxd = mapdist[functype]

    elif functype == "Muller":
        #  A = 4232, a = 1.279 and k = -0.22 (R = 0.993).
        d = 156.5 - 82.5 * np.log10(f / 1000.0)
        # d = d[::-1]
        dpct = d
        if maxd is None:
            maxd = mapdist[functype]  # convert normailzed, and scale by assumed cochlear length
        d = (
            d / normed[functype]
        ) * maxd  # convert normalized, and scale by assumed cochlear length

    elif functype == "Ehret":
        d = np.log10(f / 875.0) / 0.306  # value is absolute but assumes 7mm length
        print("Ehrtd: f, d: ", f, d)
        dpct = 100*d/7.0
        if maxd is None:
            maxd = mapdist[functype]
        d = maxd - (d / 7.0) * maxd  # convert normalized, and scale by assumed cochlear length

    elif functype == "Muniak":
        d = 78.43 * np.log10(f / 1000.0) - 49.96
        dpct = 100 - d
        if maxd is None:
            maxd = mapdist[functype]  # convert normailzed, and scale by assumed cochlear length
        d = maxd - (d / 100.0) * maxd  # convert normalized, and scale by assumed cochlear length
   
    elif functype == "Wang":
        d = 70.422 * np.log10((f / 1000.0) * 2.4) - 34.69
        print("wang d range : ", np.min(d), np.max(d))
        print("frange: ", np.min(f), np.max(f))
        dpct = d
        if maxd is None:
            maxd = mapdist[functype]
        d = (d / 100.0) * maxd

    else:
        raise ValueError(f"Unknown cochlear frequency function: {functype}")
    return d, dpct


def plot_fmaps():

    figure, ax = mpl.subplots(2, 2, figsize=(8,10))

    colors = ["b", "g", "c", "m", "k"]
    for ipl, functype in enumerate(["Ou", "Muller", "Ehret", "Muniak"]):  #, "Wang"]):
        d_pct = np.linspace(0, 100, 100)
        fp, minf, maxf = cochlear_percent_distance_to_frequency(functype, d_pct)
        d_mm = d_pct*mapdist[functype]/100.
        print(functype, d_mm)
        fr = cochlear_distance_to_frequency(functype, d_mm)
        # d2, dpct = cochlear_frequency_to_distance(functype, fr)
        ax[0, 0].plot(d_mm, fr/1000, color=colors[ipl], label=functype)
        ax[0, 1].plot(fr/1000, d_mm, color=colors[ipl], label=functype)
        ax[1, 0].plot(d_pct, fp/1000, color=colors[ipl], label=functype)
        ax[1, 1].plot(fp/1000, d_pct, color=colors[ipl], label=functype)
    for a in ax.ravel():
        a.spines.top.set(visible=False)
        a.spines.right.set(visible=False)
        a.grid(True)
    ax[0, 0].set_title("Frequency vs. Distance")
    ax[0, 1].set_title("Distance vs. Frequency")
    ax[0, 0].set_xlabel("Distance from apex (mm)")
    ax[0, 0].set_ylabel("Frequency (kHz)")
    ax[0, 1].set_xlabel("Frequency (kHz)")
    ax[0, 1].set_ylabel("Distance from apex (mm)")

    ax[1, 0].set_title("Frequency vs. % Distance")
    ax[1, 1].set_title("% Distance vs. Frequency")
    ax[1, 0].set_xlabel("Distance from apex (%)")
    ax[1, 0].set_ylabel("Frequency (kHz)")
    ax[1, 1].set_xlabel("Frequency (kHz)")
    ax[1, 1].set_ylabel("Distance from apex (%)")
    # ax[1, 0].set_xlim(0, 100)
    # ax[1, 1].set_xlim(0, 100)
    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 0].legend()
    ax[1, 1].legend()
    
    figure.suptitle("Cochlear Frequency Map Comparisons")

    mpl.text(
        0.98,
        0.01,
        f"pbm {datetime.datetime.now().strftime('%Y.%m.%d::%H:%M')}",
        ha="right",
        fontsize=8,
        transform=figure.transFigure,
    )
    mpl.tight_layout
    mpl.show()


if __name__ == "__main__":

    plot_fmaps()
