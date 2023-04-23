import re
# clean up the age string to match (mostly) the ISO standard
def ISO8601_age(agestr):
    """Convert somewhat random age designators to ISO standard, e.g.:
        postnatal day 30 mouse = P30D  (or P30W, or P3Y)
        Ranges are P1D/P3D if bounded, or P12D/ if not known but have lower bound.

    Params:
        agestr (str): age string from the file

    Returns:
        str: sanitized age string
    """
    q = False
    ish = False
    if isinstance(agestr, (float, int)):
        agestr = str(agestr)
    agestr = agestr.strip()
    if agestr.endswith("?"):
        agestr = agestr[:-1]
        q = True
    if agestr.endswith("ish"):
        agestr = agestr[:-3]
        ish = True
    if agestr == "?" or len(agestr) == 0:
        agestr = "0"    
    agestr = agestr.replace('p', 'P')
    agestr = agestr.replace('d', 'D')
    if 'P' not in agestr:
        agestr = 'P' + agestr
    if 'D' not in agestr:
        agestr = agestr + "D"
    if agestr == "PD":
        agestr = "P0D"  # no age specified
    if q:
        agestr = agestr + " ?"  # add back modifiers
    if ish:
        agestr = agestr + " ish"
    return agestr

def age_as_int(agestr):
    astr = re.sub('\D', '', agestr)
    return(int(astr))

# old version
def parse_ages(agestr:str):
    """
    Systematize the age representation
    """
    adat = []
    for a in agestr:
        a = a.strip()
        if a.endswith("?"):
            a = a[:-1]
        if a.endswith("ish"):
            a = a[:-3]
        if a == "?" or len(a) == 0:
            a = "0"
        if a.startswith("p") or a.startswith("P"):
            try:
                a = int(a[1:])
            except:
                a = 0
        elif a.endswith("d") or a.endswith("D"):
            a = int(a[:-1])
        elif a == " ":
            a = 0

        else:
            a = int(a)
        adat.append(a)
    return adat