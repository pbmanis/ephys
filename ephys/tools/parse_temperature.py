import re
import pint

UR = pint.UnitRegistry
URR = UR()
# clean up the temperature string to return an Int.
Q_ = URR.Quantity

def parse_temp(tempstr):
    """Convert somewhat temperature designators input by users to a standard, e.g.:
        37.0C = 37
        25C = 25
        room temp = 22
        22-24C = 23

    Params:
        temp(str): temperature string from the file

    Returns:
        str: sanitized temperature string
    """
    if '-' in tempstr or "to" in tempstr:
        raise ValueError("Temperature range not supported yet.")
    
    if isinstance(tempstr, (float, int)):
        temp = int(tempstr)
        if temp <= 41:
            deg = temp*URR.degC
        elif temp <= 100:
            deg = temp*URR.degF
        elif temp <= 320:
            deg = temp*URR.degK

    tempstr = tempstr.strip()
    if tempstr in ["room temp", "RT", "room temperature"]:
        return 22
    if tempstr.lower().find("c") >= 0:
        tempstr = float(tempstr.lower().replace("c", ""))
        deg = Q_(tempstr, "celsius")
    elif tempstr.lower().find("f") >= 0:
        tempstr = float(tempstr.lower().replace("f", ""))
        deg = Q_(tempstr, "fahrenheits")
    elif tempstr.lower().find("k") >= 0:
        tempstr = float(tempstr.lower().replace("k", ""))
        deg = Q_(tempstr, "kelvin")
    print("deg: ", deg)
    # if "C" in tempstr:
    #     tempstr = tempstr.replace("C", "")
    #     deg = float(tempstr)*UR.degC
    # if "c" in tempstr:
    #     tempstr = tempstr.replace("c", "")
    #     deg = float(tempstr)*UR.degC
    # if "F" in tempstr:
    #     tempstr = tempstr.replace("F", "")
    #     deg = float(tempstr)*UR.degF
    # if "f" in tempstr:
    #     tempstr = tempstr.replace("f", "")
    #     deg = float(tempstr)*UR.degF
    # if "K" in tempstr:
    #     tempstr = tempstr.replace("K", "")
    #     deg = float(tempstr)*UR.degK
    # if "k" in tempstr:
    #     tempstr = tempstr.replace("k", "")
    #     deg = float(tempstr)*UR.degK
    
    # finally, convert temperature to int, celsius.
    print("1: ", deg)
    at = deg.to(URR.degC)
    print("2: ", at)
    at = float(at.magnitude)
    print(at)
    return int(at)

if __name__ == "__main__":
    print(parse_temp("37.0C"))
    print(parse_temp("25C"))
    print(parse_temp("98F"))
    print(parse_temp("room temp"))
    print(parse_temp("22-24C"))