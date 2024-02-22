import re

"""pull the layer definition from a text string and generate a roman-numeral 
representation for the cortical layers.
assumes that the layers are preceded by "layer ", and then roman/arabic numbers
separated by / or - (if split)

"""


def build_layers_re():
    """Build the regular expressions

    Returns:
        tuple of re: regular expressions for roman and arabic layer definitions
    """
    lab_text = r"(?P<text>layer)[\s]*"
    layers = "IV|V?I{0,3}"
    layers_n = r"\d{1}"
    sep = r"([\/-]{0,1})"
    lab_layer1 = f"(?P<layer>{layers:s})"
    lab_layer2 = f"(?P<layer2>{layers:s})"
    lab_layer1n = f"(?P<layer>{layers_n:s})"
    lab_layer2n = f"(?P<layer2>{layers_n:s})"
    rel = re.compile(f"{lab_text:s}{lab_layer1:s}{sep:s}{lab_layer2:s}", re.IGNORECASE)
    rel2 = re.compile(f"{lab_text:s}{lab_layer1n:s}{sep:s}{lab_layer2n:s}", re.IGNORECASE)
    return rel, rel2


nums = ["1", "2", "3", "4", "5", "6"]


def convert_to_Roman(number):
    number = int(number)  # force representation as int
    num = [1, 4, 5, 9, 10, 40, 50, 90, 100, 400, 500, 900, 1000]
    sym = ["I", "IV", "V", "IX", "X", "XL", "L", "XC", "C", "CD", "D", "CM", "M"]
    i = 12
    roman = ""
    while number:
        div = number // num[i]
        number %= num[i]

        while div:
            roman += sym[i]
            div -= 1
        i -= 1
    return roman


def convert_to_Arabic(rletter):
    # limited: just map from letters to numbers:

    rdict = {"I": "1", "II": "2", "III": "3", "IV": "4", "V": "5", "VI": "6"}
    if rletter in nums:
        return str(rletter)
    else:
        if rletter not in rdict.keys():
            raise ValueError(
                f"Layer lettering <{rletter:s}> not in known list of layers: ", rdict.keys()
            )
        return rdict[rletter]


def parse_layer(notetext: str):
    """Parese the later text depeinding on the input, and replace arabic numbers
    with roman numerals.

    Args:
        notetext (str): _description_

    Returns:
        [str, None]: Layer string (e.g., II/III) if successful, None if no layer to be parsed in the string
    """
    lay = None
    rel, rel2 = build_layers_re()
    m = rel.search(notetext)
    m2 = rel2.search(notetext)
    # print('m: ', m.groups())
    # if m2 is not None:
    #     print('m2: ', m2.groups()) # reassemble:
    if m is None:
        return None  # raise ValueError("\nFailed to parse: ", notetext)
    if m2 is not None:
        l1 = m2.group("layer")
        l2 = m2.group("layer2")
    elif m is not None:
        l1 = m.group("layer")
        l2 = m.group("layer2")
    print("l1, l2: ", l1, l2)
    if len(l1) == 0:
        return None
    if l1 not in nums and len(l1) > 0:
        l1 = convert_to_Arabic(l1)

    if len(l2) > 0 and l2 not in nums:
        l2 = convert_to_Arabic(l2)

    if len(l2) > 0:
        lay = f"L{l1:s}/{l2:s}".upper()
    else:
        lay = f"L{l1:s}".upper()
    return lay
