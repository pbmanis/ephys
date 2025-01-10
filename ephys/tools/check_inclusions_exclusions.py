# check inclusions/exclusions in the configuration file
#

from pathlib import Path
from pylibrary.tools import cprint as CP

def include_exclude(cell_id: str, exclusions: dict, inclusions: dict, allivs: list):
    """
    Get the included and remove the excluded data from the list of IVs.
    """
    validivs = check_exclusions(cell_id, exclusions, allivs)
    validivs, additional_ivs, additional_iv_records = check_inclusions(
        cell_id, inclusions, validivs
    )
    return validivs, additional_ivs, additional_iv_records

def check_exclusions(cell_id: str, exclusions: dict, allivs: list):
    validivs = []
    if exclusions is not None:
        if cell_id in exclusions:
            print("cell_id is in exclusions")
            # exclude ALL protocols for this cell
            if exclusions[cell_id] in ["all", ["all"]]:
                msg = "All protocols for this cell are excluded from analysis in the configuration file."
                CP.cprint("r", msg)
                return  # nothing to do - entire cell is excluded.
            # exclude all protocols except those in the "exceptions" key
            elif exclusions[cell_id] in ["allexcept", ["allexcept"]]:
                if (
                    "exceptions" not in exclusions[cell_id].keys()
                ):  # confirm that there is an exceptions key
                    raise ValueError("No 'exceptions' key in the 'allexcept' exclusion")
                print("   ... Appending excepted protocols from 'allexcept': ")
                for protocol in allivs:
                    if Path(protocol).name in exclusions[cell_id]["exceptions"]:
                        validivs.append(protocol)
            else:
                print("   ... Appending valid protocols: ")
                print("   all ivs is : ", allivs)
                for protocol in allivs:
                    print("    protocol name: ", Path(protocol).name)
                    print("    exclusions: ", exclusions[cell_id]["protocols"])
                    if Path(protocol).name not in exclusions[cell_id]["protocols"]:
                        print("    adding valid protocol: ", protocol)
                        validivs.append(protocol)
                print("After appending, validivs now is: ", validivs)
                # validivs.append([protocol for protocol in allivs if Path(protocol).name not in exclusions[cell_id]['protocols']])
        else:
            validivs = allivs
        print("after Exclusions, validivs is: ", validivs)
    else:
        print("iv_analysis: analyze_ivs:: No exclusions for this cell: ", cell_id)
        validivs = allivs
    return validivs


def check_inclusions(cell_id: str, inclusions: dict, validivs: list):
    additional_ivs = []
    additional_iv_records = None
    if inclusions is not None:
        # Add ivs that are in the inclusions list. These may include data that is from
        # protocols that did not run to completion, but which might still have useful data
        if cell_id in inclusions:
            for iprot, protocol in enumerate(inclusions[cell_id]["protocols"]):
                protopath = str(Path(cell_id, protocol))
                additional_ivs.append(protopath)
                if additional_iv_records is None:
                    additional_iv_records = {
                        protopath: [cell_id, inclusions[cell_id]["records"][iprot]]
                    }
                else:
                    additional_iv_records[protopath] = [
                        cell_id,
                        inclusions[cell_id]["records"][iprot],
                    ]
            print("   Inclusions: additional ivs: ", additional_ivs)
            print("   Inclusions: additional iv records: ", additional_iv_records)
            validivs.extend(additional_ivs)
    return validivs, additional_ivs, additional_iv_records
