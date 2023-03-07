import re
import ephys.tools.parse_layers as PL

notes = [
    "   layer I ",
    "   layer II ",
    "    layer   III",
    "  layer IV",
    "  layer V",
    "layer VI",
    "layer II/III",
    "layer V/VI",
    "layer   IV/V",
    "layer II-III",
    "layer 2/3",
    "layer 5/6",
    "layer 4/5",
    "layer VIII/vii",
    "yoda is a dog",
]



def run_tests():

    for note in notes:
        lay = PL.parse_layer(note)
        if lay is not None:
            print(f"{note:>16s}:   {lay:<8s}")
        else:
            print(f"{note:>16s}:   {'no match':<8s}")

if __name__ == '__main__':
    run_tests()