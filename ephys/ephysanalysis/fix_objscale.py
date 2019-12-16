"""
Fix the objective scale factor
Give "stated objective", "actual objective"

('objective', '4x 0.1na ACHROPLAN')

"""
import numpy as np
import argparse
from pathlib import Path
from pyqtgraph import configfile
import pprint
from collections import OrderedDict
import datetime
pp = pprint.PrettyPrinter(indent=4)
# retiga:

CineScale = 1.0
refscale = [(1./CineScale)*6.54e-6, -(1./CineScale)*6.54e-6]
print(refscale)
sfactors = [4., 10., 20., 40., 63.]

objlist = {4: '4x 0.1na ACHROPLAN',
        10: '10x 0.3na W N-ACHROPLAN',
        20: '20x 0.5na W N-ACHROPLAN',
        40: '40x 0.8na ACHROPLAN',
        63: '63x 0.9na ACHROPLAN',
    }


bp = '/Volumes/Pegasus/ManisLab_Data3/Kasten_Michael/NF107Ai32Het'

# fn = '2019.11.15_000/slice_001/cell_001'
# changelist = {'image_001.tif': [4, 10], 'image_002.tif': [4, 10], 'video_000.ma': [4, 10]}

# fn = '2019.04.16_000/slice_002/cell_000'
# changelist = {'image_009.tif': [4, 10], 'image_010.tif': [4, 10], 'image_011.tif': [4, 10],'image_012.tif': [4, 10],'image_013.tif': [4, 10],}
# changelist_type = 'old'

changelist_type = 'new'  # construct from minimal information
fromto = [4, 20]
cl_videos = range(2, 18)  # last number must be 1 greater than last in the list you want to chane...
cl_images = range(2, 19)
changelist = OrderedDict()
# fn = '2019.12.09_001/slice_001'
# for v in cl_videos:
#     changelist[f"video_{v:03d}.ma"] = fromto
# for i in cl_images:
#     changelist[f"image_{i:03d}.tif"] = fromto
#

fn = '2019.12.09_001/slice_000'
fromto = [4, 20]
cl_videos = range(1, 7)  # last number must be 1 greater than last in the list you want to chane...
cl_images = range(2, 4)
for v in cl_videos:
    changelist[f"video_{v:03d}.ma"] = fromto
for i in cl_images:
    changelist[f"image_{i:03d}.tif"] = fromto
p = Path(bp, fn, '.index')

def write_index(p, index):
    configfile.writeConfigFile(index, p)

def read_index(p, write=False):
    d = datetime.datetime.now()
    dstr = d.strftime('%Y-%m-%d %H:%M:%S')
    index = configfile.readConfigFile(p)
    print(list(index.keys()))
    for k in list(index.keys()):
        if k in list(changelist.keys()):
            print('Index: ', k)
            print('Old objective: ', index[k]['objective']) # pp.pprint(index[k] )
            oldobj = index[k]['objective']
            print('   Old trans: ', index[k]['transform'])
            print('   Old devtrans: ', index[k]['deviceTransform'])
            binning = index[k]['binning']
            newobj = changelist[k][1]
            fnewobj = float(newobj)
            index[k]['transform']['scale'] = (binning[0]*refscale[0]/fnewobj, binning[1]*refscale[1]/fnewobj, 1.0)
            index[k]['deviceTransform']['scale'] = (binning[0]*refscale[0]/fnewobj, binning[1]*refscale[1]/fnewobj, 1.0)
            index[k]['objective'] = objlist[newobj]
            index[k]['note'] = f'Objective scale corrected from {oldobj:s} to {objlist[newobj]:s} on {dstr:s} by PBM'
            print('New objective: ', index[k]['objective']) # pp.pprint(index[k] )
            print('   Newtrans: ', index[k]['transform'])
            print('   Newdevtrans: ', index[k]['deviceTransform'])
            print('   Added Note: ', index[k]['note'])
            print('----------------------------')
            
    # for k in list(index.keys()):
    #     if k in list(changelist.keys()):
    #         print('newindex: ', k)
    #         print('newobj: ', index[k]['objective']) # pp.pprint(index[k] )
    #         print('newtrans: ', index[k]['transform'])
    #         print('newdevtrans: ', index[k]['deviceTransform'])
    #         print(index[k]['note'])
    
    if write:
        write_index(p, index)
        print('.index file has been modified')

    else:
        print('Dry Run: .index file was NOT modified')


def main():
    parser = argparse.ArgumentParser(description='Plot maps with traces on top')
    parser.add_argument('-w', '--write', action='store_true', dest='write',
                        help='Rewrite the .index file (otherwise, we just do a dry run)')

    args = parser.parse_args()
    
    read_index(p, args.write)

if __name__ == '__main__':
    main()




