from glob import glob

import matplotlib
import matplotlib.pyplot as mpl
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Arial']})

import ngauge
from ngauge import Neuron

files = [ Neuron.from_swc(x) for x in glob('test_Filament_0.swc') ]

funcs = [f for f in dir(files[0]) if not f.startswith("__")]
exclude = ["to_swc", "plot", "plot3d", "plot3d_mpl", "plot_mpl", "rotate", "scale", "translate", "iter_all_points", "None", "all_segment_lengths",
           "all_path_angles", "from_swc", "from_swc_text", "get_main_branch", ] 
for method in funcs:
    if method in exclude:
        continue
    print(method)
    try:
        print(getattr(files[0], method)())
    except:
        pass
         



# compute some values
# def compute_branching(file):
#     return file.branching_points()

# sns.boxplot( [ x.max_branching_order() for x in files ], color=(52/256, 66/256, 123/256) )
# mpl.show()
# # Reformat so it is the right size
# fig = mpl.gcf()
# fig.set_size_inches(2.4,0.8)
# fig.subplots_adjust(bottom=.3,left=.15)
# # plt.savefig('3E.pdf', dpi=300, transparent=True)
# mpl.show()


# # Select a neuron
# file = files[0]

# # Plot 
# fig = file.plot(fig=None, ax=None, color=(52/256, 66/256, 123/256) )

# # Format and save
# ax = fig.get_axes()[0]
# ax.axis('off')
# mpl.show()