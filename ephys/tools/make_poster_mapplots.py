from dataclasses import dataclass
from pathlib import Path
from typing import Union
import ast

import ephys.datareaders as DR
import ephys.ephys_analysis as EP
import ephys.mapanalysistools as MAT
import ephys.mini_analyses as MINIS

import ephys.mapanalysistools.plot_maps as plot_maps
import ephys.mapanalysistools.plot_map_data as plot_map_data
import matplotlib.pyplot as mpl
import src.CBA_maps as NM
import numpy as np
import pandas as pd
import pylibrary.plotting.plothelpers as PH

AR = DR.acq4_reader.acq4_reader()
PMAP = plot_maps.PlotMaps()
PMD = MAT.plot_map_data.PlotMapData()

################################
cellname = "Pyramidal"
cellno = 151
################################
# cellname = "Cartwheel"
# cellno = 1
################################

@dataclass
class CellP:
    protocol: str = ""
    LPF: float = 3000.0
    stepi: float = 150.0
    image: Union[int, None] = None
    sliceimage: Union[int, str, None] = None  # refers to
    sliceimageloc: str = (
        "slice"  # "cell" for cell directory, "slice" for slice directory
    )
    video: Union[int, None] = None

class MPP():
    def __init__(self):
        self.image_info = None
        self.video_info = None
        self.slice_info = None
        self.slice_extent = None
        self.image_extent = None
        self.video_extent = None
        self.map_extent = None
        self.videofile = None
        self.rasterize=False
        artsub = False
        self.ExptStruct = NM.experiments["NF107Ai32_Het"]

        # retrieve all of the excel/pkl tables that we will need
        # dataSummary file (a table of all cells/protocols)
        main_table = Path(self.ExptStruct["analyzeddatapath"], self.ExptStruct["directory"], self.ExptStruct["datasummaryFilename"])
        if not main_table.is_file():
            raise ValueError("Cannot find main table: ", main_table)
        self.df_main_table = pd.read_pickle(main_table)
        
        # The "selected Maps table holds specific information related to selected cells and their maps
        self.table_file = Path(self.ExptStruct["analyzeddatapath"], self.ExptStruct["directory"], self.ExptStruct["selectedMapsTable"])
        if not self.table_file.is_file():
            raise ValueError("Cannot find table file: ", self.table_file)
        self.example_table = pd.read_excel(self.table_file)

        # self.map_table_file = Path(Path(self.ExptStruct["analyzeddatapath"], self.ExptStruct["map_annotationFilename"]))
        # if not self.map_table_file.is_file():
        #     raise ValueError("Cannot find main table: ", self.map_table_file)
        # map_table = pd.read_excel(self.map_table_file)

        # select the cell from the "example table"
        print(self.example_table.cellname == "Cartwheel")
        self.cell = self.example_table.loc[
            (self.example_table.cellname == cellname) & (self.example_table.cellno == cellno)
        ]
        print("Chosen cell ID: ", self.cell.cellID.values[0])
        self.protopath = Path(self.ExptStruct["rawdatapath"],
            self.ExptStruct["directory"], self.cell.cellID.values[0], self.cell.map.values[0])
        if not self.protopath.is_dir():
            raise ValueError(f"Protocol was not found: {str(self.protopath):s}")
        protoname = self.protopath.name
        self.cellpath = Path(*self.protopath.parts[:-1])
        if not self.cellpath.is_dir():
            raise ValueError("cell path was not found: ", self.cellpath)

        # get the corresponding  cell information from the main table

        self.df_cell = self.df_main_table[self.df_main_table["cell_id"] == str(self.cell.cellID.values[0])]
        if len(self.df_cell) == 0:
            raise ValueError(f"Failed to find cell in main database: {str(self.cell.cellID.values[0]):s}")
        # Get the map annotation file - holds analysis parameters
        with open(
            Path(self.ExptStruct["analyzeddatapath"], self.ExptStruct["directory"], self.ExptStruct["map_annotationFilename"]), "rb"
        ) as fexcel:
           self.df_maps = pd.read_excel(fexcel)
        self.df_maps =self.df_maps.assign(cell_id="")  # assign some groupings
        self.df_maps =self.df_maps.apply(self.make_cell_id, axis=1)

        # then select just this cell and protocol
        self.df_map = self.df_maps[self.df_maps.cell_id == self.cell.cellID.values[0]]
        self.df_map = self.df_map[self.df_map.map == self.protopath.name]
        print('Map: ', self.df_map)
        # get the action potential rejection criteria (just by amplitude)
        self.AP_Rejects = self.cell["AP_reject"].values[0]
        if pd.isnull(self.AP_Rejects):
            self.AP_Rejects = None
        else:
            self.AP_Rejects = [float(s) for s in self.AP_Rejects.split(",")]
            if np.diff(self.AP_Rejects) == 0:
                raise ValueError("rejection must have a range")

        if protoname.startswith("Map_NewBlueLaser_VC_10Hz") and artsub:
            self.artifact_filename = Path(
                self.ExptStruct["artifactPath"],
                (self.ExptStruct["artifactFilenames"]["Map_NewBlueLaser_VC_10Hz"]),
            )
        elif protoname.startswith("Map_NewBlueLaser_VC_Singles") and artsub:
            self.artifact_filename = Path(
                self.ExptStruct["artifactPath"],
                (self.ExptStruct["artifactFilenames"]["Map_NewBlueLaser_VC_Singles"]),
            )
        else:
            print("No artifact subtraction")
            self.artifact_filename = None

    def max_intensity(self, videofile):
        vdata = AR.getImage(filename=videofile)
        # print(AR.Image_scale, AR.Image_region, AR.Image_binning, AR.Image_pos)
        return vdata, {
            "scale": AR.Image_scale,
            "region": AR.Image_region,
            "binning": AR.Image_binning,
            "position": AR.Image_pos,
        }


    def make_extent(self, imagedict, usetable:bool=False):
        """make an extent for imshow from the image metainfo

        Args:
            imagedict (_type_): image .info that includes position, scaling and binning information
            from the camera image

        Returns:
            _type_: list of points describing the "extent" of the image
        """
        if usetable and not pd.isnull(self.cell.x0.values[0]):
            x0 = self.cell.x0.values[0]
            x1 = self.cell.x1.values[0]
            y0 = self.cell.y0.values[0]
            y1 = self.cell.y1.values[0]
            print("Using table extents: ", [x0, x1, y0, y1])
            return [x0, x1, y0, y1]
        else:
            x0 = imagedict["position"][0] + imagedict["scale"][0]/imagedict["binning"][0]
            x1 = imagedict["position"][0] + imagedict["scale"][0] * imagedict["region"][2]/imagedict["binning"][0]
            y0 = imagedict["position"][1] + imagedict["scale"][1]/imagedict["binning"][1]
            y1 = imagedict["position"][1] + imagedict["scale"][1] * imagedict["region"][3]/imagedict["binning"][1]
            return [x0, x1, y0, y1]



    def plot_a_protocol(self, protopath, dc, pars, image):
        print(protopath, image)
        PMAP.setPars(pars)
        PMAP.setWindow(
            dc["x0"].values[0], dc["x1"].values[0], dc["y0"].values[0], dc["y1"].values[0]
        )
        PMAP.setOutputFile(
            Path(self.ExptStruct["directory"], "Example Maps", f"{cellname:s}_map.pdf")
        )
        prots = {"ctl": dc.cellID}

        PMAP.setProtocol(protopath, image=image)
        # print('calling plot_maps')
        PMAP.plot_maps(prots, linethickness=1.0)

    def insert_calbar(
        self, 
        ax: object,
        barlength: int = 200,
        barunitscale: float = 1.0,
        extent: Union[list, tuple] = [0.0, 1.0, 0.0, 1.0],
        color: str = "black",
    ):
        scale = barunitscale
        scaled_barlen = scale * barlength
        # print("origin: ", extent)
        x_wid = extent[1] - extent[0]  # x width of image
        y_ht = extent[3] - extent[2]  # y height of image
        # print("Lx, Ly: ", x_wid, y_ht)
        x0 = extent[1] - 0.025 * x_wid
        y0 = extent[2] - 0.05 * y_ht
        # print('x0, y0, scaled_barlen: ', x0, y0, scaled_barlen)

        ax.plot([x0 - scaled_barlen, x0], [y0, y0], color=color, linewidth=3, clip_on=False)
        micro = r"$\mu m$"
        ax.text(
            x0 - scaled_barlen / 2.0,
            y0 - 0.06 * y_ht,
            f"{barlength:d} {micro:s}",
            # ax.text(0.8, -0.05, f"{barlength:d} {micro:s}",
            fontsize=12,
            va="center",
            ha="center",
            color=color,
        )


    # find the cell in the het maps directory and set the analysis parameters
    def make_cell_id(self, row):
        if pd.isnull(row.subdirectory):
            row.cell_id = str(Path(row.date, row.slice_slice, row.cell_cell))
        else:
            row.cell_id = str(Path(row.subdirectory, row.date, row.slice_slice, row.cell_cell))
        return row

    def get_slice_image(self):
        self.slice_data = None
        self.slice_imagedict = None
        if not pd.isnull(self.cell.sliceimage.values[0]):
            slimage = self.cell.sliceimage.values[0]
            if slimage.startswith(
                "."
            ):  # then it is an image in the cell directory, not the slice directory
                self.slice_imagefile = Path(self.cellpath, slimage[1:] + ".tif")
            else:
                self.slice_imagefile = Path(
                    *Path(self.cellpath).parts[:-1], self.cell.sliceimage.values[0] + ".tif"
                )
            self.slice_data = AR.getImage(self.slice_imagefile).view(np.ndarray)
            
            self.slice_info = {
                "position": AR.Image_pos,
                "scale": AR.Image_scale,
                "region": AR.Image_region,
                "binning": AR.Image_binning,
            }
            self.slice_extent = self.make_extent(self.slice_info, usetable=False)
            print("Retrieved Slice Image ",self. slice_imagefile)
            print("with info: ", self.slice_info)


    def get_cell_video(self, videofilename):
        """Convert a video stack to an image with maximal projection

        Args:
            videofilename (_type_): _description_
        """
        self.videofile = Path(self.cellpath, f"{videofilename:s}.ma")
        self.video_data, self.video_info = self.max_intensity(self.videofile)
        self.video_data = np.max(self.video_data, axis=0)  # max intensity projection
        if self.video_data is not None:
            self.video_extent = self.make_extent(self.video_info, usetable=True)

        print("Retrieved Cell Video stack and flattened: ", self.videofile)
    
    def get_cell_image(self):
        imagefilename = self.cell.cellimage.values[0]
        self.image_data = None
        self.imagefile = None

        if not pd.isnull(imagefilename) and imagefilename.startswith("image"):
            self.imagefile = Path(self.cellpath, f"{imagefilename:s}.tif")
            self.image_data = AR.getImage(self.imagefile)
            self.image_info = {
                "position": AR.Image_pos,
                "scale": AR.Image_scale,
                "region": AR.Image_region,
                "binning": AR.Image_binning,
            }
            self.image_extent = self.make_extent(self.image_info, usetable=True)
            print("Retrieved image file ", self.imagefile)

        elif not pd.isnull(imagefilename) and imagefilename.startswith("video"):
            self.get_cell_video(imagefilename)
            
        # if self.slice_data is not None:
        #     slice_bkextent = self.make_extent(self.slice_data, self.slice_info)
        #     iminfo = slice_bkextent
        #     mpl.imshow(np.fliplr(self.slice_data), cmap="gray_r", vmax=np.max(self.slice_data), origin="lower", alpha=0.8,
        #         extent = slice_bkextent)#  ax = PMD.plotspecs["A"])


    def get_background_images(self):
        self.get_slice_image()
        self.get_cell_image()

    def add_label(self, AM, PMD):

        label = f"{self.cell.cellID.values[0]:s}, {self.cell.map.values[0]:s} {self.cell['cellname'].values[0]:s}"
        label += f" {self.df_cell['age'].values[0]:s} {self.df_cell['sex'].values[0]:s} "
        label += f"{self.df_cell['internal'].values[0]:s} {self.df_cell['temperature'].values[0]:s}"
        # get VC Comp information to add to notes on page
        # dict_keys(['WCCompValid', 'WCEnabled', 'WCResistance', 'WCCellCap', 'RsCompCorrection', 'CompEnabled', 'CompCorrection', 'CompBW'])

        print(dir(self.AR))
        comp = self.AM.AR.WCComp
        if comp['WCEnabled']:
            comp_label = f" Rs: {comp['WCResistance']*1e-6:.1f}M$\Omega$ Corr: {int(comp['RsCompCorrection']):2d}% Cm: {comp['WCCellCap']*1e12:.1f} pF"
        else:
            comp_label = f" Uncompensated"
        label += comp_label    
        mpl.text(
            0.95, 0.01, label, ha="right", fontsize=8, transform=PMD.P.figure_handle.transFigure
        )

    def plot_cell_image(self, ax:object, data: np.array, info: dict, origin:list, cell_position:list=None,
            invert:bool=True, show_scannerbox:bool=False, show_scannerpoints:bool=False,
            usetable:bool=False, slice:bool=False, gamma:float=None):

        extent = self.make_extent(info, usetable=usetable)
        extent = [extent[0], extent[1], extent[3], extent[2]]  # flip so matplplotlib places data on image correctly
        if slice and self.offset is not None:
            extent =[extent[0]-self.offset[0], extent[1]-self.offset[0],
                     extent[2]-self.offset[1], extent[3]-self.offset[1]]
            
        PH.noaxes(ax)

        SCInfo = plot_maps.ScannerInfo(self.AM.AR)
        if show_scannerbox:
            ax.plot(SCInfo.scboxw[0,:], SCInfo.scboxw[1,:], 'r-', linewidth=0.3)
    
        if show_scannerpoints:
            PMAP.plot_scanner_locations(
                ax=ax,
                scpos=SCInfo.scanner_positions,
                color="b",
                size=3,
                marker="o",
                alpha=0.5,
            )

        vmin=np.min(data)
        vmax=np.max(data)
        if usetable:
            grextent = None
        else:
            grextent = extent
        if invert:
            cmap = "gray_r"
        else:
            cmap = "gray"
        if gamma is not None:
            data = PMD.gamma_correction(data, gamma)
  
        ax.imshow(
            data,
            cmap=cmap,
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
            origin="upper",  # standard for camera image
            # alpha=1.0,
            extent=grextent,
        )  
        if usetable:
            ax.set_xlim(self.cell.x0.values[0], self.cell.x1.values[0])
            ax.set_ylim(self.cell.y0.values[0], self.cell.y1.values[0])
        # grextent = [self.cell.x0.values[0], self.cell.x1.values[0], self.cell.y0.values[0], self.cell.y1.values[0]]
        if not slice:
            grextent = [self.cell.x0.values[0], self.cell.x1.values[0], self.cell.y0.values[0], self.cell.y1.values[0]]
        self.insert_calbar(
            ax,
            barlength=200,
            barunitscale=1e-6,
            extent=grextent,
            color="black",
        )
        print("cell cell position: ", cell_position)
        if not slice:
            marker = 'o'
            size=10
            radius = 25e-6
            edgecolor = 'r'
            facecolor = "None"
        else:
            marker = (8, 2, 0)
            size=4
            radius = 0.1e-4
            edgecolor = "r"
            facecolor = "r"

        if cell_position is not None:
            ax.add_patch(
                mpl.Circle((cell_position[0], cell_position[1]),
                radius=radius,
                facecolor = facecolor,
                edgecolor = edgecolor,
                alpha=0.5,
                )
            )
            
            # ax.plot(
            #     cell_position[0],
            #     cell_position[1],
            #     marker=marker,
            #     # color="r",
            #     markerfacecolor = facecolor,
            #     markeredgecolor = edgecolor,
            #     alpha=0.5,
            #     markersize=size,
            # )

    

    def analyze(self):
        # if artifact_file is not None:
    #     df_art = pd.read_pickle(artifact_file)

        rotation = 0.0
        notch_f = self.cell.notch_freqs.values[0]
        if not pd.isnull(notch_f):
            notch_f = ast.literal_eval(notch_f)
        print("Notch freqs: ", notch_f)
        AM = MAT.analyze_map_data.AnalyzeMap()
            # load up the analysis modules (don't allow multiple instances to exist)
        self.SP = EP.spike_analysis.SpikeAnalysis()
        self.RM = EP.rm_tau_analysis.RmTauAnalysis()
        self.AR = DR.acq4_reader.acq4_reader()
        self.MA = MINIS.minis_methods.MiniAnalyses()
        self.AM = MAT.analyze_map_data.AnalyzeMap(rasterize=self.rasterize)
        self.AM.configure(
            reader=self.AR,
            spikeanalyzer=self.SP,
            rmtauanalyzer=self.RM,
            minianalyzer=self.MA,
        )

        self.AM.sign = "-"
        self.AM.overlay_scale = 1.0
        self.AM.set_taus((float(self.df_map.tau1.values[0]) * 1e-3, float(self.df_map.tau2.values[0]) * 1e-3,
        float(3*self.df_map.tau1.values[0]) * 1e-3, float(3*self.df_map.tau2.values[0]) * 1e-3))
        self.AM.Pars.threshold = float(self.df_map.threshold.values[0])
        self.AM.Pars.scale_factor = 1e12
        self.AM.Pars.stepi = self.cell.I_step.values[0]
        self.AM.set_LPF(self.cell.LPF.values[0])
        if not pd.isnull(notch_f):
            self.AM.set_notch(notch_f)

        self.AM.set_artifact_path(self.ExptStruct["artifactPath"])
        self.AM.set_artifact_filename(self.artifact_filename)
        self.AM.set_artifact_suppression(True)


        results = self.AM.analyze_one_map(self.protopath)
        PMD.set_Pars_and_Data(self.AM.Pars, self.AM.Data, self.MA)
        if not pd.isnull(self.cell.cellpos.values[0]):
            cell_position = [float(f) for f in self.cell.cellpos.values[0].split(",")]
        else:
            cell_position = None
        if not pd.isnull(self.cell.offset.values[0]):
            self.offset = [float(f) for f in self.cell.offset.values[0].split(",")]
        else:
            self.offset = None

        if self.imagefile is None and self.slice_imagedict is not None:
            imagefile = self.slice_imagefile
        elif self.imagefile is not None:
            imagefile = self.imagefile
        elif self.videofile is not None:
            imagefile = self.videofile
        zs_imagefile = None
        PMD.display_one_map(
            self.protopath,
            results=results,
            datatype=self.AM.Pars.datatype,
            imagefile=zs_imagefile,
            rotation=rotation,
            measuretype="ZScore",
            zscore_threshold=1.96,  # 1.96 is 5% criteria
            plotmode="publication",
            markers={"soma": cell_position},
            plot_minmax=self.AP_Rejects,
            cal_height=float(self.cell.calbar.values[0].split(",")[1]),
        )
        self.add_label(AM, PMD)

        PMAP.setWindow(
            self.cell["x0"].values[0],
            self.cell["x1"].values[0],
            self.cell["y0"].values[0],
            self.cell["y1"].values[0],
        )
        zs_ax = PMD.P.axdict[PMD.panels["map_panel"]]
        origin = np.array((PMAP.xlim, PMAP.ylim)).ravel()
        if cell_position is not None:
            zs_ax.plot(
                cell_position[0], cell_position[1], marker=(8, 2, 0), color="w", markersize=9
            )
  
        if zs_imagefile is not None:
            extent = self.make_extent(self.image_info)
        else:
            xl = zs_ax.get_xlim()
            yl = zs_ax.get_ylim()
            extent = [xl[0], xl[1], yl[0], yl[1]]

        self.insert_calbar(
            zs_ax,
            barlength=200,
            barunitscale=1e-6,
            extent=extent,
            color="black",
        )
        PH.noaxes(zs_ax)
        map_xlim = zs_ax.get_xlim()
        map_ylim = zs_ax.get_ylim()  # use these to scale the image

        sl_ax =  PMD.P.axdict[PMD.panels["slice_image_panel"]]
        im_ax = PMD.P.axdict[PMD.panels["cell_image_panel"]]
        if self.slice_info is not None:

            self.plot_cell_image(ax=sl_ax, data=self.slice_data, info=self.slice_info,
                origin=origin, cell_position=cell_position, invert=False,
                show_scannerbox=True, show_scannerpoints=False, usetable=False, slice=True)
        gamma = self.cell.gamma.values[0]
        if pd.isnull(gamma):
            gamma = None
        else:
            gamma = float(gamma)
        if self.image_info is not None:
            self.plot_cell_image(ax=im_ax, data=self.image_data, info=self.image_info,
                origin=origin, cell_position=cell_position, show_scannerbox=True,
                show_scannerpoints=False, usetable=False, gamma=gamma)

        if self.video_info is not None:
            self.plot_cell_image(ax=im_ax, data=self.video_data, info=self.video_info,
                origin=origin, cell_position=cell_position,
                show_scannerbox=True, show_scannerpoints=False, usetable=False)
        print("cell position: ", cell_position)
        if not pd.isnull(self.cell.x0.values[0]):
            x0 = self.cell.x0.values[0]
            x1 = self.cell.x1.values[0]
            y0 = self.cell.y0.values[0]
            y1 = self.cell.y1.values[0]
            print("Using table extents: ", [x0, x1, y0, y1])
            im_ax.set_xlim(x0, x1)
            im_ax.set_ylim(y0, y1)


        # last thing - if IV_traces is not empty, plot the traces in an inset with a little calbar. 
        if not pd.isnull(self.cell.IV_traces.values[0]):
            self.iv_protopath = Path(self.ExptStruct["rawdatapath"],
                self.cell.cellID.values[0], str(self.cell.IV_protocol.values[0]))
            AR.setProtocol(self.iv_protopath)
            traces = ast.literal_eval(self.cell.IV_traces.values[0])
            traces = [int(x) for x in traces]
            if not pd.isnull(self.cell.IV_cal.values[0]):
                iv_cal = self.cell.IV_cal.values[0]
                iv_cal = ast.literal_eval(iv_cal)
                iv_cal = [float(x) for x in iv_cal]
            else:
                iv_cal = [250, 20]
            if not pd.isnull(self.cell.IV_pos.values[0]):
                iv_pos = ast.literal_eval(self.cell.IV_pos.values[0])
                iv_pos = [float(x) for x in iv_pos]
            else:
                iv_pos = [0.49, 0.82]
            if not pd.isnull(self.cell.IV_Ilims.values[0]):
                iv_ilims = ast.literal_eval(self.cell.IV_Ilims.values[0])
                iv_ilims = [float(x) for x in iv_ilims]
            else:
                iv_ilims = None
            if not pd.isnull(self.cell.IV_caly.values[0]):
                iv_calypos= ast.literal_eval(self.cell.IV_caly.values[0])
                iv_calypos = [float(x) for x in iv_calypos]
            else:
                iv_calypos = [-20.0, 0.5]
            if not pd.isnull(self.cell.IV_caldirection.values[0]):
                iv_caldir= str(self.cell.IV_caldirection.values[0])
            else:
                iv_caldir = "right"
            AR.getData()
            maxt = int(np.max(AR.time_base)*1e3)
            ax2 = PMD.P.figure_handle.add_axes([iv_pos[0],iv_pos[1],0.1,0.12])
            ax2.patch.set_alpha(0)
            PH.noaxes(ax2)
            PH.calbar(ax2, calbar=[maxt-iv_cal[0], iv_calypos[0], iv_cal[0], iv_cal[1]], orient=iv_caldir,
                 unitNames={"x": "ms", "y": "mV"}, fontsize=6, linewidth=1)
            ax3 = PMD.P.figure_handle.add_axes([iv_pos[0],iv_pos[1],0.1,0.03])
            ax3.patch.set_alpha(0)
            PH.noaxes(ax3)
            PH.calbar(ax3, calbar=[maxt-iv_cal[0], iv_calypos[1], iv_cal[0], 1], orient=iv_caldir,
                 unitNames={"x": "ms", "y": "nA"}, fontsize=6, linewidth=1)


            for tr in traces:
                ax2.plot(AR.time_base*1e3, 1e3*AR.traces.view(np.ndarray)[tr], color='k', linewidth=0.33)
                ax3.plot(AR.time_base*1e3, 1e9*AR.cmd_wave.view(np.ndarray)[tr], color='k', linewidth=0.33)
            ax2.set_ylim(-120, 50)
            ax2.set_xlim(0, maxt)
            ax3.set_xlim(0, maxt)
            if iv_ilims is not None:
                ax3.set_ylim(iv_ilims[0], iv_ilims[1])

        mpl.savefig(
            Path(self.ExptStruct["analyzeddatapath"], f"{cellname:s}_{cellno:03d}_poster.pdf")
        )
        mpl.show()

def main():
    mpp = MPP()  # initialize
    mpp.get_background_images()
    mpp.analyze()

if __name__ == "__main__":
    main()