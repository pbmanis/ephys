import numpy as np
import pyqtgraph as pg
import scipy.spatial
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.Point import Point

# import matplotlib.pyplot as plt

# from matplotlib.backend_bases import MouseEvent
# from matplotlib.widgets import Cursor


class AnnotatedCursorPG(pg.GraphicsObject):
    """
    PYQTGRAPH version.

    Based on the matplotlib example, with the following change (pbmanis):
    1. The cursor can be set to stick to the line, or be free.
    2. When sticking to the line, we compute the closest point to the 
    cursor position on the line.

    A crosshair cursor like `~matplotlib.widgets.Cursor` with a text showing
    the current coordinates.

    For the cursor to remain responsive you must keep a reference to it.
    The data of the axis specified as *dataaxis* does not have to be
    in ascending order. 
    
    Parameters
    ----------
    line : `matplotlib.lines.Line2D`
        The plot line from which the data coordinates are displayed.

    numberformat : `python format string <https://docs.python.org/3/\
    library/string.html#formatstrings>`_, optional, default: "{0:.4g};{1:.4g}"
        The displayed text is created by calling *format()* on this string
        with the two coordinates.

    offset : (float, float) default: (5, 5)
        The offset in display (pixel) coordinates of the text position
        relative to the cross-hair.

    dataaxis : {"x", "y"}, optional, default: "x"
        If "x" is specified, the vertical cursor line sticks to the mouse
        pointer. The horizontal cursor line sticks to *line*
        at that x value. The text shows the data coordinates of *line*
        at the pointed x value. If you specify "y", it works in the opposite
        manner. But: For the "y" value, where the mouse points to, there might
        be multiple matching x values, if the plotted function is not biunique.
        Cursor and text coordinate will always refer to only one x value.
        So if you use the parameter value "y", ensure that your function is
        biunique.

    Other Parameters
    ----------------
    textprops : `matplotlib.text` properties as dictionary
        Specifies the appearance of the rendered text object.

    **cursorargs : `matplotlib.widgets.Cursor` properties
        Arguments passed to the internal `~matplotlib.widgets.Cursor` instance.
        The `matplotlib.axes.Axes` argument is mandatory! The parameter
        *useblit* can be set to *True* in order to achieve faster rendering.

    """

    def __init__(
        self,
        line,
        ax,  # plot instance
        numberformat="{0:.4g};{1:.4g}",
        mode: str = "free",
        offset=(5, 5),
        color='y',
        dataaxis="x",
        textprops=None,
        report_func=None,
        **cursorargs,
    ):
        """__init__ method for AnnotatedCursor.

        Parameters
        ----------
        line : _type_
            _description_
        numberformat : str, optional
            _description_, by default "{0:.4g};{1:.4g}"
        mode : str, optional
            _description_, by default "free"
            if Mode is "free", then the cursor can go anywhere in the plot and the
            value will be shown.
            if Mode is "stick", then the cursor will stick to the line as best it can.
        offset : tuple, optional
            _description_, by default (5, 5)
        dataaxis : str, optional
            _description_, by default 'x'
        textprops : _type_, optional
            _description_, by default None
        """
        assert mode in ["free", "stick"]

        if textprops is None:
            textprops = {}
        self.ax = ax
        self.line = np.array(line.getData())

        # The format string, on which .format() is called for creating the text
        self.numberformat = numberformat
        # the mode ("free" or "stick)
        self.mode = mode
        # Text position offset
        self.offset = np.array(offset)
        # The axis in which the cursor position is looked up
        self.dataaxis = dataaxis
        self.cursor_position = [0, 0]
        self.text_follows_cursor = False
        self.tracker_vLine = None  # add a tracker point on another plot
        self.tracking_cursor = None
        self.report_func = report_func

        # Create invisible animated text
        self.text = pg.LabelItem("0, 0")
        self.text.setParentItem(self.ax.graphicsItem())
        self.text.anchor(itemPos=(0.0, 0.0), parentPos=(0.1, 0.1), offset=offset)

        # The position at which the cursor was last drawn
        self.lastdrawnplotpoint = None
        # cross hair
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color=color, width=0.7))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(color=color, width=0.7))
        self.ax.addItem(self.vLine, ignoreBounds=True)
        self.ax.addItem(self.hLine, ignoreBounds=True)
        self.vb = self.ax.getViewBox()
        super().__init__()
        self.ax.scene().sigMouseMoved.connect(self.mouseMoved)
        self.ax.scene().sigMouseClicked.connect(self.mouseClicked)

    def set_tracker_from(self, cursor, orientation='v', color="r"):
        """Set a tracker line on another plot.
        The line position is controlled by the "host" axis
        Parameters
        ----------
        target_plot : _type_
            _description_
        color : str, optional
            _description_, by default "r"
        """
        self.tracking_cursor = cursor
        if orientation == 'v':
            angle = 90
        else:
            angle = 0.
        self.tracking_cursor.tracker_vLine = pg.InfiniteLine(
            angle=angle, movable=False, pen=pg.mkPen('m', width=0.7))
        self.tracking_cursor.ax.addItem(self.tracking_cursor.tracker_vLine)

    def mouseMoved(self, event):
        if not self.ax.sceneBoundingRect().contains(
            event
        ):  # check if mouse is in our current axis scene
            return  # if not, just return
        
        mousePoint = self.vb.mapSceneToView(
            event
        )  # map the mouse position to the data in the window


        # this might belong in the init section to be faster,
        # but if the line is changed...
        # self.minx = np.min(self.line[0, :])
        # self.miny = np.min(self.line[1, :])
        # self.maxx = np.max(self.line[0, :])
        # self.maxy = np.max(self.line[1, :])

        # # check if the mouse value is in the range of the data
        # in_window:bool = (
        #     mousePoint.x() >= self.minx
        #     and mousePoint.x() <= self.maxx
        #     and mousePoint.y() >= self.miny
        #     and mousePoint.y() <= self.maxy
        # )

        if self.mode == "free":
            "The cursor is free to move anywhere in the plot"
            self.text.setText(
                "<span style='font-size: 14pt; font_color: #aaaaff'>x=%0.3f, y1=%0.3f</span>"
                % (mousePoint.x(), mousePoint.y())
            )
            self.cursor_position = [mousePoint.x(), mousePoint.y()]
        elif self.mode == "stick":
            # the cursor sticks to the closest point on the line
            # if not in_window:
            #     return
            distance = scipy.spatial.distance.cdist(
                self.line.T, np.array([[mousePoint.x(), mousePoint.y()]]), "euclidean"
            )
            index = np.argmin(distance)
            self.cursor_position = [self.line[0,index], self.line[1,index]]

        if self.mode in ["stick", "free"]:
            self.vLine.setPos(self.cursor_position[0])
            self.hLine.setPos(self.cursor_position[1])
            self.text.setText(
                "<span style='font-size: 14p; font_color: #008888'>x=%0.3f, y1=%0.3f</span>"
                % (self.cursor_position[0], self.cursor_position[1])
            )
        
        if self.tracking_cursor is not None:
            self.tracking_cursor.tracker_vLine.setPos(self.cursor_position[0])
        return

    def mouseClicked(self, event):
        # if self.reportdock is None:
        #     return
        if not self.ax.sceneBoundingRect().contains(
            event.scenePos()
        ):  # not our event
            return
        pos = self.cursor_position  # get the cursor position
        if self.report_func is None:
            print(f"Clicked at {pos}")   
        else:
            self.report_func(pos)     

        


class test:
    def __init__(self):
        pass

    def make_graph(self):
        # generate layout
        print("test")
        self.app = pg.mkQApp("testing cursor")
        self.win = pg.GraphicsLayoutWidget(show=True, title="Test Cursor")
        self.p1 = self.win.addPlot(row=1, col=0)
        self.data1 = 0.1 * np.random.random(size=10000)

        line = self.p1.plot(self.data1, pen="r")
        cursor = AnnotatedCursorPG(line=line, ax=self.p1, mode="free")

        return cursor


if __name__ == "__main__":

    ct = test()
    cursor = ct.make_graph()
    # if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    pg.exec()
