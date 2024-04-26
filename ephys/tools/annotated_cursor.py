import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backend_bases import MouseEvent
from matplotlib.widgets import Cursor
import scipy.spatial


class AnnotatedCursor(Cursor):
    """
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

    def __init__(self, line, numberformat="{0:.4g};{1:.4g}", mode:str="free", offset=(5, 5),
                 dataaxis='x', textprops=None, **cursorargs):
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
        # The line object, for which the coordinates are displayed
        self.line = line
        self.lines = np.array([line.get_xdata(), line.get_ydata()]).T
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
        # First call baseclass constructor.
        # Draws cursor and remembers background for blitting.
        # Saves ax as class attribute.
        super().__init__(**cursorargs)

        # Default value for position of text.
        self.set_position(self.line.get_xdata()[0], self.line.get_ydata()[0])
        # Create invisible animated text
        self.text = self.ax.text(
            self.ax.get_xbound()[0],
            self.ax.get_ybound()[0],
            "0, 0",
            animated=bool(self.useblit),
            visible=False, **textprops)

        # The position at which the cursor was last drawn
        self.lastdrawnplotpoint = None

    def onmove(self, event):
        """
        Overridden draw callback for cursor. Called when moving the mouse.
        """

        # Leave method under the same conditions as in overridden method
        if self.ignore(event):
            self.lastdrawnplotpoint = None
            return
        if not self.canvas.widgetlock.available(self):
            self.lastdrawnplotpoint = None
            return

        # If the mouse left the drawable area, we make the text invisible.
        # The baseclass will redraw complete canvas after, which makes both text
        # and cursor disappear.
        if event.inaxes != self.ax:
            self.lastdrawnplotpoint = None
            self.text.set_visible(False)
            super().onmove(event)
            return

        # Get the coordinates, which should be displayed as text,
        # if the event coordinates are valid.
        plotpoint = None
        # print("event.xdata", event.xdata, "event.ydata", event.ydata)
        if event.xdata is not None and event.ydata is not None:
            # Get plot point related to current x position.
            # These coordinates are displayed in text.
            plotpoint = self.set_position(event.xdata, event.ydata)
            # Modify event, such that the cursor is displayed on the
            # plotted line, not at the mouse pointer,
            # if the returned plot point is valid
            if plotpoint is not None:
                event.xdata = plotpoint[0]
                event.ydata = plotpoint[1]
                # print("plotpoint is not None", plotpoint)
            else:
                return # ignore

        # If the plotpoint is given, compare to last drawn plotpoint and
        # return if they are the same.
        # Skip even the call of the base class, because this would restore the
        # background, draw the cursor lines and would leave us the job to
        # re-draw the text.
        if plotpoint is not None and plotpoint == self.lastdrawnplotpoint:
            return

        # Now call the baseclass, which will redraw the canvas and the cursor.
        # Due to blitting,
        # the added text is removed in this call, because the
        # background is redrawn.
        super().onmove(event)

        # Check if the display of text is still necessary.
        # If not, just return.
        # This behaviour is also cloned from the base class.
        if not self.get_active() or not self.visible:
            print("cursor not active!")
            return

        # Draw the widget, if event coordinates are valid.
        if plotpoint is not None:
            # Update position and displayed text.
            # Position: Where the event occurred.
            # Text: Determined by set_position() method earlier
            # Position is transformed to pixel coordinates,
            # an offset is added there and this is transformed back.
            temp = [event.xdata, event.ydata]

            if self.text_follows_cursor:  # if the text follows the cursor
                temp2 = self.ax.transData.transform(temp)
                temp2 = temp2 + self.offset
                temp2 = self.ax.transData.inverted().transform(temp2)
                self.text.set_position(temp2)
            else:  # set text poisiton to be fixed in the lower left corner
                temp2 = self.ax.transAxes.transform([self.ax.get_xbound()[0], self.ax.get_ybound()[0]])
                # temp2 = temp2 + np.array([5, 5])
                temp2 = self.ax.transAxes.inverted().transform(temp2)
                self.text.set_position([self.ax.get_xbound()[0], self.ax.get_ybound()[0]]) # temp2)
                
            self.text.set_text(self.numberformat.format(*plotpoint))
            self.text.set_visible(self.visible)
            xy = self.ax.transData.transform(temp)
            self.set_position(xy[0], xy[1])

            # Tell base class, that we have drawn something.
            # Baseclass needs to know, that it needs to restore a clean
            # background, if the cursor leaves our figure context.
            self.needclear = True

            # Remember the recently drawn cursor position, so events for the
            # same position (mouse moves slightly between two plot points)
            # can be skipped
            self.lastdrawnplotpoint = plotpoint
        # otherwise, make text invisible
        else:
            self.text.set_visible(False)

        # Draw changes. Cannot use _update method of baseclass,
        # because it would first restore the background, which
        # is done already and is not necessary.
        if self.useblit:
            self.ax.draw_artist(self.text)
            self.canvas.blit(self.ax.bbox)
        else:
            # If blitting is deactivated, the overridden _update call made
            # by the base class immediately returned.
            # We still have to draw the changes.
            self.canvas.draw_idle()

    def set_position(self, xpos, ypos):
        """
        Finds the coordinates, which have to be shown in text.

        The behaviour depends on the *dataaxis* attribute. Function looks
        up the matching plot coordinate for the given mouse position.

        Parameters
        ----------
        xpos : float
            The current x position of the cursor in data coordinates.
            Important if *dataaxis* is set to 'x'.
        ypos : float
            The current y position of the cursor in data coordinates.
            Important if *dataaxis* is set to 'y'.

        Returns
        -------
        ret : {2D array-like, None}
            The coordinates which should be displayed.
            *None* is the fallback value.
        """
        self.cursor_position = [xpos, ypos]
        if self.mode == "free":
            return self.cursor_position
        
        # If position is valid and in valid plot data range.
        if self.dataaxis == 'x':
            pos = xpos
            # data = self.line.get_xdata()
            lim = self.ax.get_xlim()
        elif self.dataaxis == 'y':
            pos = ypos
            # data = self.line.get_ydata()
            lim = self.ax.get_ylim()
        if self.mode == "stick" and (pos is not None and lim[0] <= pos <= lim[-1]):
        # Get plot line data
        # xdata = self.line.get_xdata()
        # ydata = self.line.get_ydata()
        
        # The dataaxis attribute decides, in which axis we look up which cursor
        # coordinate.
            if self.dataaxis == 'x':
                # pos = xpos
                # data = xdata
                lim = self.ax.get_xlim()
                if lim[0] <= xpos <= lim[-1]:
                    y = scipy.spatial.distance.cdist(self.lines, np.array([[xpos], [ypos]]).T, 'euclidean')
                    index = np.argmin(y)
                    self.cursor_position = [self.lines[index][0], self.lines[index][1]]
                    return self.cursor_position
                    # index = np.searchsorted(xdata[x_sorted], xpos)
                    # if index < 0 or index >= len(xdata):
                    #     return None
                    # self.cursor_position[0] = xdata[x_sorted[index]]
            # elif self.dataaxis == 'y':
            #     pos = ypos
            #     # data = ydata
            #     lim = self.ax.get_ylim()
            #     if lim[0] <= ypos <= lim[-1]:
            #         y_sorted = np.argsort(ydata)
            #         index = np.searchsorted(ydata[y_sorted], ypos)
            #         if index < 0 or index >= len(ydata):
            #             return None
            #         self.cursor_position[1] = ydata[y_sorted[index]]
            else:
                raise ValueError(f"The data axis specifier {self.dataaxis} should "
                                f"be 'x' or 'y'")

        return None

    def clear(self, event):
        """
        Overridden clear callback for cursor, called before drawing the figure.
        """

        # The base class saves the clean background for blitting.
        # Text and cursor are invisible,
        # until the first mouse move event occurs.
        super().clear(event)
        if self.ignore(event):
            return
        self.text.set_visible(False)

    def _update(self):
        """
        Overridden method for either blitting or drawing the widget canvas.

        Passes call to base class if blitting is activated, only.
        In other cases, one draw_idle call is enough, which is placed
        explicitly in this class (see *onmove()*).
        In that case, `~matplotlib.widgets.Cursor` is not supposed to draw
        something using this method.
        """

        if self.useblit:
            super()._update()