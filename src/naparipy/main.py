
from PyQt5.Qt import QApplication
"""
A PyQt-based application for analyzing and visualizing imaging data using napari viewer.
This module provides a main window interface that allows users to:
- Load and process image data
- Split multi-channel images
- Create and visualize ROI-based traces
- Apply filters and plot data
Classes:
    mainWindow: Main application window deriving from pyqtgraph's GraphicsView
Methods:
    stackedPlot(a): Helper function to create stacked plots by offsetting traces
The mainWindow class provides the following key functionality:
- Parameter tree for user controls
- Napari viewer integration for image visualization
- Plot window for trace visualization
- ROI-based analysis capabilities
- Channel splitting for multi-channel data
- Real-time plot updates
- Savitzky-Golay filtering options
Dependencies:
    - PyQt5
    - pyqtgraph
    - napari
    - numpy
    - scipy
Usage:
    Run this module directly to launch the GUI application:
    ```python
    python main.py
    ```
"""
from PyQt5.QtCore import Qt
from pyqtgraph.parametertree import Parameter, ParameterTree
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore,QtWidgets
import sys
import numpy as np
from scipy.signal import find_peaks, savgol_filter
import napari

pg.setConfigOptions(antialias=True)
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

COLORS = ['k','b','r','g','orange','y','cyan']

_instance = QApplication.instance()
if not _instance:
    _instance = QApplication([])
app = _instance


def stackedPlot(a):
    a2 = a.copy()
    for i in range(1,a2.shape[1]):
        a2[:,i] = a2[:,i] + np.nanmax(a2[:,i-1])

    return a2

class mainWindow(pg.GraphicsView):
    """
    A PyQtGraph-based main window for visualizing and analyzing image data with napari integration.
    This class creates a window containing a plot area and parameter controls for image processing
    and visualization. It interfaces with napari viewer for image display and ROI selection.
    Parameters
    ----------
    parent : QWidget, optional
        Parent widget. Default is None.
    useOpenGL : bool, optional
        Whether to use OpenGL for rendering. Default is None.
    background : str, optional
        Background color setting. Default is 'default'.
    Attributes
    ----------
    layout : pg.GraphicsLayout
        Main layout for the plot area
    outerlayout : pg.GraphicsLayout
        Outer layout container
    plot : pg.PlotItem
        Main plot widget
    viewer : napari.Viewer
        Napari viewer instance for image visualization
    p : Parameter
        Parameter tree containing all UI controls
    t : ParameterTree
        Tree widget displaying all parameters
    Methods
    -------
    changeLayersCb()
        Updates layer selection dropdown menus when napari layers change
    splitChannelsCb()
        Splits selected image into two channels and creates ratio images
    plotCb()
        Plots intensity traces from selected ROIs in the image
    """

    def __init__(self, parent=None, useOpenGL=None, background='default'):
        super().__init__(parent, useOpenGL, background)
        self.setWindowTitle('Your title') 
        self.setGeometry(0,0,300,200)
        self.layout = pg.GraphicsLayout()
        self.outerlayout = pg.GraphicsLayout()
        self.setCentralItem(self.layout)
        self.sc = self.scene()
        self.sc2 = self.layout.scene()
        self.plot = self.layout.addPlot(row=0,col=0)
        #self.plot.scene().sigMouseClicked.connect(self.mouseClickEventLine)

        params = [{'name':'File opening','type':'group','children':[
        # {'name':'Strain','type':'list','values':['6N','Repaired']},
            {'name':'Open thorlabs raw','type':'file'},
    
        ]},
        {'name':'Image processing', 'type' : 'group','children':[
            {'name':'Split 2 channels','type':'action'},
        ]},
        {'name':'Plot', 'type' : 'group','children':[
            {'name':'Plot','type':'action'},
            {'name':'Image layer','type':'list','value':''},
            {'name':'ROI layer','type':'list','value':''},
            {'name':'Type','type':'list','values':['F','dF/F0']},
            {'name':'F0 frame','type':'int','value':0},
            {'name':'Framerate','type':'float','value':1.0},
            {'name':'Savgol filter order','type':'int'},
            {'name':'Stacked','type':'bool'}
        ]}
        
        ]

        
        self.p = Parameter.create(name='params', type='group', children=params)

        self.t = ParameterTree()
        self.t.setParameters(self.p, showTop=False)
        self.t.setGeometry(1300,300,300,700)
        self.t.show()
        self.viewer = napari.Viewer()

        #Make connections
        #Viewers events
        self.viewer.layers.events.inserted.connect(self.changeLayersCb)
        self.viewer.layers.events.removed.connect(self.changeLayersCb)
        self.viewer.layers.events.changed.connect(self.changeLayersCb)
            #change layer list when you change one name
        @self.viewer.layers.events.inserted.connect
        def _on_insert(event):
            layer = event.value
            if layer.__class__ == napari.layers.labels.labels.Labels:
                layer.data = layer.data.max(0)# reduce dimension of labels to 2
            @layer.events.name.connect
            def _on_rename(name_event):
                self.changeLayersCb()
        
        #Image processing events:
        self.p.keys()['Image processing'].keys()['Split 2 channels'].sigActivated.connect(self.splitChannelsCb)
#            print(self.p.param('File opening','Open thorlabs raw'))

        #Plot events
        self.p.keys()['Plot'].keys()['Plot'].sigActivated.connect(self.plotCb)
        

        self.show()

    def changeLayersCb(self):
        names = [l.name for l in self.viewer.layers]

        self.p.keys()['Plot'].keys()['Image layer'].setLimits(names)
        self.p.keys()['Plot'].keys()['ROI layer'].setLimits(names)

    def splitChannelsCb(self):
        data =self.viewer.layers[self.p.param('Plot','Image layer').value()].data
        if data.shape[0]%2 !=0:
            data = data[:-1,:,:]
        data1 = data[::2,:,:]
        data2 = data[1::2,:,:]

        
        self.viewer.add_image(data1,name='Channel 1')
        self.viewer.add_image(data2,name='Channel 2')
        self.viewer.add_image(data1/data2,name='Ratio')
        self.viewer.add_image(data2/data1,name='Ratio 2')

    def plotCb(self):
        maskLayer = self.p.param('Plot','ROI layer').value()
        imageLayer = self.p.param('Plot','Image layer').value()
        mask = self.viewer.layers[maskLayer].data
        l = self.viewer.layers[maskLayer]
        self.plot.clear()
        traces = []
        for i in range(1,mask.max()+1):
            if np.size(np.argwhere(mask==i))>0:
                z = self.viewer.layers[imageLayer].data[:,np.argwhere(mask==i)[:,0],np.argwhere(mask==i)[:,1]].mean(1)
                traces.append(z)
                
        traces = np.array(traces).T
        print(traces.shape)
        if self.p.param('Plot','Savgol filter order').value()>0:
            order = self.p.param('Plot','Savgol filter order').value()
            for i in range(traces.shape[1]):
                traces[:,i] = savgol_filter(traces[:,i],order,1)

        if self.p.param('Plot','Stacked').value():
            traces = stackedPlot(traces)

        for i in range(traces.shape[1]):
            color = l.get_color(i+1)*255
            color = color.astype(np.uint16)
            pen = pg.mkPen(color)
            self.plot.plot(traces[:,i],pen=pen,name=str(i))

                #color = l.get_color(i)*255
                #color = "rgb({} , {} ,{}, {})".format( *color)


                #s

if __name__ == '__main__':
    win = mainWindow()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_() 

