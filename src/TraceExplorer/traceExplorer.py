"""
A PyQt5-based GUI application for exploring and analyzing trace data, particularly focused on peak detection and visualization.
This module provides a comprehensive interface for:
- Loading and managing trace data from various file formats
- Interactive peak detection and manipulation
- Visualization of stacked plots with customizable parameters
- Color-coded grouping and categorization of traces
- Real-time parameter adjustments and plot updates
The main window includes:
- A central plotting area for trace visualization
- Parameter trees for controlling data loading and visualization options
- Peak detection controls with adjustable parameters
- Interactive peak addition and deletion capabilities
Key features:
- Support for multiple data groups and cell types
- Customizable peak detection parameters
- Interactive plot manipulation
- Data export functionality
- Keyboard shortcuts for common operations
Dependencies:
    - PyQt5
    - pyqtgraph
    - pandas
    - numpy
    - scipy
    - shapely
Note: This module is part of the In-Vivo-Analysis project and is designed for analyzing
biological trace data, particularly for neural and hair cell activity analysis.
"""

from PyQt5.Qt import QApplication
from PyQt5.QtCore import Qt
from pyqtgraph.parametertree import Parameter, ParameterTree
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore,QtWidgets
import sys
import os
import pandas as pd
import numpy as np
import ast
from scipy.signal import find_peaks
import shapely
from scipy.signal import peak_widths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from traceUtilities import rollingMedianCorrection
pg.setConfigOptions(antialias=True)
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

COLORS = ['k','b','r','g','orange','y','cyan']

def mad_zscore(trace):
    """Normalize a trace using MAD-based z-score (robust to outliers/peaks)."""
    med = np.median(trace)
    mad = np.median(np.abs(trace - med))
    return (trace - med) / (1.4826 * mad + 1e-10)

def normalize_peak_positions(value):
    """Convert peak position cell value into a clean list[int]."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    if isinstance(value, str):
        text = value.strip()
        if text == '' or text.lower() in ['nan', 'none']:
            return []
        try:
            value = ast.literal_eval(text)
        except Exception:
            try:
                return [int(float(text))]
            except Exception:
                return []

    if isinstance(value, (np.ndarray, list, tuple, pd.Series)):
        out = []
        for item in value:
            if item is None or (isinstance(item, float) and np.isnan(item)):
                continue
            try:
                out.append(int(float(item)))
            except Exception:
                continue
        return out

    try:
        return [int(float(value))]
    except Exception:
        return []

def parse_float_or_none(value):
    """Parse a value into float, returning None when the value is empty/invalid."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == '' or value.lower() in ['nan', 'none']:
            return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None

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
    A PyQtGraph-based window for visualizing and analyzing trace data with interactive features.
    This class creates a main window containing a plot area and parameter trees for configuring
    trace visualization and peak detection. It allows loading trace data from files, displaying
    multiple traces with optional stacking, detecting and annotating peaks, and interactive
    modification of peak positions.
    Attributes:
        layout (pg.GraphicsLayout): Main graphics layout containing the plot
        outerlayout (pg.GraphicsLayout): Outer graphics layout
        plot (pg.PlotItem): Main plot item for displaying traces
        allCorrTraces (pd.DataFrame): Correlation data for traces
        colorsDict (dict): Mapping of hue values to colors for trace plotting
        master (pd.DataFrame): Master data containing trace metadata
        alltraces (pd.DataFrame): Trace data
        currentIds (numpy.array): Currently displayed trace IDs
        celltypes (list): List of cell types to display
        group1List (numpy.array): Unique values for first grouping variable 
        group2List (numpy.array): Unique values for second grouping variable
    Parameters:
        parent (QWidget): Parent widget
        useOpenGL (bool): Whether to use OpenGL for rendering
        background (str): Background color specification
    Key Methods:
        loadMasterCb(): Loads master data file and sets up parameter options
        loadTracesCb(): Loads trace data file
        plotCb(): Updates plot with current parameter settings
        mouseClickEventLine(): Handles mouse clicks for adding peaks
        mouseClickEvent(): Handles mouse clicks for deleting peaks
        on_guessButton_clicked(): Performs automated peak detection
        modifyPointsCb(): Toggles point modification mode
        saveCb(): Saves modified data to file
    Key Features:
    - Interactive peak detection and modification
    - Trace stacking visualization
    - Multiple grouping options for traces
    - Peak width visualization
    - Color coding by groups
    - Keyboard shortcuts for navigation
    """

    def __init__(self, parent=None, useOpenGL=None, background='default'):
        super().__init__(parent, useOpenGL, background)
        self.setWindowTitle('Your title') 
        self.setGeometry(0,0,1300,1000)
        self.layout = pg.GraphicsLayout()
        self.outerlayout = pg.GraphicsLayout()
        self.setCentralItem(self.layout)
        self.sc = self.scene()
        self.sc2 = self.layout.scene()
        self.plot = self.layout.addPlot(row=0,col=0)
        self.plot.scene().sigMouseClicked.connect(self.mouseClickEventLine)
        self.allCorrTraces = None
        self.colorsDict = {}


        # #self.plot.addItem(self.cursorlabel)
        # self.proxy = pg.SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=60, slot=self.update_crosshair)
        # self.mouse_x = None
        # self.mouse_y = None

        params = [
        # {'name':'Strain','type':'list','values':['6N','Repaired']},
            {'name':'Master file','type':'file','value':''},
            {'name':'Traces file','type':'file','value':''},
            {'name':'Correlation file','type':'file','value':''},
            {'name':'Cell ID column','type':'list','values':{}},
            {'name':'Group 1','type':'list','values':[]},
            {'name':'Group 2','type':'list','values':[]},
            {'name':'Hue group','type':'list','values':[]},
            {'name':'Cell types','type':'text','value':''},
            {'name':'Group 1 select','type':'slider','limits':[0,1]},
            {'name':'Group 2 select','type':'slider','limits':[0,1]},

            {'name':'Plot','type':'action'},
            # {'name':'dF/F0','type':'bool','value':True},
            # {'name':'F0 Frame','type':'int','value':0,'limits':[0,None]},
            {'name':'Stacked plot','type':'bool','value':True},
            {'name':'Peaks','type':'bool','value':True},
            {'name':'File','type':'text','value':''},
            {'name':'Show peak half width','type':'bool','value':False},
            {'name':'Peak group','type':'list','values':['Peak positions','Peak positions hq']},
            #{'name':'Group 1','type':'list','values':[1,3,5,6,9,12]},
            #{'name':'Prev mouse','type':'action'},
            ]

        params2 = [
                {'name':'Initial guess of peaks','type':'action'},
                {'name':'Use MAD z-score','type':'bool','value':False},
                {'name':'Prominence','type':'slider','limits':[0,10],'step':0.01,'value':0.2},
                {'name':'Distance','type':'slider','limits':[0,300]},
                {'name':'Height','type':'slider','limits':[0,10],'step':0.05},
                {'name':'Correlation','type':'slider','limits':[0,1],'step':0.05},
                {'name':'Min duration (s)','type':'float','value':0,'limits':[0,10],'step':0.05},
                {'name':'Delete points','type':'bool','value':False},
                {'name':'Add points','type':'bool','value':False},
                {'name':'Save','type':'action'},
        ]

        ## Create tree of Parameter objects
        self.p = Parameter.create(name='params', type='group', children=params)
        self.t = ParameterTree()
        self.t.setParameters(self.p, showTop=False)
        self.t.setGeometry(1300,300,300,700)
        
        self.p2 = Parameter.create(name='Peak detection', type='group', children=params2)
        self.t2 = ParameterTree()
        self.t2.setParameters(self.p2, showTop=True)
        self.t2.setGeometry(1600,300,300,400)


        #connections
        self.p.keys()['Master file'].sigValueChanged.connect(self.loadMasterCb)
        self.p.keys()['Traces file'].sigValueChanged.connect(self.loadTracesCb)
        self.p.keys()['Correlation file'].sigValueChanged.connect(self.loadCorrelationCb)
        self.p.keys()['Plot'].sigActivated.connect(self.plotCb)
        self.p.keys()['Group 1'].sigValueChanged.connect(self.changeGroupsCb)
        self.p.keys()['Group 2'].sigValueChanged.connect(self.changeGroupsCb)
        self.p.keys()['Hue group'].sigValueChanged.connect(self.changeGroupsCb)
        self.p.keys()['Group 1 select'].sigValueChanged.connect(self.changeGroup1Cb)
        self.p.keys()['Cell types'].sigValueChanged.connect(self.changeCellType)
        self.p.keys()['Show peak half width'].sigValueChanged.connect(self.plotCb)
        
        self.p2.keys()['Initial guess of peaks'].sigActivated.connect(self.on_guessButton_clicked)
        self.p2.keys()['Delete points'].sigValueChanged.connect(self.modifyPointsCb)
        self.p2.keys()['Add points'].sigValueChanged.connect(self.modifyPointsCb)
        self.p2.keys()['Save'].sigActivated.connect(self.saveCb)

        self.show()
        self.t.show()
        self.t2.show()

    def modifyPointsCb(self):
        if (self.p2['Delete points']) or (self.p2['Add points']):
            self.cursor = Qt.CrossCursor
            #self.crosshair_v = pg.InfiniteLine(angle=90, movable=False)
            #self.crosshair_h = pg.InfiniteLine(angle=0, movable=False)
            #self.plot.addItem(self.crosshair_v, ignoreBounds=True)
            #self.plot.addItem(self.crosshair_h, ignoreBounds=True)
            #self.cursorlabel = pg.TextItem()
            # self.cursor = Qt.BlankCursor    
            self.plot.setCursor(self.cursor)
        else:
            self.plot.setCursor(Qt.ArrowCursor)


    def mouseClickEvent(self,points,ev):
        """
        Handle mouse click events for trace plotting.
        This method processes mouse clicks when in 'Delete points' mode, allowing users to remove peak
        points from the trace data.
        Parameters:
            points : PlotDataItem
                The plot data item containing the trace points. The name attribute should be formatted
                as '{trace_id}_peaks'.
            ev : MouseClickEvent
                Mouse click event object containing position information of the click.
        Returns:
            None
        Notes:
            - Only functions when 'Delete points' mode is active
            - Updates the master DataFrame by removing the clicked peak position
            - Automatically refreshes the plot after peak deletion
        """
        if self.p2['Delete points']:
            traceId = points.name().split('_peaks')[0]
            print(traceId)
            el = self.master.loc[self.master[self.p['Cell ID column']]==traceId]
            index = el.index[0]

            peaks = self.master.loc[self.master[self.p['Cell ID column']]==traceId,'Peak positions'].values[0]
            #print(peaks)
            #print(ev[0].pos()[0])
            try:
                peaks.remove(ev[0].pos()[0])
            except ValueError:
                peaks.remove(int(ev[0].pos()[0]))
            #print(peaks)
            self.master.at[index,'Peak positions'] = peaks
            self.plotCb()
        else:
            pass
    
    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_D:
            if self.p2['Delete points'] and not self.p2['Add points']:
                self.p2['Delete points'] = False
                self.p2['Add points'] = True
            elif not self.p2['Delete points'] and self.p2['Add points']:
                self.p2['Delete points'] = False
                self.p2['Add points'] = False                
            elif not self.p2['Delete points'] and not self.p2['Add points']:
                self.p2['Delete points'] = True
        elif ev.key() == Qt.Key_W:
            self.p['Group 2 select'] = self.p['Group 2 select'] + 1
            self.plotCb()
        elif ev.key() == Qt.Key_Q:
            self.p['Group 2 select'] = self.p['Group 2 select'] - 1
            self.plotCb()

        elif ev.key() == Qt.Key_A:
            self.p['Group 1 select'] = self.p['Group 1 select'] - 1
            self.plotCb()

        elif ev.key() == Qt.Key_S:
            self.p['Group 1 select'] = self.p['Group 1 select'] + 1
            self.plotCb()
        elif ev.key() == Qt.Key_E:
            if self.p['Peak group'] == 'Peak positions':
                self.p['Peak group'] = 'Peak positions hq'
                self.plotCb()
            elif self.p['Peak group'] == 'Peak positions hq':
                self.p['Peak group'] = 'Peak positions'
                self.plotCb()
        return super().keyPressEvent(ev)
    

    def findNearestPeak(self,trace,point,window=10,negative=False):
        if negative == False:
            return np.argmax(trace[point-window:point+window])+ (point-window)
        else:
            return np.argmin(trace[point-window:point+window])+ (point-window)

    def mouseClickEventLine(self,ev):
        """
        Event handler for mouse click events on the plot line.
        This method processes left-click events when 'Add points' is enabled, adding new peak positions
        to the closest trace to the click point.
        Parameters:
        ----------
        ev : QMouseEvent
            The mouse event object containing click information.
        Notes:
        -----
        - Only processes left mouse button clicks (button 1)
        - Only active when 'Add points' is enabled
        - Uses Shapely for geometric calculations
        - Updates the master DataFrame with new peak positions
        - Automatically refreshes the plot via plotCb()
        The method:
        1. Converts click position to plot coordinates
        2. Finds the closest trace to the click point
        3. Determines the nearest peak position
        4. Updates the master DataFrame with the new peak
        5. Refreshes the plot
        """
        if (ev.button() == 1) and (self.p2['Add points']):
            mousePoint = self.plot.vb.mapSceneToView(ev._scenePos)
            point = shapely.Point(mousePoint.x(),mousePoint.y())
            # find closest trace to the click
            mindistance = 1e23
            for item in self.plot.dataItems:
                if item.name() in self.currentIds:
                    #print(item.name())
                    x,y = item.getData()
                    
                    line = shapely.LineString(np.vstack((x,y)).T)
                    if point.distance(line)<mindistance:
                        mindistance = point.distance(line)
                        closestItem = item
           # print(closestItem.name())
            x,trace = closestItem.getData()
            pointToAdd = self.findNearestPeak(np.array(trace),int(mousePoint.x()))
            print(pointToAdd)

            el = self.master.loc[self.master[self.p['Cell ID column']]==closestItem.name()]
            index = el.index[0]
            peaks = el['Peak positions'].values[0]
            newPeaks = list(set(peaks + [pointToAdd]))
            self.master.at[index,'Peak positions'] = newPeaks
            self.plotCb()

    def saveCb(self):
        self.master.to_excel(self.p['Master file']+'new.xlsx')

    def on_guessButton_clicked(self):
        """
        Updates peak detection parameters and finds peaks in traces based on user-defined criteria.
        This method processes each trace in the dataset using scipy's find_peaks function with
        specified parameters for prominence, distance, and height. If correlation traces are
        available, it also filters peaks based on correlation threshold (i.e., average coherence of pixel composing the ROI).
        Parameters are stored in self.p2 parameter dictionary:
            Height (float): Minimum peak height threshold
            Prominence (float): Minimum peak prominence
            Distance (int): Minimum distance between peaks
            Correlation (float): Minimum correlation threshold for peak validation
        Updates the following columns in self.master DataFrame:
            - Peak prominence
            - Peak min distance
            - Peak min height
            - Peak correlation
            - Peak positions
        After processing, updates the plot via self.plotCb()
        Note: 
            - Correlation filtering uses a window of 1.5*fps before and 1*fps after each peak
            - Peaks too close to trace edges are excluded from correlation filtering
        """

        xwHeight = self.p2['Height']
        prominence = self.p2['Prominence']
        distance = self.p2['Distance']
        xwCorrelation = self.p2['Correlation']
        minDuration = self.p2['Min duration (s)']

        use_zscore = self.p2['Use MAD z-score']

        if 'Peak positions' not in self.master.columns:
            self.master['Peak positions'] = [[] for _ in range(self.master.shape[0])]
        self.master['Peak positions'] = self.master['Peak positions'].astype(object)

        for j,row in self.el.iterrows():
            dff0 =  self.alltraces[row['Cell ID']].dropna().values
            if self.allCorrTraces is not None:
                correlation = self.allCorrTraces[row['Cell ID']].values
            else:
                correlation = None

            # Optionally normalize per-cell before peak detection
            if use_zscore:
                # Remove slow drift before computing MAD so drifty traces don't inflate the noise estimate
                corrected = rollingMedianCorrection(dff0, rollingN=2000)
                detect_trace = mad_zscore(corrected)
            else:
                detect_trace = dff0

            if xwHeight ==0:
                pheight = None
            else:
                pheight = xwHeight
            if minDuration > 0:
                pwidth = int(minDuration * row['fps'])
            else:
                pwidth = None
            peaks = list(find_peaks(detect_trace,prominence=prominence,distance=distance, height=pheight, width=pwidth)[0])

            self.master.loc[j,'Peak prominence'] = prominence
            self.master.loc[j,'Peak min distance'] = distance
            self.master.loc[j,'Peak min height'] = xwHeight
            self.master.loc[j, 'Peak correlation'] = xwCorrelation
            self.master.loc[j, 'Use MAD z-score'] = bool(use_zscore)
            self.master.loc[j, 'Min duration (s)'] = minDuration
            fps1 = int(row['fps']*1.5)
            fps2 = int(row['fps'])

            if correlation is not None:
                if xwCorrelation!=0:
                    for index,p in reversed(list(enumerate(peaks))):
                        if (p>fps1) & (p<correlation.size-fps2):
                            if correlation[p-fps1:p+fps2].max()<xwCorrelation:
                                peaks.pop(index)

            self.master.at[j,'Peak positions'] = peaks
        self.plotCb()

    def loadMasterCb(self):
        """
        Loads and processes the master file (CSV or Excel) containing cell data and updates parameter settings.
        This method performs the following operations:
        1. Loads the master file specified in self.p['Master file']
        2. Converts 'Peak positions' and 'Peak positions hq' columns to literal Python expressions if they exist
        3. Updates GUI parameters with available column names and default values
        4. Sets default peak detection parameters from the master file
        Returns:
            None
        Raises:
            Exception: If neither CSV nor Excel file can be loaded
        """
        try:
            self.master = pd.read_csv(self.p['Master file'])
        except:
            self.master = pd.read_excel(self.p['Master file'])

        if self.master is None or self.master.shape[0] == 0:
            self.celltypes = []
            columns = list(self.master.columns) if self.master is not None else []
            self.p.keys()['Cell ID column'].setLimits(columns)
            self.p.keys()['Group 1'].setLimits(columns)
            self.p.keys()['Group 2'].setLimits([' '] + columns)
            self.p.keys()['Hue group'].setLimits([' '] + columns)
            return

        try:
            self.master['Peak positions'] = self.master['Peak positions'].apply(ast.literal_eval)
        except:
            pass

        try:
            self.master['Peak positions hq'] = self.master['Peak positions hq'].apply(ast.literal_eval)
        except:
            pass

        if 'Peak positions' not in self.master.columns:
            self.master['Peak positions'] = [[] for _ in range(self.master.shape[0])]
        self.master['Peak positions'] = self.master['Peak positions'].apply(normalize_peak_positions).astype(object)

        if 'Peak positions hq' in self.master.columns:
            self.master['Peak positions hq'] = self.master['Peak positions hq'].apply(normalize_peak_positions).astype(object)


        celltypes = self.master['Cell type'].unique()
        self.celltypes = celltypes
        celltypes = [str(i) for i in celltypes]
        celltypes = ','.join(celltypes)
        

        #self.p.keys()['Cell ID column'].setOpts(value={'1':'2123'})
        columns = list(self.master.columns)
        self.p.keys()['Cell ID column'].setLimits(columns)
        self.p.keys()['Group 1'].setLimits(columns)
        self.p.keys()['Group 2'].setLimits([' '] + columns)
        self.p.keys()['Hue group'].setLimits([' '] + columns)
       # self.p.keys()['Peak group'].setLimits(list(self.master.columns))

        if 'Cell ID' in self.master.columns:
            self.p.keys()['Cell ID column'].setValue('Cell ID')
            self.p.keys()['Group 2'].setValue('Cell ID')
        if 'Folder' in self.master.columns:
            self.p.keys()['Group 1'].setValue('Folder')
        if 'Peak positions' in self.master.columns:
            self.p.keys()['Peak group'].setValue('Peak positions')

        self.p.keys()['Cell types'].setValue(celltypes)
        
        self.changeGroupsCb()

        if 'Peak prominence' in self.master.columns:
            value = parse_float_or_none(self.master['Peak prominence'].values[0])
            if value is not None:
                self.p2.keys()['Prominence'].setValue(value)
        if 'Peak min distance' in self.master.columns:
            value = parse_float_or_none(self.master['Peak min distance'].values[0])
            if value is not None:
                self.p2.keys()['Distance'].setValue(value)
        if 'Peak min height' in self.master.columns:
            value = parse_float_or_none(self.master['Peak min height'].values[0])
            if value is not None:
                self.p2.keys()['Height'].setValue(value)
        if 'Peak correlation' in self.master.columns:
            value = parse_float_or_none(self.master['Peak correlation'].values[0])
            if value is not None:
                self.p2.keys()['Correlation'].setValue(value)
        if 'Use MAD z-score' in self.master.columns:
            value = self.master['Use MAD z-score'].values[0]
            if isinstance(value, str):
                value = value.strip().lower() in ['1', 'true', 'yes', 'y', 't']
            elif pd.isna(value):
                value = False
            else:
                value = bool(value)
            self.p2.keys()['Use MAD z-score'].setValue(value)
        if 'Min duration (s)' in self.master.columns:
            value = parse_float_or_none(self.master['Min duration (s)'].values[0])
            if value is not None:
                self.p2.keys()['Min duration (s)'].setValue(value)


    def loadTracesCb(self):
        try:
            self.alltraces = pd.read_csv(self.p['Traces file'])
        except:
            self.alltraces = pd.read_excel(self.p['Traces file'])   

    def loadCorrelationCb(self):
        try:
            self.allCorrTraces = pd.read_csv(self.p['Correlation file'])
        except:
            self.allCorrTraces = pd.read_excel(self.p['Correlation file'])   


    def plotCb(self):
        """
        Plots selected data based on group selections and their parameters.
        The function handles the plotting of traces based on one or two group selections:
        - Supports peak detection and visualization
        - Supports stacked plot mode
        - Can show peak half-widths
        - Allows color coding by hue groups
        Parameters are drawn from self.p dictionary including:
        - Group 1/2 select: Index of selected groups
        - Cell ID column: Column name containing cell identifiers
        - Peaks: Boolean for peak visualization
        - Show peak half width: Boolean for displaying peak width markers
        - Stacked plot: Boolean for stacked plot mode
        - Hue group: Column name for color coding
        The function updates:
        - Current cell IDs
        - Peak detection parameters
        - Plot visualization with traces and optional peak markers
        - File name display
        The plot includes:
        - Individual traces with optional color coding
        - Peak markers if enabled
        - Peak half-width markers if enabled
        - Interactive hoverable scatter points for peaks
        Returns:
            None
        """
        group1Index = self.p['Group 1 select']
        group2Index = self.p['Group 2 select']

        #print(celltypes)

        # self.master= self.master[self.master['Cell type']==1]
        if group2Index == 0:
            self.el = self.master.loc[(self.master[self.p['Group 1']]==self.group1List[group1Index]) & (self.master['Cell type'].isin(self.celltypes))]
            
            #Set value of peak detection
            value = parse_float_or_none(self.el['Peak prominence'].values[0])
            if value is not None:
                self.p2.keys()['Prominence'].setValue(value)
            value = parse_float_or_none(self.el['Peak min distance'].values[0])
            if value is not None:
                self.p2.keys()['Distance'].setValue(value)
            value = parse_float_or_none(self.el['Peak min height'].values[0])
            if value is not None:
                self.p2.keys()['Height'].setValue(value)
            value = parse_float_or_none(self.el['Peak correlation'].values[0])
            if value is not None:
                self.p2.keys()['Correlation'].setValue(value)
            if 'Min duration (s)' in self.el.columns:
                value = parse_float_or_none(self.el['Min duration (s)'].values[0])
                if value is not None:
                    self.p2.keys()['Min duration (s)'].setValue(value)

            self.currentIds = self.el[self.p['Cell ID column']].values
            if len(self.colorsDict)>0:
                hues = self.el[self.p['Hue group']].values
                print(hues)
            else:
                hues = None
            #print(self.alltraces[ids.values].dropna())
            y = self.alltraces[self.currentIds].dropna().values
            x = np.arange(y.shape[0])

            # if self.p['dF/F0']:
            #     f0frame = self.p['F0 Frame']
            #     for i in range(y.shape[1]):
            #         f0 = y[f0frame:f0frame+5,i]
            #         y[:,i] = (y[:,i]-f0)/f0

            if self.p['Stacked plot']:
                y=stackedPlot(y)

            self.plot.clear()
            for i in range(y.shape[1]):
                trace = y[:,i]
                if hues is not None:
                    pen = self.colorsDict[hues[i]]
                else:
                    pen = 'k'

                self.plot.plot(x,trace,pen=pen,name=str(self.currentIds[i]))
                xpeak = np.array(normalize_peak_positions(self.el.iloc[i][self.p['Peak group']]), dtype=int)
                xpeak = xpeak[(xpeak >= 0) & (xpeak < trace.shape[0])]

                if self.p['Peaks']:
                    sp = pg.ScatterPlotItem(x[xpeak],trace[xpeak],name=str(self.currentIds[i])+'_peaks',hoverable=True,hoversize=20)
                    sp.sigClicked.connect(self.mouseClickEvent)

                    self.plot.addItem(sp)
                    if self.p['Show peak half width']:
                        results_full = peak_widths(trace,xpeak,rel_height=0.5)
                        sp2 = pg.ScatterPlotItem(x[results_full[2].astype(int)],trace[results_full[2].astype(int)],hoverable=True,hoversize=20,symbol='+')
                        sp3 = pg.ScatterPlotItem(x[results_full[3].astype(int)],trace[results_full[3].astype(int)],hoverable=True,hoversize=20,symbol='t')
                        self.plot.addItem(sp2)       
                        self.plot.addItem(sp3)        
            self.p.keys()['File'].setValue(self.el['Folder'].values[0])
        else:
            self.el = self.master.loc[(self.master[self.p['Group 1']]==self.group1List[group1Index]) & (self.master[self.p['Group 2']]==self.group2List[group2Index-1])& (self.master['Cell type'].isin(self.celltypes))]
            self.currentIds = self.el[self.p['Cell ID column']].values
            
            #Set value of peak detection
            value = parse_float_or_none(self.el['Peak prominence'].values[0])
            if value is not None:
                self.p2.keys()['Prominence'].setValue(value)
            value = parse_float_or_none(self.el['Peak min distance'].values[0])
            if value is not None:
                self.p2.keys()['Distance'].setValue(value)
            value = parse_float_or_none(self.el['Peak min height'].values[0])
            if value is not None:
                self.p2.keys()['Height'].setValue(value)
            value = parse_float_or_none(self.el['Peak correlation'].values[0])
            if value is not None:
                self.p2.keys()['Correlation'].setValue(value)
            if 'Min duration (s)' in self.el.columns:
                value = parse_float_or_none(self.el['Min duration (s)'].values[0])
                if value is not None:
                    self.p2.keys()['Min duration (s)'].setValue(value)
            
            
            #print(self.alltraces[ids.values].dropna())
            y = self.alltraces[self.currentIds].dropna().values
            if len(self.colorsDict)>0:
                hues = self.el[self.p['Hue group']].values
                print(hues)
            else:
                hues = None
            x = np.arange(y.shape[0])

            # if self.p['dF/F0']:
            #     f0frame = self.p['F0 Frame']
            #     for i in range(y.shape[1]):
            #         f0 = y[f0frame:f0frame+5,i]
            #         y[:,i] = (y[:,i]-f0)/f0
                    
            if self.p['Stacked plot']:
                y=stackedPlot(y)

            self.plot.clear()
            for i in range(y.shape[1]):
                trace = y[:,i]
                if hues is not None:
                    pen = self.colorsDict[hues[i]]
                else:
                    pen = 'k'
                self.plot.plot(x,trace,pen=pen,name=str(self.currentIds[i]))
                xpeak = np.array(normalize_peak_positions(self.el.iloc[i][self.p['Peak group']]), dtype=int)
                xpeak = xpeak[(xpeak >= 0) & (xpeak < trace.shape[0])]

                if self.p['Peaks']:
                    sp = pg.ScatterPlotItem(x[xpeak],trace[xpeak],name=str(self.currentIds[i])+'_peaks',hoverable=True,hoversize=20)
                    sp.sigClicked.connect(self.mouseClickEvent)
                    self.plot.addItem(sp)


                    if self.p['Show peak half width']:
                        results_full = peak_widths(trace,xpeak,rel_height=0.5)

                        sp2 = pg.ScatterPlotItem(x[results_full[2].astype(int)],trace[results_full[2].astype(int)],hoverable=True,hoversize=20,symbol='+')
                        sp3 = pg.ScatterPlotItem(x[results_full[3].astype(int)],trace[results_full[3].astype(int)],hoverable=True,hoversize=20,symbol='t')
                        self.plot.addItem(sp2)       
                        self.plot.addItem(sp3)    

            try:
                self.p.keys()['File'].setValue(str(self.group2List[group2Index-1].values[0]))
            except AttributeError:
                self.p.keys()['File'].setValue(str(self.group2List[group2Index-1]))

    def changeGroupsCb(self):
        if self.master is None or self.master.shape[0] == 0:
            self.group1List = np.array([])
            self.group2List = np.array([])
            self.colorsDict = {}
            self.p.keys()['Group 1 select'].setLimits((0,0))
            self.p.keys()['Group 1 select'].setValue(0)
            self.p.keys()['Group 2 select'].setLimits((0,0))
            self.p.keys()['Group 2 select'].setValue(0)
            return

        self.group1List = self.master[self.p['Group 1']].unique()
        self.p.keys()['Group 1 select'].setLimits((0,len(self.group1List)-1))
        self.p.keys()['Group 1 select'].setValue(0)

        try:
            self.group2List = self.master.loc[self.master[self.p['Group 1']] == self.group1List[self.p['Group 1 select']],self.p['Group 2']].unique()
            self.p.keys()['Group 2 select'].setLimits((0,len(self.group2List)))
            self.p.keys()['Group 2 select'].setValue(0)
        except KeyError:
            self.group2List = []
            self.p.keys()['Group 2 select'].setValue(0)
        try:
            if self.p['Hue group'] != ' ':
                levels = self.master[self.p['Hue group']].unique()
                self.colorsDict = {}
                for j,l in enumerate(levels):
                    self.colorsDict[l] = pg.mkPen(COLORS[j])
      
        except KeyError:
            self.colorsDict = {}
            

    def changeGroup1Cb(self):
        self.group2List = self.master.loc[(self.master[self.p['Group 1']] == self.group1List[self.p['Group 1 select']]) & (self.master['Cell type'].isin(self.celltypes)),self.p['Group 2']].unique()
        self.p.keys()['Group 2 select'].setLimits((0,len(self.group2List)))
        self.p.keys()['Group 2 select'].setValue(0)
 

    def changeCellType(self):
        try:
            self.celltypes = self.p['Cell types'].split(',')
            self.celltypes = [int(c) for c in self.celltypes]
        except ValueError:
            self.celltypes = self.master['Cell type'].unique()
            celltypes = self.master['Cell type'].unique()
            celltypes = [str(i) for i in celltypes]
            celltypes = ','.join(celltypes)
            self.p.keys()['Cell types'].setValue(celltypes)

        #print(self.celltypes)
        self.changeGroup1Cb()

if __name__ == '__main__':
    win = mainWindow()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()