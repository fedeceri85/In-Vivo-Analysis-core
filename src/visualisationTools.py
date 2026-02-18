import ipywidgets as widgets
import panel as pn
from ipywidgets import interact, Dropdown
import pylab as plt
import numpy as np
import seaborn as sns
import pandas as pd
import traceUtilities as tu
from scipy.signal import savgol_filter
import os
from movieTools import thorlabsFile, getPreviewImage,getNImages,calculateFrameIntervalsToRemove, getImgInfo, savefolder
from scipy.signal import argrelmin,argrelmax
import tifffile
import plotly.graph_objs as go
import mass_ts
from pathlib import Path
import traceback
from bokeh.plotting import figure as bk_figure
from bokeh.models import (ColumnDataSource, BoxAnnotation, CustomJS,
                           BoxSelectTool)
from bokeh.events import SelectionGeometry

def jumpFramesFinder(master,allminima,allmaxima,correctionReferenceTraceDf,tb):
    """
    Interactive tool to find and correct jumps in fluorescence movies.
    This function creates an interactive widget interface for visualizing and processing multiple traces
    from imaging data, with options for jump correction, motion correction, and trace filtering. Jump corrected
    data is saved in the `processedMovies` subfolder of each image as 1-motionCorrected.tif. Motion corrected data (produced by the batchMotionCorrection.ipynb notebook)
    is saved in the same folder as 1-jumpCorrected-mc.tif. 
    Parameters
    ----------
    master : pandas.DataFrame
        DataFrame containing metadata for each recording, including folder paths and correction parameters
    allminima : pandas.DataFrame
        DataFrame storing minima points for each folder/recording
    allmaxima : pandas.DataFrame 
        DataFrame storing maxima points for each folder/recording
    correctionReferenceTraceDf : pandas.DataFrame
        DataFrame containing reference traces used for correction
    tb : thorlabsFile object
        Instance of thorlabsFile class used for loading and displaying movies
    Returns
    -------
    None
        Displays interactive widget interface in Jupyter notebook
    
    The interface allows:
    - Loading original, jump-corrected and motion-corrected movies
    - Manual selection of frames to remove
    - Template-based artifact detection
    - Saving corrected movies
    - Tracking processing status with validation indicators
    """

    identifiers = master['Folder'].unique()
    xw = widgets.IntSlider(min=0,max=np.size(identifiers)-1,step=1,value=0,continuous_update=False,description='Trace #')
    xwLeft = widgets.IntSlider(min=0,max=50,step=1,value=0,continuous_update=False,description='Left win')
    xwRight = widgets.IntSlider(min=0,max=50,step=1,value=0,continuous_update=False,description='Right win')
    xwMinimaOrder = widgets.IntSlider(min=0,max=400,step=1,value=50,continuous_update=False,description='Order')
    lbl1 = widgets.Label('Minima',layout= widgets.Layout(display="flex", justify_content="center"))
    smoothOrderInt = widgets.IntSlider(min=1,max=31,step=2,value=11,continuous_update=False,description='Smooth. ord.')
    #xwslice = widgets.IntSlider(min=0,max=2,step=1,value=0,continuous_update=False,description='Current slice',layout=widgets.Layout(width='95%'))
    xwMaxLeft = widgets.IntSlider(min=0,max=50,step=1,value=0,continuous_update=False,description='Left win')
    xwMaxRight = widgets.IntSlider(min=0,max=50,step=1,value=0,continuous_update=False,description='Right win')
    xwMaximaOrder = widgets.IntSlider(min=0,max=400,step=1,value=50,continuous_update=False,description='Order')
    lbl2 = widgets.Label('Maxima',layout= widgets.Layout(display="flex", justify_content="center"))
    xwUpdateRate = widgets.IntSlider(min=1,max=15,step=1,value=4,continuous_update=False,description='Update interval')
    xwTemplateSlider =  widgets.IntSlider(min=0,max=1000,step=1,value=0,continuous_update=False,description='Template strength')
    lbl3 = widgets.Label('Template search',layout= widgets.Layout(display="flex", justify_content="center"))
    pbar = widgets.IntProgress(min=0,max=1,bar_style='success',description='Progress')
    ddMenu =  widgets.Dropdown(options=['Z', 'C', 'D','E','F','/media/marcotti-lab'], value='Z', description='Drive:', disabled=False)
    

    prevButton =  widgets.Button(description='<',button_style = 'primary')
    nextButton =  widgets.Button(description='>',button_style = 'primary')
    b = widgets.Button(description='Jump corr original',button_style = 'primary')
    b2 = widgets.Button(description='Load original')
    b4 = widgets.Button(description='Load jump-corrected')
    b4bis = widgets.Button(description='Quick load jump-corrected')
    b5 = widgets.Button(description='Load motion-corrected')
    b3 = widgets.Button(description='Save File',button_style='Success', icon='check')
    lblCurrent= widgets.Label(value="None")
    valid1 = widgets.Valid(value= False, description ='Jump corr')
    valid2 = widgets.Valid(value=False,description='Motion corr')
    valid3 = widgets.Valid(value=False,description='Rois')
    valid4 = widgets.Valid(value=False,description='Traces')
    valid5 = widgets.Valid(value=False,description='Annotations')

    button = widgets.Button(
        description='Delete selected frames',
        disabled=False,
        button_style='warning', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
       )
    buttonTemplate = widgets.Button(
        description='Create template',
        disabled=False,
        button_style='info', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Click me',
       )   

    frameStartInt = widgets.BoundedIntText(value=0, min=0, max=10**6, step=1, description='Frame start:')
    frameEndInt   = widgets.BoundedIntText(value=0, min=0, max=10**6, step=1, description='Frame end:')
    buttonManualInterval = widgets.Button(
        description='Delete interval',
        disabled=False,
        button_style='warning',
        tooltip='Remove the frame interval defined by Frame start / Frame end',
    )

    firstFrameInt = widgets.BoundedIntText(value=1, min=1, max=10**6, step=1, description='First frame:')
    lastFrameInt  = widgets.BoundedIntText(value=1, min=1, max=10**6, step=1, description='Last frame:')
    buttonSetFirstLast = widgets.Button(
        description='Set first-last',
        disabled=False,
        button_style='info',
        tooltip='Write first-last to master and refresh the plot',
    )

    def process_original_cbk(b):
        """
        Callback function to process and load original movie frames with jump corrections.
        This function processes movie frames based on selected parameters and applies jump corrections.
        It loads the movie file, applies specified frame intervals for removal, and handles layer management.
        Parameters
        ----------
        b : widget button
            The button widget that triggered the callback.
        Notes
        -----
        The function performs the following operations:
        - Retrieves movie folder path and frame range from master dataframe
        - Calculates frame intervals to remove based on minima/maxima and window parameters
        - Applies additional correction intervals if available
        - Loads and displays the movie with Gaussian filter applied
        - Removes specified layers ('Masks', 'Avg', 'Annotations') if they exist

        """
        el = master.loc[xw.value]
        workingFolder = el['Folder']
        if ddMenu.value !='Z':
            if os.name == 'posix':
                workingFolder = Path(ddMenu.value) / Path(workingFolder[2:].replace('\\', '/').lstrip('/'))
            else:
                workingFolder = ddMenu.value + workingFolder[1:]
      

        try:
            firstFrame,lastFrame = el['first-last'].split('-')
            firstFrame = int(firstFrame)-1
            lastFrame = int(lastFrame)
        except:
            firstFrame,lastFrame = [0, getImgInfo(workingFolder)[2]]
            if el['nChannels']==2:
                lastFrame = lastFrame//2

        winLeft = el['Window left']
        winRight = el['Window right']
        thisMinima = allminima[el['Folder']]
        thisMinima = thisMinima[thisMinima!=0]
    
        winMaxLeft = el['Window Max left']
        winMaxRight = el['Window Max right']
        thisMaxima = allmaxima[el['Folder']]
        thisMaxima = thisMaxima[thisMaxima!=0]

        try:
            nChannels = el['nChannels']
        except:
            nChannels = 1

        frameIntervalsToRemove = calculateFrameIntervalsToRemove(jumpFrames=thisMinima,winLeft=winLeft,winRight=winRight, jumpFramesMax=thisMaxima, winMaxLeft=winMaxLeft, winMaxRight=winMaxRight)
        try:
            frameIntervalsToRemove.extend(el['ExtraCorrectionIntervals'])
        except TypeError:
            pass
        try:
            frameIntervalsToRemove.extend(el['TemplateIntervals'])
        except TypeError:
            pass

        lblCurrent.value = 'Jump-corrected movie'

        try:
            spatialGaussian = int(el['SpatialGaussian'])
        except:
            spatialGaussian = 2

        try:
            temporalGaussian = int(el['TemporalGaussian'])
        except:
            temporalGaussian = 2

        tb.loadFile(workingFolder, applyGaussian = True,nChannels=nChannels,spatialGaussian=spatialGaussian,temporalGaussian=temporalGaussian)
        pbar.max = tb.nFrames
        tb.loadFrameInterval(firstFrame, lastFrame,frameIntervalsToRemove=frameIntervalsToRemove,pbar=pbar)
        try:
            tb.app.layers.remove('Masks')
        except:
            pass
        try:
            tb.app.layers.remove('Avg')
        except:
            pass   
        try:
            tb.app.layers.remove('Annotations')
        except:
            pass   

        if nChannels==2:
            tb.loadFile(workingFolder, applyGaussian = True,nChannels=nChannels,spatialGaussian=spatialGaussian,temporalGaussian=temporalGaussian)

            tb.loadFrameInterval(firstFrame, lastFrame,frameIntervalsToRemove=frameIntervalsToRemove,pbar=pbar, layerName='Image channel 2',channel=2)


    b.on_click(process_original_cbk)

    def load_original_cbk(b):
        el = master.loc[xw.value]
        workingFolder = el['Folder']
        if ddMenu.value !='Z':
            if os.name == 'posix':
                workingFolder = Path(ddMenu.value) / Path(workingFolder[2:].replace('\\', '/').lstrip('/'))
            else:
                workingFolder = ddMenu.value + workingFolder[1:]
        try:
            firstFrame,lastFrame = el['first-last'].split('-')
            firstFrame = int(firstFrame)-1
            lastFrame = int(lastFrame)
        except:
            firstFrame,lastFrame = [0, getImgInfo(workingFolder)[2]]

        try:
            nChannels = el['nChannels']
        except:
            nChannels = 1

        tb.loadFile(workingFolder, applyGaussian = True,nChannels=nChannels)
        pbar.max = tb.nFrames
        tb.loadFrameInterval(firstFrame, lastFrame,frameIntervalsToRemove=None,pbar=pbar)

        try:
            tb.app.layers.remove('Masks')
        except:
            pass
        try:
            tb.app.layers.remove('Avg')
        except:
            pass   
        try:
            tb.app.layers.remove('Annotations')
        except:
            pass   


    b2.on_click(load_original_cbk)
    
    def saveProcessed(b):
        el = master.loc[xw.value]
        workingFolder = el['Folder']
        if ddMenu.value !='Z':
            if os.name == 'posix':
                workingFolder = Path(ddMenu.value) / Path(workingFolder[2:].replace('\\', '/').lstrip('/'))
            else:
                workingFolder = ddMenu.value + workingFolder[1:]
        if lblCurrent.value == 'Jump-corrected movie':
            
                outFolder = os.path.join(workingFolder,savefolder)

                if not os.path.exists(outFolder):
                    os.makedirs(outFolder)
                

                outfile = os.path.join(outFolder,'1-jumpCorrected.tif')
                tifffile.imwrite(outfile, tb.app.layers['Image'].data)
                valid1.value = True
                
                if el['nChannels']==2:
                    valid1.value = False
                    outfile2 = os.path.join(outFolder,'1-jumpCorrected-channel2.tif')
                    tifffile.imwrite(outfile2, tb.app.layers['Image channel 2'].data)
                    valid1.value = True

    b3.on_click(saveProcessed)

    def loadJumpCorr(b):
        el = master.loc[xw.value]
        workingFolder = el['Folder']
        if ddMenu.value !='Z':
            if os.name == 'posix':
                workingFolder = Path(ddMenu.value) / Path(workingFolder[2:].replace('\\', '/').lstrip('/'))
            else:
                workingFolder = ddMenu.value + workingFolder[1:]
        #try:
        outFolder = os.path.join(workingFolder,savefolder,'1-jumpCorrected.tif')
            
        #tb.loadQuickLook(outFolder)
        tb.loadFromTiff(outFolder,nChannels=el['nChannels'],channel=1)
        if el['nChannels']==2:
            outFolder2 = os.path.join(workingFolder,savefolder,'1-jumpCorrected-channel2.tif')
            tb.loadFromTiff(outFolder2,title='Image channel 2',nChannels=el['nChannels'],channel=2)
            
        #except:
            #print('Cannot open jump-corrected movie')
        try:
            tb.app.layers.remove('Masks')
        except:
            pass
        try:
            tb.app.layers.remove('Avg')
        except:
            pass   
        try:
            tb.app.layers.remove('Annotations')
        except:
            pass   

    b4.on_click(loadJumpCorr)

    def quickLoadJumpCorr(b):
        '''
        Load a jumpCorrected file without loading it in memory for fast access.
        '''
        el = master.loc[xw.value]
        workingFolder = el['Folder']
        if ddMenu.value !='Z':
            if os.name == 'posix':
                workingFolder = Path(ddMenu.value) / Path(workingFolder[2:].replace('\\', '/').lstrip('/'))
            else:
                workingFolder = ddMenu.value + workingFolder[1:]
        #try:
        outFolder = os.path.join(workingFolder,savefolder,'1-jumpCorrected.tif')
            
        tb.loadQuickLook(outFolder)
        #tb.loadFromTiff(outFolder)
        #except:    
        try:
            tb.app.layers.remove('Masks')
        except:
            pass
        try:
            tb.app.layers.remove('Avg')
        except:
            pass   
        try:
            tb.app.layers.remove('Annotations')
        except:
            pass   


    b4bis.on_click(quickLoadJumpCorr)

    def loadMotionCorr(b):
        el = master.loc[xw.value]
        workingFolder = el['Folder']
        if ddMenu.value !='Z':
            if os.name == 'posix':
                workingFolder = Path(ddMenu.value) / Path(workingFolder[2:].replace('\\', '/').lstrip('/'))
            else:
                workingFolder = ddMenu.value + workingFolder[1:]
        #try:
        outFolder = os.path.join(workingFolder,savefolder,'1-jumpCorrected-mc.tif')
            
        #tb.loadQuickLook(outFolder)
        tb.loadFromTiff(outFolder)
        #except:
            #print('Cannot open jump-corrected movie')

        #remove the avg and Masks layers if they exist
        try:
            tb.app.layers.remove('Masks')
        except:
            pass
        try:
            tb.app.layers.remove('Avg')
        except:
            pass
        try:
            tb.app.layers.remove('Annotations')
        except:
            pass   

        
    b5.on_click(loadMotionCorr)

    output = widgets.Output()

    #Generate traces for plotly plot to display fluorescence trace and select minima and maxima.
    trace1 = go.Scatter(
        x = [],
        y= [],
        mode= 'lines',
        name = 'Original',
        marker=dict(
            color=(0,0,0,0),
            size=0,
            opacity= 0,
            ))
    

    trace2 = go.Scatter(
        x = [],
        y= [],
        mode= 'lines',
        name = 'Corrected')
    
    trace3 = go.Scatter(
        x = [],
        y= [],
        mode= 'markers',
        marker=dict(
            color='teal',
            size=12,
            symbol='x-thin',
            line=dict(
                color='black',
                width=1,
                
            )),
        name = 'Jumps'
            )

    trace3Max = go.Scatter(
        x = [],
        y= [],
        mode= 'markers',
        marker=dict(
            color='teal',
            size=12,
            symbol='x-thin',
            line=dict(
                color='red',
                width=1,            
            )),
        name = 'Jumps Up'
            )
       
    trace4 = go.Scatter(
        x = [],
        y= [],
        mode= 'markers',
        marker=dict(
            color='black',
            size=12,
            line=dict(
                color='black',
                width=1
            )),
        name = 'Remove'
            )
    
    trace5 = go.Scatter(
        x = [0,0],
        y= [-10,10],
        mode= 'lines',
        
        line=dict(color='black', 
                  width=2,
                dash='dash'),
        name = ''
            )
    #Plotly figure widget
    fig = go.FigureWidget(data=[trace1,trace2,trace3,trace4,trace5,trace3Max],layout=go.Layout({ 'autosize':True,'height':750,
                                                                      #  'xaxis':{'rangeslider':{'visible':True}},
                                                                        }))
    global counter
    counter = 0

    #update the slider every N frames
    def update_slider(event):
            global counter
            counter = counter +1
            everyN = xwUpdateRate.value
            if counter%everyN == 0:
                fig.data[4]['x'] = [tb.app.dims.current_step[0], tb.app.dims.current_step[0]]
                fig.data[4]['y'] = [min(fig.data[0]['y'])*0.9, max(fig.data[0]['y'])*1.1]
                counter = 0


    tb.app.dims.events.connect(update_slider)
    sliderStack = widgets.VBox([xw,smoothOrderInt,valid1,valid2,valid3,valid4,valid5,
                                 widgets.HBox([firstFrameInt, lastFrameInt, buttonSetFirstLast])])
    buttonStack = widgets.VBox([lblCurrent,b,b2,b4,b4bis,b5,b3])
    buttonStack2 = widgets.VBox([button,lbl3,buttonTemplate,xwTemplateSlider,
                                  widgets.HBox([frameStartInt, frameEndInt,buttonManualInterval])])
    buttonStack3 = widgets.HBox([prevButton, nextButton])
    ui = widgets.VBox([widgets.HBox([sliderStack, widgets.VBox([lbl1,xwMinimaOrder,xwLeft,xwRight]),
                                     widgets.VBox([lbl2,xwMaximaOrder,xwMaxLeft,xwMaxRight]), buttonStack2,buttonStack,output,
                                     widgets.VBox([xwUpdateRate,buttonStack3,pbar]),ddMenu]),
                                       widgets.VBox([fig])])

  

    def selectionFunction(t,p,box):
        #print(p)
        #print(t)
        print(box.xrange)
        fstart = int(box.xrange[0])
        fend = int(box.xrange[1])
        fig.data[3]['x'] = np.arange(fstart,fend)
        fig.data[3]['y'] = fig.data[0]['y'][np.arange(fstart,fend)]
        
    fig.data[0].on_selection(selectionFunction)



    def on_value_change(change):
        master.loc[xw.value,'Window left'] = xwLeft.value
        master.loc[xw.value,'Window right'] = xwRight.value
        master.loc[xw.value,'Minima order'] = xwMinimaOrder.value
        master.loc[xw.value,'Window Max left'] = xwMaxLeft.value
        master.loc[xw.value,'Window Max right'] = xwMaxRight.value
        master.loc[xw.value,'Maxima order'] = xwMaximaOrder.value

    def on_button_clicked(b):

        intervals =  master.at[xw.value,'ExtraCorrectionIntervals']
        if intervals.__class__!= list:
            intervals = []
 

        interval = [int(fig.data[3]['x'][0] )   ,int(fig.data[3]['x'][-1]) ]
        
        intervals.append(interval)
        master.at[xw.value,'ExtraCorrectionIntervals'] = intervals
        fig.data[3]['x'] = []
        fig.data[3]['y'] = []
        f(xw.value,xwLeft.value,xwRight.value,xwMinimaOrder.value,xwMaxLeft.value,xwMaxRight.value,xwMaximaOrder.value)
      #  print('Deleted')

    def on_manual_interval_clicked(b):
        intervals = master.at[xw.value, 'ExtraCorrectionIntervals']
        if intervals.__class__ != list:
            intervals = []
        interval = [frameStartInt.value, frameEndInt.value]
        intervals.append(interval)
        master.at[xw.value, 'ExtraCorrectionIntervals'] = intervals
        f(xw.value, xwLeft.value, xwRight.value, xwMinimaOrder.value, xwMaxLeft.value, xwMaxRight.value, xwMaximaOrder.value)

    buttonManualInterval.on_click(on_manual_interval_clicked)

    def on_set_first_last_clicked(b):
        master.at[xw.value, 'first-last'] = f'{firstFrameInt.value}-{lastFrameInt.value}'
        f(xw.value, xwLeft.value, xwRight.value, xwMinimaOrder.value,
          xwMaxLeft.value, xwMaxRight.value, xwMaximaOrder.value)

    buttonSetFirstLast.on_click(on_set_first_last_clicked)

    def on_template_clicked(b):
        global distance_profile
        global template
        template =  fig.data[3]['y']
        distance_profile = mass_ts.mass2(fig.data[0]['y'],template)
        fig.data[3]['x'] = []
        fig.data[3]['y'] = []
        f(xw.value,xwLeft.value,xwRight.value,xwMinimaOrder.value,xwMaxLeft.value,xwMaxRight.value,xwMaximaOrder.value)
           
    buttonTemplate.on_click(on_template_clicked)
    button.on_click(on_button_clicked)
    
    def on_prev_clicked(change):
        z,x, y = tb.app.dims.current_step
        if z>0:
            tb.app.dims.current_step = (z-1,x,y)
            fig.data[4]['x'] = [z-1, z-1]
            fig.data[4]['y'] = [min(fig.data[0]['y'])*0.9, max(fig.data[0]['y'])*1.1]

    def on_next_clicked(change):
        z,x, y = tb.app.dims.current_step
        if z<tb.nFrames:
            tb.app.dims.current_step = (z+1,x,y)
            fig.data[4]['x'] = [z+1, z+1]
            fig.data[4]['y'] = [min(fig.data[0]['y'])*0.9, max(fig.data[0]['y'])*1.1]
    prevButton.on_click(on_prev_clicked)
    nextButton.on_click(on_next_clicked)

    xwLeft.observe(on_value_change, names='value')
    xwRight.observe(on_value_change, names='value')
    xwMinimaOrder.observe(on_value_change, names='value')

    xwMaxLeft.observe(on_value_change, names='value')
    xwMaxRight.observe(on_value_change, names='value')
    xwMaximaOrder.observe(on_value_change, names='value')

    def template_value_changed(change):
        global distance_profile
        global template
        distance_profile2 = distance_profile.copy()
        idcs = np.argsort(distance_profile2)
        intervals = []
        i=0
        while i<=xwTemplateSlider.value:
            intervals.append([idcs[0],idcs[0]+template.size])
            distance_profile2[idcs[0]:idcs[0]+template.size] = distance_profile2.max()
            idcs = np.argsort(distance_profile2)
            i = i+1

        master.at[xw.value,'TemplateIntervals'] = intervals        
        f(xw.value,xwLeft.value,xwRight.value,xwMinimaOrder.value,xwMaxLeft.value,xwMaxRight.value,xwMaximaOrder.value)
    xwTemplateSlider.observe(template_value_changed,names='value')

    def smooth_value_changed(change):
        master.loc[xw.value,'SmoothOrder'] = smoothOrderInt.value
        f(xw.value,xwLeft.value,xwRight.value,xwMinimaOrder.value,xwMaxLeft.value,xwMaxRight.value,xwMaximaOrder.value)
    smoothOrderInt.observe(smooth_value_changed,names='value')

    #Main function executed by ipywidget interactive output 
    def f(x, window_size_left, window_size_right, minima_order, windowMax_size_left,windowMax_size_right,maxima_order,drive=None):
        """
        Updates and visualizes motion correction analysis data for a given dataset.
        Called by widgets.interactive_output
        """

        el = master.loc[x]
        workingFolder = el['Folder']
        if ddMenu.value !='Z':
            if os.name == 'posix':
                workingFolder = Path(ddMenu.value) / Path(workingFolder[2:].replace('\\', '/').lstrip('/'))
            else:
                workingFolder = ddMenu.value + workingFolder[1:]
        
      

        fig.layout.title.text = str(workingFolder)
        lblCurrent.value = 'None'
        #check wheteher it has been motion corrected
        outFolder = os.path.join(workingFolder,savefolder,'1-jumpCorrected.tif')
        if os.path.exists(outFolder):
            valid1.value=True
        else:
            valid1.value= False
            
        outFolder = os.path.join(workingFolder,savefolder,'1-jumpCorrected-mc.tif')
        if os.path.exists(outFolder):
            valid2.value=True
        else:
            valid2.value= False

        outFolder = os.path.join(workingFolder,savefolder,'Masks.tif')
        if os.path.exists(outFolder):
            valid3.value=True
        else:
            valid3.value= False

        outFolder = os.path.join(workingFolder,savefolder,'traces.csv')
        if os.path.exists(outFolder):
            valid4.value=True
        else:
            valid4.value= False

        outFolder = os.path.join(workingFolder,savefolder,'Annotations.tif')
        if os.path.exists(outFolder):
            valid5.value=True
        else:
            valid5.value= False

        #reset the extra points to remove
        fig.data[3]['x'] = []
        fig.data[3]['y'] = []

       

        try:
            firstFrame,lastFrame = el['first-last'].split('-')
            firstFrame = int(firstFrame)-1
            lastFrame = int(lastFrame)
        except:
            firstFrame,lastFrame = [0,-1]

        # Sync first/last frame widgets (1-based display)
        firstFrameInt.value = firstFrame + 1
        if lastFrame > 0:
            lastFrameInt.value = lastFrame

        try:
            smoothOrder = el['SmoothOrder']
            if (smoothOrder%2)==0:
                smoothOrder = smoothOrder+1
                master.loc[xw.value,'SmoothOrder'] = smoothOrder
        
        except:
            smoothOrder = 11
        smoothOrderInt.value = smoothOrder

        if os.path.exists(os.path.join(workingFolder,'corrReference.npy')):
                r,s,t = tu.loadRoisFromFile(os.path.join(workingFolder,'corrReference.npy'))
                s = s[firstFrame*el['nChannels']:lastFrame*el['nChannels'],:]
                if smoothOrder!=1:
                    for i in np.arange(s.shape[1]):
                        s[:,i] = savgol_filter(s[:,i],smoothOrder,1)
                if el['nChannels']==2:
                    s = s[::2,:]    
                ttrace = s.mean(1)
        elif os.path.exists(os.path.join(workingFolder,'corrReference.csv')):
                s = pd.read_csv(os.path.join(workingFolder,'corrReference.csv'))
                s = s['Mean'].values
                s = s[firstFrame*el['nChannels']:lastFrame*el['nChannels']]
                if smoothOrder!=1:
                    s = savgol_filter(s,smoothOrder,1)
                if el['nChannels']==2:
                    s = s[::2]
                ttrace = s 
        else:
            raise ValueError('corrReference not found')       
        # elif  not pd.isna(el['rois']):
        #     if os.path.exists(os.path.join(workingFolder,el['rois'])):
        #         r,s,t = tu.loadRoisFromFile(os.path.join(workingFolder,el['rois']))
        #         s = s[firstFrame:lastFrame,:]
        #         for i in np.arange(s.shape[1]):
        #             s[:,i] = savgol_filter(s[:,i],11,1)
        #         ttrace = s.mean(1)
        # else:
        #     try:
        #         ttrace = correctionReferenceTraceDf[el['Folder']].values
        #         ttrace = ttrace[~pd.isna(ttrace)]
        #         ttrace = ttrace[firstFrame:lastFrame]
        #     except:
        #         print('no roi found')
        #         prevImg = getPreviewImage(workingFolder)
        #         height,width = prevImg.shape

        #         movie = thorlabsFile(workingFolder,applyGaussian=False,showViewer=False)
        #         movie.loadFrameInterval(firstFrame,lastFrame)

        #         s = movie.array.mean(2).mean(1)
        #         s = np.expand_dims(s,1)
        #         s = s[firstFrame:lastFrame,:]   

        #         for i in np.arange(s.shape[1]):
        #             s[:,i] = savgol_filter(s[:,i],11,1)
        
        #         ttrace = s.mean(1)
        correctionReferenceTraceDf[el['Folder']] = np.pad(ttrace,(0,60000-len(ttrace)),mode='constant',constant_values=np.nan)
        
        try:         
            if not np.isnan(el['Minima order']):

                window_left = int(el['Window left'])
                window_right = int(el['Window right'])
                minima_order = int(el['Minima order'])
    
                xwLeft.value = window_left
                xwRight.value = window_right
                xwMinimaOrder.value = minima_order
        except:
            pass

        try:         
            if not np.isnan(el['Maxima order']):

                windowMax_left = int(el['Window Max left'])
                windowMax_right = int(el['Window Max right'])
                maxima_order = int(el['Maxima order'])
    
                xwMaxLeft.value = windowMax_left
                xwMaxRight.value = windowMax_right
                xwMaximaOrder.value = maxima_order
        except:
            pass
            
            
        minima = argrelmin(ttrace,order=minima_order )[0]
        maxima = argrelmax(ttrace,order=maxima_order )[0]


        if window_size_left>minima[0]:
            minima = minima[1:]
        if (window_left==0 )and (window_right==0):
            minima = []
        try:
            allminima[el['Folder']] = np.pad(minima,(0,1000-len(minima)),'constant')
        except:
            print('Cannot save minima')
        left = 0
        ttrace2 = np.zeros(ttrace.shape) - 10000

        for elmin in minima:

            ttrace2[left:elmin-window_size_left] = ttrace[left:elmin-window_size_left]
            left = elmin+window_size_right
        ttrace2[left:] = ttrace[left:] # add last bit
       #ttrace2=np.array(ttrace2)
        ttrace2[left:] = ttrace[left:] # add last bit


        if windowMax_size_left>maxima[0]:
            maxima = maxima[1:]
        if (windowMax_left==0 )and (windowMax_right==0):
            maxima = []
        #try:
        allmaxima[el['Folder']] = np.pad(maxima,(0,1000-len(maxima)),'constant')
        #except:
        #    print('Cannot save maxima')
     

        for elmin in maxima:

            ttrace2[elmin-windowMax_size_left:elmin+windowMax_size_right] = -10000




        try:
            extraIntervals = el['ExtraCorrectionIntervals']

        except:
            extraIntervals = []
        try:
            for interval in extraIntervals:
                 
                    ttrace2[interval[0]:interval[1]] = -10000
        except (TypeError,):
            pass
        
        
        try:
            extraIntervals = el['TemplateIntervals']

        except:
            extraIntervals = []
        try:
            for interval in extraIntervals:
                 
                    ttrace2[interval[0]:interval[1]] = -10000
        except (TypeError,):
            pass


        ttrace2[ttrace2 == - 10000] = np.nan

        with fig.batch_update():
            fig.data[0]['x'] = np.arange(ttrace.shape[0])
            fig.data[0]['y'] = ttrace

            fig.data[1]['x'] = np.arange(ttrace2.shape[0])
            fig.data[1]['y'] = ttrace2

            fig.data[2]['x'] = minima
            fig.data[2]['y'] = ttrace[minima]

            fig.data[5]['x'] = maxima
            fig.data[5]['y'] = ttrace[maxima]

        try:
            fig.data[4]['y'] = [min(fig.data[0]['y'])*0.9, max(fig.data[0]['y'])*1.1]
        except:
            pass


        

    out = widgets.interactive_output(f, {'x': xw,'window_size_left':xwLeft,'window_size_right':xwRight,
                                        'minima_order':xwMinimaOrder, 'windowMax_size_left':xwMaxLeft, 'windowMax_size_right':xwMaxRight,'maxima_order':xwMaximaOrder,'drive':ddMenu})

    display(ui, out)


class JumpFramesFinderPanel:
    """
    Class-based implementation of the jumpFramesFinderPanel tool.
    Encapsulates state and UI logic for jump correction visualization.
    """
    def __init__(self, master, allminima, allmaxima, tb, corrFilename=None, jumpFrameFilename=None, jumpFrameMaxFilename=None):
        self.master = master
        self.allminima = allminima
        self.allmaxima = allmaxima
        self.tb = tb
        self.corrFilename = corrFilename
        self.jumpFrameFilename = jumpFrameFilename
        self.jumpFrameMaxFilename = jumpFrameMaxFilename
        self.identifiers = master['Folder'].unique()
        self._ntraces = max(0, np.size(self.identifiers) - 1)
        
        # Internal state for template matching
        self._distance_profile = [None]
        self._template = [None]
        
        # Napari cursor counter
        self._panel_counter = 0

        self._init_widgets()
        self._init_plot()
        self._init_layout()
        self._init_callbacks()

        # Initial plot
        self.update_plot(0)

    def _init_widgets(self):
        # Spinners (IntInput) for compact layout
        self.xw             = pn.widgets.IntInput(name='Trace #',      value=0,  start=0, end=self._ntraces, step=1)
        self.smoothOrderInt = pn.widgets.IntInput(name='Smooth. ord.', value=11, start=1, end=31,       step=2)

        self.xwLeft         = pn.widgets.IntInput(name='Left win',  value=0,  start=0, end=50,  step=1)
        self.xwRight        = pn.widgets.IntInput(name='Right win', value=0,  start=0, end=50,  step=1)
        self.xwMinimaOrder  = pn.widgets.IntInput(name='Order',     value=50, start=0, end=400, step=1)

        self.xwMaxLeft      = pn.widgets.IntInput(name='Left win',  value=0,  start=0, end=50,  step=1)
        self.xwMaxRight     = pn.widgets.IntInput(name='Right win', value=0,  start=0, end=50,  step=1)
        self.xwMaximaOrder  = pn.widgets.IntInput(name='Order',     value=50, start=0, end=400, step=1)

        self.xwUpdateRate     = pn.widgets.IntInput(name='Update interval',   value=4,  start=1,  end=15,   step=1)
        self.xwTemplateSlider = pn.widgets.IntInput(name='Template strength', value=0,  start=0,  end=1000, step=1)

        self.ddMenu = pn.widgets.Select(name='Drive', options=['Z', 'C', 'D', 'E', 'F', '/media/marcotti-lab'], value='Z')

        self.prevButton  = pn.widgets.Button(name='<',    button_type='primary', width=50)
        self.nextButton  = pn.widgets.Button(name='>',    button_type='primary', width=50)
        
        self.bJumpCorr   = pn.widgets.Button(name='Jump corr original', button_type='primary')
        self.bLoadOrig   = pn.widgets.Button(name='Load original')
        self.bLoadJump   = pn.widgets.Button(name='Load jump-corrected')
        self.bQuickJump  = pn.widgets.Button(name='Quick load jump-corrected')
        self.bLoadMC     = pn.widgets.Button(name='Load motion-corrected')
        self.bSave       = pn.widgets.Button(name='Save File', button_type='success')
        self.buttonSaveAnalysis = pn.widgets.Button(name='Save Parameters (CSVs)', button_type='success')

        self.lblCurrent  = pn.pane.Str('None')

        self.valid1 = pn.indicators.BooleanStatus(value=False, color='success', width=20, height=20, name='Jump corr')
        self.valid2 = pn.indicators.BooleanStatus(value=False, color='success', width=20, height=20, name='Motion corr')
        self.valid3 = pn.indicators.BooleanStatus(value=False, color='success', width=20, height=20, name='Rois')
        self.valid4 = pn.indicators.BooleanStatus(value=False, color='success', width=20, height=20, name='Traces')
        self.valid5 = pn.indicators.BooleanStatus(value=False, color='success', width=20, height=20, name='Annotations')

        self.pbar = pn.indicators.Progress(name='Progress', value=0, max=100, bar_color='success', width=200)

        self.buttonDeleteSel  = pn.widgets.Button(name='Delete selected frames', button_type='warning')
        self.buttonUndo       = pn.widgets.Button(name='Undo last delete',       button_type='warning')
        self.buttonTemplate   = pn.widgets.Button(name='Create template',        button_type='primary')

        self.frameStartInt = pn.widgets.IntInput(name='Frame start:', value=0, start=0)
        self.frameEndInt   = pn.widgets.IntInput(name='Frame end:',   value=0, start=0)
        self.buttonManualInterval = pn.widgets.Button(name='Delete interval', button_type='warning')

        self.firstFrameInt     = pn.widgets.IntInput(name='First frame:', value=1, start=1)
        self.lastFrameInt      = pn.widgets.IntInput(name='Last frame:',  value=1, start=1)
        self.buttonSetFirstLast = pn.widgets.Button(name='Set first-last', button_type='primary')

    def _init_plot(self):
        self.src_orig   = ColumnDataSource(data=dict(x=[], y=[]))
        self.src_corr   = ColumnDataSource(data=dict(x=[], y=[]))
        self.src_min    = ColumnDataSource(data=dict(x=[], y=[]))
        self.src_sel    = ColumnDataSource(data=dict(x=[], y=[]))
        self.src_max    = ColumnDataSource(data=dict(x=[], y=[]))
        self.cursor_src = ColumnDataSource(data=dict(x=[0, 0], y=[-1e9, 1e9]))
        
        # Stores the selected x-range so Python can read it at button-click time
        self.sel_range_src = ColumnDataSource(data=dict(x0=[None], x1=[None]))

        self.fig = bk_figure(height=380, sizing_mode='stretch_width',
                             tools='box_zoom,xbox_select,pan,wheel_zoom,reset,save',
                             active_drag='box_zoom')
        self.fig.title.text = ''

        self.fig.line('x', 'y', source=self.src_orig, color='steelblue', line_width=1)
        self.fig.line('x', 'y', source=self.src_corr, color='orange',    line_width=1.5)
        self.fig.scatter('x', 'y', source=self.src_min, color='teal', size=10, marker='x')
        self.fig.scatter('x', 'y', source=self.src_sel, color='black', size=10, marker='x')
        self.fig.scatter('x', 'y', source=self.src_max, color='red',  size=10, marker='x')
        self.fig.line('x', 'y', source=self.cursor_src, color='black', line_width=2, line_dash='dashed')

        # Dark red shaded box for the selected interval
        self.sel_box = BoxAnnotation(left=0, right=0,
                                     fill_alpha=0.35, fill_color='darkred',
                                     line_color='darkred', line_alpha=0.8, line_width=1.5)
        self.fig.add_layout(self.sel_box)

        # CustomJS on SelectionGeometry
        sel_js = CustomJS(args=dict(box=self.sel_box, rng=self.sel_range_src), code="""
            const x0 = cb_obj.geometry.x0;
            const x1 = cb_obj.geometry.x1;
            box.left  = x0;
            box.right = x1;
            rng.data = {x0: [x0], x1: [x1]};
        """)
        self.fig.js_on_event(SelectionGeometry, sel_js)

    def _init_layout(self):
        # Widget sizing
        _IW  = 90    # spinner width
        _BW  = 170   # button width
        _COL = 200   # column width

        for w in (self.xw, self.smoothOrderInt, self.xwLeft, self.xwRight, self.xwMinimaOrder,
                  self.xwMaxLeft, self.xwMaxRight, self.xwMaximaOrder, self.xwUpdateRate,
                  self.xwTemplateSlider, self.firstFrameInt, self.lastFrameInt, 
                  self.frameStartInt, self.frameEndInt):
            w.width = _IW

        for w in (self.bJumpCorr, self.bLoadOrig, self.bLoadJump, self.bQuickJump, 
                  self.bLoadMC, self.bSave, self.buttonSaveAnalysis, self.buttonDeleteSel, self.buttonUndo, 
                  self.buttonSetFirstLast, self.buttonManualInterval):
            w.width = _BW
        
        self.ddMenu.width = 150

        # Compact status row
        status_row = pn.Row(
            pn.pane.Str('JC:', width=22), self.valid1,
            pn.pane.Str('MC:', width=22), self.valid2,
            pn.pane.Str('R:',  width=16), self.valid3,
            pn.pane.Str('T:',  width=16), self.valid4,
            pn.pane.Str('A:',  width=16), self.valid5,
            margin=(2, 0),
        )

        # Layout columns
        slider_col = pn.Column(
            self.xw, self.smoothOrderInt,
            pn.Row(self.firstFrameInt, self.lastFrameInt),
            self.buttonSetFirstLast,
            width=_COL,
        )
        minima_col = pn.Column(
            pn.pane.Str('─ Minima ─', margin=(4, 0)),
            self.xwMinimaOrder, self.xwLeft, self.xwRight,
            width=_COL,
        )
        maxima_col = pn.Column(
            pn.pane.Str('─ Maxima ─', margin=(4, 0)),
            self.xwMaximaOrder, self.xwMaxLeft, self.xwMaxRight,
            width=_COL,
        )
        load_col = pn.Column(
            self.lblCurrent,
            self.bJumpCorr, self.bLoadOrig, self.bLoadJump, self.bQuickJump, 
            self.bLoadMC,
            width=_BW + 10,
        )
        save_col = pn.Column(
            pn.pane.Str('─ Save ─', margin=(4, 0)),
            self.bSave,
            self.buttonSaveAnalysis,
            width=_BW + 10,
        )
        edit_col = pn.Column(
            pn.pane.Str('─ Edit ─', margin=(4, 0)),
            self.buttonDeleteSel,
            self.buttonUndo,
            width=_BW + 10,
        )
        nav_col = pn.Column(
            self.xwUpdateRate,
            pn.Row(self.prevButton, self.nextButton, margin=(2, 0)),
            self.pbar,
            self.ddMenu,
            width=_COL,
        )

        controls = pn.FlexBox(
            slider_col, minima_col, maxima_col, edit_col, load_col, save_col, nav_col,
            flex_wrap='wrap', align_content='flex-start',
            styles={'gap': '12px'},
        )
        
        # Add status_row to the bottom
        self.layout = pn.Column(controls, pn.pane.Bokeh(self.fig, sizing_mode='stretch_width'), status_row)

    def _init_callbacks(self):
        # Watchers - using value_throttled for spinners
        self.xwLeft.param.watch(self._on_minima_params, 'value_throttled')
        self.xwRight.param.watch(self._on_minima_params, 'value_throttled')
        self.xwMinimaOrder.param.watch(self._on_minima_params, 'value_throttled')
        self.xwMaxLeft.param.watch(self._on_maxima_params, 'value_throttled')
        self.xwMaxRight.param.watch(self._on_maxima_params, 'value_throttled')
        self.xwMaximaOrder.param.watch(self._on_maxima_params, 'value_throttled')
        self.smoothOrderInt.param.watch(self._on_smooth, 'value_throttled')
        self.xw.param.watch(self._on_trace, 'value_throttled')
        self.xwTemplateSlider.param.watch(self._on_template_slider, 'value_throttled')
        self.ddMenu.param.watch(self._on_drive, 'value')

        # Button clicks
        self.bJumpCorr.on_click(self._process_original)
        self.bLoadOrig.on_click(self._load_original)
        self.bSave.on_click(self._save_processed)
        self.buttonSaveAnalysis.on_click(self._save_analysis)
        self.bLoadJump.on_click(self._load_jump_corr)
        self.bQuickJump.on_click(self._quick_load_jump_corr)
        self.bLoadMC.on_click(self._load_motion_corr)
        self.prevButton.on_click(self._on_prev)
        self.nextButton.on_click(self._on_next)
        self.buttonDeleteSel.on_click(self._delete_selected)
        self.buttonUndo.on_click(self._undo_last_interval)
        self.buttonManualInterval.on_click(self._delete_manual_interval)
        self.buttonSetFirstLast.on_click(self._set_first_last)
        self.buttonTemplate.on_click(self._create_template)

        # Napari cursor connection
        try:
             self.tb.app.dims.events.connect(self._update_cursor)
        except:
             pass

    def show(self):
        return self.layout.servable()

    # --- Helper methods ---
    def _resolve_folder(self, el):
        workingFolder = el['Folder']
        if self.ddMenu.value != 'Z':
            if os.name == 'posix':
                workingFolder = Path(self.ddMenu.value) / Path(workingFolder[2:].replace('\\', '/').lstrip('/'))
            else:
                workingFolder = self.ddMenu.value + workingFolder[1:]
        return workingFolder

    def _parse_first_last(self, el, workingFolder):
        try:
            firstFrame, lastFrame = el['first-last'].split('-')
            firstFrame = int(firstFrame) - 1
            lastFrame  = int(lastFrame)
        except:
            tr = getImgInfo(workingFolder)
            if tr:
                firstFrame, lastFrame = 0, tr[2]
            else:
                firstFrame, lastFrame = 0, 1000 # Fallback
            if el.get('nChannels', 1) == 2:
                lastFrame = lastFrame // 2
        return firstFrame, lastFrame

    def _remove_layers(self, *names):
        for name in names:
            try:
                self.tb.app.layers.remove(name)
            except Exception:
                pass

    def _clear_selection(self):
        self.sel_range_src.data = dict(x0=[None], x1=[None])
        self.sel_box.left  = 0
        self.sel_box.right = 0

    # --- Callbacks ---
    def update_plot(self, x=None):
        try:
            self._update_plot_inner(x)
        except Exception as e:
            print(f'[JumpFramesFinderPanel] Error: {e}')
            traceback.print_exc()

    def _update_plot_inner(self, x=None):
        if x is None:
            x = self.xw.value
        
        # Ensure x is int
        x = int(x)
        if x not in self.master.index:
            return

        el = self.master.loc[x]
        workingFolder = self._resolve_folder(el)

        self.fig.title.text = str(workingFolder)
        self.lblCurrent.object = 'None'

        # Update status indicators
        self.valid1.value = os.path.exists(os.path.join(workingFolder, savefolder, '1-jumpCorrected.tif'))
        self.valid2.value = os.path.exists(os.path.join(workingFolder, savefolder, '1-jumpCorrected-mc.tif'))
        self.valid3.value = os.path.exists(os.path.join(workingFolder, savefolder, 'Masks.tif'))
        self.valid4.value = os.path.exists(os.path.join(workingFolder, savefolder, 'traces.csv'))
        self.valid5.value = os.path.exists(os.path.join(workingFolder, savefolder, 'Annotations.tif'))

        # Reset selection
        self.src_sel.data = dict(x=[], y=[])
        self._clear_selection()

        # Parse first/last
        try:
            firstFrame, lastFrame = str(el['first-last']).split('-')
            firstFrame = int(firstFrame) - 1
            lastFrame  = int(lastFrame)
        except:
            firstFrame, lastFrame = 0, -1

        # Sync first/last frame widgets
        self.firstFrameInt.value = firstFrame + 1
        if lastFrame > 0:
            self.lastFrameInt.value = lastFrame

        # Smooth order
        try:
            smoothOrder = el['SmoothOrder']
            if (smoothOrder % 2) == 0:
                smoothOrder += 1
                self.master.loc[x, 'SmoothOrder'] = smoothOrder
        except:
            smoothOrder = 11
        self.smoothOrderInt.value = int(smoothOrder)

        # Load trace
        ttrace = np.array([])
        if os.path.exists(os.path.join(workingFolder, 'corrReference.npy')):
            r, s, t = tu.loadRoisFromFile(os.path.join(workingFolder, 'corrReference.npy'))
            s = s[firstFrame * el['nChannels']:lastFrame * el['nChannels'], :]
            if smoothOrder != 1:
                for i in np.arange(s.shape[1]):
                    s[:, i] = savgol_filter(s[:, i], smoothOrder, 1)
            if el['nChannels'] == 2:
                s = s[::2, :]
            ttrace = s.mean(1)
        elif os.path.exists(os.path.join(workingFolder, 'corrReference.csv')):
            s = pd.read_csv(os.path.join(workingFolder, 'corrReference.csv'))
            s = s['Mean'].values
            s = s[firstFrame * el['nChannels']:lastFrame * el['nChannels']]
            if smoothOrder != 1:
                s = savgol_filter(s, smoothOrder, 1)
            if el['nChannels'] == 2:
                s = s[::2]
            ttrace = s

        # Sync slider values from master
        try:
            if not np.isnan(el['Minima order']):
                self.xwLeft.value        = int(el['Window left'])
                self.xwRight.value       = int(el['Window right'])
                self.xwMinimaOrder.value = int(el['Minima order'])
        except: pass
        
        try:
            if not np.isnan(el['Maxima order']):
                self.xwMaxLeft.value      = int(el['Window Max left'])
                self.xwMaxRight.value     = int(el['Window Max right'])
                self.xwMaximaOrder.value  = int(el['Maxima order'])
        except: pass

        window_size_left  = self.xwLeft.value
        window_size_right = self.xwRight.value
        minima_order      = self.xwMinimaOrder.value
        windowMax_size_left  = self.xwMaxLeft.value
        windowMax_size_right = self.xwMaxRight.value
        maxima_order         = self.xwMaximaOrder.value

        minima = argrelmin(ttrace, order=minima_order)[0]
        maxima = argrelmax(ttrace, order=maxima_order)[0]

        if len(minima) and window_size_left > minima[0]:
            minima = minima[1:]
        if (self.xwLeft.value == 0) and (self.xwRight.value == 0):
            minima = []
        try:
            self.allminima[el['Folder']] = np.pad(minima, (0, 1000 - len(minima)), 'constant')
        except:
            print('Cannot save minima')

        ttrace2 = np.zeros(ttrace.shape) - 10000
        left = 0
        for elmin in minima:
            ttrace2[left:elmin - window_size_left] = ttrace[left:elmin - window_size_left]
            left = elmin + window_size_right
        ttrace2[left:] = ttrace[left:]

        if len(maxima) and windowMax_size_left > maxima[0]:
            maxima = maxima[1:]
        if (self.xwMaxLeft.value == 0) and (self.xwMaxRight.value == 0):
            maxima = []
        self.allmaxima[el['Folder']] = np.pad(maxima, (0, 1000 - len(maxima)), 'constant')

        for elmin in maxima:
            ttrace2[elmin - windowMax_size_left:elmin + windowMax_size_right] = -10000

        try:
            for interval in el['ExtraCorrectionIntervals']:
                ttrace2[interval[0]:interval[1]] = -10000
        except (TypeError, KeyError):
            pass
        try:
            for interval in el['TemplateIntervals']:
                ttrace2[interval[0]:interval[1]] = -10000
        except (TypeError, KeyError):
            pass

        ttrace2[ttrace2 == -10000] = np.nan

        self.src_orig.data = dict(x=list(np.arange(ttrace.shape[0])),  y=list(ttrace))
        self.src_corr.data = dict(x=list(np.arange(ttrace2.shape[0])), y=list(ttrace2))
        self.src_min.data  = dict(x=list(minima), y=list(ttrace[minima] if len(minima) else []))
        self.src_max.data  = dict(x=list(maxima), y=list(ttrace[maxima] if len(maxima) else []))

        # Update cursor range
        try:
            ymin = float(np.nanmin(ttrace)) * 0.9
            ymax = float(np.nanmax(ttrace)) * 1.1
            self.cursor_src.data = dict(x=self.cursor_src.data['x'], y=[ymin, ymax])
        except: pass

    def _update_cursor(self, event):
        self._panel_counter += 1
        if self._panel_counter % self.xwUpdateRate.value == 0:
            try:
                frame = self.tb.app.dims.current_step[0]
                self.cursor_src.data = dict(x=[frame, frame], y=self.cursor_src.data['y'])
            except: pass
            self._panel_counter = 0

    def _on_minima_params(self, event):
        self.master.loc[self.xw.value, 'Window left']   = self.xwLeft.value
        self.master.loc[self.xw.value, 'Window right']  = self.xwRight.value
        self.master.loc[self.xw.value, 'Minima order']  = self.xwMinimaOrder.value
        self.update_plot()

    def _on_maxima_params(self, event):
        self.master.loc[self.xw.value, 'Window Max left']  = self.xwMaxLeft.value
        self.master.loc[self.xw.value, 'Window Max right'] = self.xwMaxRight.value
        self.master.loc[self.xw.value, 'Maxima order']     = self.xwMaximaOrder.value
        self.update_plot()

    def _on_smooth(self, event):
        self.master.loc[self.xw.value, 'SmoothOrder'] = self.smoothOrderInt.value
        self.update_plot()

    def _on_trace(self, event):
        self.update_plot(self.xw.value)

    def _on_drive(self, event):
        self.update_plot()

    def _on_template_slider(self, event):
        if self._distance_profile[0] is None:
            return
        dp = self._distance_profile[0].copy()
        tmpl = self._template[0]
        if tmpl is None: return
        
        idcs = np.argsort(dp)
        intervals = []
        i = 0
        while i <= self.xwTemplateSlider.value:
            intervals.append([idcs[0], idcs[0] + tmpl.size])
            dp[idcs[0]:idcs[0] + tmpl.size] = dp.max()
            idcs = np.argsort(dp)
            i += 1
        self.master.at[self.xw.value, 'TemplateIntervals'] = intervals
        self.update_plot()

    def _process_original(self, event):
        el = self.master.loc[self.xw.value]
        workingFolder = self._resolve_folder(el)
        firstFrame, lastFrame = self._parse_first_last(el, workingFolder)
        thisMinima = self.allminima[el['Folder']]; thisMinima = thisMinima[thisMinima != 0]
        thisMaxima = self.allmaxima[el['Folder']]; thisMaxima = thisMaxima[thisMaxima != 0]
        nChannels = el.get('nChannels', 1)
        frameIntervalsToRemove = calculateFrameIntervalsToRemove(
            jumpFrames=thisMinima, winLeft=el['Window left'], winRight=el['Window right'],
            jumpFramesMax=thisMaxima, winMaxLeft=el['Window Max left'], winMaxRight=el['Window Max right']
        )
        try: frameIntervalsToRemove.extend(el['ExtraCorrectionIntervals'])
        except TypeError: pass
        try: frameIntervalsToRemove.extend(el['TemplateIntervals'])
        except TypeError: pass
        
        self.lblCurrent.object = 'Jump-corrected movie'
        spatialGaussian  = int(el.get('SpatialGaussian',  2))
        temporalGaussian = int(el.get('TemporalGaussian', 2))
        
        self.tb.loadFile(workingFolder, applyGaussian=True, nChannels=nChannels,
                         spatialGaussian=spatialGaussian, temporalGaussian=temporalGaussian)
        self.pbar.max   = self.tb.nFrames
        self.pbar.value = 0
        self.tb.loadFrameInterval(firstFrame, lastFrame, frameIntervalsToRemove=frameIntervalsToRemove)
        self._remove_layers('Masks', 'Avg', 'Annotations')
        if nChannels == 2:
            self.tb.loadFile(workingFolder, applyGaussian=True, nChannels=nChannels,
                             spatialGaussian=spatialGaussian, temporalGaussian=temporalGaussian)
            self.tb.loadFrameInterval(firstFrame, lastFrame, frameIntervalsToRemove=frameIntervalsToRemove,
                                      layerName='Image channel 2', channel=2)

    def _load_original(self, event):
        el = self.master.loc[self.xw.value]
        workingFolder = self._resolve_folder(el)
        firstFrame, lastFrame = self._parse_first_last(el, workingFolder)
        nChannels = el.get('nChannels', 1)
        self.tb.loadFile(workingFolder, applyGaussian=True, nChannels=nChannels)
        self.pbar.max = self.tb.nFrames; self.pbar.value = 0
        self.tb.loadFrameInterval(firstFrame, lastFrame, frameIntervalsToRemove=None)
        self._remove_layers('Masks', 'Avg', 'Annotations')

    def _save_processed(self, event):
        el = self.master.loc[self.xw.value]
        workingFolder = self._resolve_folder(el)
        if self.lblCurrent.object == 'Jump-corrected movie':
            outFolder = os.path.join(workingFolder, savefolder)
            os.makedirs(outFolder, exist_ok=True)
            tifffile.imwrite(os.path.join(outFolder, '1-jumpCorrected.tif'), self.tb.app.layers['Image'].data)
            self.valid1.value = True
            if el.get('nChannels', 1) == 2:
                self.valid1.value = False
                tifffile.imwrite(os.path.join(outFolder, '1-jumpCorrected-channel2.tif'),
                                 self.tb.app.layers['Image channel 2'].data)
                self.valid1.value = True

    def _save_analysis(self, event):
        try:
            if self.corrFilename:
                self.master.to_csv(self.corrFilename)
            if self.jumpFrameFilename:
                self.allminima.to_csv(self.jumpFrameFilename)
            if self.jumpFrameMaxFilename:
                self.allmaxima.to_csv(self.jumpFrameMaxFilename)
            print("Analysis saved.")
        except Exception as e:
            print(f"Error saving analysis: {e}")

    def _load_jump_corr(self, event):
        el = self.master.loc[self.xw.value]
        workingFolder = self._resolve_folder(el)
        outFolder = os.path.join(workingFolder, savefolder, '1-jumpCorrected.tif')
        self.tb.loadFromTiff(outFolder, nChannels=el.get('nChannels', 1), channel=1)
        if el.get('nChannels', 1) == 2:
            outFolder2 = os.path.join(workingFolder, savefolder, '1-jumpCorrected-channel2.tif')
            self.tb.loadFromTiff(outFolder2, title='Image channel 2', nChannels=el['nChannels'], channel=2)
        self._remove_layers('Masks', 'Avg', 'Annotations')

    def _quick_load_jump_corr(self, event):
        el = self.master.loc[self.xw.value]
        workingFolder = self._resolve_folder(el)
        outFolder = os.path.join(workingFolder, savefolder, '1-jumpCorrected.tif')
        self.tb.loadQuickLook(outFolder)
        self._remove_layers('Masks', 'Avg', 'Annotations')

    def _load_motion_corr(self, event):
        el = self.master.loc[self.xw.value]
        workingFolder = self._resolve_folder(el)
        outFolder = os.path.join(workingFolder, savefolder, '1-jumpCorrected-mc.tif')
        self.tb.loadFromTiff(outFolder)
        self._remove_layers('Masks', 'Avg', 'Annotations')

    def _on_prev(self, event):
        try:
            z, x, y = self.tb.app.dims.current_step
            if z > 0:
                self.tb.app.dims.current_step = (z - 1, x, y)
                self.cursor_src.data = dict(x=[z - 1, z - 1], y=self.cursor_src.data['y'])
        except: pass

    def _on_next(self, event):
        try:
            z, x, y = self.tb.app.dims.current_step
            if z < self.tb.nFrames:
                self.tb.app.dims.current_step = (z + 1, x, y)
                self.cursor_src.data = dict(x=[z + 1, z + 1], y=self.cursor_src.data['y'])
        except: pass

    def _delete_selected(self, event):
        x0 = self.sel_range_src.data['x0'][0]
        x1 = self.sel_range_src.data['x1'][0]
        if x0 is None or x1 is None:
            return
        intervals = self.master.at[self.xw.value, 'ExtraCorrectionIntervals']
        if not isinstance(intervals, list):
            intervals = []
        intervals.append([int(x0), int(x1)])
        self.master.at[self.xw.value, 'ExtraCorrectionIntervals'] = intervals
        self._clear_selection()
        self.update_plot()

    def _undo_last_interval(self, event):
        intervals = self.master.at[self.xw.value, 'ExtraCorrectionIntervals']
        if isinstance(intervals, list) and len(intervals) > 0:
            intervals.pop()
            self.master.at[self.xw.value, 'ExtraCorrectionIntervals'] = intervals
            self.update_plot()

    def _delete_manual_interval(self, event):
        intervals = self.master.at[self.xw.value, 'ExtraCorrectionIntervals']
        if not isinstance(intervals, list):
            intervals = []
        intervals.append([self.frameStartInt.value, self.frameEndInt.value])
        self.master.at[self.xw.value, 'ExtraCorrectionIntervals'] = intervals
        self.update_plot()

    def _set_first_last(self, event):
        if self.master['first-last'].dtype != object:
            self.master['first-last'] = self.master['first-last'].astype(object)
        self.master.at[self.xw.value, 'first-last'] = f'{self.firstFrameInt.value}-{self.lastFrameInt.value}'
        self.update_plot()

    def _create_template(self, event):
        x0 = self.sel_range_src.data['x0'][0]
        x1 = self.sel_range_src.data['x1'][0]
        if x0 is None or x1 is None:
            return
        i0 = int(x0); i1 = int(x1)
        ys = np.array(self.src_orig.data['y'])[i0:i1]
        if not len(ys):
            return
        self._template[0] = ys
        self._distance_profile[0] = mass_ts.mass2(np.array(self.src_orig.data['y']), self._template[0])
        self._clear_selection()
        self.update_plot()


def jumpFramesFinderPanel(master, allminima, allmaxima, tb, corrFilename=None, jumpFrameFilename=None, jumpFrameMaxFilename=None):
    """
    Panel-based interactive tool to find and correct jumps in fluorescence movies.
    Now implemented as a wrapper around JumpFramesFinderPanel class.
    """
    pn.extension()
    app = JumpFramesFinderPanel(master, allminima, allmaxima, tb, corrFilename, jumpFrameFilename, jumpFrameMaxFilename)
    return app.show()

# Class definition for JumpFramesFinderPanel will be inserted above this function.


import itertools
import ipywidgets as widgets
from ipywidgets import interact, Dropdown
import plotly.graph_objs as go
import pandas as pd 

colors2 = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]

def simpleTracePlotter(alltraces,xScaleFactor = 1, xlabel = 'Time (s)',showCorrelation=True):
    """
    Creates an interactive plot of multiple traces using ipywidgets with sliders for navigation.
    Parameters
    ----------
    alltraces : pd.DataFrame
        DataFrame containing traces where keys are trace names and values are lists of traces.
        Each list of traces will be plotted separately.
        The traces should be in pandas DataFrame format with time values either in a 'Time (s)' column
        or implicitly defined by the row index.
    xScaleFactor : float, optional
        Scale factor for the x-axis (time axis). Default is 1.
    xlabel : str, optional
        Label for the x-axis. Default is 'Time (s)'.
    showCorrelation : bool, optional
        If True, calculates and displays correlation between traces. Default is True.
    Returns
    -------
    None
      
    Notes
    -----
    The plot includes:
    - Interactive navigation through different recordings
    - Option to view all traces stacked or individual traces
    - Color-coded traces for better visualization
    - Automatic time axis generation if not provided
    - Correlation calculation between traces

    """
    identifiers = list(alltraces.keys())
    xw = widgets.IntSlider(min=0,max=np.size(identifiers),step=1,value=0,continuous_update=False,description='Recording #')
    xw2 = widgets.IntSlider(min=0,max=100,step=1,value=0,continuous_update=False,description='Trace #')

    prevButton = widgets.Button(description = 'Previous')
    prevButton.style.button_color = 'pink' 
    nextButton = widgets.Button(description = 'Next')
    nextButton.style.button_color = 'pink' 
    
    hbox = widgets.HBox((prevButton,nextButton))
    fig = go.FigureWidget(data=[],layout=go.Layout({ 'autosize':True,'height':750, 'xaxis_title': xlabel }))

    output = widgets.Output()
    ui = widgets.VBox([xw,xw2,hbox,fig, output])


    def on_next_clicked(click):
        if xw.value< xw.max:
            xw.value = xw.value+1
    nextButton.on_click(on_next_clicked)


    def on_prev_clicked(click):
        if xw.value> xw.min:
            xw.value = xw.value - 1
    prevButton.on_click(on_prev_clicked)


    def f(x,x2):
        colors = itertools.cycle(colors2)
        
       
        print(identifiers[x])
        
        dff0s = alltraces[identifiers[x]]
        if 'Time (s)' in dff0s.columns:
            time = dff0s['Time (s)']
            x = time/xScaleFactor
            dff0s = dff0s.drop('Time (s)',axis=1)
        else:
            time = np.arange(dff0s.shape[0])
            x = np.arange(dff0s.shape[0])

        if x2 ==0:
            fig.data = []
            datatoAdd = []
            traces = tu.stackedPlot(dff0s.values)
            for j in range(dff0s.shape[1]):
                trace = traces[:,j]
                traceName = dff0s.columns[j]
                color = next(colors)

                datatoAdd.append(go.Scatter(x=x,y= trace, mode= 'lines',line=dict(color=color, width=2,),  name ='',showlegend=False, ))



            fig.add_traces(datatoAdd)

        else:
            fig.data = []
            

            trace = dff0s.values[:,x2-1]
            traceName = dff0s.columns[x2-1]
            color = next(colors)
            if 'Time (s)' in dff0s.columns:
                time = dff0s['Time (s)']
                x = time/xScaleFactor
            else:
                time = np.arange(dff0s.shape[0])
                x = np.arange(dff0s.shape[0])

            datatoAdd = go.Scatter(x=x,y= trace, mode= 'lines',line=dict(color=color, width=2,),  name ='',showlegend=False, )
            fig.add_trace(datatoAdd)

        fps = 1/(np.diff(time).mean())
        tu.calculateCorrelation(dff0s,5*60*fps)
        
    out = widgets.interactive_output(f, {'x': xw,'x2':xw2})
    display(ui,out)
