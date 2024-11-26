import ipywidgets as widgets
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
    #xwslice = widgets.IntSlider(min=0,max=2,step=1,value=0,continuous_update=False,description='Current slice',layout=widgets.Layout(width='95%'))
    xwMaxLeft = widgets.IntSlider(min=0,max=50,step=1,value=0,continuous_update=False,description='Left win')
    xwMaxRight = widgets.IntSlider(min=0,max=50,step=1,value=0,continuous_update=False,description='Right win')
    xwMaximaOrder = widgets.IntSlider(min=0,max=400,step=1,value=50,continuous_update=False,description='Order')
    lbl2 = widgets.Label('Maxima',layout= widgets.Layout(display="flex", justify_content="center"))
    xwUpdateRate = widgets.IntSlider(min=1,max=15,step=1,value=4,continuous_update=False,description='Update interval')
    xwTemplateSlider =  widgets.IntSlider(min=0,max=1000,step=1,value=0,continuous_update=False,description='Template strength')
    lbl3 = widgets.Label('Template search',layout= widgets.Layout(display="flex", justify_content="center"))
    pbar = widgets.IntProgress(min=0,max=1,bar_style='success',description='Progress')
    ddMenu =  widgets.Dropdown(options=['Z', 'C', 'D','E','F'], value='Z', description='Drive:', disabled=False)
    

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
            workingFolder = ddMenu.value + workingFolder[1:]
        
      

        try:
            firstFrame,lastFrame = el['first-last'].split('-')
            firstFrame = int(firstFrame)-1
            lastFrame = int(lastFrame)
        except:
            firstFrame,lastFrame = [0, getImgInfo(workingFolder)[2]]

        winLeft = el['Window left']
        winRight = el['Window right']
        thisMinima = allminima[el['Folder']]
        thisMinima = thisMinima[thisMinima!=0]

        winMaxLeft = el['Window Max left']
        winMaxRight = el['Window Max right']
        thisMaxima = allmaxima[el['Folder']]
        thisMaxima = thisMaxima[thisMaxima!=0]


        frameIntervalsToRemove = calculateFrameIntervalsToRemove(lastFrame-firstFrame,jumpFrames=thisMinima,winLeft=winLeft,winRight=winRight, jumpFramesMax=thisMaxima, winMaxLeft=winMaxLeft, winMaxRight=winMaxRight)
        try:
            frameIntervalsToRemove.extend(el['ExtraCorrectionIntervals'])
        except TypeError:
            pass
        try:
            frameIntervalsToRemove.extend(el['TemplateIntervals'])
        except TypeError:
            pass

        lblCurrent.value = 'Jump-corrected movie'
        tb.loadFile(workingFolder, applyGaussian = True)
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

    b.on_click(process_original_cbk)

    def load_original_cbk(b):
        el = master.loc[xw.value]
        workingFolder = el['Folder']
        if ddMenu.value !='Z':
            workingFolder = ddMenu.value + workingFolder[1:]
        try:
            firstFrame,lastFrame = el['first-last'].split('-')
            firstFrame = int(firstFrame)-1
            lastFrame = int(lastFrame)
        except:
            firstFrame,lastFrame = [0, getImgInfo(workingFolder)[2]]

        tb.loadFile(workingFolder, applyGaussian = True)
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
            workingFolder = ddMenu.value + workingFolder[1:]
        if lblCurrent.value == 'Jump-corrected movie':
            
                outFolder = os.path.join(workingFolder,savefolder)

                if not os.path.exists(outFolder):
                    os.makedirs(outFolder)
                

                outfile = os.path.join(outFolder,'1-jumpCorrected.tif')
                tifffile.imwrite(outfile, tb.app.layers['Image'].data)
                valid1.value = True
    b3.on_click(saveProcessed)

    def loadJumpCorr(b):
        el = master.loc[xw.value]
        workingFolder = el['Folder']
        if ddMenu.value !='Z':
            workingFolder = ddMenu.value + workingFolder[1:]
        #try:
        outFolder = os.path.join(workingFolder,savefolder,'1-jumpCorrected.tif')
            
        #tb.loadQuickLook(outFolder)
        tb.loadFromTiff(outFolder)
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
    sliderStack = widgets.VBox([xw,valid1,valid2,valid3,valid4,valid5])
    buttonStack = widgets.VBox([lblCurrent,b,b2,b4,b4bis,b5,b3])
    buttonStack2 = widgets.VBox([button,lbl3,buttonTemplate,xwTemplateSlider])
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


    #Main function executed by ipywidget interactive output 
    def f(x, window_size_left, window_size_right, minima_order, windowMax_size_left,windowMax_size_right,maxima_order,drive=None):
        """
        Updates and visualizes motion correction analysis data for a given dataset.
        Called by widgets.interactive_output
        """

        el = master.loc[x]
        workingFolder = el['Folder']
        if ddMenu.value !='Z':
            workingFolder = ddMenu.value + workingFolder[1:]
        
      

        fig.layout.title.text = workingFolder
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

            
        if os.path.exists(os.path.join(workingFolder,'corrReference.npy')):
                r,s,t = tu.loadRoisFromFile(os.path.join(workingFolder,'corrReference.npy'))
                s = s[firstFrame:lastFrame,:]
                for i in np.arange(s.shape[1]):
                    s[:,i] = savgol_filter(s[:,i],11,1)
                ttrace = s.mean(1)
        elif  not pd.isna(el['rois']):
            if os.path.exists(os.path.join(workingFolder,el['rois'])):
                r,s,t = tu.loadRoisFromFile(os.path.join(workingFolder,el['rois']))
                s = s[firstFrame:lastFrame,:]
                for i in np.arange(s.shape[1]):
                    s[:,i] = savgol_filter(s[:,i],11,1)
                ttrace = s.mean(1)
        else:
            try:
                ttrace = correctionReferenceTraceDf[el['Folder']].values
                ttrace = ttrace[~pd.isna(ttrace)]
                ttrace = ttrace[firstFrame:lastFrame]
            except:
                print('no roi found')
                prevImg = getPreviewImage(workingFolder)
                height,width = prevImg.shape

                movie = thorlabsFile(workingFolder,applyGaussian=False,showViewer=False)
                movie.loadFrameInterval(firstFrame,lastFrame)

                s = movie.array.mean(2).mean(1)
                s = np.expand_dims(s,1)
                s = s[firstFrame:lastFrame,:]   

                for i in np.arange(s.shape[1]):
                    s[:,i] = savgol_filter(s[:,i],11,1)
        
                ttrace = s.mean(1)
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
