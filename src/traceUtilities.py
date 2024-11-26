#Trace Utilities Module
#A collection of utility functions for processing calcium imaging traces and ROI data.

import numpy as np
import os 
import pandas as pd
from scipy.signal import savgol_filter, argrelmin
from scipy.interpolate import interp1d

def loadRoisFromFile(filename):
    """
    Load ROIs (Regions of Interest) and their associated data from a NumPy file.
    This function loads ROI data including coordinates, traces and timestamps from a .npy file.
    This ROIs are only used for jump correction and are not used for the analysis of the traces.
    This files are automatically generated using the Robopy software. 
    The file should contain a structured array with 'ROIS', 'traces' and 'times' fields.
    Parameters
    ----------
    filename : str
        Path to the .npy file containing ROI data

    Returns
    -------
    tuple
        A tuple containing:
        - roboRois : list
            List of ROI objects with coordinates and color information
        - roiprofile : ndarray
            Array containing trace data for each ROI
        - times : ndarray
            Array of timestamps associated with the traces

        Note that only the 'roiprofile' array is used for subsequent analysis
    Notes
    -----
    The input .npy file should be saved with protocol 2 compatibility and contain
    a structured array with bytes-encoded keys.
    """

    roiprofile = None
    times = None
    roiBounds = None


    R = np.load(filename,encoding='bytes',allow_pickle=True)
    R = np.expand_dims(R,0)[0]
    ROIS = R[b'ROIS']
    roiprofile = R[b'traces']
    times = R[b'times']
    roboRois = []

    for roi in ROIS:

        c = roi[b'Coordinates']
        color = roi[b'Color']
        isValidRoi = True


    if roiBounds is None:
        return roboRois, roiprofile, times
        
		
def calculatedFF0(traces,f0Frames = [0,5]):
    """
    Calculates ΔF/F0 (delta F / F0) for fluorescence traces.
    Parameters
    ----------
    traces : numpy.ndarray
        2D array of fluorescence traces where rows are time points and columns are ROIs/cells
    f0Frames : list or str, optional
        If list: [start_frame, end_frame] defining window for F0 calculation
        If 'percentile': uses 5th percentile of entire trace as F0
        Default is [0,5]
    Returns
    -------
    numpy.ndarray
        2D array of ΔF/F0 values with same dimensions as input traces
    Notes
    -----
    ΔF/F0 is calculated as (F - F0)/F0 where:
    - F is the raw fluorescence trace
    - F0 is either mean of specified frames or 5th percentile of trace
    """
    if f0Frames == 'percentile':
        f0 = np.nanpercentile(traces,5,axis=0)
    else:
        f0 = np.nanmean(traces[f0Frames[0]:f0Frames[1],:],0)
    
    f0 = np.tile(f0,(traces.shape[0],1))
    dff0 = (traces-f0)/f0
    return dff0


def stackedPlot(traces):
    '''
    Plot traces in a stacked plot
    '''
    out = traces.copy()
    for i in range(1,traces.shape[1]):
        out[:,i] = traces[:,i] + np.nanmax(out[:,i-1])
    return out

def rollingMedianCorrection(traces,rollingN = 1000):
    """
    Remove slow drifts in calcium traces by subtracting the rolling median.
    This function calculates and subtracts the rolling median from calcium traces to remove slow
    drifts that could affect correlation calculations. Works with both 1D and 2D arrays.
    Parameters
    ----------
    traces : numpy.ndarray
        Input calcium traces. Can be either a 1D array (single trace) or 2D array (multiple traces)
    rollingN : int, optional
        Window size for calculating rolling median. Default is 1000 samples
    Returns
    -------
    numpy.ndarray
        Drift-corrected calcium traces with same shape as input. 
        For 2D input: returns corrected traces as a 2D array
        For 1D input: returns corrected trace as a 1D array
    Notes
    -----
    Uses pandas rolling median calculation with zero minimum periods to handle
    edge cases at the start of the traces.
    """

    traces2 = pd.DataFrame(traces)
    traces2 = traces2 - traces2.rolling(rollingN,min_periods=0).median()
    if traces.ndim ==2:
        traces2 = traces2.values #+ traces[:5,:].mean(0)
        return traces2
    elif traces.ndim == 1:
        traces2 = traces2.values #+ traces[0]
        return traces2[:,0]

def fillMissingValues(dff0):
    """
    Fill missing values in dff0 traces where values stay constant due to removed frames using linear interpolation.
    This function detects segments where the difference between consecutive values is zero,
    indicating removed frames, and fills these gaps using linear interpolation between valid points.
    A window size of 5 points before and after each gap is used for interpolation.
    Parameters
    ----------
    dff0 : pandas.DataFrame
        DataFrame containing fluorescence traces where each column represents a different trace
        and each row represents a time point.
    Returns
    -------
    pandas.DataFrame
        The input DataFrame with gaps filled using linear interpolation.
    Notes
    -----
    The function uses a linear interpolation method as other interpolation methods have been
    found to be less effective for this specific application.
    If an IndexError occurs during interpolation of any segment, that segment is skipped.
    """

    dz = np.diff(dff0.iloc[:,0])
    keep = np.argwhere(dz!=0)[:,0]
    discard = np.argwhere(dz==0)[:,0]

    for i in range(dff0.shape[1]):
        trace = dff0.values[:,i]
        size = 5
        for j in np.argwhere(np.diff(keep)!=1)[:,0]: #find the index where the gap between frames to keep is larger that 1
        
            try:            
                firstInterval = [keep[k] for k in np.arange(j-size+1,j+1)]
                secondInterval = [keep[k] for k in np.arange(j+1,j+size+1)]
                interval = firstInterval + secondInterval
                

                x1 = np.arange(size)
                x2 = np.arange(size)+keep[j+1] - keep[j] + size
                x = np.hstack((x1,x2))
                f = interp1d(x,trace[interval],kind=1)
                newx = np.arange(x[-1]+1)
                newy = f(newx)
                dff0.values[keep[j]:keep[j+1],i]=newy[size:-size]
            except IndexError:
                pass

    return dff0	

def getRawImage(sequence,n,width,height):
	frameSize = width*height*2
	offset = int(n * frameSize)
	sequence.seek(offset)
	st = sequence.read(frameSize)

	nparray = np.fromstring(st,dtype = np.uint16).reshape((height,width))
	return nparray


def getSequenceAsArray(filename,width,height):
	sequence = open(filename,'rb')
	frameSize = width*height*2
	nbytes = os.path.getsize(filename)
	nframes = int(nbytes/frameSize)
	allimg = []
	for i in range(nframes):
		#print(i)
		allimg.append(getRawImage(sequence,i,width,height))
	sequence.close()
	return np.array(allimg)


def returnJumpFrames(master,savgolFilter=True,savgolOrder =11):
    """
    Calculate the positions of "jumps" (minima) in traces and return them as frame values.
    This function processes trace data from multiple recordings, identifies local minima
    in the averaged traces, and returns them as frame indices in a pandas DataFrame.
    Parameters
    ----------
    master : pandas.DataFrame
        DataFrame containing information about recordings with columns:
        - 'Folder': path to recording folder
        - 'rois': ROI file name
        - 'first-last': string with format 'first-last' frame numbers (optional)
        - 'Minima order': integer for minimum detection sensitivity
    savgolFilter : bool, optional
        Whether to apply Savitzky-Golay filtering to traces (default=True)
    savgolOrder : int, optional
        Window length for Savitzky-Golay filter (default=11)
    Returns
    -------
    pandas.DataFrame
        DataFrame where each column corresponds to a recording folder and contains
        frame indices of detected minima, padded with zeros to length 1000
    Notes
    -----
    - If 'first-last' is not specified, entire trace is used
    - Minima are detected using scipy.signal.argrelmin
    - Failed minima calculations are printed as error messages
    """

    allminima = pd.DataFrame()

    for _, el in master.iterrows():

        _, s, _ = loadRoisFromFile(os.path.join(el['Folder'],el['rois']))
        try:
            firstFrame,lastFrame = el['first-last'].split('-')
        except:
            firstFrame,lastFrame = [0, s.shape[0]]
        
        firstFrame = int(firstFrame)-1
        lastFrame = int(lastFrame)
        s = s[firstFrame:lastFrame,:]
        if savgolFilter:
            for i in np.arange(s.shape[1]):
                s[:,i] = savgol_filter(s[:,i],savgolOrder,1)
				


		#plot(tu.stackedPlot(tu.calculatedFF0(s)))
        ttrace = s.mean(1)
        minima_order =el['Minima order']
        minima = argrelmin(ttrace,order=minima_order )[0]
        try:
            allminima[el['Folder']] = np.pad(minima,(0,1000-len(minima)),'constant')
        except ValueError:
            print('Cannot calculate minima for '+el['Folder'])

    return allminima


import tifffile
import os
import pandas as pd
import numpy as np


def calculatePixelRollingCorr(folder, window, downsample=4, addNoise=False, maskfilename='Masks.tif'):
    """
    Calculates the time-dependent average correlation between pixels within each ROI.
    This function analyzes temporal correlations between pixels in regions of interest (ROIs)
    from a motion-corrected time series. It can help distinguish genuine signals from artifacts
    by examining the coherence of pixel activity within ROIs.
    Parameters
    ----------
    folder : str
        Path to the main directory containing the processed movies and masks
    window : int
        Size of the temporal window (in frames) for calculating rolling correlations
    downsample : int, optional
        Factor by which to downsample the spatial dimensions (default=4)
    addNoise : bool, optional
        Whether to add Gaussian noise to periods with no signal change (default=False)
    maskfilename : str, optional
        Name of the mask file containing ROI definitions (default='Masks.tif')
    Returns
    -------
    pandas.DataFrame
        DataFrame containing correlation time series for each ROI.
        Columns are named 'ROI_1', 'ROI_2', etc.
    Notes
    -----
    The function attempts to load motion-corrected data from either '1-jumpCorrected-mc.tif'
    or '1-jumpCorrected.tif' in the 'processedMovies' subdirectory.
    When addNoise=True, Gaussian noise with standard deviation = 1/3 of the signal's std
    is added to timepoints with no signal change to prevent undefined correlations.
    """

    try:
        filename = os.path.join(folder,'processedMovies','1-jumpCorrected-mc.tif')
        data = tifffile.imread(filename)[:,::downsample,::downsample]
    except FileNotFoundError:
        filename = os.path.join(folder,'processedMovies','1-jumpCorrected.tif')
        data = tifffile.imread(filename)[:,::downsample,::downsample]
        
    filename = os.path.join(folder,'processedMovies',maskfilename)
    masks = tifffile.imread(filename)[::downsample,::downsample]

    df = pd.DataFrame()
    for i in range(1,masks.max()+1):   
        d = data[:,masks==i]
        
        dff0 = d.mean(1)
        dz = np.diff(dff0)
        keep = np.argwhere(dz!=0)[:,0]
        discard = np.argwhere(dz==0)[:,0]  
        out = [0]*window
        if addNoise:
            d[discard,:] = d[discard,:] + np.random.normal(0,d.std(0)/3,(discard.size,d.shape[1])) # add noise to avoid zeros for small windows
        for j in range(window,d.shape[0]-window):
           # if j in keep:
                out.append(np.corrcoef(d[j-window:j+window,:].T).mean())
           # else:
           #     out.append(np.nan)
        df['ROI_'+str(i)] = out
            
    return df


def concatenateRecordings(el, alltraces, rollingMedianCorrectionNumber=None):
    """
    Concatenates traces from recordings in a sequence based on matching ROI numbers.
    Args:
        el (pandas.DataFrame): DataFrame containing metadata for cells from different recordings in the sequence.
                                Must include columns: 'Cell ID', 'Number in sequence', 'Matched RoiN', 'fps'
        alltraces (dict): Dictionary containing trace data for each cell, with Cell IDs as keys
        rollingMedianCorrectionNumber (int, optional): Window size for rolling median correction. Defaults to None.
    Returns:
        pandas.DataFrame: Concatenated traces with the following properties:
            - Columns are ROI numbers
            - Rows represent timepoints
            - 1000-point nan gaps between sequences
            - Last column is 'Time (s)'
            - Traces are aligned based on sequence number
            - Optional rolling median correction applied if specified
    Notes:
        - Preserves temporal alignment between different recordings
        - Handles missing ROIs in sequences by filling with NaN values
        - Automatically trims trailing NaN values
        - Sorts output columns (ROIs) in ascending order
    """
    cellIDs = list(el['Cell ID'].values)


    maxSequenceN = el['Number in sequence'].max()
    processedCells = []
    sequenceSizes = {} # keep track of the number of frames in the different sequences in a multiSequence series
    sequenceNumbers = pd.DataFrame() # keep track of the sequence numbers comprising each trace
    
    #Determine the sizes of the different sequences
    for cellid in cellIDs:
        if cellid not in processedCells:
            processedCells.append(cellid)

            #cellIDs.remove(cellid)
            #print(cellIDs)
            this_el = el[el['Cell ID']== cellid]
            this_dff0s = alltraces[cellid].dropna()
            sequenceN = this_el['Number in sequence'].values[0]
            matchedRoiN = this_el['Matched RoiN'].values[0]
            this_sequenceNumbers = [sequenceN]
            sequenceSizes[float(sequenceN)] = this_dff0s.size
            for j in range(sequenceN+1,maxSequenceN+1):
                next_el = el[(el['Matched RoiN']==matchedRoiN) & (el['Number in sequence']==j) ]
                if next_el.shape[0]!=0:
                    this_sequenceNumbers.append(j)
                    
                    next_cellID = next_el['Cell ID'].values[0]
                    processedCells.append(next_cellID)
                    #this_dff0s = np.hstack((this_dff0s,[np.nan]*1000,alltraces[next_cellID].dropna()))
                    sequenceSizes[float(j)] = alltraces[next_cellID].dropna().size
                
    #Build the traces and keep them aligned
    dff0s=pd.DataFrame()
    processedCells = []
    for cellid in cellIDs:
        if cellid not in processedCells:
            processedCells.append(cellid)

            this_el = el[el['Cell ID']== cellid]
            this_dff0s = alltraces[cellid].dropna()

            sequenceN = this_el['Number in sequence'].values[0]
            matchedRoiN = this_el['Matched RoiN'].values[0]
            this_sequenceNumbers = [sequenceN]
                
            if sequenceN == 1:
                    concatenatedTrace = np.hstack((this_dff0s,[np.nan]*1000))
            else:
                    concatenatedTrace = np.array([])
                    for j in range(1,sequenceN):
                        concatenatedTrace = np.hstack((concatenatedTrace,[np.nan]*(sequenceSizes[float(j)]+1000)))
                    concatenatedTrace = np.hstack((concatenatedTrace,this_dff0s,[np.nan]*1000))
                
            for j in range(sequenceN+1,maxSequenceN+1):
                    
                    
                    
                    next_el = el[(el['Matched RoiN']==matchedRoiN) & (el['Number in sequence']==j) ]
                    if next_el.shape[0]!=0:
                        this_sequenceNumbers.append(j)
                        
                        next_cellID = next_el['Cell ID'].values[0]
                        processedCells.append(next_cellID)
                        concatenatedTrace = np.hstack((concatenatedTrace,alltraces[next_cellID].dropna(),[np.nan]*1000,))
                    else:
                        concatenatedTrace = np.hstack((concatenatedTrace,[np.nan]*(sequenceSizes[float(j)]+1000)))
                        
            dff0s[matchedRoiN] = concatenatedTrace
            sequenceNumbers[matchedRoiN] = this_sequenceNumbers + [np.nan]*(maxSequenceN-len(this_sequenceNumbers))
    
    
    #remove the final nans
    maxNFrames = 0
    df2 = dff0s[::-1].isna()
    for _,colu in df2.items():
        this_maxNFrames = colu.where(colu==False).dropna().index[0]
        if this_maxNFrames> maxNFrames:
            maxNFrames = this_maxNFrames             
    dff0s = dff0s.iloc[:maxNFrames,:] 

    dff0s = dff0s.reindex(sorted(dff0s.columns), axis=1)
    
    if rollingMedianCorrectionNumber is not None:
            dff0s.loc[:,:] = rollingMedianCorrection(dff0s.values,rollingN=rollingMedianCorrectionNumber)
    time = np.arange(dff0s.shape[0])/el['fps'].values[0]
    dff0s['Time (s)'] = time
    return dff0s


import seaborn as sns
def calculateCorrelation(dff0s,min_period,rollingMedianCorrectionNumber = 2000,drawCorrMatrixLabels = True):
    """
    Calculate correlation matrix between signals and display it as a heatmap.
    Parameters
    ----------
    dff0s : pandas.DataFrame
        DataFrame containing the signals to correlate.
    min_period : int
        Minimum number of valid observations required to calculate correlation.
    rollingMedianCorrectionNumber : int, optional
        Window size for rolling median correction. If None, no correction is applied.
        Default is 2000.
    drawCorrMatrixLabels : bool, optional
        Whether to display axis labels in the correlation matrix heatmap.
        Default is True.
    Returns
    -------
    pandas.DataFrame
        Correlation matrix of the input signals.
    Notes
    -----
    The function performs the following steps:
    1. Applies rolling median correction if specified
    2. Calculates correlation matrix
    3. Displays upper triangle of correlation matrix as a heatmap
    4. Uses green color palette for visualization
    The heatmap's color scale ranges from 0 (low correlation) to 1 (high correlation).
    """
    #min_period = 5*60*el['fps'].values[0]
    dff0s2 = dff0s.copy()
    if rollingMedianCorrectionNumber is not None:
        dff0s2.iloc[:,:] = rollingMedianCorrection(dff0s.values,rollingN=rollingMedianCorrectionNumber) # Subtract median to remove wobbling for corr calculation
    values = dff0s2.corr(min_periods=min_period)

    mask = np.triu(np.ones_like(values, dtype=bool))
    cmap =  sns.color_palette('Greens_r', as_cmap=True).copy()
    if drawCorrMatrixLabels:
        sns.heatmap(values,mask=mask,vmin=0,vmax=1,cmap=cmap)
    else:
        sns.heatmap(values,mask=mask,vmin=0,vmax=1,cmap=cmap,xticklabels=False,yticklabels=False,cbar_kws={'ticks':[]})

    return values