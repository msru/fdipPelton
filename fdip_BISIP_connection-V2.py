#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:45:33 2022

@author: sadegh
Synthetic model and forward modeling for checking BISIP inversion code
"""
#%%   FORWARD 
###############################################################################

import numpy as np
import pygimli as pg
import pygimli.meshtools as mt
import pybert as pb
from pygimli.physics import ert

# %% load data(sheme) container
data = ert.createData(elecs=np.arange(5.1), schemeName='dd')  # will change it to 50 later
# A B N M
# data["a"], data["b"] = data["b"], data["a"]
data["k"] = ert.geometricFactors(data)
print(data)
# print(data["k"][:5])
# pg.x(data)    # number of electrods and their positions
# %% create geometry (poly object)
geo = mt.createWorld(start=[-70, 0], end=[110, -50], worldMarker=True, marker=0)
# Create a heterogeneous block
blockA = mt.createRectangle(start=[-35, -3], end=[-5, -10.0],
                           marker=1,  boundaryMarker=2, area=0.1)
blockB = mt.createRectangle(start=[40, -3], end=[70, -10.0],
                           marker=2,  boundaryMarker=3, area=0.1)
# Merge geometrical entities
plc = geo + blockA + blockB
for sen in data.sensors():   # to refine the model at the surface
    plc.createNode(sen)      # we create a node at any sensor position
     
pg.show(plc, boundaryMarker=True);
# raise SystemExit()
# mesh.cellMarkers()   # number of cells in the mesh
# res[mesh.cellMarkers()]  # resistiviry of each cell
# %% save the anomalies 
# We save the anomalies for later plotting lines on results
ano = plc
ano.exportPLC("geo.poly")
# %% mesh the geometry
mesh = mt.createMesh(plc)
pg.show(mesh, markers=True, showMesh=True);
# %% Cole Cole Parameters for each layer
# the synthetic model
frvec = [0.1, 1., 10., 100., 1000.]
# frvec = [0.156, 0.312, 0.625, 1.25, 2.5, 5, 10, 20, 40, 80, 125,
#          250, 500, 1000]  # SIP256C frequencies #frequency vector
rhovec = np.array([100, 200, 400]) # resistivity in Ohmm
mvec = np.array([0.8, 0.8, 0.7])    # chargeability in V/V # chargables are non-zero
tauvec = np.array([1.0, 1.0, 0.6])  # time constants in s
cvec = np.array([0.25, 0.4, 0.5])   # relaxation exponent
fdip = pb.FDIP(f=frvec, data=data) 
# %% Plot True Cole-Cole values
if 0:
    fig, ax = pg.plt.subplots(nrows=4, figsize=(8, 12), sharex=True, sharey=True)
    cm = mesh.cellMarkers()
    pg.show(mesh, rhovec[cm], ax=ax[0], cMin=0, cMax=500, logScale=True,
        cMap="Spectral_r")
    pg.show(mesh, mvec[cm], ax=ax[1], cMin=0, cMax=0.8, logScale=0, cMap="plasma")
    pg.show(mesh, tauvec[cm], ax=ax[2], cMin=0.01, cMax=1, logScale=1, cMap="magma")
    pg.show(mesh, cvec[cm], ax=ax[3], cMin=0, cMax=0.5, logScale=0, cMap="viridis")
    ax[0].set_xlim(-70, 105)
    ax[0].set_ylim(-50, 0);

# %% check complex resistivity
# Frequency-domain Cole-Cole impedance for each anomaly:
# Z = (1. - m * (1. - relaxationTerm(f, tau, c, a))) * rho
from pygimli.physics.SIP.models import modelColeColeRho
res = modelColeColeRho(frvec[0], rhovec, mvec, tauvec, cvec)
print('Z=',res)    

# %% FDIP simulation
# pybert.FDIP.simulate can do the forward modelling for many frequencies taking a Cole-Cole model as input
# now the actual simulation
fdip.simulate(mesh, rhovec, mvec, tauvec, cvec, noiseAbs=0, noiseLevel=0.0, verbose=True); # noiseLevel=error percentage of the model

nDatapoint = 2
spec=fdip.getDataSpectrum(nDatapoint) # Return SIP spectrum class for single data number
spec.showData()
# %% Show pseudosections of a single frequency data
fdip.showSingleFrequencyData(1);
# fdip.showSingleFrequencyData(13);
# %%  Show decay curves
fdip.showDataSpectra(data, ab=[1, 2]); # resistivity phase is normaly negative / conductivity phase is normaly pasitive
# %% save rhoa, phia and array data
fdip.basename = "synthSlag"
fdip.saveData()

# %% save pdf files
# fdip.generateDataPDF()
# fdip.generateSpectraPDF()
# %% Forward Modelling Results 
rhoa = np.copy(fdip.RHOA)  # make a copy for not to be noisified every time we call it
# rhoa.shape
phia = np.copy(fdip.PHIA)
# phia.shape
ampError = 0.05                           # relative error level (in percent)
ampNoise = ampError * pg.randn(rhoa.shape)  # ampNoise: add Gaussian noise to error
rhoa *= ampNoise + 1.0   # noisified data   # why +1 ?
phiError = 0.005         # 5 miliradians
phiNoise = phiError * pg.randn(phia.shape)  # phiNoise: add Gaussian noise to error
phia += phiNoise         # noisified data

#%% extracting the required matrix for BISIP   - ONE datapoint
# The order of the columns: Frequency, Amplitude, Phase_shift, Amplitude_error, Phase_error
# nDatapoint =  int(input('Please enter the datapiont´s number between 0 and 1175: \n'))
headers = np.array([('freq', 'amp', 'phia', 'amp_err', 'phia_err')])
frarray = np.array(frvec)   # frequency array

rhoaDatapoint = rhoa[(nDatapoint),]   # Rhoa for one datapoint
# rhoaDatapoint /= max(rhoaDatapoint)
amp_errDatapoint = rhoaDatapoint * ampError
phiaDatapoint = -phia[nDatapoint,] * 1000 # Phia for one datapoint # needs to be neative for BISIP
phia_errDatapoint = np.ones_like(phiaDatapoint) * phiError * 1000 # *1000 since unit is mrad

SIPDatapoint = np.flipud(np.array(list(zip(*[frarray, rhoaDatapoint, phiaDatapoint,
                                       amp_errDatapoint, phia_errDatapoint ])))) # SIP data without header/ Frequency descending
# SIPDataAllWithoutf = np.transpose((np.array(list(zip(*[rhoaAllDatapoint, phiaAllDatapoint,
#                                         amp_errAllDatapoint, phia_errAllDatapoint ]))))) # SIP data without header and frvec/ Frequency descending

SIPDatapointHeder = (np.vstack([headers, SIPDatapoint])) #  SIP data with header

# saving SIPDAta as a csv file
np.savetxt('SIPData.csv', SIPDatapointHeder,fmt='%s', delimiter=',')
# check00000
# %%   extracting the required matrix for BISIP   - All datapoints
frvecAll = np.tile(np.flipud(frarray),len(rhoa))
rhoaAllDatapoint = (np.fliplr(rhoa)).ravel() #using all of the apparent resistivity values
phiaAllDatapoint = (np.fliplr(phia)).ravel()
amp_errAllDatapoint = rhoaAllDatapoint * ampError
phia_errAllDatapoint = np.ones_like(phiaAllDatapoint) * phiError * 1000 # *1000 since unit is mrad
SIPDataAll = (np.array(list(zip(*[frvecAll, rhoaAllDatapoint, phiaAllDatapoint,
                                        amp_errAllDatapoint, phia_errAllDatapoint ])))) # SIP data without header/ Frequency descending
SIPDataAllHeder = np.vstack([headers, SIPDataAll]) #  SIP data with header
# saving SIPDAta as a csv file
np.savetxt('SIPDataAll.csv', SIPDataAllHeder,fmt='%s', delimiter=',')


#%% save the 3D array
# function for spliting each datapoint
def arraySplit(array):
    return np.array(np.vsplit(array, len(rhoa)))
SIPDataForBISIP = arraySplit(SIPDataAll)

# saving a 3D table
with open('SIPDataForBISIP3D.csv', 'w') as outfile:
    # Write a header for the text to be more readabe
    # outfile.write('# Array shape: {0}n'.format(SIPDataForBISIP.shape))
    
    # Iterating through a Ndimensional array produces slices along
    # the last axis. This is equivalent to array[i,:,:] in this case
    for array_slice in SIPDataForBISIP:

        # The formatting string indicates that I'm writing out
        # the values in left-justified columns 7 characters in width
        # with 2 decimal places.  
        np.savetxt(outfile, array_slice,  fmt='%s')

        # Writing out a break to indicate different slices...
        #outfile.write('# New slice')
# Read the array 
ReloadArray = np.loadtxt('SIPDataForBISIP3D.csv')

# Go back to 3D knowing the original shape of the array
new_SIPDataForBISIP = ReloadArray.reshape(np.shape(SIPDataForBISIP))

#%%  INVERSION 
# BISIP 


#%%
# This will get one of the example data files in the BISIP package
filepath = 'SIPDataAll.csv'
from bisip import fdipPeltonNew
# model=np.delete(np.array([rho, m, tau, c]), 0 , axis=1)
# model = [[rhovec[ii], mvec[ii], tauvec[ii], cvec[ii]] for ii in range(len(rhovec))]
model = np.concatenate([rhovec, mvec, tauvec,cvec])
BISIPmodel = fdipPeltonNew(filepath=filepath, mesh=mesh, frvec=frvec, data=data,
                        n_cells=3, nsteps=1000, nwalkers=32) #initializing the class
output = BISIPmodel.forward(model) #calling the function
sdfsfsdf

print(output.shape)
#%%
# sgfsdfsdfsf
# Fit the model to this data file
BISIPmodel.fit()
# model._data     # load data

#%%

# import os
# import numpy as np
# from bisip import PeltonColeCole
# from bisip import DataFiles
# # #%%
# # This will get one of the example data files in the BISIP package
# #data_files = DataFiles()
# filepath = 'SIPDataAll.csv'

# # Define MCMC parameters and ColeCole model
# model = PeltonColeCole(filepath=filepath,
#                        nwalkers=64,  # number of MCMC walkers
#                        nsteps=2000,  # number of MCMC steps
#                        headers=1,  # number of lines to skip the header line and the 8 highest measurement frequencies
#                        ph_units='mrad', # The units of the phase shift measurements
#                        n_modes=1   # The number of ColeCole modes to use for the inversion
#                        )

#%% Visualizing the parameter traces
# Plot the parameter traces
fig = BISIPmodel.plot_traces(discard=100)
fig = BISIPmodel.plot_traces(chain=None, discard=100, thin=2, flat=True)
#%%
# Get chains of all walkers, discarding first 500 steps
chain = BISIPmodel.get_chain(discard=100)
print(chain.shape)  # (nsteps, nwalkers, ndim)
fig = BISIPmodel.plot_traces(discard=500)

#%% 
# Get chains of all walkers, discarding first 500 steps,
# thinning by a factor of 2 and flattening the walkers
chain = BISIPmodel.get_chain(discard=100, thin=2, flat=True)
print(chain.shape)  # (nsteps*nwalkers/thin, ndim)
# ndim refers to the number of dimensions of the model (the number of parameters)
#%% Plotting models over data
# Use the chain argument to plot the model
# for a specific flattened chain
fig = BISIPmodel.plot_fit(chain)
# The plot will also display the median model as a red line and the 95% highest
# probability density interval as dotted lines.
# You can then use the `fig` matplotlib object to save the figure or make
# adjustments according to your personal preference. For example:
# fig.savefig('fit_figure_sample_K389172.png', dpi=144, bbox_inches='tight')
#%%
# Use the discard argument to discard samples
# note that we do not need to pass the chain argument
fig = model.plot_fit_pa(discard=500, p=[14, 50, 86])  # 14=lower, 50=median, 86=higher
#%% Printing best parameters, Print the mean and std of the parameters values
values = model.get_param_mean(chain=chain)
uncertainties = model.get_param_std(chain=chain)

for n, v, u in zip(model.param_names, values, uncertainties):
  print(f'{n}: {v:.5f} +/- {u:.5f}')
#real-r0-value = r0*model.data['norm_factor']
#%% Inspecting the posterior
# visualize the posterior distribution of all parameters using a corner plot 
fig = model.plot_corner(chain=chain)

#%%
# Get the lower, median and higher percentiles
results = model.get_param_percentile(chain=chain, p=[2.5, 50, 97.5])
# Join the list of parameter names into a comma separated string
headers = ','.join(model.param_names)
# Save to csv with numpy
# The first row is the 2.5th percentile, 2nd the 50th (median), 3rd the 97.5th.
# Parameter names will be listed in the csv file header.
print(headers)

print(results)

np.savetxt('quickstart_results.csv', results, header=headers,
           delimiter=',', comments='')
#%% 
#Plots the input data.
fig = model.plot_data()
#%%
# Plot histograms of the MCMC simulation chains.
fig = model.plot_histograms()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 16:21:43 2022

@author: sadegh
"""

