import numpy as np 
import scipy as sp 
import pandas as pd 
# from netCDF4 import Dataset 

from fair_scm import *
# from fair_inverse_revised import *

from pandas import DataFrame
from statsmodels.api import OLS
import statsmodels.tools.tools 

# from scipy.optimize import curve_fit



# from matplotlib import pyplot as plt

def calc_gwi(forc_in, obs, obs_years, reg_type='mon',base_low=1850.,base_high=1900, RWF_val=1.6/2.75, d_array=np.array([4.1,239.0]),r_vals=np.array([32.4,0.019,4.165])):

	#Express the observations relative to the base period 
    obs = obs - np.mean(obs[np.logical_and(obs_years>=base_low,obs_years<(base_high+1))])

    #Split the forcing up into years, total forcing and anthro forcing components
    years = forc_in[0,:]
    tot_forc = forc_in[1,:]
    ant_forc = forc_in[2,:]

    #Integrate anthropogenic and natural forcing with inputted FaIRv1.0 parameters
    C, t_nat = fair_scm(other_rf=tot_forc-ant_forc, tcrecs=np.array([1.6,1.6/RWF_val]), r0=r_vals[0], rC=r_vals[1], rT=r_vals[2], d=d_array)
    C, t_anthro = fair_scm(other_rf=ant_forc, tcrecs=np.array([1.6,1.6/RWF_val]), r0=r_vals[0], rC=r_vals[1], rT=r_vals[2], d=d_array)
    #Express relative to the centre of the base period
    t_nat = t_nat - np.mean(t_nat[np.logical_and(years>=base_low,years<base_high+1)])
    t_anthro = t_anthro - np.mean(t_anthro[np.logical_and(years>=base_low,years<base_high+1)])
    # -----------------------------------------------

    # Prepare the temperatures run through FaIR, so they lie on same year-grid as observations, so they can be compared
    # -----------------------------------------------
    #Interpolate the annual forced responses to the grid of the observed data
    if reg_type !='mon':
        t_nat = np.interp(obs_years+0.5, years+0.5, t_nat)
        t_anthro = np.interp(obs_years+0.5, years+0.5, t_anthro)
    else:
        t_nat = np.interp(obs_years, years+0.5, t_nat)
        t_anthro = np.interp(obs_years, years+0.5, t_anthro)

    #Linearly project the final half year
    t_anthro[obs_years>(years[-1]+0.5)] = 12*(t_anthro[obs_years<=(years[-1]+0.5)][-1] - t_anthro[obs_years<=(years[-1]+0.5)][-2]) * (obs_years[obs_years>(years[-1]+0.5)] - obs_years[obs_years<=(years[-1]+0.5)][-1]) \
    +t_anthro[obs_years<=(years[-1]+0.5)][-1]
    t_nat[obs_years>(years[-1]+0.5)] = 12*(t_nat[obs_years<=(years[-1]+0.5)][-1] - t_nat[obs_years<=(years[-1]+0.5)][-2]) * (obs_years[obs_years>(years[-1]+0.5)] - obs_years[obs_years<=(years[-1]+0.5)][-1]) \
    +t_nat[obs_years<=(years[-1]+0.5)][-1]
    # -----------------------------------------------

    #Use scipy defined OLS regression function to complete OLD regression of observations data on natural and anthropogenic warming with a constant
    y = np.copy(obs)
    x = DataFrame({'x1': (t_anthro), 'x2': (t_nat)})
    
    # add constant vector on to dataframe we will fit to temp observations
    x = statsmodels.tools.tools.add_constant(x)
    # complete OLS regression of anthropogenic and natural temperatures (found from FaIR integrated best estimate forcing) onto given observed temperature dataset.
    model = OLS(y, x)
    result = model.fit()
    # collect output scaling factors for anthro and natural temperature timeseries
    sf = result.params

    #Form scaled anthropgenic warming index
    awi = t_anthro * sf['x1']
    #Scaled natural warming index
    nwi = t_nat * sf['x2']
    #Scaled total externally forced warming index

    return awi, nwi, sf['x1'], sf['x2']

# fig,ax = plt.subplots(5,10,figsize=(50,20))

# fig1,ax1 = plt.subplots(figsize=(8,6))

def fit_awi_nwi_to_natural_variability(forc_in=False):

	awis_nat = np.zeros((50,2016))
	awi_sf_nat = np.zeros((50))
	nwis_nat = np.zeros((50,2016))
	nwi_sf_nat = np.zeros((50))

	if forc_in == False:
		forc_best = np.genfromtxt('./Annualforcings_Mar2014_GHGrevised.txt', skip_header=4)
		forc_in = np.array([forc_best[:,0], forc_best[:,13], forc_best[:,14]])

	for i in np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52]):
	    txt_f = './CMIP5_IV_'+str(i)+'.txt'
	    natural_variability_txt = np.genfromtxt(txt_f)
	    
	    tas = natural_variability_txt[1,:]
	    tas_years = natural_variability_txt[0,:]
	    # ax[(i-1)%5,int((i-1)/5)].plot(tas_years, tas)
	    
	    awi, nwi, sf_awi, sf_nwi = calc_gwi(forc_in, tas[:(2018-1850)*12], np.arange(1850,2018,1/12))
	    
	    if np.abs(awi[-1]) < 0.15:
		    # ax[(i-1)%5,int((i-1)/5)].plot(np.arange(0,int(awi.size/12),1/12), awi, color='orange')
		    # ax[(i-1)%5,int((i-1)/5)].plot(np.arange(0,int(nwi.size/12),1/12), nwi, color='darkblue')
		    
		    awis_nat[i-1,:] = awi
		    nwis_nat[i-1,:] = nwi
		    awi_sf_nat[i-1] = sf_awi
		    nwi_sf_nat[i-1] = sf_nwi
	    
	    # ax1.plot(np.arange(0,int(awi.size/12),1/12), awi, color='orange')
	    # ax1.plot(np.arange(0,int(nwi.size/12),1/12), nwi, color='darkblue')

	return  awis_nat, awi_sf_nat, nwis_nat, nwi_sf_nat













