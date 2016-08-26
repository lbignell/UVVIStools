# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 12:00:18 2016

@author: lbignell
"""

import numpy as np
#import os
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['axes.labelsize'] = 28
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['axes.titlesize'] = 28
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['figure.subplot.bottom'] = 0.15
mpl.rcParams['figure.subplot.left'] = 0.15
#Path is file path, celllength is the thickness of the UV-VIS cell.
#base units are cm
def ReadUVVIS(path, rindex_WL, rindex_material, rindex_glass, celllength=10):
    data = np.genfromtxt(path, delimiter=" ", skip_header=2)
    if len(np.shape(data)) != 2:
        data = np.genfromtxt(path, delimiter="\t", skip_header=2)
    if len(np.shape(data)) != 2:
        print("Couldn't read data file!")
        return []
    data = np.transpose(data)
    rindex_glass_interp = np.interp(data[0], rindex_WL, rindex_glass)
    rindex_material_interp = np.interp(data[0], rindex_WL, rindex_material)
    Tempty, Tfull = TransmissionFactor(data[0], rindex_material_interp, 
                                       rindex_glass_interp)
    data[1] = -np.log((Tempty/Tfull)*10**(-data[1]))/celllength
    #data[1] = np.multiply(data[1], 1/celllength) #normalise to 1/cm units.
    AbsAt600 = 0.002018 #cm, from Segelstien.#0.00231 #cm, from Daya Bay
    normidx = np.searchsorted(data[0], 600)
    #minval = min(data[1])
    data[1] = np.add(data[1], AbsAt600 - data[1][normidx])
    return data

def ReadUVVIS_nocorr(path, celllength=10):
    data = np.genfromtxt(path, delimiter=" ", skip_header=2)
    if len(np.shape(data)) != 2:
        data = np.genfromtxt(path, delimiter="\t", skip_header=2)
    if len(np.shape(data)) != 2:
        print("Couldn't read data file!")
        return []
    data = np.transpose(data)
    data[1] = -np.log(10**(-data[1]))/celllength
    #data[1] = np.multiply(data[1], 1/celllength) #normalise to 1/cm units.
    AbsAt600 = 0.002018 #cm, from Segelstien.#0.00231 #cm, from Daya Bay
    normidx = np.searchsorted(data[0], 600)
    #minval = min(data[1])
    data[1] = np.add(data[1], AbsAt600 - data[1][normidx])
    return data

def PltRelWater(datax,datay,waterx,watery,normWL,aqfrac,
                celllength=1,newfig=True,**kwargs):
    '''
    Plot WbLS with aqueous fraction aqfrac, normalizing using water data.
    The input data are NOT corrected for the UV-VIS's base-10 wierdness.
    '''
    datay_fixed = -np.log(10**-datay)/celllength
    normidx_data = np.searchsorted(datax,normWL)
    WaterAbs = watery[np.searchsorted(waterx, normWL)]
    datay_corr = np.add(datay_fixed, (WaterAbs*aqfrac - 
                                        datay_fixed[normidx_data]))

    if newfig:
        fig = plt.figure()
    else:
        fig = plt.gcf()

    ax = plt.plot(datax, datay_corr, **kwargs)
    return fig, ax

#pathlength units are cm.
def CalculateSensitivity(WL, data, ref, CkovSpec, QE, pathlength=100):
    '''
    Returns the detection probability for the 'data' water relative to some reference.
    
    The arguments must have the same wavelength span and binning as CkovSpec.
    Arguments:

    - Wavelength in nm    
    
    - data, ref the attenuation coeff in (1/cm).

    - CkovSpec is the un-normalised Cherenkov Spectrum.

    - QE is the un-normalised PMT QE.

    - pathlength is the optical pathlength.
    '''
    IonI0 = np.exp(np.multiply(-1*pathlength,
                               np.subtract(data,ref)))
    #print(IonI0[0:10])
    QE_interp = np.interp(WL, QE[0], QE[1])
    #print((WL[100:110],QE_interp[100:110]))
    Others = np.multiply(CkovSpec, QE_interp)
    #print(sum(Others))
    NormOthers = np.divide(Others, sum(Others))    
    return np.multiply(IonI0,NormOthers)

def TransmissionFactor(WL, rindex_material, rindex_glass):
    '''
    Calculate the transmission through the cuvettes, including the wavelength
    dependence, assuming an empty reference cuvette, using fresnel eqn.
    Note:
    rindex_material and rindex_glass must be the same length and refer
    to the same wavelength range as each other and the data that will be
    corrected.
    '''
    rindex_air = np.zeros(len(WL)) + 1

    Tempty = np.zeros(len(WL)) + 1
    Tfull = np.zeros(len(WL)) + 1    
    
    #Calculate the reflectance factors; note the recursion.
    #The beam hits the outer front wall of the cuvette.
    Tempty = Tempty - fresnelreflect_norminc(rindex_air, rindex_glass)
    Tfull = Tfull - fresnelreflect_norminc(rindex_air, rindex_glass)
    
    #the beam hits the inner front wall of the cuvette
    Tempty = Tempty - fresnelreflect_norminc(rindex_glass, rindex_air)
    Tfull = Tfull - fresnelreflect_norminc(rindex_glass, rindex_material)
    
    #the beam hits the inner back wall of the cuvette
    Tempty = Tempty - fresnelreflect_norminc(rindex_air, rindex_glass)
    Tfull = Tfull - fresnelreflect_norminc(rindex_material, rindex_glass)
    
    #the beam hits the outer back wall of the cuvette
    Tempty = Tempty - fresnelreflect_norminc(rindex_glass, rindex_air)
    Tfull = Tfull - fresnelreflect_norminc(rindex_glass, rindex_air)
    
    return Tempty, Tfull    
    
def fresnelreflect_norminc(n1,n2):
    return ((n1-n2)/(n1+n2))**2