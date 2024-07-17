# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:54:53 2022

@author: demooij
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib as mpl
import paths

mpl.use('macosx')
mpl.rcParams['figure.figsize'] = 12, 8
mpl.rc('image', interpolation='nearest', origin='lower', cmap = 'gray')

#from pylab import rcParams
from scipy import interpolate
from astropy import units as u
from astropy import constants as const
from astropy.stats import sigma_clip
from astropy.stats.sigma_clipping import sigma_clipped_stats
from scipy.interpolate import interp1d
from numpy.polynomial import Polynomial as P
from scipy import signal
from scipy.optimize import curve_fit
import scipy.stats as stats
import time
import gc

## this is to convolve an arbitrary spectrum in (wave, spec) with velocity broadening effects. 
## dv is in km/s (which here we're setting to c/R)
def wavegrid(lmin,lmax,R):
    '''lmin is the minimum wavelength, lmax is the maximum wavelength you want, R is the spectral resolution
    returns a wavelength vector that is constant in velocity space.
    then interpolate to this wavelength grid'''
    dx=np.log(1.+1./R) ## the log10 gives you constant velocity spacing
    x=np.arange(np.log(lmin),np.log(lmax),dx)
    return np.exp(x)


## Read in the model from a file and convert the crossesction to opacity for a given columndensity.
def get_mdl(fn='CN/CN_0300K.npy',N=1e15,FWHM_kernel=3e5/1.1e5):
    
    ## set sampling for convolution to 0.25 km/s
    res_interp=12e5
    
    c = np.load(paths.data/fn)
    w = np.load(paths.data/'CN/CN_wavelengths.npy') #np.arange(3850, 3885, 0.0001)
    tau = N * c
    prof = np.exp(-tau)
    
      
    ## Sample the model at a constant resolution for convolution
    w_mdl=wavegrid(np.min(w)-1., np.max(w)+1.,res_interp)
    prof_i=interp1d(w,prof,kind='linear',bounds_error=False,fill_value=1.)
    mdl=prof_i(w_mdl)

    ## generate a Gaussian kernel with the FWHM at the instrumental resolution & convolve model
    ## *** TBD: Allow other line broadening (e.g. based on Ca lines or even for different resolutions) ***
    print(FWHM_kernel)
    x_kernel=np.arange(-30.,30.001,(3e5/res_interp))
    kernel=np.exp( - ((x_kernel)**2)/2./(FWHM_kernel/2.35)**2)
    kernel=kernel/np.sum(kernel)
    mdl_conv=np.convolve(mdl,kernel,mode='same')
    idx=(w_mdl >= np.min(w))*(w_mdl <= np.max(w))

    ## set up interpolation object for the model to use in the CCF code
    imdl_conv=interp1d(w_mdl[idx],mdl_conv[idx],kind='linear',bounds_error=False,fill_value=1.)

    return w,prof,imdl_conv



##Read in data
def read_data(order=[0,1,2,3,4,5,6,7]):
    vsys=20.    #Systemic velocity of beta Pic
    nspec = fits.getdata(paths.data/'2d_nspec.fits')
    spec = fits.getdata(paths.data/'2d_spec.fits')
    wlen = fits.getdata(paths.data/'2d_wavelength.fits')/(1.+vsys/3e5)
    bary = fits.getdata(paths.data/'2d_baryvel.fits')
    obsdate = fits.getdata(paths.data/'2d_obsdate.fits')
    return nspec,spec[:,order,:],wlen[order,:],bary,obsdate


## Estimate residual blaze
def remove_resid_blaze(pixels,spec,star_spec,ord=3,poly_order=4):
    # order start and order end (the pixel point where there is flux in the order)
    # for the first six orders
    o_start=np.array([ 225,  15,  26,  26,  26,  26])
    o_end  =np.array([3670,4050,4060,4060,4060,2500])
    
    # take the ratio of the spectrum to the mean spectrum...
    spec_cor=spec[:,ord,:]/star_spec[np.newaxis,ord,:]

    # fit a polynomial to the this ration (should be flat if there was no small residuals)
    # and then divide it out to make this order relatively flat
    # TODO add sigma clipping for this
    for i in range(spec.shape[0]):
        x=pixels.copy()
        y=spec_cor[i,:].copy()
        idx=(y>0)*np.isfinite(y)*(x>o_start[ord])*(x<o_end[ord])
        l=np.polyfit(x[idx]-x[0],y[idx],poly_order)
        spec_cor[i,:]=spec_cor[i,:]/np.polyval(l,x-x[0])
    return spec_cor



####################################################################################
###### Only used for finding comets ################################################
####################################################################################



## Estimate blaze
def simple_blaze(wlen,master_blaze,nskip,nbins,binsize):
    ww0=wlen[nskip:nskip+nbins*binsize].reshape(nbins,binsize).copy()
    co=master_blaze[nskip:nskip+nbins*binsize].reshape(nbins,binsize).copy()

    mf=np.nanmean(co,axis=1)
    ww=np.nanmean(ww0,axis=1)

    for it in range(nbins):
        f_tmp=co[it,:]
        l=np.polyfit(ww0[it,:],co[it,:]-mf[it],1)
        res=co[it,:]-mf[it] - np.polyval(l,ww0[it,:])
        sig=np.median(np.abs(res-np.median(res)))

        idx=np.abs(res/sig)>4.
        if np.sum(idx)>0:
            f_tmp[idx]=np.nan
        co[it,:]=f_tmp.copy()

    mf=np.nanmean(co,axis=1)
    ww=np.nanmean(ww0,axis=1)

    flux_int=interpolate.interp1d(ww,mf,kind='cubic',fill_value='extrapolate',bounds_error=False)
    interp_flux = flux_int(wlen)

    return interp_flux



## Higher order blaze variation corrections
def blaze_var_cor(wlen, cor,nskip,nbins,binsize,n_frm,poly_ord):
    for frm in range(n_frm):
        ww0=wlen[nskip:nskip+nbins*binsize].reshape(nbins,binsize).copy()
        co=cor[frm,nskip:nskip+nbins*binsize].reshape(nbins,binsize).copy()
        mf=np.nanmean(co,axis=1)
        ww=np.nanmean(ww0,axis=1)

        for it in range(nbins):
            f_tmp=co[it,:]
            res=co[it,:]-mf[it]
            sig=np.median(np.abs(res-np.median(res)))
            idx=np.abs(res/sig)>4.
            if np.sum(idx)>0:
                f_tmp[idx]=np.nan
            co[it,:]=f_tmp.copy()

        mf=np.nanmean(co,axis=1)
        ww=np.nanmean(ww0,axis=1)

        l=np.polyfit(ww,mf,poly_ord)
        cor[frm,:]=cor[frm,:]/np.polyval(l,wlen)

    return cor



## Find the comets in the Ca II H line
def find_comets(wlen,spec_m,ord_FEB=7,use_all=True):
    #Select the order with Ca II H line
    wlen_FEB = wlen[ord_FEB]
    spec_FEB = np.transpose(spec_m[:,ord_FEB,:])
    
    #remove outliers to get a good sigma-clipped spectrum for stationary features
    filtered_data = sigma_clip(spec_FEB, sigma_lower=1.0, sigma_upper=5, maxiters=5, axis=1, copy=True, masked=True)
    spec_Ca_only = np.ma.mean(filtered_data, axis=1)
    spec_FEB_norm = spec_FEB/spec_Ca_only[:,np.newaxis]
    
    # look between 3965.74 and 3968.43 as well as 3968.58 and 3971.0, find the minima (=maximum absorption)
    # range chosen to avoid most of the core. 
    # use_all will look for blueshifted comets too
    if use_all:
        m_p = (wlen_FEB > 3968.58) * (wlen_FEB < 3971.)  + (wlen_FEB > 3965.68) * (wlen_FEB < 3968.43)
    else:
        m_p = (wlen_FEB > 3968.58) * (wlen_FEB < 3971.0)
    wlen_peak = wlen_FEB[m_p]
    spec_peak = 1. - (spec_FEB_norm[m_p])
    
    wheremax = np.argmax(spec_peak, axis=0)
    ampl_FEB_clip = np.max(spec_peak, axis=0)
    wlen_FEB_clip = wlen_peak[wheremax]

    # we now have the heights of the peaks in what_max, and the positions of the peaks in wlen_max
    # we make two separate sets to make: FEBs and no FEBs
    # 0.6 means 60% strength for pickup as an FEB
    # 0.15 means less than 15% strength for no FEB
    FEB_sel =   (ampl_FEB_clip > 0.6)
    star_sel =  (ampl_FEB_clip < 0.15)
    print("Number of strong comets: {0:}".format(np.sum(FEB_sel)))
    print("Number of weak comets: {0:}".format(np.sum(star_sel)))
    return FEB_sel,star_sel,wlen_FEB_clip

   

####################################################################################
####################################################################################
####################################################################################


##Set up a simple CCF code. N.B. This is the unweighted CCF
def c_correlate(l_data,data,imodel,dv=np.linspace(-50,50,101)):
    npix=dv.shape[0]
    ccf=np.zeros((data.shape[0],npix))
    ccf0=np.zeros((data.shape[0],npix))
    dat=data-np.mean(data,axis=1)[:,np.newaxis]

    
    z0=1.-dv/3e5
    for i,z in enumerate(z0):
        mdl=imodel(l_data*z)
        mdl=mdl-np.mean(mdl)
        ccf[:,i]=np.sum( data*mdl[np.newaxis,:],axis=1)
        ccf0[:,i]=ccf[:,i].copy()


    return ccf,ccf0




## Inject the model into the data 
def inject_mdl(l_data,data,imodel,dv=np.zeros(51)):
    sim_dat=data.copy()
    z0=1.-dv/3e5
    for i in range(data.shape[0]):
        mdl=imodel(l_data*z0[i])
        sim_dat[i,:]=data[i,:]*mdl
    return sim_dat



def generate_CCFs_orders(wlens,data_for_ccf,spec,pixels,broadening=5.,binsize=80,binsize2=400,nskip=20,T_gas=10,N=1e12,v_cutoff=50,v_comet=0.,idx_star=0):
    
    ## Set order start and end pixels to exclude bad edges (need to optimize!)
    o_start=np.array([ 225,  15,  26,  26,  26,  26])
    o_end  =np.array([3670,4050,4060,4060,4060,2500])

    ## define arrays to store ccfs 
    v_ccf=np.arange(-250.,250.,1.)
    m_ccf=np.zeros((data_for_ccf.shape[0],wlens.shape[0],v_ccf.shape[0]))
    m_sim_ccf=np.zeros((data_for_ccf.shape[0],wlens.shape[0],v_ccf.shape[0]))

    indexing=np.arange(wlens.shape[1])
    # sim_dat is an array to inject the model into
    sim_dat=np.zeros((data_for_ccf.shape))
    print(sim_dat.shape)
    # star_spec is the specra with NO comets used to divide out the stellar spectrum
    star_spec=np.mean(spec[idx_star,:,:],0)

    for ord in range(wlens.shape[0]): # loop over the orders
        
        ## get the opacities and convert to line-profile
        w,prof,imdl_conv=get_mdl('CN/CN_{0:04d}K.npy'.format(T_gas),N=N,FWHM_kernel=3e5/1.1e5)
        w_sim,prof_sim,imdl_conv_sim=get_mdl('CN/CN_{0:04d}K.npy'.format(T_gas),N=N,FWHM_kernel=broadening)

        ## Calculate CCF for data
        ## Filter out the strongest circumstellar Fe I lines, that could contaminate the signal.
        idx=(indexing >= o_start[ord])*(indexing<o_end[ord])*(np.abs(wlens[ord,:]-3860.18)>0.07)*(np.abs(wlens[ord,:]-3856.63)>0.07)
        
        #ccf,ccf0=c_correlate(wlens[ord,o_start[ord]:o_end[ord]],data_for_ccf[:,ord,o_start[ord]:o_end[ord]],imdl_conv,dv=v_ccf)
        ccf,ccf0=c_correlate(wlens[ord,idx],data_for_ccf[:,ord,idx],imdl_conv,dv=v_ccf)
        m_ccf[:,ord,:]=ccf.copy()

        
        ## Inject model into data 
        sim_dat[:,ord,:]=inject_mdl(wlens[ord,:],spec[:,ord,:],imdl_conv_sim,dv=v_comet)
        sim_dat[:,ord,:]=remove_resid_blaze(pixels,sim_dat,star_spec,ord,poly_order=4)

        # cross correlate simulated spectra
        sim_ccf,sim_ccf0=c_correlate(wlens[ord,idx],sim_dat[:,ord,idx],imdl_conv,dv=v_ccf)
        m_sim_ccf[:,ord,:]=sim_ccf.copy()

    return v_ccf,m_ccf,m_sim_ccf






def gauss(x,A,mu,sig,offset):
    return(offset+A*np.exp( - (x-mu)**2/2./sig**2))

def gen_gauss(v_c):
    def n_gauss(x,A,sig,offset):
        return(offset+A*np.exp( - (x-v_c)**2/2./sig**2))
    return n_gauss


def estimate_significance(v_ccf,ccf,v_c=0.,dv=5.,T_gas=10.,N=1e12,ord=3,sim=False):
    # 15 km/s from the line central position is the outside limit
    v_outer = 15

    idx_outside=(np.abs(v_ccf-v_c)>dv)*(np.abs(v_ccf-v_c)<v_outer)
    idx_both=(np.abs(v_ccf-v_c)<v_outer)
    idx_line=np.abs(v_ccf-v_c)<dv
    if len(ccf.shape)==1:
        # fit a gaussian to the central peak of the CCF
        # and take curvfit uncertainty as our estimate of the error on the detection
        l=np.polyfit(v_ccf[idx_outside],ccf[idx_outside],1)
        mdl=np.polyval(l,v_ccf)
        rms=np.std(ccf[idx_outside]/mdl[idx_outside])
        popt,pcov=curve_fit(gauss,v_ccf[idx_both],ccf[idx_both]/mdl[idx_both],p0=[0.,v_c,dv/2.35,0.],sigma=rms*np.ones(v_ccf[idx_both].shape))
        perr=np.sqrt(np.diag(pcov))
    else:
        
        fit=np.zeros((ccf.shape[0]+1,3))
        err=np.zeros((ccf.shape[0]+1,3))
        
        #f,ax=plt.subplots(1)
        ccf_cor=ccf.copy()

        for i in range(ccf.shape[0]): # fit linear function in background region
            l=np.polyfit(v_ccf[idx_outside],ccf[i,idx_outside],1)
            mdl=np.polyval(l,v_ccf)

            # use the rms of the resiudal to get an error estimate
            rms=np.std(ccf[i,idx_outside]/mdl[idx_outside])
            ccf_cor[i,:]=ccf[i,:]-mdl
            try:
                popt,pcov=curve_fit(gen_gauss(v_c),v_ccf[idx_both],ccf[i,idx_both]-mdl[idx_both],p0=[0.,dv/2.35,0.],sigma=rms*np.ones(v_ccf[idx_both].shape))
                perr=np.sqrt(np.diag(pcov))              
                fit[i,:]=popt
                err[i,:]=perr
            except:
                print("no convergence")    
            #ax.plot(v_ccf[idx_both],ccf[i,idx_both]-mdl[idx_both])
            
        # taking the weighted mean of all CCFs     
        rms_o=np.std(ccf_cor[:,idx_outside],axis=1)
        w=1./rms_o**2
        m_ccf=np.sum(w[:,np.newaxis]*ccf_cor,axis=0)/np.sum(w)
        sig=1./np.sqrt(np.sum(w))

        # refit again using sig as the error on the weighted mean
        try:
            popt,pcov=curve_fit(gen_gauss(v_c),v_ccf[idx_both],m_ccf[idx_both],p0=[0.,dv/2.35,0.],sigma=sig*np.ones(v_ccf[idx_both].shape))
            perr=np.sqrt(np.diag(pcov))              
            fit[-1,:]=popt
            err[-1,:]=perr
        except:
            print("comb, no convergence")    
        return fit,err



def fold_comet(v_ccf,ccf,sim_ccf,T_gas,N,binsize,idx_comets,v_comets):

    ccf_comets=ccf
    sim_ccf_comets=sim_ccf

    # velocity range over which we try to line up comets
    v_phasefold=np.arange(-90,90.1)
    
    # output arrays
    phasefold=np.zeros((ccf.shape[0],ccf.shape[1],v_phasefold.shape[0]))
    sim_phasefold=np.zeros((ccf.shape[0],ccf.shape[1],v_phasefold.shape[0]))

    for ord in range(ccf.shape[1]): # all orders...
        for i in range(phasefold.shape[0]): # for each night...
            ccf_i=interp1d(v_ccf,ccf[i,ord,:],kind='linear',bounds_error=False,fill_value=np.nan)
            sim_ccf_i=interp1d(v_ccf,sim_ccf[i,ord,:],kind='linear',bounds_error=False,fill_value=np.nan)
            phasefold[i,ord,:]=ccf_i(v_phasefold+v_comets[i])
            sim_phasefold[i,ord,:]=sim_ccf_i(v_phasefold+v_comets[i])

        if ord == 3: # the CN spectral order
            fit_sim,err_sim=estimate_significance(v_phasefold,sim_phasefold[:,ord,:],v_c=0.,dv=5.,T_gas=T_gas,N=N,sim=True)
            fit,err=estimate_significance(v_phasefold,phasefold[:,ord,:],v_c=0.,dv=5.,T_gas=T_gas,N=N)

    return v_phasefold,phasefold,sim_phasefold,fit,err,fit_sim,err_sim




## run ccf for a range of temperatures
def run_ccf_ord_multi_temp(wlens,data_for_ccf,spec,pixels,f,binsize=81,binsize2=400,nskip=25,v_cutoff=50,v_comet=0.,idx_comet=0,idx_star=0,use_stellar_frame=False):
    ## Define a range of column densities to use (mainly injection)
    N0=np.hstack([np.arange(1,10)*1e11,np.arange(1,10)*1e12,np.arange(1,10)*1e13,np.arange(1,10)*1e14,np.arange(1,10)*1e15])
 
    # short run version for quick examination
    #N0=np.hstack([np.arange(1,4)*1e13])
    #N0=np.asarray([1e11,5e11,1e12,5e12,1e13,5e13,1e14,5e14,1e15])
    if use_stellar_frame:
        fappend='_stellar'
    else:
        fappend=''
    T_gas0=np.asarray([10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000])
    # short run version for quick examination
    #T_gas0=np.asarray([30,300, 1000,2000])

    # tried different line broadenings for the CN model (km/s)
    broadening0=np.array([5.,10.,15.])

    # how many total iterations to print progress
    ntot = np.size(N0)*np.size(T_gas0)*np.size(broadening0)
    count = 1
    for k,broadening in enumerate(broadening0): # line broadening...
       for i,T_gas in enumerate(T_gas0): # CN gas temperature...
           for j,N in enumerate(N0): # column density...
               print('Running model {0} of {1} models for N0={2:.2e}, T_gas={3:4d}K and v_broad={4:.1f}km/s...'.format(count,ntot,N,T_gas,broadening))
               v_ccf,ccf,sim_ccf=generate_CCFs_orders(wlens,data_for_ccf,spec,pixels,broadening=broadening,binsize=binsize,binsize2=binsize2,nskip=nskip,T_gas=T_gas,N=N,v_cutoff=v_cutoff,v_comet=v_comet,idx_star=idx_star)
       
               plt.figure()
               plt.plot(v_ccf,ccf[-1,3,:])
               plt.plot(v_ccf,sim_ccf[-1,3,:])
               plt.savefig(path.figures/"orig_ccf_{0:04d}_{1:.2e}_{2:}.png".format(T_gas,N,fappend))
               plt.close('all')

            # align everything up in the comet rest frame
               v_phase,phase,sim_phase,fit,err,fit_sim,err_sim=fold_comet(v_ccf,ccf[:-1],sim_ccf[:-1],T_gas,N,binsize,idx_comet,v_comet)
       
               f.write("{0:.0f}  {1:.1e} {2:.1f}  {3:.2f}  {4:.2f}\n".format(T_gas, N, broadening, fit[-1,0]/err[-1,0],fit_sim[-1,0]/err_sim[-1,0]) )
       
               if count==1:
                   fit_cube=np.zeros((T_gas0.shape[0],N0.shape[0],broadening0.shape[0],fit.shape[0],fit.shape[1]))
                   fit_err_cube=np.zeros((T_gas0.shape[0],N0.shape[0],broadening0.shape[0],fit.shape[0],fit.shape[1]))
                   sim_fit_cube=np.zeros((T_gas0.shape[0],N0.shape[0],broadening0.shape[0],fit.shape[0],fit.shape[1]))
                   sim_fit_err_cube=np.zeros((T_gas0.shape[0],N0.shape[0],broadening0.shape[0],fit.shape[0],fit.shape[1]))
                   ccf_cube=np.zeros((T_gas0.shape[0],N0.shape[0],broadening0.shape[0],ccf.shape[0],ccf.shape[2]))
                   sim_ccf_cube=np.zeros((T_gas0.shape[0],N0.shape[0],broadening0.shape[0],ccf.shape[0],ccf.shape[2]))
                   phase_cube=np.zeros((T_gas0.shape[0],N0.shape[0],broadening0.shape[0],phase.shape[0],phase.shape[2]))
                   sim_phase_cube=np.zeros((T_gas0.shape[0],N0.shape[0],broadening0.shape[0],phase.shape[0],phase.shape[2]))
       
               plt.figure()
               plt.plot(v_phase,phase[-1,3,:])
               plt.plot(v_phase,sim_phase[-1,3,:])
               plt.savefig(path.figures/"phase_{0:04d}_{1:.2e}_{2:.1f}_{3:}.png".format(T_gas,N,broadening,fappend))
               plt.close('all')
               plt.figure()
               plt.plot(v_ccf,ccf[-1,3,:])
               plt.plot(v_ccf,sim_ccf[-1,3,:])
               plt.savefig(path.figures/"ccf_{0:04d}_{1:.2e}_{2:.1f}_{3:}.png".format(T_gas,N,broadening,fappend))
               plt.close('all')
               
               fit_cube[i,j,k,:,:]=fit.copy()
               fit_err_cube[i,j,k,:,:]=err.copy()
               sim_fit_cube[i,j,k,:,:]=fit_sim.copy()
               sim_fit_err_cube[i,j,k,:,:]=err_sim.copy()
               ccf_cube[i,j,k,:,:]=ccf[:,3,:].copy()
               sim_ccf_cube[i,j,k,:,:]=sim_ccf[:,3,:].copy()
               phase_cube[i,j,k,:,:]=phase[:,3,:].copy()
               sim_phase_cube[i,j,k,:,:]=sim_phase[:,3,:].copy()
               count = count + 1

    hdu=fits.PrimaryHDU(fit_cube)
    hdu.writeto(paths.data/"fit_cube{0:}.fits".format(fappend),overwrite=True)
    hdu=fits.PrimaryHDU(fit_err_cube)
    hdu.writeto(paths.data/"fit_err_cube{0:}.fits".format(fappend),overwrite=True)
    hdu=fits.PrimaryHDU(sim_fit_cube)
    hdu.writeto(paths.data/"sim_fit_cube{0:}.fits".format(fappend),overwrite=True)
    hdu=fits.PrimaryHDU(sim_fit_err_cube)
    hdu.writeto(paths.data/"sim_fit_err_cube{0:}.fits".format(fappend),overwrite=True)
    hdu=fits.PrimaryHDU(ccf_cube)
    hdu.writeto(paths.data/"ccf_cube{0:}.fits".format(fappend),overwrite=True)
    hdu=fits.PrimaryHDU(sim_ccf_cube)
    hdu.writeto(paths.data/"sim_ccf_cube{0:}.fits".format(fappend),overwrite=True)
    hdu=fits.PrimaryHDU(v_ccf)
    hdu.writeto(paths.data/"v_ccf.fits".format(fappend),overwrite=True)
    hdu=fits.PrimaryHDU(phase_cube)
    hdu.writeto(paths.data/"phase_cube{0:}.fits".format(fappend),overwrite=True)
    hdu=fits.PrimaryHDU(sim_phase_cube)
    hdu.writeto(paths.data/"sim_phase_cube{0:}.fits".format(fappend),overwrite=True)

'''
MAIN CODE FOR RUNNING FITS
MAIN CODE FOR RUNNING FITS
MAIN CODE FOR RUNNING FITS
MAIN CODE FOR RUNNING FITS
MAIN CODE FOR RUNNING FITS
MAIN CODE FOR RUNNING FITS
'''

CN_ord=3

# two binsizes - one for small scale and one for larger scales... determined empirically
binsize=81 # in pixels

nskip=25 # how many pixels to skip at the start of an echelle order to avoid edge effects

binsize2=400 # in pixels...

# read it all in
nspec,spec,wlen,bary,obsdate=read_data()

## Find the strongest comet in the Ca II H line for each of the spectra, before doing so, remove the overall line-shape by taking the envelope for each point
ord_FEB=7
cor=spec.copy()
for i in range(spec.shape[2]):
    for ord in range(8):
       flux=spec[:,ord,i].copy()
       flux.sort()
       cor[:,ord,i]=spec[:,ord,i]/np.mean(flux[-10:])
    
# we split the spectra into 'comets' and 'no comets'
FEB_idx,star_idx,FEB_wave=find_comets(wlen,cor,ord_FEB=ord_FEB,use_all=True)

# rest wavelength for CaII H line in the stellar rest frame
lam0=3968.469

# converting the FEB features into a relative velocity for the comet
v_comet=3e5*(FEB_wave-lam0)/lam0

# stack both the mean spectrum and individual spectra for orders 0 to 6 
# (cut down on time and memory)
data_for_ccf=np.vstack((cor[:,0:6,:],np.mean(cor[:,0:6,:],axis=0)[np.newaxis,:,:]))
wlens=wlen[0:6,:]


##HARPS Fiber change in June 2015 (JD~2457180) Current script avoids the affected region,
# but might be possible to improve this by selecting low activity pre- and post-fiberchange. 
idx_pre=obsdate<57200.
idx_post=obsdate>=57200.

# residual blaze corrections
pixels=np.arange(wlens.shape[1])
star_spec_pre=np.median(spec[star_idx*idx_pre,:,:],axis=0)
star_spec_post=np.median(spec[star_idx*idx_post,:,:],axis=0)
spec_cor=spec.copy()
# pre and post fiber change, because the blaze function shifted a bit(?) implied due to a seen RBV jump/line shape profile
for ord in range(6):
    spec_cor[idx_pre,ord,:]=remove_resid_blaze(pixels,spec[idx_pre,:,:],star_spec_pre,ord,poly_order=4)
    spec_cor[idx_post,ord,:]=remove_resid_blaze(pixels,spec[idx_post,:,:],star_spec_post,ord,poly_order=4)

# CHECK POINT to see that this looks normal. REMOVE in final version    
data_for_ccf=spec_cor[:,0:6,:]
hdu=fits.PrimaryHDU(data_for_ccf)
hdu.writeto(paths.data/'tmp.fits',overwrite=True)

# Do the analysis for the strongest comets selected using find_comets()
f = open(paths.data/"results_HARPS.txt", "w")
run_ccf_ord_multi_temp(wlens,data_for_ccf,spec,pixels,f,binsize=binsize,binsize2=binsize2,nskip=nskip,v_cutoff=50,v_comet=v_comet,idx_comet=FEB_idx,idx_star=star_idx)
f.close()








# DIAGNOSTIC PLOT    
###Perform autocorrelation of the model to show the strength and pattern in the different orders
order_start=np.array([ 225,  15,  26,  26,  26,  26])
order_end  =np.array([3670,4050,4060,4060,4060,2500])

dv=np.linspace(-300,300,601)
col=['r','g','b','c','y','m','k','orange']
T_gas0=np.asarray([10, 20, 50, 100, 200, 300, 1000, 2000])
f,ax=plt.subplots(4,3,figsize=(32,20))
for i,T in enumerate(T_gas0):
    w,prof,imdl=get_mdl('CN/CN_{0:04d}K.npy'.format(T),N=1e13)
    for j in range(1,5):
        p2=imdl(wlen[j,order_start[j]:order_end[j]])
        ccf,ccf0=c_correlate(wlen[j,order_start[j]:order_end[j]],np.vstack((p2,p2)),imdl,dv=dv)
        ax[j-1,0].plot(dv,ccf[0,:],col[i],label='T={0:}K,O={1:}'.format(T,j))
        ax[j-1,1].plot(dv,ccf[0,:]/np.max(ccf[0,:]),col[i],label='T={0:}K,O={1:}'.format(T,j))
        ax[j-1,2].plot(wlen[j,order_start[j]:order_end[j]],p2/np.max(p2),col[i])
ax[0,0].legend()
ax[1,0].legend()
ax[2,0].legend()
ax[3,0].legend()
f.tight_layout()
f.savefig(paths.figures/'ccf_self.png')

