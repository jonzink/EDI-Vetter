#!/usr/bin/python

### EDI-Vetter -  Created by Jon Zink ###
### Devolped on python 3.7.1 ###

### If you make use of this code, please cite: ###
### J. K. Zink et al. 2020



import copy
import numpy as np
import pandas as pd
from numpy import ma
import terra.tfind as tfind
import terra.tval as tval
import terra.prepro as prepro
import emcee
from scipy.stats import norm, uniform
from scipy import special
from lmfit import minimize, fit_report
import batman
from astropy.stats import mad_std

import warnings
warnings.filterwarnings("ignore")

gravConstant=6.674e-11*1.98855e30*(86400)**2/(6.957e8)**3
snrThreshold=8.68

class parameters:
    """Initialize Vetting.

    The parameters object itself is just a container object. Different
    codes can perform module operations on the parmameters object.
    
    Args:
        lc (Required[pandas.DataFrame]): Light curve. Must have the 
            following columns: t, f, ferr, fmask. Setting equal to None
            is done for reading from disk
        
        per (Required[float]): best estimate of transit period in units
            of days.
        
        t0 (Required[float]): best estimate of transit mid-point in units
             of days.
        
        tdur (Required[float]): best estimate of transit duration in units
             of days.
        
        radRatio (Optional[float]): best estimate of the planet to star 
            radius ratio.
        
        radStar (Optional[float]): radius of the stellar host in solar units.
    
        uradStar (Optional[float]): uncertainty of stellar host radius in solar
            units.
        
        massStar (Optional[float]): mass of stellar host in solar units.
        
        umassStar (Optional[float]): uncertainty of stellar host mass in solar
            units.
    
        limbDark (Optional[array]): array of the two quadratic limb darkening 
            parameters for the stellar host
    

    Example:
    
        # Working with the parameters
    
        >>> params=EDI_Vetter.paramaters(per=8.261, t0=2907.645, tdur=.128, lc=lc)
        >>> params=EDI_Vetter.MCfit(params)
        >>> params=EDI_Vetter.Go(params,delta_mag=2.7, delta_dist=1000, photoAp=25)

    """
    
        
    lc_required_columns = ['t','f','ferr','fmask']
    
    def _get_fm(self):
        """Convenience function to return masked flux array"""
        fm = ma.masked_array(
            self.lc.f.copy(), self.lc.fmask.copy(), fill_value=0 )
        fm -= ma.median(fm)
        return fm
    
    def __init__(self,per=None,t0=None,tdur=None,radRatio=None, radStar=None, uradStar=.1, massStar=None, umassStar=.1, limbDark=None, lc=None):
        super(parameters,self).__init__()
        # if type(lc)==type(None):
        #     return
        self.lc=lc
            
        for col in self.lc_required_columns:
            assert list(lc.columns).index(col) >= 0, \
                "light curve lc must contain {}".format(col)
        
        #Transit Parameters in units of days (Required)
        self.per=per
        self.t0=t0
        self.tdur=tdur


        
        if (per is None) | (t0 is None) | (tdur is None):
            print("ERROR: You must specify the MES (SNR), Period, T0, and Transit Duration. While they will be fit later, good starting points are needed.")
            return 
        
        #Transit Parameters(Optional)
        #This parameter will be fit later, but it helps if you have good starting guess.
        self.radRatio=radRatio
        
        #Stellar Parameters in solar units (Optional)
        self.radStar=radStar
        self.uradStar=uradStar
        self.massStar=massStar
        self.umassStar=umassStar
        self.limbDark=limbDark

        if (radStar is None) | (massStar is None) | (limbDark is None):
            print("WARNING: One or more of the stellar parameters are missing, assuming solar values")
            
            if (radStar is None):
                self.radStar=1
                self.uradStar=1
            if (massStar is None):
                self.massStar=1
                self.umassStar=1
            if (limbDark is None):
                self.limbDark=[.49,.16]
        
        self.Mes=grid_search_NewMES(self)
        
        if self.Mes==0:
            print("ERROR: No signal was detected at the input Period, T0, and Transit duration location")
            return 
            
                                   

def MCfit(params, removeOutliers=True):
    
    if removeOutliers:
    #######Remove Outliers
        fm = params._get_fm()
        isOutlier = prepro.isOutlier(fm, [-1e3,10], interp='constant')
        params.lc['isOutlier'] = isOutlier
        params.lc['fmask'] = fm.mask | isOutlier | np.isnan(fm.data)
    
    
   # Compute initial parameters. Fits are more robust if we star with
   # transits that are too wide as opposed to to narrow

    per = params.per
    t0 = params.t0
    tdur = params.tdur
    Mes= params.Mes
    dt=np.nanmean(params.lc.t[1:]-params.lc.t[:-1])
    

    apl=(per**2*gravConstant*params.massStar/(4*np.pi**2))**(1/3)/params.radStar
    ap_err=(((per**2*gravConstant/(4*np.pi**2))**(1/3)*1/params.radStar*1/3*params.massStar**(-2/3)*params.umassStar)**2
        +((per**2*params.massStar*gravConstant/(4*np.pi**2))**(1/3)*-1*params.radStar**(-2)*params.uradStar)**2)**.5

    if (tdur/per*apl*3.14)>1:
        b=.1
    else:
        b = (1-(tdur/per*apl*3.14)**2)**.5
           
    if np.isnan(apl):
        apl=(per**2*gravConstant*1/(4*np.pi**2))**(1/3)/1
        ap_err=(((per**2*gravConstant/(4*np.pi**2))**(1/3)*1/1*1/3*1**(-2/3)*1)**2+((per**2*1*gravConstant/(4*np.pi**2))**(1/3)*-1*1**(-2)*1)**2)**.5
        b=0.1
    else:
        pass


    # Grab data, perform local detrending, and split by tranists.
    lcdt = params.lc.copy()

    lcdt = lcdt[~lcdt.fmask]

    time_base = (t0-np.min(lcdt.t))-per/2+np.min(lcdt.t)
    lcdt['t_shift'] = lcdt['t'] - time_base
    t = np.array(lcdt.t_shift)
    ferr=lcdt["ferr"]//np.nanmedian(lcdt.f)
    f = np.array(lcdt.f)/np.nanmedian(lcdt.f)

    ####5-sigma clipping
    
    if params.radRatio is None:
        if Mes>50:
            try:
                rp=(1-np.nanmin(f[(t%per>=(t0-time_base)%per-.5*tdur) & (t%per<=(t0-time_base)%per+.5*tdur)]))**.5
            except:
                rp=(1-np.nanmean(f[(t%per>=(t0-time_base)%per-.5*tdur) & (t%per<=(t0-time_base)%per+.5*tdur)]))**.5


        elif t0-time_base-np.min(t)>3*dt:
            rp=(1-np.mean(f[(t%per>=(t0-time_base)%per-.5*tdur) & (t%per<=(t0-time_base)%per+.5*tdur)]))**.5
        else:
            rp = np.sqrt(abs(Mes)*np.nanmedian(ferr))*.95/np.sqrt(80/per)

        if rp>.75:
            rp=.75
        elif np.isnan(rp):
            rp=0.02
    else:
        rp=params.radRatio        
    
    ndim = 5
    nwalkers = 100
    pos_min = np.array([(t0-time_base)*.99999,rp*.90,0,per*.9999,apl*.999])
    pos_max = np.array([(t0-time_base)*1.00001,rp*1.0,b,per*1.0001,apl+10])
    psize = pos_max - pos_min
    pos = [pos_min + psize*np.random.rand(ndim) for i in range(nwalkers)]


    def lnprior(theta):
        a1,a2,a3,a5,a6= theta
        a11=uniform.pdf(a1,np.min(t),(np.max(t)-np.min(t)))
        a33=uniform.pdf(a2,0,2)
        a44=uniform.pdf(a3,0,1)
        a66=uniform.pdf(a5,0,np.max(t))
        a77=uniform.pdf(a6,0,200)
        ## A Minor prior on the semi-major axis to refelct the estimates of steller density
        return(np.log(a11*a33*a44*a66*a77)-(a6-apl)**2/(2*ap_err**2))


    def lnlike(theta, x, y):
        a1,a2,a3,a5,a6= theta

        time=(x)%a5

        paramsBatman = batman.TransitParams()
        b=a3
        paramsBatman.per = a5                #orbital period
        paramsBatman.t0 = a1                     #time of inferior conjunction
        paramsBatman.rp = a2                      #planet radius (in units of stellar radii)
        paramsBatman.a = a6        #semi-major axis (in units of stellar radii)
        if b<paramsBatman.a and b>0 and paramsBatman.a>0: #semi-major axis (in units of stellar radii)
            paramsBatman.inc =np.arccos(b/paramsBatman.a)*180/np.pi  #orbital inclination (in degrees)
            paramsBatman.ecc = 0                      #eccentricity
            paramsBatman.w = 90                 #longitude of periastron (in degrees)
            paramsBatman.limb_dark = "quadratic"
            paramsBatman.u = params.limbDark       #limb darkening model

            model_tran = batman.TransitModel(paramsBatman, time,transittype="primary")    #initializes model
            flux = model_tran.light_curve(paramsBatman)

            chiSq=-(flux-y)**2/((ferr)**2)
            return np.sum(chiSq)
        else:
            return -np.inf

    def lnprob(theta, x, y):
        lp = lnprior(theta)
        lk = lnlike(theta,x,y)
        if not np.isfinite(lp):
            return -np.inf
        if not np.isfinite(lk):
            return -np.inf
        return lp + lk

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t, f), threads=1, a=2)

    nburnsteps = 250
    nsteps=100

    width = 1
    
    ###Burn in
    result=sampler.run_mcmc(pos, nburnsteps)

    pos,prob,state=result
    sampler.reset()

    ########## perform MCMC
    result=sampler.run_mcmc(pos, nsteps)

    samples = sampler.flatchain
    samples.shape

    params.fit_t0=np.median(samples[:,0]+time_base)
    params.fit_ut0=np.std(samples[:,0]+time_base)
    params.fit_rp=np.median(samples[:,1])
    params.fit_urp=np.std(samples[:,1])
    params.fit_b=np.median(samples[:,2])
    params.fit_ub=np.nanstd(samples[:,2])
    params.fit_P=np.median(samples[:,3])
    params.fit_uP=np.nanstd(samples[:,3])
    params.fit_apl=np.median(samples[:,4])
    params.fit_uapl=np.std(samples[:,4])
    params.fit_tdur=np.arcsin(((1+params.fit_rp)**2-params.fit_b**2)**.5/params.fit_apl)*params.fit_P/np.pi

    if np.isnan(params.fit_tdur) or np.isinf(params.fit_tdur):
        params.fit_tdur=np.arcsin(1)*params.fit_P/np.pi
    
    #####Measure Transit Depth
    
    tfold=t%params.fit_P

    paramsBatman = batman.TransitParams()

    b=params.fit_b
    paramsBatman.per = params.fit_P               #orbital period
    paramsBatman.t0 = (params.fit_t0-time_base)%paramsBatman.per                  #time of inferior conjunction
    paramsBatman.rp = params.fit_rp                     #planet radius (in units of stellar radii)
    paramsBatman.a = params.fit_apl        #semi-major axis (in units of stellar radii)
    paramsBatman.inc =np.arccos(b/paramsBatman.a)*180/np.pi  #orbital inclination (in degrees)
    paramsBatman.ecc = 0                      #eccentricity
    paramsBatman.w = 90                 #longitude of periastron (in degrees)
    paramsBatman.limb_dark = "quadratic"
    paramsBatman.u = params.limbDark   #limb darkening model

    rang=np.linspace(0,paramsBatman.per, num=10000)

    model_tran = batman.TransitModel(paramsBatman, rang)    #initializes model
    flux = model_tran.light_curve(paramsBatman)
    
    params.tranDepth=1-np.min(flux)
    return params 

            

def Go(params,delta_mag=float("Inf"),delta_dist=float("Inf"), photoAp=1):
    """Initialize All Vetting Metrics.

    The Go function runs all of the EDI-Vetter metrics on the transit signal.  
    
    Args:
        params (Required[object]): the transit parameters need to assess
             the validity of the signal
        
        delta_mag (optional[float]): magnitude difference between target star
             and potential contaminate source in the Gaia G band
        
        delta_dist (optional[float]): distance between the target star and the
             potential contaminate source in arc-seconds
        
        photoAp (optional[int]): number of pixels used for the target aperture.


    """
    

    params.FalsePositive=False
        #############   Run EDI-Vetter   ########
        ############# Exoplanet Detection Indicator - Vetter ######

    print("""
     ___________ _____      _   _      _   _            
    |  ___|  _  \_   _|    | | | |    | | | |           
    | |__ | | | | | |______| | | | ___| |_| |_ ___ _ __ 
    |  __|| | | | | |______| | | |/ _ \ __| __/ _ \ '__|
    | |___| |/ / _| |_     \ \_/ /  __/ |_| ||  __/ |   
    \____/|___/  \___/      \___/ \___|\__|\__\___|_|
    """)

    params=fluxContamination(params,delta_mag,delta_dist, photoAp)
    params=outlierTransit(params)
    params=individual_transits(params)
    params=even_odd_transit(params)
    params=uniqueness_test(params)
    params=ephemeris_wonder(params)
    params=check_SE(params)
    params=harmonic_test(params)
    params=period_alias(params,cycle=True)
    params=phase_coverage(params)
    params=tdur_max(params)
    
    if params.fluxContaminationFP | params.outlierTransitFP | params.TransMaskFP | params.even_odd_transit_misfit | params.uniquenessFP | params.SeFP | params.eph_slipFP | params.harmonicFP | params.phaseCoverFP | params.tdurFP :
        params.FalsePositive=True
    else:
        params.FalsePositive=False
    
    print("==========================================")
    print("            Vetting Report") 
    print("==========================================")   
    print("        Flux Contamination : " + str(params.fluxContaminationFP))        
    print("         Too Many Outliers : " + str(params.outlierTransitFP))   
    print("  Too Many Transits Masked : " + str(params.TransMaskFP))   
    print("Odd/Even Transit Variation : " + str(params.even_odd_transit_misfit))   
    print("      Signal is not Unique : " + str(params.uniquenessFP))
    print("   Secondary Eclipse Found : " + str(params.SeFP))
    print(" Transit Mid-point Slipped : " + str(params.eph_slipFP))
    print("     Strong Harmonic Found : " + str(params.harmonicFP))
    print("Low Transit Phase Coverage : " + str(params.phaseCoverFP) )
    print("Transit Duration Too Large : " + str(params.tdurFP))
    print("==========================================") 
    print("Signal is a False Positive : "+  str(params.FalsePositive))
             
        
    return params

    

    

def fluxContamination(params,delta_mag,delta_dist, photoAp):
    """Flux Contamination test
    Look for transit contamination from nearby stars.
    
    Args:
        params : Normal transit parameters
        delta_mag (float): the difference in magnitudes between the target star and the potential contaminate in the Gaia G band 
        delta_dist (float): the distance between the potentially contaminating source and the target star in arcsecond.
        photoAp (int): The number of pixels used in the aperture of the flux measurements. 

    """

    fit_rp=params.fit_rp
    fit_b=params.fit_b
    deltDist=(np.sqrt(photoAp/np.pi)+1)*3.98
    fluxRatio=10**(delta_mag/-2.5)
    fTotStar=1+fluxRatio*1/2*(1+special.erf((deltDist-delta_dist)/(2.55*np.sqrt(2))))
    params.flux_ratio=fTotStar  
    
    if deltDist>20.4:
        params.fluxContaminationFP=True
    
    elif (fit_rp*np.sqrt(fTotStar) + fit_b)>1.04:
        params.fluxContaminationFP=True
        
    elif (fit_rp)*np.sqrt(fTotStar)>0.3:
        params.fluxContaminationFP=True
    else:
        params.fluxContaminationFP=False
    
    return params    
        

def outlierTransit(params):
    
    # global params
    fit_tdur=params.fit_tdur
    fit_P=params.fit_P
    fit_t0=params.fit_t0
    fit_rp=params.fit_rp
    fit_b=params.fit_b
    fit_apl=params.fit_apl
    mes=params.Mes
    
    lcdt = params.lc.copy() 
    time_base = (fit_t0-np.min(lcdt.t))-fit_P/2+np.min(lcdt.t)  
    lcdt['t_shift'] = lcdt['t'] - time_base
    t = np.array(lcdt.t_shift)
    f = np.array(lcdt.f)/np.median(lcdt.f)
    ferr=lcdt.ferr
    
    paramsBatman = batman.TransitParams()

    b=fit_b
    paramsBatman.per = fit_P                #orbital period
    paramsBatman.t0 = fit_t0- time_base                   #time of inferior conjunction
    paramsBatman.rp = fit_rp                      #planet radius (in units of stellar radii)
    paramsBatman.a = fit_apl        #semi-major axis (in units of stellar radii)
    paramsBatman.inc =np.arccos(b/paramsBatman.a)*180/np.pi  #orbital inclination (in degrees)
    paramsBatman.ecc = 0                      #eccentricity
    paramsBatman.w = 90                 #longitude of periastron (in degrees)
    paramsBatman.limb_dark = "quadratic"
    paramsBatman.u = params.limbDark      #limb darkening model

    model_tran = batman.TransitModel(paramsBatman, t,transittype="primary")    #initializes model
    flux = model_tran.light_curve(paramsBatman)
    
    madG=flux-f
    
    stCheck=mad_std(madG[(t%fit_P>=((fit_t0-time_base)%fit_P-.5*fit_tdur)) & (t%fit_P<=((fit_t0-time_base)%fit_P+.5*fit_tdur))])
    
    newMask=np.where((t%fit_P>=((fit_t0-time_base)%fit_P-.5*fit_tdur)) & (t%fit_P<=((fit_t0-time_base)%fit_P+.5*fit_tdur)) & (madG>3*mad_std(f)),True, False)

    params.lc['fmask'] = params.lc['fmask'] | newMask
    
    if len(newMask[newMask])>np.round(1/3*(np.floor((np.max(t)-(fit_t0-time_base))/fit_P)+1)):
        params.outlierTransitFP=True
    elif len(newMask[newMask])>6:
        params.outlierTransitFP=True
    elif stCheck>(0.4*mes-1.764)*mad_std(f):
        params.outlierTransitFP=True
    else:
        params.outlierTransitFP=False   
        
    return params              
    

def individual_transits(params):
    
    params.TransMask=False
    fit_tdur=params.fit_tdur
    fit_P=params.fit_P
    fit_t0=params.fit_t0
    fit_rp=params.fit_rp
    fit_b=params.fit_b
    fit_apl=params.fit_apl
    mes=params.Mes
    
    
    lcdt = params.lc.copy()
    lcdt = lcdt[~lcdt.fmask]
    time_base = (fit_t0-np.min(lcdt.t))-fit_P/2+np.min(lcdt.t) 
    lcdt['t_shift'] = lcdt['t'] - time_base
    
    t = np.array(lcdt.t_shift)
    f = np.array(lcdt.f)/np.median(lcdt.f)
    ferr=np.array(lcdt.ferr)
    meddt=np.nanmean(params.lc.t[1:]-params.lc.t[:-1])
    params.dt=meddt
       ####5-sigma clipping

    
    numTran=np.int(np.floor((np.max(t)-(fit_t0-time_base))/fit_P)+1)
    calcTrans=numTran
    sES=np.zeros(numTran)
    for idxx in range(numTran):
        
        tTran=t[(lcdt.t_shift>=(fit_t0)+idxx*fit_P-time_base-fit_P*.5) & (lcdt.t_shift<=(fit_t0)+idxx*fit_P-time_base+fit_P*.5)]
        fTran=f[(lcdt.t_shift>=(fit_t0)+idxx*fit_P-time_base-fit_P*.5) & (lcdt.t_shift<=(fit_t0)+idxx*fit_P-time_base+fit_P*.5)]
        ferrTran=ferr[(lcdt.t_shift>=(fit_t0)+idxx*fit_P-time_base-fit_P*.5) & (lcdt.t_shift<=(fit_t0)+idxx*fit_P-time_base+fit_P*.5)]
    
    
        tm = tval.TransitModel(fit_P, fit_t0-time_base, fit_rp, fit_apl, fit_b, params.limbDark[0], params.limbDark[1],)
        tm.lm_params['rp'].min = 0.01*fit_rp
        tm.lm_params['rp'].max = 2.0 * fit_rp
        tm.lm_params['b'].min = 0.0
        tm.lm_params['b'].max = 1.0
        tm.lm_params['apl'].min = 0.0
        tm.lm_params['apl'].max = 200.0
        tm.lm_params['t0'].min = tm.lm_params['t0'] - .1
        tm.lm_params['t0'].max = tm.lm_params['t0'] + .1
        tm.lm_params['per'].min = tm.lm_params['per'] - .1
        tm.lm_params['per'].max = tm.lm_params['per'] + .1
        tm_initial = copy.deepcopy(tm)

        method = 'lbfgsb'
        #tm.lm_params.pretty_print()
        try:
            out = minimize(tm.residual, tm.lm_params, args=(tTran, fTran, ferrTran), method=method)

            # Store away best fit parameters
            par = out.params
            BICmodelTran=out.bic
        except:
            BICmodelTran=-np.inf    


        sm = tval.FPModel1(1)
        sm.sin_params['offSet'].min = 0.9
        sm.sin_params['offSet'].max = 1.1

        sm_initial = copy.deepcopy(sm)

        method = 'lbfgsb'
        #sm.sin_params.pretty_print()
        try:
            out = minimize(sm.residual, sm.sin_params, args=(tTran, fTran, ferrTran), method=method)

            # Store away best fit parameters
            par = out.params
            BICmodel1=out.bic

        except:
            BICmodel1=BICmodelTran    



        sm = tval.FPModel2(1,fit_rp**2,10,fit_t0-time_base)
        sm.sin_params['offSet'].min = 0.9
        sm.sin_params['offSet'].max = 1.1
        sm.sin_params['dVal'].min = 0
        sm.sin_params['dVal'].max = 0.6
        sm.sin_params['aVal'].min = 0
        sm.sin_params['aVal'].max = 75
        sm.sin_params['t0'].min = fit_t0-time_base - .5
        sm.sin_params['t0'].max = fit_t0-time_base + .5

        sm_initial = copy.deepcopy(sm)

        method = 'lbfgsb'
        #sm.sin_params.pretty_print()
        try:
            out = minimize(sm.residual, sm.sin_params, args=(tTran, fTran, ferrTran), method=method)

            # Store away best fit parameters
            par = out.params
            BICmodel2=out.bic

        except:
            BICmodel2=BICmodelTran     

        sm = tval.FPModel3(1,fit_rp**2,10,fit_t0-time_base,-10)
        sm.sin_params['offSet'].min = 0.9
        sm.sin_params['offSet'].max = 1.1
        sm.sin_params['dVal'].min = 0
        sm.sin_params['dVal'].max = 0.6
        sm.sin_params['aVal'].min = 0
        sm.sin_params['aVal'].max = 75
        sm.sin_params['t0'].min = fit_t0-time_base - .5
        sm.sin_params['t0'].max = fit_t0-time_base + .5
        sm.sin_params['beta'].min = -75
        sm.sin_params['beta'].max = 75

        sm_initial = copy.deepcopy(sm)

        method = 'lbfgsb'
        #sm.sin_params.pretty_print()
        try:
            out = minimize(sm.residual, sm.sin_params, args=(tTran, fTran, ferrTran), method=method)

            # Store away best fit parameters
            par = out.params
            BICmodel3=out.bic

        except:
            BICmodel3=BICmodelTran
        
        sm = tval.FPModel4(1,fit_rp**2,10,fit_t0-time_base,fit_tdur)
        sm.sin_params['offSet'].min = 0.9
        sm.sin_params['offSet'].max = 1.1
        sm.sin_params['dVal'].min = 0
        sm.sin_params['dVal'].max = 0.6
        sm.sin_params['aVal'].min = 0
        sm.sin_params['aVal'].max = 75
        sm.sin_params['t0'].min = fit_t0-time_base - .5
        sm.sin_params['t0'].max = fit_t0-time_base + .5
        sm.sin_params['tau'].min = 0
        sm.sin_params['tau'].max = 5

        sm_initial = copy.deepcopy(sm)

        method = 'lbfgsb'
        #sm.sin_params.pretty_print()
        try:
            out = minimize(sm.residual, sm.sin_params, args=(tTran, fTran, ferrTran), method=method)

            # Store away best fit parameters
            par = out.params
            if BICmodel4 < BICmodelTran*.5: 
                BICmodel4=BICmodelTran
            else:
                BICmodel4=out.bic   

        except:
            BICmodel4=BICmodelTran    
        

        newT=t
        newF=f
        meddt=np.nanmean(params.lc.t[1:]-params.lc.t[:-1])
        newMask=np.zeros((len(newF)), dtype=bool)
        newMask=np.where((lcdt.t_shift>=(fit_t0)+idxx*fit_P-time_base-fit_P*.5) & (lcdt.t_shift<=(fit_t0)+idxx*fit_P-time_base+fit_P*.5),True, False)
        resid=np.where((lcdt.t_shift>=(fit_t0)+idxx*fit_P-time_base-.5*fit_tdur) & (lcdt.t_shift<=(fit_t0)+idxx*fit_P-time_base+.5*fit_tdur),(1-newF), 0)
        resid=np.where(resid<0,-(resid)**2,resid**2)
        sumRes=np.sum(resid)**.5
        if idxx==0:
            residTot=np.where((lcdt.t_shift%fit_P>=(fit_t0-time_base)%fit_P-.5*fit_tdur) & (lcdt.t_shift%fit_P<=(fit_t0-time_base)%fit_P+.5*fit_tdur),(1-newF), 0)
            residTot=np.where(residTot<0,-(residTot)**2,residTot**2)
            sumResTot=np.sum(residTot)**.5
        
        
        if np.isnan(sumRes):
            sES[idxx]=0
        if np.isnan(sumResTot):
            sES[idxx]=1    
        else:    
            sES[idxx]=sumRes/sumResTot
    
        numCad=len(lcdt.t_shift[(lcdt.t_shift>=(fit_t0)+idxx*fit_P-time_base-.5*fit_tdur) & (lcdt.t_shift<=(fit_t0)+idxx*fit_P-time_base+.5*fit_tdur)])
        numExpect=np.floor(fit_tdur/params.dt)
    


        if .6*numExpect>=numCad:
            isMask=(lcdt.t_shift>=(fit_t0)+idxx*fit_P-time_base-.75*fit_tdur) & (lcdt.t_shift<=(fit_t0)+idxx*fit_P-time_base+.75*fit_tdur)
            params.lc['fmask'] = params.lc['fmask'] | isMask
            params.TransMask=True
            calcTrans=calcTrans-1
        elif BICmodel1+10 < BICmodelTran:
            calcTrans=calcTrans-1
            isMask=(lcdt.t_shift>=(fit_t0)+idxx*fit_P-time_base-.75*fit_tdur) & (lcdt.t_shift<=(fit_t0)+idxx*fit_P-time_base+.75*fit_tdur)
            params.lc['fmask'] = params.lc['fmask'] | isMask
            params.TransMask=True
        elif BICmodel2+10 < BICmodelTran and mES/np.sqrt(numTran)>4:
            calcTrans=calcTrans-1
            isMask=(lcdt.t_shift>=(fit_t0)+idxx*fit_P-time_base-.75*fit_tdur) & (lcdt.t_shift<=(fit_t0)+idxx*fit_P-time_base+.75*fit_tdur)
            params.lc['fmask'] = params.lc['fmask'] | isMask
            params.TransMask=True
        elif BICmodel3+10 < BICmodelTran and mES/np.sqrt(numTran)>4:
            calcTrans=calcTrans-1
            isMask=(lcdt.t_shift>=(fit_t0)+idxx*fit_P-time_base-.75*fit_tdur) & (lcdt.t_shift<=(fit_t0)+idxx*fit_P-time_base+.75*fit_tdur)
            params.lc['fmask'] = params.lc['fmask'] | isMask
            params.TransMask=True
        elif BICmodel4+10 < BICmodelTran and mES/np.sqrt(numTran)>4:
            calcTrans=calcTrans-1
            isMask=(lcdt.t_shift>=(fit_t0)+idxx*fit_P-time_base-.75*fit_tdur) & (lcdt.t_shift<=(fit_t0)+idxx*fit_P-time_base+.75*fit_tdur)
            params.lc['fmask'] = params.lc['fmask'] | isMask
            params.TransMask=True       
        else:
            pass
            
    if calcTrans<3:
        params.TransMaskFP=True
    else:
        params.TransMaskFP=False

    if np.max(sES)>=0.8:
        params.TransMaskFP=True
    else:
        pass

    newMES=grid_search_NewMES(params)
    
    if newMES<snrThreshold:
        params.TransMaskFP=True
    else:
        pass                    
                
    return params    

def grid_search_NewMES(params): 
    """Run the grid based search

    Args:
        P1 (Optional[float]): Minimum period to search over. Default is 0.5
        P2 (Optional[float]): Maximum period to search over. Default is half 
            the time baseline
        **kwargs : passed to grid.periodogram

    Returns:
        None
    
    """
    try:
        fit_tdur=params.fit_tdur
        fit_P=params.fit_P
        fit_t0=params.fit_t0
    except:
        fit_tdur=params.tdur
        fit_P=params.per
        fit_t0=params.t0
        
    lcdt = params.lc.copy()
   
    t = np.array(params.lc.t)
    dt = t[1:] - t[:-1]
    meddt = np.median(dt)
    loca=np.where(dt>meddt+0.0001)[0]
    
    ##Fill data gaps with masked points
    while len(loca)>0:
        i=0
        var=list(params.lc.head())
        #print (var)
        try:
            line = pd.DataFrame({var[0]: params.lc[var[0]][loca[i]+1]-meddt,
             var[1]: params.lc[var[1]][loca[i]+1],
             var[2]: params.lc[var[2]][loca[i]+1],
             var[3]: params.lc[var[3]][loca[i]+1],
             var[4]: params.lc[var[4]][loca[i]+1],
             var[5]: params.lc[var[5]][loca[i]+1],
             var[6]: params.lc[var[6]][loca[i]+1],
             var[7]: params.lc[var[7]][loca[i]+1],
             var[8]: params.lc[var[8]][loca[i]+1],
             var[9]: params.lc[var[9]][loca[i]+1]} , index=[loca[i]+1])
        except:
            line = pd.DataFrame({var[0]: params.lc[var[0]][loca[i]+1]-meddt,
             var[1]: params.lc[var[1]][loca[i]+1],
             var[2]: params.lc[var[2]][loca[i]+1],
             var[3]: params.lc[var[3]][loca[i]+1],
             var[4]: params.lc[var[4]][loca[i]+1]}, index=[loca[i]+1])
                 

        params.lc = pd.concat([params.lc[:loca[i]+1], line, params.lc[loca[i]+1:]], sort=False).reset_index(drop=True)
        t = np.array(params.lc.t)
        dt = t[1:] - t[:-1]
        meddt = np.median(dt)
        loca=np.where(dt>meddt+0.0001)[0]    
    

    
    fm = params._get_fm()

    grid = tfind.Grid(t, fm)
    Pcad1 = fit_P / meddt - 1
    Pcad2 = fit_P / meddt + 1
  
    
    pgram_params = [dict(Pcad1=Pcad1, Pcad2=Pcad2, twdG=[fit_tdur/meddt])]
    try:
        pgram = grid.periodogram(pgram_params,mode='max')
        row = pgram.sort_values('s2n').iloc[-1]

        SNR=row.s2n
    except:
        SNR=0    

    return SNR  
    
def even_odd_transit(params):
    
    def add_phasefold(params):
        return tval.add_phasefold(params.lc, params.lc.t, params.fit_P, params.fit_t0,1)

    fit_tdur=params.fit_tdur
    fit_P=params.fit_P
    fit_t0=params.fit_t0
    fit_rp=params.fit_rp
    fit_b=params.fit_b
    fit_apl=params.fit_apl
    apl=(fit_P**2*gravConstant*params.massStar/(4*np.pi**2))**(1/3)/params.radStar
    ap_err=(((fit_P**2*gravConstant/(4*np.pi**2))**(1/3)*1/params.radStar*1/3*params.massStar**(-2/3)*params.umassStar)**2
        +((fit_P**2*params.massStar*gravConstant/(4*np.pi**2))**(1/3)*-1*params.radStar**(-2)*params.uradStar)**2)**.5
    mes=params.Mes    

     
    lc = add_phasefold(params)
    
    lcdt = params.lc.copy()
    lcdt = lcdt[~lcdt.fmask]  
    time_base = (fit_t0-np.min(lcdt.t))-fit_P/2+np.min(lcdt.t)   
    lcdt['t_shift'] = lcdt['t'] - time_base
    t = np.array(lcdt.t_shift)
    f = np.array(lcdt.f)/np.median(lcdt.f)
    ferr=lcdt.ferr
    
    feven=f[params.lc.cycle_m1[params.lc.fmask==False]%2==0]
    teven=t[params.lc.cycle_m1[params.lc.fmask==False]%2==0]
    ferr=ferr[params.lc.cycle_m1[params.lc.fmask==False]%2==0]
    
    evenMask=np.where((teven%fit_P>=((fit_t0-time_base)%fit_P-.5*fit_tdur)) & (teven%fit_P<=((fit_t0-time_base)%fit_P+.5*fit_tdur)),True, False)
    
    try:
        if mes>50:
            t0Even=teven[feven==np.min(feven[evenMask])]
            t0Even=t0Even[0]
            rpEven=np.sqrt(1-np.min(feven[evenMask]))
        else:
            t0Even=fit_t0-time_base    
            rpEven=np.sqrt(1-np.mean(feven[evenMask]))
            
    except:
        t0Even=fit_t0-time_base
        rpEven=fit_rp
    
    if np.isnan(rpEven):
        rpEven=fit_rp
    if np.isnan(t0Even):
        t0Even=fit_t0-time_base
           
             
    
    ndim = 4
    nwalkers = 100
    pos_min = np.array([t0Even*.99,rpEven*.95,fit_b*.99,fit_apl*.999])
    pos_max = np.array([t0Even*1.01,rpEven*1.0,fit_b,fit_apl*1.01])
    psize = pos_max - pos_min
    pos = [pos_min + psize*np.random.rand(ndim) for i in range(nwalkers)]


    def lnpriorRR(theta):
        a1,a2,a3,a6= theta
        a11=uniform.pdf(a1,np.min(t),(np.max(t)-np.min(t)))
        a33=uniform.pdf(a2,0,2)
        a44=uniform.pdf(a3,0,1)
        a77=uniform.pdf(a6,0,200)
        return(np.log(a11*a33*a44*a77)-(a6-apl)**2/(2*ap_err**2))


    def lnlikeRR(theta, x, y):
        a1,a2,a3,a6= theta

        time=(x)%fit_P

        paramsBatman = batman.TransitParams()

        b=a3
        paramsBatman.per = fit_P                #orbital period
        paramsBatman.t0 = a1                     #time of inferior conjunction
        paramsBatman.rp = a2                      #planet radius (in units of stellar radii)
        paramsBatman.a = a6        #semi-major axis (in units of stellar radii)
        if b<paramsBatman.a and b>0 and paramsBatman.a>0: #semi-major axis (in units of stellar radii)
            paramsBatman.inc =np.arccos(b/paramsBatman.a)*180/np.pi  #orbital inclination (in degrees)
            paramsBatman.ecc = 0                      #eccentricity
            paramsBatman.w = 90                 #longitude of periastron (in degrees)
            paramsBatman.limb_dark = "quadratic"
            paramsBatman.u = params.limbDark      #limb darkening model
            model_tran = batman.TransitModel(paramsBatman, time,transittype="primary")    #initializes model
            flux = model_tran.light_curve(paramsBatman)

            chiSqr=-(flux-y)**2/((ferr)**2)
            return np.sum(chiSqr)
        else:
            return -np.inf

    def lnprobRR(theta, x, y):
        lp = lnpriorRR(theta)
        lk = lnlikeRR(theta,x,y)
        if not np.isfinite(lp):
            return -np.inf
        if not np.isfinite(lk):
            return -np.inf
        return lp + lk

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobRR, args=(teven, feven), threads=1, a=2)

    nburnsteps = 150
    nsteps=75  

    width = 1
#Burn in
    result=sampler.run_mcmc(pos, nburnsteps)
    pos,prob,state=result
    sampler.reset()

########## perform MCMC
    result=sampler.run_mcmc(pos, nsteps)
    samples = sampler.flatchain
    samples.shape

    params.fit_t0_even=np.median(samples[:,0]+time_base)
    params.fit_ut0_even=np.std(samples[:,0]+time_base)  
    params.fit_rp_even=np.median(samples[:,1])
    params.fit_urp_even=np.std(samples[:,1])
    
    ferr=lcdt.ferr
    
    fodd=f[params.lc.cycle_m1[params.lc.fmask==False]%2==1]
    todd=t[params.lc.cycle_m1[params.lc.fmask==False]%2==1]
    ferr=ferr[params.lc.cycle_m1[params.lc.fmask==False]%2==1]    
    oddMask=np.where((todd%fit_P>=((fit_t0-time_base)%fit_P-.5*fit_tdur)) & (todd%fit_P<=((fit_t0-time_base)%fit_P+.5*fit_tdur)),True, False)
    
    try:
        if mes>50:
            t0Odd=todd[fodd==np.min(fodd[evenMask])]
            t0Odd=t0Odd[0]
            rpOdd=np.sqrt(1-np.min(fodd[oddMask]))
        else:
            t0Odd=fit_t0-time_base        
            rpOdd=np.sqrt(1-np.mean(fodd[oddMask]))
    except:
        t0Odd=fit_t0-time_base
        rpOdd=fit_rp
        
    if np.isnan(rpOdd):
        rpOdd=fit_rp
    if np.isnan(t0Odd):
        t0Odd=fit_t0-time_base     
        
    
    pos_min = np.array([t0Odd*.99,rpOdd*.95,fit_b*.9,fit_apl*.999])
    pos_max = np.array([t0Odd*1.01,rpOdd*1.0,fit_b,fit_apl*1.01])
    psize = pos_max - pos_min
    pos = [pos_min + psize*np.random.rand(ndim) for i in range(nwalkers)]
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobRR, args=(todd, fodd), threads=1, a=2)
    ###Burn in
    result=sampler.run_mcmc(pos, nburnsteps)
    pos,prob,state=result
    sampler.reset()

    ########## perform MCMC
    result=sampler.run_mcmc(pos, nsteps)
    samples = sampler.flatchain
    samples.shape

    params.fit_t0_odd=np.median(samples[:,0]+time_base)
    params.fit_ut0_odd=np.std(samples[:,0]+time_base)  
    params.fit_rp_odd=np.median(samples[:,1])
    params.fit_urp_odd=np.std(samples[:,1])
    
    MesMask=np.where((t%fit_P>=((fit_t0-time_base)%fit_P-.15*fit_tdur)) & (t%fit_P<=((fit_t0-time_base)%fit_P+.15*fit_tdur)),True, False)
    rpMes=np.std(f[MesMask])

    if (params.fit_urp_odd==None):
        params.fit_urp_odd=params.fit_rp_odd  
    if (params.fit_urp_even==None):
        params.fit_urp_even=params.fit_rp_even
    if (params.fit_ut0_odd==None):
        params.fit_ut0_odd=params.fit_t0_odd  
    if (params.fit_ut0_even==None):
        params.fit_ut0_even=params.fit_t0_even
        
    if abs(params.fit_rp_even-params.fit_rp_odd)>5*np.sqrt(params.fit_urp_odd**2+params.fit_urp_even**2):
        params.even_odd_transit_misfit=True
    elif rpMes>5*np.median(lcdt.ferr):
        params.even_odd_transit_misfit=True               
    else:
        params.even_odd_transit_misfit=False
    
    return params
     
def uniqueness_test(params):

    fit_tdur=params.fit_tdur
    fit_P=params.fit_P
    fit_t0=params.fit_t0
    fit_rp=params.fit_rp
    fit_b=params.fit_b
    fit_apl=params.fit_apl
    apl=(fit_P**2*gravConstant*params.massStar/(4*np.pi**2))**(1/3)/params.radStar
    ap_err=(((fit_P**2*gravConstant/(4*np.pi**2))**(1/3)*1/params.radStar*1/3*params.massStar**(-2/3)*params.umassStar)**2
        +((fit_P**2*params.massStar*gravConstant/(4*np.pi**2))**(1/3)*-1*params.radStar**(-2)*params.uradStar)**2)**.5
    mes=params.Mes
    

    lcdt = params.lc.copy()
    lcdt = lcdt[~lcdt.fmask] 
    time_base = (fit_t0-np.min(lcdt.t))-fit_P/2+np.min(lcdt.t)   
    lcdt['t_shift'] = lcdt['t'] - time_base
    t = np.array(lcdt.t_shift)
    f = np.array(lcdt.f)/np.median(lcdt.f)
    ferr=lcdt.ferr

    sigUniq=np.ones(3)
    
    newMask3=np.where((t%fit_P>=((fit_t0-time_base)%fit_P-.5*fit_tdur)) & (t%fit_P<=((fit_t0-time_base)%fit_P+.5*fit_tdur)),True, False)
    newTrueF=f[newMask3]
    tranDepth=1-np.mean(newTrueF)

    
    newT=t
    newF=f
    meddt=np.nanmean(lcdt.t[1:]-lcdt.t[:-1])
    newMask=np.zeros((len(newF)), dtype=bool)
    
    if 3*fit_tdur/fit_P>.10:
        newMask=np.where((newT%fit_P>=((fit_t0-time_base)%fit_P-.5*fit_tdur)) & (newT%fit_P<=((fit_t0-time_base)%fit_P+.5*fit_tdur)),True, False)
        newMask=np.where((newT%fit_P>=((fit_t0-time_base)%fit_P+fit_P/2-.5*fit_tdur)) & (newT%fit_P<=((fit_t0-time_base)%fit_P+fit_P/2+.5*fit_tdur)),True, newMask)
    else:    
        newMask=np.where((newT%fit_P>=((fit_t0-time_base)%fit_P-.5*fit_tdur)) & (newT%fit_P<=((fit_t0-time_base)%fit_P+.5*fit_tdur)),True, False)
        newMask=np.where((newT%fit_P>=((fit_t0-time_base)%fit_P+fit_P/2-.5*fit_tdur)) & (newT%fit_P<=((fit_t0-time_base)%fit_P+fit_P/2+.5*fit_tdur)),True, newMask)
    flux_red=newF[~newMask]
    time_red=newT[~newMask]
    
    try:
        minBin=np.zeros(len(time_red))
        
        for intt in range(len(time_red)):
            if time_red[intt]%fit_P<np.min(time_red%fit_P)-.5*fit_tdurGrid:
                pass
            elif time_red[intt]%fit_P>np.max(time_red)%fit_P+.5*fit_tdurGrid:
                pass
            else:
                binMed=np.where((time_red%fit_P<=time_red[intt]%fit_P+.5*fit_tdurGrid) & (time_red%fit_P>=time_red[intt]%fit_P-.5*fit_tdurGrid),flux_red,0)
                minBin[intt]=np.mean(binMed[binMed>0])  
        
        minBin=minBin[minBin>0]
                    
        bg_e=mad_std(flux_red)
        
        if bg_e<np.median(ferr):
            pass
        else:
            bg_e=np.median(ferr)
               
        F_red=mad_std(minBin)/bg_e

    except:
        bg_e=mad_std(flux_red)
        if bg_e<np.median(ferr):
            pass
        else:
            bg_e=np.median(ferr)
            
        F_red=1

                
    mES=tranDepth/bg_e    
    
    if 3*fit_tdur/fit_P>.10:
        newMask=np.where((newT%fit_P>=((fit_t0-time_base)%fit_P-.5*fit_tdur)) & (newT%fit_P<=((fit_t0-time_base)%fit_P+.5*fit_tdur)),True, False)
    else:    
        newMask=np.where((newT%fit_P>=((fit_t0-time_base)%fit_P-.5*fit_tdur)) & (newT%fit_P<=((fit_t0-time_base)%fit_P+.5*fit_tdur)),True, False)
    
    dt = newT[1:] - newT[:-1]
    loca=np.where(dt>meddt+0.0001)[0]
    
    for idxx in range(3):
        if idxx==0:
            while len(loca)>0:
                i=0
                newT=np.concatenate((newT[:loca[i]+1],newT[loca[i]+1]-meddt,newT[loca[i]+1:]), axis=None)
                newF=np.concatenate((newF[:loca[i]+1],1,newF[loca[i]+1:]), axis=None)
                newMask=np.concatenate((newMask[:loca[i]+1],True,newMask[loca[i]+1:]), axis=None)
                dt = newT[1:] - newT[:-1]
                meddt = np.median(dt)
                loca=np.where(dt>meddt+0.0001)[0]
        elif idxx==1:
            try:
                params.fit_t0_se=row.t0+time_base

                newMask=np.where((newT%row.P>=((row.t0)%row.P-1*row.tdur)) & (newT%row.P<=((row.t0)%row.P+1*row.tdur)),True, newMask)
                phaseSE1=(params.fit_t0_se-(fit_t0+fit_P/2))/fit_tdur
                phaseSE2=(params.fit_t0_se-(fit_t0-fit_P/2))/fit_tdur
                phaseSE=np.min([phaseSE1,phaseSE2])
                if phaseSE<=.1:
                    flux_red=newF[~newMask]
                    time_red=newT[~newMask]
    
                    try:
                        minBin=np.zeros(len(time_red))
                        for intt in range(len(time_red)):
                            if time_red[intt]%fit_P<np.min(time_red%fit_P)-.5*fit_tdur:
                                pass
                            elif time_red[intt]%fit_P>np.max(time_red)%fit_P+.5*fit_tdur:
                                pass
                            else:
                                binMed=np.where((time_red%fit_P<=time_red[intt]%fit_P+.5*fit_tdur) & (time_red%fit_P>=time_red[intt]%fit_P-.5*fit_tdur),flux_red,0)
                                minBin[intt]=np.mean(binMed[binMed>0])  
        
                        minBin=minBin[minBin>0]
                        bg_e=mad_std(flux_red)
        
                        if bg_e<np.median(ferr):
                            pass
                        else:
                            bg_e=np.median(ferr) 
               
                        F_red=mad_std(minBin)/bg_e

                    except:
                        bg_e=mad_std(flux_red)
                        if bg_e<np.median(ferr):
                            pass
                        else:
                            bg_e=np.median(ferr)
            
                        F_red=1
                    
            except:
                params.fit_t0_se=fit_t0+fit_P/2
                newMask=np.where((newT%fit_P>=((fit_t0-time_base+fit_P/2)%fit_P-.5*fit_tdur)) & (newT%fit_P<=((fit_t0-time_base+fit_P/2)%fit_P+.5*fit_tdur)),True, newMask)
                
        elif idxx==2:
            newF=-newF
                        
        fm = ma.masked_array(newF,mask=newMask, fill_value=0)
        fm -= ma.median(fm)

        try:
            grid = tfind.Grid(newT, fm)
            Pcad1 = (fit_P - fit_uP)/meddt
            Pcad2 = (fit_P + fit_uP)/meddt
            twd = fit_tdurGrid/meddt
            pgram_params = [dict(Pcad1=Pcad1, Pcad2=Pcad2, twdG=[twd])]
            pgram = grid.periodogram(pgram_params,mode='max')
            row = pgram.sort_values('s2n').iloc[-1]
            if row.s2n>0:
                newT2=newT[~newMask]
                newF2=newF[~newMask]
                newMask2=np.where((newT2%row.P>=((row.t0)%row.P-.5*row.tdur)) & (newT2%row.P<=((row.t0)%row.P+.5*row.tdur)),True, False)
                Fstuff=np.mean(newF2[newMask2])*-1
                sigUniq[idxx]=Fstuff/bg_e
            else:
                sigUniq[idxx]=0
        except:
            sigUniq[idxx]=0
    
    fA_1=np.sqrt(2)*special.erfcinv(fit_tdur/fit_P/10000)
    fA_2=np.sqrt(2)*special.erfcinv(fit_tdur/fit_P)
    
    if np.isnan(fA_1):
        fA_1=4
        
    
    mS1=fA_1-mES/F_red
    mS2=fA_2-(mES-sigUniq[1])
    mS3=fA_2-(mES-sigUniq[2])
    
    mS4=sigUniq[0]/F_red-fA_1
    mS5=(sigUniq[0]-sigUniq[1])-fA_2
    mS6=(sigUniq[0]-sigUniq[2])-fA_2
    
    ms7=(mES-sigUniq[0])-fA_2
    
    phaseSE1=(params.fit_t0_se-(fit_t0+fit_P/2))/fit_P
    phaseSE2=(params.fit_t0_se-(fit_t0-fit_P/2))/fit_P
    phaseSE=np.min([phaseSE1,phaseSE2])
    
    if mS1>0.0:
        params.uniquenessFP=True
    elif mS2>1: 
        params.uniquenessFP=True
    elif mS3>2:
        params.uniquenessFP=True
    else:    
        params.uniquenessFP=False
    
    if mS4>0.5 and mS5>-0.5 and mS6>-0.5:
        if ms7>1 and phaseSE<=.1:
            params.SE_found=True

        else:
            params.uniquenessFP=True
            params.SE_found=True 
    else:
        params.SE_found=False
    return params   


def ephemeris_wonder(params):

    fit_tdur=params.fit_tdur
    fit_P=params.fit_P
    fit_t0=params.fit_t0
    grid_t0=params.t0
    
    if abs(grid_t0-fit_t0)>0.5*fit_tdur:
        params.eph_slipFP=True
    else:
        params.eph_slipFP=False
        
    return params 

def check_SE(params):
    
    if params.SE_found==True:
        
        # Perform global fit. Set some common-sense limits on parameters
        fit_tdur=params.fit_tdur
        fit_P=params.fit_P
        fit_t0=params.fit_t0
        fit_rp=params.fit_rp
        fit_b=params.fit_b
        fit_apl=params.fit_apl
        mes=params.Mes
        
        lcdt = params.lc.copy()
        lcdt = lcdt[~lcdt.fmask]   
        time_base = (fit_t0-np.min(lcdt.t))-fit_P/2+np.min(lcdt.t)  
        lcdt['t_shift'] = lcdt['t'] - time_base
        tse = np.array(lcdt.t_shift)
        fse = np.array(lcdt.f)/np.median(lcdt.f)
        ferr=lcdt.ferr
        
        evenMask=np.where((tse%fit_P>=((params.fit_t0_se-time_base)%fit_P-.5*fit_tdur)) & (tse%fit_P<=((params.fit_t0_se-time_base)%fit_P+.5*fit_tdur)),True, False)
     
        try:
            if mes>50:
                rpSE=np.sqrt(1-np.min(fse[evenMask]))
            else: 
                rpSE=np.sqrt(1-np.mean(fse[evenMask]))
             
        except:
            rpSE=fit_rp*.1
     
        if np.isnan(rpSE):
            rpSE=fit_rp*.1

        if abs((params.fit_t0_se)+.5*fit_P-fit_t0)<=fit_P/10:
            tm = tval.TransitModel(fit_P, params.fit_t0_se - time_base, rpSE, fit_apl, fit_b, params.limbDark[0], params.limbDark[1], )
        elif abs((params.fit_t0_se)-.5*fit_P-fit_t0)<=fit_P/10:
            tm = tval.TransitModel(fit_P, params.fit_t0_se - time_base, rpSE, fit_apl, fit_b, params.limbDark[0], params.limbDark[1], )    
        else: 
            tm = tval.TransitModel(fit_P, fit_t0 - time_base+.5*fit_P, rpSE, fit_apl, fit_b, params.limbDark[0], params.limbDark[1], )   
        tm.lm_params['rp'].min = 0.0
        tm.lm_params['rp'].max = 2.0 * fit_rp
        tm.lm_params['b'].min = 0.0
        tm.lm_params['b'].max = 1.0
        tm.lm_params['apl'].min = 0.0
        tm.lm_params['apl'].max = 200.0
        tm.lm_params['t0'].min = tm.lm_params['t0'] - fit_tdur
        tm.lm_params['t0'].max = tm.lm_params['t0'] + fit_tdur
        tm.lm_params['per'].min = tm.lm_params['per'] - .1
        tm.lm_params['per'].max = tm.lm_params['per'] + .1

        tm_initial = copy.deepcopy(tm)

        method = 'least_squares'
        # tm.lm_params.pretty_print()

        out = minimize(tm.residual, tm.lm_params, args=(tse, fse, ferr), method=method)

        # Store away best fit parameters
        par = out.params
        t0 = par['t0'].value + time_base
        params.fit_rp_se=par['rp'].value
        params.fit_urp_se=par['rp'].stderr
        params.fit_t0_se=par['t0'].value

 
   
        if 0.1*fit_rp**2<params.fit_rp_se**2 and params.fit_urp_se!=None:
            if fit_b>=.9 and params.fit_rp_se/params.fit_urp_se>2:
                params.SE_found=True
                params.SeFP=True
            else:
                params.SE_found=True
                params.SeFP=False
    else:
         params.SeFP=False
    
    return params            

def harmonic_test(params):
    #####Check if harmonic!!!!
    fit_tdur=params.fit_tdur
    fit_P=params.fit_P
    fit_t0=params.fit_t0
    fit_rp=params.fit_rp
    fit_depth=params.tranDepth
    mes=params.Mes
    
    if fit_P>=.5:
        
        lcdt = params.lc.copy()
        lcdt = lcdt[~lcdt.fmask]   
        time_base = (fit_t0-np.min(lcdt.t))-fit_P/2+np.min(lcdt.t)  
        lcdt['t_shift'] = lcdt['t'] - time_base
        t = np.array(lcdt.t_shift)
        f = np.array(lcdt.f)/np.median(lcdt.f)
        ferr=lcdt.ferr

        
        
        sm = tval.CosModel(-fit_depth, 2*np.pi*(fit_t0-time_base)/fit_P , 1, 2*np.pi/(fit_P))
        sm.sin_params['amp'].max = -0.0001*fit_depth
        sm.sin_params['amp'].min = -5.0 * fit_depth
        sm.sin_params['offSet'].min = np.min(t)
        sm.sin_params['offSet'].max = np.min(t)+fit_P
        sm.sin_params['YoffSet'].min = .9
        sm.sin_params['YoffSet'].max = 1.1


        sm_initial = copy.deepcopy(sm)
        method = 'least_squares'
        out = minimize(sm.residual, sm.sin_params, args=(t, f, ferr), method=method)

        # Store away best fit parameters
        par = out.params
        ampVal1=par['amp'].value
        ampStd1=par['amp'].stderr
        if ampStd1==None:
            ampStd1=ampVal1
        
        sm = tval.CosModel(-fit_depth, 2*np.pi*(fit_t0-time_base)/(fit_P/2), 1, 2*np.pi/(fit_P/2))
        sm.sin_params['amp'].max = -0.0001*fit_depth
        sm.sin_params['amp'].min = -5.0 * fit_depth
        sm.sin_params['offSet'].min = np.min(t)
        sm.sin_params['offSet'].max = np.min(t)+fit_P
        sm.sin_params['YoffSet'].min = .9
        sm.sin_params['YoffSet'].max = 1.1

        sm_initial = copy.deepcopy(sm)
        out = minimize(sm.residual, sm.sin_params, args=(t, f, ferr), method=method)

        # Store away best fit parameters
        par = out.params
        ampVal2=par['amp'].value
        ampStd2=par['amp'].stderr
        if ampStd2==None:
            ampStd2=ampVal2
        
        sm = tval.CosModel(-fit_depth, 2*np.pi*(fit_t0-time_base)/(fit_P*2) , 1, 2*np.pi/(fit_P*2))
        sm.sin_params['amp'].max = -0.0001*fit_depth
        sm.sin_params['amp'].min = -5.0 * fit_depth
        sm.sin_params['offSet'].min = np.min(t)
        sm.sin_params['offSet'].max = np.min(t)+fit_P
        sm.sin_params['YoffSet'].min = .9
        sm.sin_params['YoffSet'].max = 1.1


        sm_initial = copy.deepcopy(sm)

        out = minimize(sm.residual, sm.sin_params, args=(t, f, ferr), method=method)

        # Store away best fit parameters
        par = out.params
        ampVal3=par['amp'].value
        ampStd3=par['amp'].stderr
        
        sm = tval.CosModel(-fit_depth, 2*np.pi*(fit_t0-time_base)/(fit_tdur*2) , 1, 2*np.pi/(fit_tdur*2))
        sm.sin_params['amp'].max = -0.0001*fit_depth
        sm.sin_params['amp'].min = -5.0 * fit_depth
        sm.sin_params['offSet'].min = np.min(t)
        sm.sin_params['offSet'].max = np.min(t)+fit_P
        sm.sin_params['YoffSet'].min = .9
        sm.sin_params['YoffSet'].max = 1.1


        sm_initial = copy.deepcopy(sm)

        out = minimize(sm.residual, sm.sin_params, args=(t, f, ferr), method=method)

        # Store away best fit parameters
        par = out.params
        ampVal4=par['amp'].value
        ampStd4=par['amp'].stderr
        
        sm = tval.CosModel(-fit_depth, 2*np.pi*(fit_t0-time_base)/(fit_tdur*4) , 1, 2*np.pi/(fit_tdur*4))
        sm.sin_params['amp'].max = -0.0001*fit_depth
        sm.sin_params['amp'].min = -5.0 * fit_depth
        sm.sin_params['offSet'].min = np.min(t)
        sm.sin_params['offSet'].max = np.min(t)+fit_P
        sm.sin_params['YoffSet'].min = .9
        sm.sin_params['YoffSet'].max = 1.1

        sm_initial = copy.deepcopy(sm)
        out = minimize(sm.residual, sm.sin_params, args=(t, f, ferr), method=method)

        # Store away best fit parameters
        par = out.params
        ampVal5=par['amp'].value
        ampStd5=par['amp'].stderr
        
        sm = tval.CosModel(-fit_depth, 2*np.pi*(fit_t0-time_base)/(fit_tdur), 1, 2*np.pi/(fit_tdur))
        sm.sin_params['amp'].max = -0.0001*fit_depth
        sm.sin_params['amp'].min = -5.0 * fit_depth
        sm.sin_params['offSet'].min = np.min(t)
        sm.sin_params['offSet'].max = np.min(t)+fit_P
        sm.sin_params['YoffSet'].min = .9
        sm.sin_params['YoffSet'].max = 1.1


        sm_initial = copy.deepcopy(sm)
        out = minimize(sm.residual, sm.sin_params, args=(t, f, ferr), method=method)

        # Store away best fit parameters
        par = out.params
        ampVal6=par['amp'].value
        ampStd6=par['amp'].stderr
        
        
        
        if ampStd6==None:
            ampStd6=ampVal6
        if ampStd5==None:
            ampStd5=ampVal5    
        if ampStd4==None:
            ampStd4=ampVal4    
        if ampStd3==None:
            ampStd3=ampVal3
        if ampStd2==None:
            ampStd2=ampVal2
        if ampStd1==None:
            ampStd1=ampVal1                    
        if abs(ampVal1/ampStd1)>50 or abs(ampVal2/ampStd2)>50 or abs(ampVal3/ampStd3)>50 or abs(ampVal4/ampStd4)>50 or abs(ampVal5/ampStd5)>50 or abs(ampVal6/ampStd6)>50:
            params.harmonicFP=True
        elif (abs(ampVal1)> fit_depth and abs(ampVal1)>2*abs(ampStd1)):
            params.harmonicFP=True
        elif (abs(ampVal2)> fit_depth and abs(ampVal2)>2*abs(ampStd2)):
            params.harmonicFP=True
        elif (abs(ampVal3)> fit_depth and abs(ampVal3)>2*abs(ampStd3)):
            params.harmonicFP=True
        elif (abs(ampVal4)> fit_depth and abs(ampVal4)>2*abs(ampStd4)):
            params.harmonicFP=True
        elif (abs(ampVal5)> fit_depth and abs(ampVal5)>2*abs(ampStd5)):
            params.harmonicFP=True
        elif (abs(ampVal6)> fit_depth and abs(ampVal6)>2*abs(ampStd6)):
            params.harmonicFP=True  
        else:
            params.harmonicFP=False
    else:
        params.harmonicFP=False
        
    return params    

           

def period_alias(params, cycle=False): 

    if hasattr(params,'period_alias_cycle'):
        pass
    else:
        params.period_alias_cycle=0
        
    per_cyc=params.period_alias_cycle    
    fit_tdur=params.fit_tdur
    fit_P=params.fit_P
    fit_t0=params.fit_t0
    fit_rp=params.fit_rp
    fit_b=params.fit_b
    fit_apl=params.fit_apl
    fit_depth=params.tranDepth
    mes=params.Mes
    
    lcdt = params.lc.copy()
    lcdt = lcdt[~lcdt.fmask]   
    time_base = (fit_t0-np.min(lcdt.t))-fit_P/2+np.min(lcdt.t)     
    lcdt['t_shift'] = lcdt['t'] - time_base
    t = np.array(lcdt.t_shift)
    f = np.array(lcdt.f)/np.median(lcdt.f)
    ferr=lcdt.ferr

    bm_params = batman.TransitParams()
    bm_params.b = fit_b
    bm_params.per = fit_P
    bm_params.t0 = fit_t0-time_base
    bm_params.rp = fit_rp
    bm_params.u = [params.limbDark[0], params.limbDark[1]]  #orbital inclination (in degrees)
    bm_params.a = fit_apl
    bm_params.inc = np.arccos(bm_params.b/bm_params.a)*180/np.pi 
    bm_params.ecc = 0  
    bm_params.w = 90 
    bm_params.limb_dark = "quadratic" 
    model_tran = batman.TransitModel(bm_params, t, transittype="primary")
    _model = model_tran.light_curve(bm_params)
    
    likelihood_p0=np.mean(((f-_model)/ferr)**2)

    bm_params.per = fit_P/2
    bm_params.a = fit_apl/2
    bm_params.inc = np.arccos(bm_params.b/bm_params.a)*180/np.pi 
    model_tran = batman.TransitModel(bm_params, t, transittype="primary")
    _model = model_tran.light_curve(bm_params)
    
    likelihood_p0_half=np.mean(((f-_model)/ferr)**2)
    
    # apl=((fit_P/3)**2*gravConstant*pipe.star_mass/(4*np.pi**2))**(1/3)/pipe.star_rad
    bm_params.per = fit_P/3
    bm_params.a = fit_apl/3
    bm_params.inc = np.arccos(bm_params.b/bm_params.a)*180/np.pi 
    
    model_tran = batman.TransitModel(bm_params, t, transittype="primary")
    _model = model_tran.light_curve(bm_params)
    
    likelihood_p0_third=np.mean(((f-_model)/ferr)**2)
    
    # apl=((fit_P*2)**2*gravConstant*pipe.star_mass/(4*np.pi**2))**(1/3)/pipe.star_rad
    bm_params.per = fit_P*2
    bm_params.a = fit_apl*2
    bm_params.inc = np.arccos(bm_params.b/bm_params.a)*180/np.pi 
    
    model_tran = batman.TransitModel(bm_params, t, transittype="primary")
    _model = model_tran.light_curve(bm_params)
    
    likelihood_p0_double=np.mean(((f-_model)/ferr)**2)
    
    # apl=((fit_P*3)**2*gravConstant*pipe.star_mass/(4*np.pi**2))**(1/3)/pipe.star_rad
    bm_params.per = fit_P*3
    bm_params.a = fit_apl*3
    bm_params.inc = np.arccos(bm_params.b/bm_params.a)*180/np.pi 

    model_tran = batman.TransitModel(bm_params, t, transittype="primary")
    _model = model_tran.light_curve(bm_params)
    
    likelihood_p0_trip=np.mean(((f-_model)/ferr)**2)
    
    likeArray=np.array([likelihood_p0_trip+likelihood_p0*.05,likelihood_p0_double+likelihood_p0*.05,likelihood_p0_half+likelihood_p0*.05,likelihood_p0_third+likelihood_p0*.05,likelihood_p0])
    
    if likelihood_p0==np.min(likeArray):
        params.period_alias_cycle=per_cyc+1

    else:    
        bolRatio=likeArray==np.min(likeArray)
        indBol=np.where(bolRatio==True)[0][0]

        if indBol==0:
            params.period_alias_cycle=per_cyc+1
            params.per=3*fit_P
            if ((per_cyc+1)<3) & (cycle==True):
                MCfit(params)
            
        elif indBol==1:
            params.period_alias_cycle=per_cyc+1
            params.per=2*fit_P
            if ((per_cyc+1)<3) & (cycle==True):
                MCfit(params)
            
        elif indBol==2:
            params.period_alias_cycle=per_cyc+1
            params.per=0.5*fit_P
            if ((per_cyc+1)<3) & (cycle==True):
                MCfit(params)
            
        elif indBol==3:
            params.period_alias_cycle=per_cyc+1
            params.per=1/3*fit_P
            if ((per_cyc+1)<3) & (cycle==True):
                MCfit(params)
            
        else:
            params.period_alias_cycle=per_cyc+1       
    return params 


def phase_coverage(params):

    fit_P=params.fit_P
    fit_t0=params.fit_t0
    fit_tdur=params.fit_tdur
  
    lcdt = params.lc.copy()
    lcdt = lcdt[~lcdt.fmask]
    time_base = (fit_t0-np.min(lcdt.t))-fit_P/2+np.min(lcdt.t)  
    lcdt['t_shift'] = lcdt['t'] - time_base
    t = np.array(lcdt.t_shift)
    t=t%fit_P
    
    newMask=np.where((t>=((fit_t0-time_base)%fit_P-1*fit_tdur)) & (t<=((fit_t0-time_base)%fit_P+1*fit_tdur)),False, True)
    newT=t[~newMask]
    newT=np.sort(newT)
    dt = newT[1:] - newT[:-1]
    
    #allowed gap
    def allow_tol(x):
        func=(16*x**4-8*x**2+2)*params.dt
        return func
        
    xVal=(newT[:-1]+dt/2-(fit_t0-time_base)%fit_P)/fit_tdur
    allow=allow_tol(xVal)
    
    if np.any(np.where(dt>=allow,True,False)):
        params.phaseCoverFP=True
    else:
        params.phaseCoverFP=False
        
    return params  
    
      
def tdur_max(params):
    fit_P=params.fit_P
    fit_t0=params.fit_t0
    fit_tdur=params.fit_tdur
    mes=params.Mes
    
    if fit_P<.5:
        params.tdurFP=True
    elif fit_tdur/fit_P>.1:
        params.tdurFP=True
    else:
        params.tdurFP=False
        
    return params    



###############################################################################################################################
################This feature are not currently implemented but is part of the full EDI-Vetter software######################### 
###############################################################################################################################

def previous_pl_check(pipe,plM):
    pipe.update_header('previousFP_m{}'.format(plM), False, "is the previous planet a FP?")
    
    if plM==1:
        pass
    else:
        for ipl in range(1,plM):
            perA=eval("pipe.fit_P_m{}".format(ipl))
            perB=eval("pipe.fit_P_m{}".format(plM))
            
            t0A=eval("pipe.fit_t0_m{}".format(ipl))
            t0B=eval("pipe.fit_t0_m{}".format(plM))
            
            tdur=eval("pipe.fit_tdur_m{}".format(ipl))

              
            if perA<perB:
                deltaP=(perA-perB)/tdur
                deltaPp=abs(deltaP-round(deltaP))
                sigmaP=np.sqrt(2)*special.erfcinv(deltaPp)
                emph=abs(t0A-t0B)/tdur
                emph2=abs(t0A-t0B+perA/2)/tdur
                emph3=abs(t0A-t0B-perA/2)/tdur
                if sigmaP>2:
                     if eval("pipe.falsePos_m{}".format(ipl)):
                         pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")
                         pipe.update_header('previousFP_m{}'.format(plM), True, "is the previous planet a FP?")
                     elif emph<1 or emph2<1 or emph3<1:
                         pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")
                         pipe.update_header('previousFP_m{}'.format(plM), True, "is the previous planet a FP?")         
                             
            else:
                deltaP=(perB-perA)/perB
                deltaPp=abs(deltaP-round(deltaP))
                sigmaP=np.sqrt(2)*special.erfcinv(deltaPp)
                emph=abs(t0A-t0B)/tdur
                emph2=abs(t0A-t0B+perA/2)/tdur
                emph3=abs(t0A-t0B-perA/2)/tdur
                if sigmaP>2:
                     if eval("pipe.falsePos_m{}".format(ipl)):
                         pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")
                         pipe.update_header('previousFP_m{}'.format(plM), True, "is the previous planet a FP?")
                     elif emph<1 or emph2<1 or emph3<1:
                         pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")
                         pipe.update_header('previousFP_m{}'.format(plM), True, "is the previous planet a FP?")     

                            
    return None                     

    