"""Pipeline

Defines the components of the TERRA pipeline.
"""

import copy
import os

import numpy as np
from numpy import ma
import pandas as pd
from lmfit import minimize, fit_report
import limbdark
import math
import time
from astropy.table import Table

# from terra.utils.hdfstore import HDFStore
# import terra.prepro as prepro
# import terra.tfind as tfind
# import terra.tval as tval

from utils.hdfstore import HDFStore
import prepro as prepro
import tfind as tfind  
import tval as tval

import batman
import emcee
from scipy.stats import norm, uniform, beta, gamma, poisson, binom, binned_statistic
import math
import sys
import exoplanet as xox
import pymc3 as pm
import theano.tensor as tt

import scipy.interpolate as interp
from scipy import special

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from matplotlib.backends.backend_pdf import PdfPages



gravc=6.674e-11*1.98855e30*(86400)**2/(6.957e8)**3

class Pipeline(HDFStore):
    """Initialize a pipeline model.

    The pipeline object itself is just a container object that can easily write
    to hdf5 using pandas. Different codes can perform module operations on the
    pipeline object.
    
    Args:
        lc (Optional[pandas.DataFrame]): Light curve. Must have the 
            following columns: t, f, ferr, fmask. Setting equal to None is 
            done for reading from disk
        header (Optional[dict]): metadata to be stored with the
            pipeline object. At a bare minimum, it must include
            the star name

    Example:
    
        # Working with the pipeline
        >>> pipe = Pipeline(lc=lc, starname='temp',header)
        >>> pipeline.preprocess(pipe) 
        >>> pipeline.grid_search(pipe) 
        >>> pipeline.data_validation(pipe)
        >>> pipeline.fit_transits(pipe)
    """
    lc_required_columns = ['t','f','ferr','fmask']
    pgram_nbins = 2000 # Bin the periodogram down to save storage space
    def __init__(self, lc=None, starname=None, header=None):
        super(Pipeline,self).__init__()
        if type(lc)==type(None):
            return 

        for col in self.lc_required_columns:
            assert list(lc.columns).index(col) >= 0, \
                "light curve lc must contain {}".format(col)

        self.update_header('starname', starname, 'String Star ID')
        for key,value in header.items():
            self.update_header(key, value,'')
        self.update_header('finished_preprocess',False,'preprocess complete?')
        self.update_header('finished_grid_search',False,'grid_serach complete?')
        self.update_header(
            'finished_data_validation',False,'Data validation complete?'
        )
        self.update_table('lc',lc,'light curve')

    def _get_fm(self):
        """Convenience function to return masked flux array"""
        fm = ma.masked_array(
            self.lc.f.copy(), self.lc.fmask.copy(), fill_value=0 )
        fm -= ma.median(fm)
        return fm

def read_hdf(hdffile, group):
    pipe = Pipeline()
    pipe.read_hdf(hdffile, group)
    return pipe

def preprocess(pipe):

    fm = pipe._get_fm()
    isOutlier = prepro.isOutlier(fm, [-1e3,10], interp='constant')
    pipe.lc['isOutlier'] = isOutlier
    pipe.lc['fmask'] = fm.mask | isOutlier | np.isnan(fm.data)
    print(("preprocess: identified {} outliers in the time domain".format(
          isOutlier.sum() )))
    print(("preprocess: {} measurements, {} are masked out".format(
        len(pipe.lc) , pipe.lc['fmask'].sum())))

    pipe.update_header('finished_preprocess',True)
    
    ##### Remove sky points C#0
    sky=np.array([89630, 89631, 89632, 89633, 89634, 89636, 89637, 89638,
       89640, 89664, 89665, 89666, 89667, 89668, 89710, 89711,
       89712, 89714, 89715, 89716, 89724, 89736, 89771, 89772,
       89794, 89795, 89796, 89797, 89806, 89807, 89808, 89809,
       89810, 89811, 89841, 89842, 89928, 89929, 90000, 90083,
       90084, 90085, 90096, 90114, 90115, 90116, 90117, 90118,
       90119, 90120, 90192, 90194, 90195, 90196, 90197, 90240,
       90312, 90313, 90314, 90315, 90389, 90395, 90396, 90398,
       90399, 90408, 90431, 90432, 90433, 90504, 90551, 90595,
       90596, 90597, 90598, 90599, 90600, 90601, 90602, 90672,
       90696, 90697, 90698, 90699, 90785, 90786, 90787, 90788,
       90817, 90903, 90996, 90997, 91021, 91032, 91057, 91078,
       91079, 91080, 91081, 91082, 91083, 91084, 91140, 91141,
       91174, 91175, 91176, 91177, 91178])

              
    newMask=np.zeros(len(pipe.lc['cad']), dtype=bool)      
    for idx in range(len(pipe.lc['cad'])):
        newMask[idx]= pipe.lc['cad'][idx] in sky
        
    pipe.lc['fmask'] = pipe.lc['fmask'] | newMask 
    
    print(("preprocess: identified {} sky masks in the time domain".format(
          len(newMask[newMask]) )))
    
    pipe.update_header('finished_preprocess',True)  
    
    return
    
def gpfit(pipe):
    pmin=5
    y=pipe.lc.f
    x=pipe.lc.t

    m = (pipe.lc.fmask==False) & np.isfinite(x) & np.isfinite(y)

    x = np.ascontiguousarray(x[m], dtype=np.float64)
    y = np.ascontiguousarray(y[m], dtype=np.float64)

    xTrue=x

    divd=np.median(y)
    y=y/np.median(y)

    yTrue=y

    errBin=binned_statistic(x,y,statistic=np.std, bins=round((np.max(x)-np.min(x))/1))[0]
    arr2_interp = interp.interp1d(np.arange(errBin.size),errBin)
    err=errBin[~np.isnan(errBin)]
    yerr=np.median(err)*y/y

    medBin=binned_statistic(x,y,statistic=np.median, bins=round((np.max(x)-np.min(x))/1))[0]
    medBin[np.isnan(medBin)]=1
    arr2_interp = interp.interp1d(np.arange(medBin.size),medBin)
    medStrech = arr2_interp(np.linspace(0,medBin.size-1,y.size))
    x=x[abs(y-medStrech)<3*yerr]
    y=y[abs(y-medStrech)<3*yerr]

    yerr=np.std(y)*y/y
    y=y-1

    for igp in range(2):
        results = xox.estimators.lomb_scargle_estimator(
            x, y, max_peaks=1, min_period=pmin, max_period=50.0,
            samples_per_peak=50)
        try:
            peak = results["peaks"][0]
            freq, power = results["periodogram"]

            with pm.Model() as model:

                # The mean flux of the time series
                mean = pm.Normal("mean", mu=0.0, sd=10.0)

                # A jitter term describing excess white noise
                logs2 = pm.Normal("logs2", mu=2*np.log(yerr[0]), sd=5.0)

                # The parameters of the RotationTerm kernel
                logamp = pm.Normal("logamp", mu=np.log(np.var(y)), sd=5.0)
                logperiod = pm.Normal("logperiod", mu=np.log(peak["period"]), sd=np.log(peak["period_uncert"]))
                logQ0 = pm.Normal("logQ0", mu=1.0, sd=10.0)
                logdeltaQ = pm.Normal("logdeltaQ", mu=2.0, sd=10.0)
                mix = pm.Uniform("mix", lower=0, upper=1.0)

                pipe.update_header('gp_Period',peak["period"])

                # Track the period as a deterministic
                period = pm.Deterministic("period", tt.exp(logperiod))

                # Set up the Gaussian Process model
                try:
                    kernel = xox.gp.terms.RotationTerm(
                        log_amp=logamp,
                        period=period,
                        log_Q0=logQ0,
                        log_deltaQ=logdeltaQ,
                        mix=mix
                    )
                except:
                    time.sleep(5)
                    kernel = xox.gp.terms.RotationTerm(
                        log_amp=logamp,
                        period=period,
                        log_Q0=logQ0,
                        log_deltaQ=logdeltaQ,
                        mix=mix
                    )


                gp = xox.gp.GP(kernel, x, 2*yerr**2 + tt.exp(logs2), J=4)

                # Compute the Gaussian Process likelihood and add it into the
                # the PyMC3 model as a "potential"
                pm.Potential("loglike", gp.log_likelihood(y - mean))

                # Compute the mean model prediction for plotting purposes
                pm.Deterministic("pred", gp.predict())

                # Optimize to find the maximum a posteriori parameters
                map_soln = pm.find_MAP(start=model.test_point)
            if np.isnan(np.sum(map_soln["pred"])):
                pipe.update_header('gp_fit',False)
                pass
            else:
                f = interp.interp1d(x,map_soln["pred"],  fill_value=0.0, bounds_error =False)
                pipe.update_header('gp_fit',True)
                if not os.path.exists('./gpFit/{}/'.format(pipe.starname)):
                    os.makedirs('./gpFit/{}/'.format(pipe.starname))
                with PdfPages('./gpFit/{}/gpFit.pdf'.format(pipe.starname)) as pdf:
                    plt.rcParams.update({'font.size': 16})
                    plt.errorbar(xTrue,yTrue,yerr=np.median(yerr)*xTrue/xTrue,color="black", fmt='o', capsize=5,zorder=1)
                    plt.plot(xTrue,f(xTrue)+1,color="red",zorder=2)
                    plt.title('GP fit')
                    pdf.savefig()
                    plt.close()
                pipe.lc.f[pipe.lc.fmask==False]=pipe.lc.f[pipe.lc.fmask==False]/divd-f(xTrue)
                break    
        except:
            pipe.update_header('gp_fit',False)
        


    return None
    
def grid_search(pipe, P1=0.5, P2=None, tdurRan=None, periodogram_mode='max'): 
    """Run the grid based search

    Args:
        P1 (Optional[float]): Minimum period to search over. Default is 0.5
        P2 (Optional[float]): Maximum period to search over. Default is half 
            the time baseline
        **kwargs : passed to grid.periodogram

    Returns:
        None
    
    """
 
    
    lcdt = pipe.lc.copy()
    
    if type(P2) is type(None):
        P2 = 0.49 * pipe.lc.t.ptp() 
    SNR=10
    plM=1
    plMF=0
    
    pipe.update_header("number_TCE", plMF, "Number of TCE found")
    pipe.update_header('finished_grid_search',True)
        
    while (SNR>=7.1 and plM<=1):
          
         #######fill spacing between jets#####
        if plM==1: 
            t = np.array(pipe.lc.t)
            dt = t[1:] - t[:-1]
            meddt = np.median(dt)
            loca=np.where(dt>meddt+0.0001)[0]

            while len(loca)>0:
                i=0
                var=list(pipe.lc.head())
                try:
                    line = pd.DataFrame({var[0]: pipe.lc[var[0]][loca[i]+1]-meddt,
                     var[1]: pipe.lc[var[1]][loca[i]+1],
                     var[2]: pipe.lc[var[2]][loca[i]+1],
                     var[3]: pipe.lc[var[3]][loca[i]+1],
                     var[4]: pipe.lc[var[4]][loca[i]+1],
                     var[5]: pipe.lc[var[5]][loca[i]+1],
                     var[6]: pipe.lc[var[6]][loca[i]+1],
                     var[7]: pipe.lc[var[7]][loca[i]+1],
                     var[8]: pipe.lc[var[8]][loca[i]+1],
                     var[9]: pipe.lc[var[9]][loca[i]+1]} , index=[loca[i]+1])
                except:
                    line = pd.DataFrame({var[0]: pipe.lc[var[0]][loca[i]+1]-meddt,
                     var[1]: pipe.lc[var[1]][loca[i]+1],
                     var[2]: pipe.lc[var[2]][loca[i]+1],
                     var[3]: pipe.lc[var[3]][loca[i]+1],
                     var[4]: pipe.lc[var[4]][loca[i]+1]}, index=[loca[i]+1])
                
                 # var[10]: pipe.lc[var[10]][loca[i]+1]}, index=[loca[i]+1])
                pipe.lc = pd.concat([pipe.lc[:loca[i]+1], line, pipe.lc[loca[i]+1:]], sort=False).reset_index(drop=True)
                t = np.array(pipe.lc.t)
                dt = t[1:] - t[:-1]
                meddt = np.median(dt)
                loca=np.where(dt>meddt+0.0001)[0]    
        else:
            ##remove higher snr planet
            pipe.lc.f[(t%row.P>=(row.t0%row.P-1.25*row.tdur)) & (t%row.P<=(row.t0%row.P+1.25*row.tdur))]=np.median(pipe.lc.f[pipe.lc.fmask==False])
            
        t = np.array(pipe.lc.t)
        dt = t[1:] - t[:-1]
        fm = pipe._get_fm()


        grid = tfind.Grid(t, fm)
        pipe.update_header('dt',grid.dt,'Exposure time (days)')
        tbase = pipe.lc.t.max() - pipe.lc.t.min()
        
        Pcad1 = P1 / meddt
        Pcad2 = P2 / meddt
        
        pgram_params = [dict(Pcad1=Pcad1, Pcad2=Pcad2, twdG=[tdurRan/meddt])]
        pgram = grid.periodogram(pgram_params,mode='max')
        pgram = pgram.query('P > 0') # cut out candences that failed

        if len(pgram) > pipe.pgram_nbins:
            log10P = np.log10(pgram.P)
            bins = np.logspace(log10P.min(),log10P.max(),pipe.pgram_nbins)
            pgram['Pbin'] = pd.cut(
                pgram.P, bins, include_lowest=True, precision=4,labels=False
                )

            # Take the highest s2n row at each period bin
            pgram = pgram.sort_values(['Pbin','s2n']).groupby('Pbin').last()
            pgram = pgram.reset_index()
        try:
            row = pgram.sort_values('s2n').iloc[-1]
            SNR=row.s2n
            numTr=(t[len(t)-1]-(row.t0))/row.P+1
        except:
            SNR=0
            numTr=0    

        if row.P>3*grid.dt and np.isfinite(SNR):
            plMF+=1
            pipe.update_header("number_TCE", plMF, "Number of TCE found")
            pipe.update_header('grid_s2n_m{}'.format(plM), row.s2n, "Periodogram peak s2n")
            pipe.update_header('grid_P_m{}'.format(plM), row.P, "Periodogram peak period")
            pipe.update_header('grid_t0_m{}'.format(plM), row.t0, "Periodogram peak transit time")
            pipe.update_header('grid_tdur_m{}'.format(plM), row.tdur, "Periodogram peak transit duration")
            #pipe.update_table('pgram',pgram,'periodogram')
            pipe.update_header('finished_grid_search',True)
            
        plM+=1
    
    pipe.lc=lcdt.copy()
    return None
    
    
def find_PhotError(pipe):
    numTce=pipe.number_TCE
    
    lcdt = pipe.lc.copy()
    try:
        lcdt = lcdt[~lcdt.fmask].drop(['fdt_t_rollmed'],axis=1)
    except:
        lcdt = lcdt[~lcdt.fmask].drop(['ferr'],axis=1)

    time_base = 0
    lcdt['t_shift'] = lcdt['t'] - time_base
    t = np.array(lcdt.t_shift)
    f = np.array(lcdt.f)/np.median(lcdt.f)
    
    
    for plMid in range(numTce):
        P = eval("pipe.grid_P_m{}".format(plMid+1))
        t0 = eval("pipe.grid_t0_m{}".format(plMid+1))
        tdur = eval("pipe.grid_tdur_m{}".format(plMid+1))
        
        if t0-tdur>np.min(t) and t0+tdur<np.max(t):
            newMask=np.where((t%P>=((t0-time_base)%P-tdur)) & (t%P<=((t0-time_base)%P+tdur)),True, False)
        else:
            newMask=np.where(((t+P/2)%P>=((t0+P/2)%P-tdur)) & ((t+P/2)%P<=((t0+P/2)%P+tdur)),True, False)    
        t=t[~newMask]
        f=f[~newMask]
    
    bg_err=np.std(f)
    
    pipe.update_header("phot_err", bg_err, "Photometric error")
    
    return None    

        
        
    
    
    
def jon_fit(pipe,plM,ap):
    
    batman_kw = dict(supersample_factor=4, exp_time=1.0/48.0)
    label_transit_kw = dict(cpad=0, cfrac=2)
    local_detrending_kw = dict(
        poly_degree=1, min_continuum=2, label_transit_kw=label_transit_kw
    )
    
    
   # Compute initial parameters. Fits are more robust if we star with
   # transits that are too wide as opposed to to narrow
    try:
        P = eval("pipe.grid_P_m{}".format(plM))
        t0 = eval("pipe.grid_t0_m{}".format(plM))
        tdur = eval("pipe.grid_tdur_m{}".format(plM))
        mES=eval("pipe.grid_s2n_m{}".format(plM))
        
        ###PULL UNCERTAINTY!
    
        apl=(P**2*gravc*pipe.star_mass/(4*np.pi**2))**(1/3)/pipe.star_rad
        ap_err=(((P**2*gravc/(4*math.pi**2))**(1/3)*1/pipe.star_rad*1/3*pipe.star_mass**(-2/3)*pipe.star_umass)**2+((P**2*pipe.star_mass*gravc/(4*math.pi**2))**(1/3)*-1*pipe.star_rad**(-2)*pipe.star_urad)**2)**.5

        if (tdur/P*apl*3.14)>1:
            b=.1
        else:
            b = (1-(tdur/P*apl*3.14)**2)**.5

        if pipe.star_mass==None and pipe.star_rad!=None:
            apl=(P**2*gravc*1/(4*np.pi**2))**(1/3)/pipe.star_rad
            ap_err=(((P**2*gravc/(4*math.pi**2))**(1/3)*1/1*1/3*1**(-2/3)*1)**2+((P**2*1*gravc/(4*math.pi**2))**(1/3)*-1*pipe.star_rad**(-2)*pipe.star_urad)**2)**.5
            b=0.1
               
        elif np.isnan(apl):
            apl=(P**2*gravc*1/(4*np.pi**2))**(1/3)/1
            ap_err=(((P**2*gravc/(4*math.pi**2))**(1/3)*1/1*1/3*1**(-2/3)*1)**2+((P**2*1*gravc/(4*math.pi**2))**(1/3)*-1*1**(-2)*1)**2)**.5
            b=0.1
        else:
            pass


        # Grab data, perform local detrending, and split by tranists.
        lcdt = pipe.lc.copy()
        try:
            lcdt = lcdt[~lcdt.fmask].drop(['fdt_t_rollmed'],axis=1)
        except:
            lcdt = lcdt[~lcdt.fmask].drop(['ferr'],axis=1)

        time_base = (t0-np.min(lcdt.t))-P/2+np.min(lcdt.t)
        lcdt['t_shift'] = lcdt['t'] - time_base
        t = np.array(lcdt.t_shift)
        f = np.array(lcdt.f)/np.median(lcdt.f)
        
        ferr=pipe.phot_err
        
        ####5-sigma clipping
        
        bg_err=pipe.phot_err
        
        if mES>50:
            rp=(1-np.min(f[(t%P>=(t0-time_base)%P-.5*tdur) & (t%P<=(t0-time_base)%P+.5*tdur)]))**.5
        elif t0-time_base-np.min(t)>3*pipe.dt:
            rp=(1-np.mean(f[(t%P>=(t0-time_base)%P-.5*tdur) & (t%P<=(t0-time_base)%P+.5*tdur)]))**.5
        else:
            rp = np.sqrt(abs(eval("pipe.grid_s2n_m{}".format(plM))*np.std(f)))*.95
        
        if rp>.75:
            rp=.75
        elif np.isnan(rp):
            rp=0.02

        
        # # Predictor variable
      #   X1 = np.random.randn(size)
      #   X2 = np.random.randn(size) * 0.2
      #
      #   # Simulate outcome variable
      #   Y = alphaaa + betaaa[0]*X1 + betaaa[1]*X2 + np.random.randn(size)*sigmaaa

        # %matplotlib inline

 #        import pymc3 as pm
 #        print('Running on PyMC3 v{}'.format(pm.__version__))
 #
 #
 #
 #
 #        x=t
 #        y=f
 #
 #        basic_model = pm.Model()
 #
 #        with basic_model:
 #
 #            # a1,a2,a3,a5,a6= theta
 #            # a11=puniform.pdf(a1,np.min(t),(np.max(t)-np.min(t)))
 #            # a33=uniform.pdf(a2,0,2)
 #            # a44=uniform.pdf(a3,0,1)
 #            # a66=uniform.pdf(a5,0,np.max(t))
 #            # a77=uniform.pdf(a6,0,200)
 #            # return(np.log(a11*a33*a44*a66*a77)-(a6-apl)**2/(2*ap_err**2))
 #
 #
 #            # Priors for unknown model parameters
 #            a1 = pm.Uniform('a1', lower=np.min(x), upper=np.max(t))
 #            a2 = pm.Uniform('a2', lower=0, upper=2)
 #            a5 = pm.Uniform('a5', lower=0, upper=40)
 #            a6 = pm.Normal('a6', mu=apl, sd=ap_err)
 #            a3 = pm.Uniform('a3', lower=0, upper=a6)
 #
 #
 #
 #            # Expected value of outcome
 #            x2=(x)%a5
 #
 #            params = batman.TransitParams()
 #
 #            b=a3
 #            params.per = a5                #orbital period
 #            params.t0 = a1                     #time of inferior conjunction
 #            params.rp = a2                      #planet radius (in units of stellar radii)
 #            params.a = a6
 #            try:
 #                       #semi-major axis (in units of stellar radii)
 #                params.inc=math.acos(b/params.a)*180/math.pi  #orbital inclination (in degrees)
 #            except:
 #                params.inc=0
 #
 #            params.ecc = 0                      #eccentricity
 #            params.w = 90                 #longitude of periastron (in degrees)
 #            params.limb_dark = "quadratic"
 #            params.u = pipe.limb_parms       #limb darkening model
 #
 #            m1 = batman.TransitModel(params, np.array(x2),transittype="primary")    #initializes model
 #            flux = m1.light_curve(params)
 #
 #            pm.Normal('Y_obs', mu=flux, sd=bg_err, observed=y)
 #                # guy=-(flux-y)**2/((bg_err)**2)
 # #                return np.sum(guy)
 # #            else:
 # #                return -np.inf
 #            trace = pm.sample(1000)
 #        pm.summary(trace)
 #        pm.summary(trace)["mean"]["a1"]
 #
 #        pipe.update_header('fit_t0_m{}'.format(plM), pm.summary(trace)["mean"]["a1"],  "Best fit transit mid-point" )
 #        pipe.update_header('fit_ut0_m{}'.format(plM), pm.summary(trace)["sd"]["a1"], "Uncertainty")
 #
 #        pipe.update_header('fit_rp_m{}'.format(plM), pm.summary(trace)["mean"]["a2"], "Best fit Rp/Rstar")
 #        pipe.update_header('fit_urp_m{}'.format(plM), pm.summary(trace)["sd"]["a2"], "Uncertainty")
 #
 #        pipe.update_header('fit_b_m{}'.format(plM), pm.summary(trace)["mean"]["a3"], "Best fit impact parameter")
 #        pipe.update_header('fit_ub_m{}'.format(plM), pm.summary(trace)["sd"]["a3"], "Uncertainty")
 #
 #        pipe.update_header('fit_P_m{}'.format(plM), pm.summary(trace)["mean"]["a5"], "Best fit Period parameter")
 #        pipe.update_header('fit_uP_m{}'.format(plM), pm.summary(trace)["sd"]["a5"], "Uncertainty")
 #
 #        pipe.update_header('fit_apl_m{}'.format(plM), pm.summary(trace)["mean"]["a6"], "Best fit Period parameter")
 #        pipe.update_header('fit_uapl_m{}'.format(plM), pm.summary(trace)["sd"]["a6"], "Uncertainty")
 #
 #
 #
        
        
        

        ndim = 5
        nwalkers = 50
        pos_min = np.array([(t0-time_base)*.99999,rp*.90,0,P*.9999,apl*.999])
        pos_max = np.array([(t0-time_base)*1.00001,rp*1.0,b,P*1.0001,apl+10])
        psize = pos_max - pos_min
        pos = [pos_min + psize*np.random.rand(ndim) for i in range(nwalkers)]
        # print("start")
        # print(t0-time_base)
        # print(rp)
        # print(b)
        # print(apl)
        # print("end")

        def lnprior(theta):
            a1,a2,a3,a5,a6= theta
            a11=uniform.pdf(a1,np.min(t),(np.max(t)-np.min(t)))
            a33=uniform.pdf(a2,0,2)
            a44=uniform.pdf(a3,0,1)
            a66=uniform.pdf(a5,0,np.max(t))
            a77=uniform.pdf(a6,0,200)
            return(np.log(a11*a33*a44*a66*a77)-(a6-apl)**2/(2*ap_err**2))


        def lnlike(theta, x, y):
            a1,a2,a3,a5,a6= theta

            x2=(x)%a5

            params = batman.TransitParams()

            b=a3
            params.per = a5                #orbital period
            params.t0 = a1                     #time of inferior conjunction
            params.rp = a2                      #planet radius (in units of stellar radii)
            params.a = a6        #semi-major axis (in units of stellar radii)
            if b<params.a and b>0 and params.a>0: #semi-major axis (in units of stellar radii)
                params.inc =math.acos(b/params.a)*180/math.pi  #orbital inclination (in degrees)
                params.ecc = 0                      #eccentricity
                params.w = 90                 #longitude of periastron (in degrees)
                params.limb_dark = "quadratic"
                params.u = pipe.limb_parms       #limb darkening model

                m1 = batman.TransitModel(params, x2,transittype="primary")    #initializes model
                flux = m1.light_curve(params)


                guy=-(flux-y)**2/((bg_err)**2)
                return np.sum(guy)
            else:
                return -np.inf
    #




        def lnprob(theta, x, y):
            lp = lnprior(theta)
            lk = lnlike(theta,x,y)
            if not np.isfinite(lp):
                return -np.inf
            if not np.isfinite(lk):
                return -np.inf
            return lp + lk

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t, f), threads=1, a=2)

        nburnsteps = 100
        nsteps=50

        width = 1
        print("Burn In")
        result=sampler.run_mcmc(pos, nburnsteps)


        pos,prob,state=result
        sampler.reset()

        ########## perform MCMC
        print("True Distribution")
        result=sampler.run_mcmc(pos, nsteps)



        samples = sampler.flatchain
        samples.shape

        numsamp=len(samples[:,0])

        likelihoodPost=lnprob([np.median(samples[:,0]),np.median(samples[:,1]),np.median(samples[:,2]),np.median(samples[:,3]),np.median(samples[:,4])],t,f)

        p1=np.sort(samples[:,0]+time_base)
        low=np.median(p1)-p1[int(.16*numsamp)]
        high=p1[int((1-.16)*numsamp)]-np.median(p1)

        pipe.update_header('fit_t0_m{}'.format(plM), np.median(p1),  "Best fit transit mid-point" )
        pipe.update_header('fit_ut0_m{}'.format(plM), np.std(p1), "Uncertainty")
        # print ("t0=",(np.median(p1),low,high))
        fit_t0=np.median(p1)

        p1=np.sort(samples[:,1])
        low=np.median(p1)-p1[int(.16*numsamp)]
        high=p1[int((1-.16)*numsamp)]-np.median(p1)

        pipe.update_header('fit_rp_m{}'.format(plM), np.median(p1), "Best fit Rp/Rstar")
        pipe.update_header('fit_urp_m{}'.format(plM), np.std(p1), "Uncertainty")
        # print ("r/R=",(np.median(p1),low,high))
        fit_rp=np.median(p1)

        p1=np.sort(samples[:,2])
        low=np.median(p1)-p1[int(.16*numsamp)]
        high=p1[int((1-.16)*numsamp)]-np.median(p1)

        pipe.update_header('fit_b_m{}'.format(plM), np.median(p1), "Best fit impact parameter")
        pipe.update_header('fit_ub_m{}'.format(plM), np.std(p1), "Uncertainty")
        # print ("b=",(np.median(p1),low,high))
        fit_b=np.median(p1)

        p1=np.sort(samples[:,3])
        low=np.median(p1)-p1[int(.16*numsamp)]
        high=p1[int((1-.16)*numsamp)]-np.median(p1)

        pipe.update_header('fit_P_m{}'.format(plM), np.median(p1), "Best fit Period parameter")
        pipe.update_header('fit_uP_m{}'.format(plM), np.std(p1), "Uncertainty")
        # print ("Per=",(np.median(p1),low,high))
        fit_P=np.median(p1)

        p1=np.sort(samples[:,4])
        low=np.median(p1)-p1[int(.16*numsamp)]
        high=p1[int((1-.16)*numsamp)]-np.median(p1)

        pipe.update_header('fit_apl_m{}'.format(plM), np.median(p1), "Best fit Period parameter")
        pipe.update_header('fit_uapl_m{}'.format(plM), np.std(p1), "Uncertainty")
        # print ("Per=",(np.median(p1),low,high))
        fit_apl=np.median(p1)
        
        p1=np.arcsin(((1+fit_rp)**2-fit_b**2)**.5/fit_apl)*fit_P/np.pi
        # low=np.median(p1)-p1[int(.16*numsamp)]
        # high=p1[int((1-.16)*numsamp)]-np.median(p1)

        pipe.update_header('fit_tdur_m{}'.format(plM), p1, "Best fit Period parameter")
        #pipe.update_header('fit_utdur_m{}'.format(plM), np.std(p1), "Uncertainty")
        # print ("Per=",(np.median(p1),low,high))
        fit_tdur=p1
        
        if np.isnan(fit_tdur) or np.isinf(fit_tdur):
            fit_tdur=tdur
            


        #############   Run EDI-Vetter   ########
        ############# Exoplanet Detection Indicator - Vetter ######
        pipe.update_header('falsePos_m{}'.format(plM), False, "is it a FP?")
        print(""" _______  ______  _________              _______ __________________ _______  _______
    (  ____ \(  __  \ \__   __/    |\     /|(  ____ \\__   __/\__   __/(  ____ \(  ____ )
    | (    \/| (  \  )   ) (       | )   ( || (    \/   ) (      ) (   | (    \/| (    )|
    | (__    | |   ) |   | | _____ | |   | || (__       | |      | |   | (__    | (____)|
    |  __)   | |   | |   | |(_____)( (   ) )|  __)      | |      | |   |  __)   |     __)
    | (      | |   ) |   | |        \ \_/ / | (         | |      | |   | (      | (\ (
    | (____/\| (__/  )___) (___      \   /  | (____/\   | |      | |   | (____/\| ) \ \__
    (_______/(______/ \_______/       \_/   (_______/   )_(      )_(   (_______/|/   \__/
                                                                                          """)

        previous_pl_check(pipe,plM)
        centroid(pipe,plM)
        individual_transits(pipe,plM,time_base)
        even_odd_transit(pipe,plM,time_base)
        uniqueness_test(pipe,plM,time_base)
        ephemeris_wonder(pipe,plM)
        check_SE(pipe,plM,time_base)
        too_big_planet(pipe,plM)
        harmonic_test(pipe,plM,time_base)
        # individual_transits(pipe,plM,time_base)
        period_alias(pipe,plM,time_base)
        phase_coverage(pipe,plM,time_base)
        tdur_max(pipe,plM,time_base)

    
        ############# Plot Transit ##################
    
        tfold=t%eval("pipe.fit_P_m{}".format(plM))
 
        params = batman.TransitParams()

        b=eval("pipe.fit_b_m{}".format(plM))
        params.per = eval("pipe.fit_P_m{}".format(plM))                #orbital period
        params.t0 = (eval("pipe.fit_t0_m{}".format(plM))-time_base)%params.per                  #time of inferior conjunction
        params.rp = eval("pipe.fit_rp_m{}".format(plM))                      #planet radius (in units of stellar radii)
        params.a = fit_apl        #semi-major axis (in units of stellar radii)
        params.inc =math.acos(b/params.a)*180/math.pi  #orbital inclination (in degrees)
        params.ecc = 0                      #eccentricity
        params.w = 90                 #longitude of periastron (in degrees)
        params.limb_dark = "quadratic"
        params.u = pipe.limb_parms   #limb darkening model

        rang=np.linspace(0,params.per, num=10000)

        m = batman.TransitModel(params, rang)    #initializes model
        flux = m.light_curve(params)

        if not os.path.exists('./TCE/{}/'.format(pipe.starname)):
            os.makedirs('./TCE/{}/'.format(pipe.starname))

        with PdfPages('./TCE/{}/m{}.pdf'.format(pipe.starname,plM)) as pdf:
            plt.rcParams.update({'font.size': 16})
            plt.errorbar(tfold,f,yerr=bg_err,color="black", fmt='o', capsize=5)
            plt.plot(rang,flux,color="red")
            plt.ylim(1-2*params.rp**2,1+3*bg_err)
            plt.xlim(params.t0-2*fit_tdur,params.t0+2*fit_tdur)
            plt.title('Transit')
            pdf.savefig()
            plt.close()

            #secondary_eclipse_search(pipe,plM)
            tfoldse=(t+params.per/2)%eval("pipe.fit_P_m{}".format(plM))

            plt.rcParams.update({'font.size': 16})
            plt.errorbar(tfoldse,f,yerr=bg_err,color="black", fmt='o', capsize=5)
            plt.ylim(1-2*params.rp**2,1+3*bg_err)
            plt.xlim(params.t0-2*fit_tdur,params.t0+2*fit_tdur)
            plt.title('Secondary Eclipse')
            pdf.savefig()
            plt.close()

            plt.rcParams.update({'font.size': 16})
            plt.scatter(tfold,f, color="black")
            plt.ylim(1-4*params.rp**2,1+3*bg_err)
            plt.xlim(0,params.per)
            plt.title('Full Folding')
            pdf.savefig()
            plt.close()
    except AttributeError:
        pipe.update_header('falsePos_m{}'.format(plM), False, "is it a FP?")    
    return None
    
    
    
def centroid(pipe,plM):
    if pipe.delta_dist_Gaia>(np.sqrt(pipe.photo_ap/np.pi)+1)*3.98 or pipe.delta_mag_Gaia>5:
        if pipe.delta_dist>(np.sqrt(pipe.photo_ap/np.pi)+1)*3.98 or pipe.delta_mag>3:
            pipe.update_header('centroidOffset_m{}'.format(plM), False, "something nearby?")
        else:
            pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")
            pipe.update_header('centroidOffset_m{}'.format(plM), True, "something nearby?")
    else:
        pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")
        pipe.update_header('centroidOffset_m{}'.format(plM), True, "something nearby?")
    return None    
        
                 
    
    
    ########################################
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

def phase_coverage(pipe,plM,time_base):
    fit_P=eval("pipe.fit_P_m{}".format(plM))
    fit_t0=eval("pipe.fit_t0_m{}".format(plM))
    fit_tdur=eval("pipe.fit_tdur_m{}".format(plM))
    
    lcdt = pipe.lc.copy()
    try:
        lcdt = lcdt[~lcdt.fmask].drop(['fdt_t_rollmed'],axis=1)
    except:
        lcdt = lcdt[~lcdt.fmask].drop(['ferr'],axis=1)    
      
    lcdt['t_shift'] = lcdt['t'] - time_base
    t = np.array(lcdt.t_shift)
    t=t%fit_P

    
    newMask=np.where((t>=((fit_t0-time_base)%fit_P-1*fit_tdur)) & (t<=((fit_t0-time_base)%fit_P+1*fit_tdur)),False, True)
    newT=t[~newMask]
    newT=np.sort(newT)
    dt = newT[1:] - newT[:-1]
    
    #allowed gap
    def allow_tol(x):
        func=(16*x**4-8*x**2+2)*pipe.dt
        return func
        
    xVal=(newT[:-1]+dt/2-(fit_t0-time_base)%fit_P)/fit_tdur
    
    allow=allow_tol(xVal)
    
    if np.any(np.where(dt>=allow,True,False)):
        pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")
        pipe.update_header('spacingFP_m{}'.format(plM), True, "large data gap found")
    else:
        pipe.update_header('spacingFP_m{}'.format(plM), False, "large data gap found")
    return None    
              

def even_odd_transit(pipe,plM,time_base):
    
     def add_phasefold(pipe,plM):
        return tval.add_phasefold(pipe.lc, pipe.lc.t, eval("pipe.fit_P_m{}".format(plM)), eval("pipe.fit_t0_m{}".format(plM)),plM)

     fit_P=eval("pipe.fit_P_m{}".format(plM))
     fit_t0=eval("pipe.fit_t0_m{}".format(plM))
     fit_rp=eval("pipe.fit_rp_m{}".format(plM))
     fit_b=eval("pipe.fit_b_m{}".format(plM))
     fit_apl=eval("pipe.fit_apl_m{}".format(plM))
     fit_tdur=eval("pipe.fit_tdur_m{}".format(plM))
     apl=(fit_P**2*gravc*pipe.star_mass/(4*np.pi**2))**(1/3)/pipe.star_rad
     ap_err=(((fit_P**2*gravc/(4*math.pi**2))**(1/3)*1/pipe.star_rad*1/3*pipe.star_mass**(-2/3)*pipe.star_umass)**2+((fit_P**2*pipe.star_mass*gravc/(4*math.pi**2))**(1/3)*-1*pipe.star_rad**(-2)*pipe.star_urad)**2)**.5
     mES=eval("pipe.grid_s2n_m{}".format(plM))
     
     lc = add_phasefold(pipe,plM)

     pipe.update_table('lc', lc)
     
     lcdt = pipe.lc.copy()
     try:
         lcdt = lcdt[~lcdt.fmask].drop(['fdt_t_rollmed'],axis=1)
     except:
         lcdt = lcdt[~lcdt.fmask].drop(['ferr'],axis=1)    
       
     lcdt['t_shift'] = lcdt['t'] - time_base
     t = np.array(lcdt.t_shift)
     f = np.array(lcdt.f)/np.median(lcdt.f)
     ferr=pipe.phot_err
        ####5-sigma clipping
     bg_err=pipe.phot_err
    
     feven=f[eval('pipe.lc.cycle_m{}'.format(plM))[pipe.lc.fmask==False]%2==0]
     teven=t[eval('pipe.lc.cycle_m{}'.format(plM))[pipe.lc.fmask==False]%2==0]
     
     
     evenMask=np.where((teven%fit_P>=((fit_t0-time_base)%fit_P-.5*fit_tdur)) & (teven%fit_P<=((fit_t0-time_base)%fit_P+.5*fit_tdur)),True, False)
     
     try:
         if mES>50:
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
     nwalkers = 50
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

         x2=(x)%fit_P

         params = batman.TransitParams()

         b=a3
         params.per = fit_P                #orbital period
         params.t0 = a1                     #time of inferior conjunction
         params.rp = a2                      #planet radius (in units of stellar radii)
         params.a = a6        #semi-major axis (in units of stellar radii)
         if b<params.a and b>0 and params.a>0: #semi-major axis (in units of stellar radii)
             params.inc =math.acos(b/params.a)*180/math.pi  #orbital inclination (in degrees)
             params.ecc = 0                      #eccentricity
             params.w = 90                 #longitude of periastron (in degrees)
             params.limb_dark = "quadratic"
             params.u = pipe.limb_parms       #limb darkening model

             m1 = batman.TransitModel(params, x2,transittype="primary")    #initializes model
             flux = m1.light_curve(params)


             guy=-(flux-y)**2/((bg_err)**2)
             return np.sum(guy)
         else:
             return -np.inf
 #

     def lnprobRR(theta, x, y):
         lp = lnpriorRR(theta)
         lk = lnlikeRR(theta,x,y)
         if not np.isfinite(lp):
             return -np.inf
         if not np.isfinite(lk):
             return -np.inf
         return lp + lk

     sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobRR, args=(teven, feven), threads=1, a=2)

     nburnsteps = 50
     nsteps=25  

     width = 1
     print("Burn In")
     result=sampler.run_mcmc(pos, nburnsteps)

     pos,prob,state=result
     sampler.reset()

     ########## perform MCMC
     print("True Distribution")
     result=sampler.run_mcmc(pos, nsteps)

     samples = sampler.flatchain
     samples.shape

     numsamp=len(samples[:,0])
     
     p1=np.sort(samples[:,0]+time_base)

     pipe.update_header('fit_t0_even', np.median(p1), "Best fit t0")
     pipe.update_header('fit_ut0_even', np.std(p1), "Uncertainty")
     # print ("t0=",(np.median(p1),low,high))

     p1=np.sort(samples[:,1])

     pipe.update_header('fit_rp_even', np.median(p1), "Best fit Rp/Rstar")
     pipe.update_header('fit_urp_even', np.std(p1), "Uncertainty")
     

            
     
    
     # # Perform global fit. Set some common-sense limits on parameters
  #
  #
  #    tm = tval.TransitModel(fit_P+.01, t0Even, rpEven, fit_apl, fit_b, pipe.limb_parms[0], pipe.limb_parms[1], )
  #    tm.lm_params['rp'].min = 0.0
  #    tm.lm_params['rp'].max = 3.0 *fit_rp
  #    tm.lm_params['b'].min = 0.0
  #    tm.lm_params['b'].max = 1.0
  #    tm.lm_params['apl'].min = 0.0
  #    tm.lm_params['apl'].max = 200.0
  #    tm.lm_params['t0'].min = t0Even - fit_tdur
  #    tm.lm_params['t0'].max = t0Even + fit_tdur
  #    tm.lm_params['per'].min = fit_P - .01
  #    tm.lm_params['per'].max = fit_P + .01
  #
  #    tm_initial = copy.deepcopy(tm)
  #
  #    method = 'leastsq'
  #    # tm.lm_params.pretty_print()
  #
  #    out = minimize(tm.residual, tm.lm_params, args=(teven, feven, ferr), method=method)
  #
  #    # Store away best fit parameters
  #    par = out.params
  #    t0 = par['t0'].value + time_base
  #    pipe.update_header('fit_rp_even', par['rp'].value, "Best fit Rp/Rstar")
  #    pipe.update_header('fit_urp_even', par['rp'].stderr, "Uncertainty")
  #    pipe.update_header('fit_rchisq_even', out.redchi, "Reduced Chi-squared")
  #
  #    tm_global = copy.deepcopy(tm)
  #    tm_global.lm_params = out.params
    
     fodd=f[eval('pipe.lc.cycle_m{}'.format(plM))[pipe.lc.fmask==False]%2==1]
     todd=t[eval('pipe.lc.cycle_m{}'.format(plM))[pipe.lc.fmask==False]%2==1]
     
     oddMask=np.where((todd%fit_P>=((fit_t0-time_base)%fit_P-.5*fit_tdur)) & (todd%fit_P<=((fit_t0-time_base)%fit_P+.5*fit_tdur)),True, False)
     
     try:
         if mES>50:
             t0Odd=todd[fodd==np.min(fodd[evenMask])]
             rpOdd=np.sqrt(1-np.min(fodd[oddMask]))
             t0Odd=t0Odd[0]
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

     print("Burn In")
     result=sampler.run_mcmc(pos, nburnsteps)

     pos,prob,state=result
     sampler.reset()

     ########## perform MCMC
     print("True Distribution")
     result=sampler.run_mcmc(pos, nsteps)

     samples = sampler.flatchain
     samples.shape

     numsamp=len(samples[:,0])
     
     p1=np.sort(samples[:,0]+time_base)

     pipe.update_header('fit_t0_odd', np.median(p1), "Best fit t0")
     pipe.update_header('fit_ut0_odd', np.std(p1), "Uncertainty")
     # print ("t0=",(np.median(p1),low,high))

     p1=np.sort(samples[:,1])

     pipe.update_header('fit_rp_odd', np.median(p1), "Best fit Rp/Rstar")
     pipe.update_header('fit_urp_odd', np.std(p1), "Uncertainty")
     
     
     
     MesMask=np.where((t%fit_P>=((fit_t0-time_base)%fit_P-.15*fit_tdur)) & (t%fit_P<=((fit_t0-time_base)%fit_P+.15*fit_tdur)),True, False)
     rpMes=np.std(f[MesMask])



     
           #
     #
     # # Perform global fit. Set some common-sense limits on parameters
     # tm = tval.TransitModel(fit_P, t0Odd, rpOdd, fit_apl, fit_b, pipe.limb_parms[0], pipe.limb_parms[1], )
     # tm.lm_params['rp'].min = 0.0
     # tm.lm_params['rp'].max = 3.0 * fit_rp
     # tm.lm_params['b'].min = 0.0
     # tm.lm_params['b'].max = 1.0
     # tm.lm_params['apl'].min = 0.0
     # tm.lm_params['apl'].max = 200.0
     # tm.lm_params['t0'].min = t0Odd - fit_tdur
     # tm.lm_params['t0'].max = t0Odd + fit_tdur
     # tm.lm_params['per'].min = fit_P - .01
     # tm.lm_params['per'].max = fit_P + .01
     #
     # tm_initial = copy.deepcopy(tm)
     #
     # method = 'leastsq'
     # # tm.lm_params.pretty_print()
     #
     # out = minimize(tm.residual, tm.lm_params, args=(todd, fodd, ferr), method=method)
     #
     # # Store away best fit parameters
     # par = out.params
     # t0 = par['t0'].value + time_base
     # pipe.update_header('fit_rp_odd', par['rp'].value, "Best fit Rp/Rstar")
     # pipe.update_header('fit_urp_odd', par['rp'].stderr, "Uncertainty")
     # pipe.update_header('fit_rchisq_odd', out.redchi, "Reduced Chi-squared")
     #
     # tm_global = copy.deepcopy(tm)
     # tm_global.lm_params = out.params
    
     if (pipe.fit_urp_odd==None):
         pipe.fit_urp_odd=pipe.fit_rp_odd  
     if (pipe.fit_urp_even==None):
         pipe.fit_urp_even=pipe.fit_rp_even
     if (pipe.fit_ut0_odd==None):
         pipe.fit_ut0_odd=pipe.fit_t0_odd  
     if (pipe.fit_ut0_even==None):
         pipe.fit_ut0_even=pipe.fit_t0_even      
        
     if abs(pipe.fit_rp_even-pipe.fit_rp_odd)>5*np.sqrt(pipe.fit_urp_odd**2+pipe.fit_urp_even**2):
         pipe.update_header('even_odd_transit_misfit_m{}'.format(plM), True, "Fold on secondary (1)")
         pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")
         
     elif abs(pipe.fit_t0_even-pipe.fit_t0_odd)>5*np.sqrt(pipe.fit_ut0_odd**2+pipe.fit_ut0_even**2):
         pipe.update_header('even_odd_transit_misfit_m{}'.format(plM), True, "Fold on secondary (2)")
         pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")
     elif rpMes>5*bg_err:
         pipe.update_header('even_odd_transit_misfit_m{}'.format(plM), True, "Fold on secondary (3)")
         pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")     
                   
     # elif abs(pipe.fit_rp_even-pipe.fit_rp_odd)<bg_err:
     #     pipe.update_header('even_odd_transit_misfit_m{}'.format(plM), False, "Fold on secondary")
     
     else:
         pipe.update_header('even_odd_transit_misfit_m{}'.format(plM), False, "Fold on secondary")
     
     return None
     
def uniqueness_test(pipe,plM,time_base):
    
    fit_tdur=eval("pipe.fit_tdur_m{}".format(plM))
    fit_tdurGrid=eval("pipe.grid_tdur_m{}".format(plM))
    fit_P=eval("pipe.fit_P_m{}".format(plM))
    fit_t0=eval("pipe.fit_t0_m{}".format(plM))
    fit_rp=eval("pipe.fit_rp_m{}".format(plM))
    fit_b=eval("pipe.fit_b_m{}".format(plM))
    fit_uP=eval("pipe.fit_uP_m{}".format(plM))
    #mES=eval("pipe.grid_s2n_m{}".format(plM))
    apl=(fit_P**2*gravc*pipe.star_mass/(4*np.pi**2))**(1/3)/pipe.star_rad

    lcdt = pipe.lc.copy()
    try:
        lcdt = lcdt[~lcdt.fmask].drop(['fdt_t_rollmed'],axis=1)
    except:
        lcdt = lcdt[~lcdt.fmask].drop(['ferr'],axis=1)    
   
    lcdt['t_shift'] = lcdt['t'] - time_base
    t = np.array(lcdt.t_shift)
    f = np.array(lcdt.f)/np.median(lcdt.f)
    ferr=pipe.phot_err
       ####5-sigma clipping
    bg_err=pipe.phot_err
    sigUniq=np.ones(3)
    
    newMask3=np.where((t%fit_P>=((fit_t0-time_base)%fit_P-.5*fit_tdurGrid)) & (t%fit_P<=((fit_t0-time_base)%fit_P+.5*fit_tdurGrid)),True, False)
    newTrueF=f[newMask3]
    tranDepth=1-np.mean(newTrueF)
    
    
    
    # mu=np.sqrt(1-fit_b**2)
#     tranDepth=fit_rp**2*(1-pipe.limb_parms[0]*(1-mu)-pipe.limb_parms[1]*(1-mu)**2)/(1-pipe.limb_parms[0]/3-pipe.limb_parms[1]/6)
    
    x=t
    y=f
    
    # errBin=binned_statistic(x,y,statistic=np.std, bins=round((np.max(x)-np.min(x))/numBin))[0]
    # arr2_interp = interp.interp1d(np.arange(errBin.size),errBin)
    # err=errBin[~np.isnan(errBin)]
    # yerr=np.median(err)*y/y
    
    # def gausKern(x,xi,length):
    #     return(np.exp(-(x-xi)**2/(2*length**2)))
    #
    # guyt=np.ones(len(t))
    #
    # params = batman.TransitParams()
    # b=eval("pipe.fit_b_m{}".format(plM))
    # params.per = eval("pipe.fit_P_m{}".format(plM))                #orbital period
    # params.t0 = (eval("pipe.fit_t0_m{}".format(plM))-time_base)                  #time of inferior conjunction
    # params.rp = eval("pipe.fit_rp_m{}".format(plM))                      #planet radius (in units of stellar radii)
    # params.a = apl        #semi-major axis (in units of stellar radii)
    # params.inc =math.acos(b/params.a)*180/math.pi  #orbital inclination (in degrees)
    # params.ecc = 0                      #eccentricity
    # params.w = 90                 #longitude of periastron (in degrees)
    # params.limb_dark = "quadratic"
    # params.u = pipe.limb_parms   #limb darkening model
    #
    #
    # m = batman.TransitModel(params, t)    #initializes model
    # flux = m.light_curve(params)-1
    #
    # fproc=f-flux
    #
    #
    # for ti in range(len(t)):
    #     guyt[ti]=np.sum(gausKern(t[ti],t,fit_tdur)*fproc)/np.sum(gausKern(t[ti],t,fit_tdur))

    # medBin=binned_statistic(x,y,statistic=np.median, bins=round((np.max(x)-np.min(x))/(fit_tdur/2)))[0]
#     medBin=medBin[~np.isnan(medBin)]
#
#     medBin=medBin[abs(medBin-np.median(medBin))<3*np.std(medBin)]
#
#     F_red=np.std(medBin)/bg_err
    
    newT=t
    newF=f
    meddt=pipe.dt
    newMask=np.zeros((len(newF)), dtype=bool)
    
    # dt = newT[1:] - newT[:-1]
    # loca=np.where(dt>meddt+0.0001)[0]
    #
    # while len(loca)>0:
    #     i=0
    #     newT=np.concatenate((newT[:loca[i]+1],newT[loca[i]+1]-meddt,newT[loca[i]+1:]), axis=None)
    #     newF=np.concatenate((newF[:loca[i]+1],1,newF[loca[i]+1:]), axis=None)
    #     newMask=np.concatenate((newMask[:loca[i]+1],True,newMask[loca[i]+1:]), axis=None)
    #     dt = newT[1:] - newT[:-1]
    #     meddt = np.median(dt)
    #     loca=np.where(dt>meddt+0.0001)[0]
    
    # fm = ma.masked_array(newF,mask=newMask, fill_value=0)
    # fm -= ma.median(fm)


    # grid = tfind.Grid(newT, fm)
    # Pcad1 = (fit_P - fit_uP)/meddt
    # Pcad2 = (fit_P + fit_uP)/meddt
    # twd = fit_tdurGrid/meddt


    # pgram_params = [dict(Pcad1=Pcad1, Pcad2=Pcad2, twdG=[twd])]
 #    try:
 #        pgram = grid.periodogram(pgram_params,mode='max')
 #        row = pgram.sort_values('s2n').iloc[-1]
 #        if row.s2n>0:
 #            mES=row.s2n
 #        else:
 #            mES=eval("pipe.grid_s2n_m{}".format(plM))
 #    except:
 #       mES=eval("pipe.grid_s2n_m{}".format(plM))
     
    
   # newF[(newT%fit_P>=((fit_t0-time_base)%fit_P-.5*fit_tdur)) & (newT%fit_P<=((fit_t0-time_base)%fit_P+.5*fit_tdur))]=np.median(newF)
    if 3*fit_tdur/fit_P>.10:
        newMask=np.where((newT%fit_P>=((fit_t0-time_base)%fit_P-1*fit_tdur)) & (newT%fit_P<=((fit_t0-time_base)%fit_P+1*fit_tdur)),True, False)
        newMask=np.where((newT%fit_P>=((fit_t0-time_base)%fit_P+fit_P/2-1*fit_tdur)) & (newT%fit_P<=((fit_t0-time_base)%fit_P+fit_P/2+1*fit_tdur)),True, newMask)
    else:    
        newMask=np.where((newT%fit_P>=((fit_t0-time_base)%fit_P-1.5*fit_tdur)) & (newT%fit_P<=((fit_t0-time_base)%fit_P+1.5*fit_tdur)),True, False)
        newMask=np.where((newT%fit_P>=((fit_t0-time_base)%fit_P+fit_P/2-1.5*fit_tdur)) & (newT%fit_P<=((fit_t0-time_base)%fit_P+fit_P/2+1.5*fit_tdur)),True, newMask)
    flux_red=newF[~newMask]
    time_red=newT[~newMask]
    
    try:
        minBin=np.zeros(len(time_red))
        
        for intt in range(len(time_red)):
            if time_red[intt]%fit_P<np.min(time_red%fit_P)+.5*fit_tdurGrid:
                pass
            elif time_red[intt]%fit_P>np.max(time_red)%fit_P+.5*fit_tdurGrid:
                pass
            else:
                binMed=np.where((time_red%fit_P<=time_red[intt]%fit_P+.5*fit_tdurGrid) & (time_red%fit_P>=time_red[intt]%fit_P-.5*fit_tdurGrid),flux_red,0)
                minBin[intt]=np.mean(binMed[binMed>0])  
        
        minBin=minBin[minBin>0]
                    
        # medBin=binned_statistic(time_red,flux_red,statistic=np.mean, bins=round((np.max(x)-np.min(x))/(fit_tdur)))[0]
        # medBinsd=binned_statistic(time_red,flux_red,statistic=np.std, bins=round((np.max(x)-np.min(x))/(fit_tdur)))[0]
        # medBin=medBin[~np.isnan(medBin)]
        # medBinsd=medBinsd[~np.isnan(medBinsd)]
        bg_e=np.std(flux_red)
        
        if bg_e<bg_err:
            pass
        else:
            bg_e=bg_err 
               
        F_red=np.std(minBin)/bg_e

    except:
        bg_e=np.std(flux_red)
        if bg_e<bg_err:
            pass
        else:
            bg_e=bg_err
            
        F_red=1

                
    mES=tranDepth/bg_e    
    
    if 3*fit_tdur/fit_P>.10:
        newMask=np.where((newT%fit_P>=((fit_t0-time_base)%fit_P-1*fit_tdur)) & (newT%fit_P<=((fit_t0-time_base)%fit_P+1*fit_tdur)),True, False)
    else:    
        newMask=np.where((newT%fit_P>=((fit_t0-time_base)%fit_P-1.5*fit_tdur)) & (newT%fit_P<=((fit_t0-time_base)%fit_P+1.5*fit_tdur)),True, False)
    
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
                pipe.update_header('fit_t0_se', row.t0+time_base, "SE t0")
                newMask=np.where((newT%row.P>=((row.t0)%row.P-1.5*row.tdur)) & (newT%row.P<=((row.t0)%row.P+1.5*row.tdur)),True, newMask)
            except:
                pipe.update_header('fit_t0_se', fit_t0+fit_P/2, "SE t0")
                newMask=np.where((newT%fit_P>=((fit_t0-time_base+fit_P/2)%fit_P-1.5*fit_tdur)) & (newT%fit_P<=((fit_t0-time_base+fit_P/2)%fit_P+1.5*fit_tdur)),True, newMask)
                
        elif idxx==2:
            newF=-newF
                        
        fm = ma.masked_array(newF,mask=newMask, fill_value=0)
        fm -= ma.median(fm)


        grid = tfind.Grid(newT, fm)
        Pcad1 = (fit_P - fit_uP)/meddt
        Pcad2 = (fit_P + fit_uP)/meddt
        twd = fit_tdurGrid/meddt


        pgram_params = [dict(Pcad1=Pcad1, Pcad2=Pcad2, twdG=[twd])]
        try:
            pgram = grid.periodogram(pgram_params,mode='max')
            row = pgram.sort_values('s2n').iloc[-1]
            if row.s2n>0:
                newT2=newT[~newMask]
                newF2=newF[~newMask]
                newMask2=np.where((newT2%row.P>=((row.t0)%row.P-.5*row.tdur)) & (newT2%row.P<=((row.t0)%row.P+.5*row.tdur)),True, False)
                Fstuff=np.mean(newF2[newMask2])*-1
                print(np.mean(newF2[newMask2]))
                sigUniq[idxx]=Fstuff/bg_e
            else:
                sigUniq[idxx]=0
        except:
            sigUniq[idxx]=0
    
    fA_1=np.sqrt(2)*special.erfcinv(fit_tdur/fit_P/10000)
    fA_2=np.sqrt(2)*special.erfcinv(fit_tdur/fit_P)
    
    if np.isnan(fA_1):
        fA_1=4
        
    # print("FA")
    # print(fA_1)
    # print(mES)
    # print(F_red)
    # print(bg_e)
    # print(bg_err)
    # print(sigUniq)
    
    
    mS1=fA_1-mES/F_red
    mS2=fA_2-(mES-sigUniq[1])
    mS3=fA_2-(mES-sigUniq[2])
    
    mS4=sigUniq[0]/F_red-fA_1
    mS5=(sigUniq[0]-sigUniq[1])-fA_2
    mS6=(sigUniq[0]-sigUniq[2])-fA_2
    
    ms7=(mES-sigUniq[0])-fA_2
    
    phaseSE1=(pipe.fit_t0_se-(fit_t0+fit_P/2))/fit_tdur
    phaseSE2=(pipe.fit_t0_se-(fit_t0-fit_P/2))/fit_tdur
    phaseSE=np.min([phaseSE1,phaseSE2])
        
            
    pipe.update_header('SE_found_m{}'.format(plM), False, "EB or hot Jup like")
    
    if mS1>0.5:
        pipe.update_header('uniqueness_m{}'.format(plM), True, "Transit !uniq. to the signal (1)")
        pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")
    elif mS2>1.5: 
        pipe.update_header('uniqueness_m{}'.format(plM), True, "Transit !uniq. to the signal (2)")
        pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")
    elif mS3>2.5:
        pipe.update_header('uniqueness_m{}'.format(plM), True, "Transit !uniq. to the signal (3)")
        pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")
    else:    
        pipe.update_header('uniqueness_m{}'.format(plM), False, "Transit !uniq. to the signal")
    
    if mS4>0.5 and mS5>-0.5 and mS6>-0.5:
        if ms7>1 and phaseSE<=1:
            pipe.update_header('SE_found_m{}'.format(plM), True, "EB or hot Jup like")
        else:
            pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")
            pipe.update_header('SE_found_m{}'.format(plM), True, "EB or hot Jup like")
             
    
    return None    
           
###TDUR>.5        
def tdur_max(pipe,plM,time_base):
    fit_tdur=eval("pipe.fit_tdur_m{}".format(plM))
    fit_P=eval("pipe.fit_P_m{}".format(plM))
    mES=eval("pipe.grid_s2n_m{}".format(plM))

    if fit_tdur/fit_P>.1:
        pipe.update_header('longTD_m{}'.format(plM), True, "Transit Duration dominates")
        pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")
    else:
        pipe.update_header('longTD_m{}'.format(plM), False, "Transit Duration dominates")
    return None    
            
        

         
    
def ephemeris_wonder(pipe,plM):
    fit_tdur=eval("pipe.fit_tdur_m{}".format(plM))
    fit_t0=eval("pipe.fit_t0_m{}".format(plM))
    grid_t0=eval("pipe.grid_t0_m{}".format(plM))
    
    if abs(grid_t0-fit_t0)>0.5*fit_tdur:
        pipe.update_header('eph_slip_m{}'.format(plM), True, "ephemeris wandering")
        pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")
    else:
         pipe.update_header('eph_slip_m{}'.format(plM), False, "ephemeris wandering")
    
    return None        
    
     
def check_SE(pipe,plM,time_base):
    
    if eval('pipe.SE_found_m{}'.format(plM))=="True":
        
        lcdt = pipe.lc.copy()
        try:
            lcdt = lcdt[~lcdt.fmask].drop(['fdt_t_rollmed'],axis=1)
        except:
            lcdt = lcdt[~lcdt.fmask].drop(['ferr'],axis=1)    

        lcdt['t_shift'] = lcdt['t'] - time_base
        t = np.array(lcdt.t_shift)
        f = np.array(lcdt.f)/np.median(lcdt.f)
        ferr=f
           ####5-sigma clipping
        bg_err=pipe.phot_err
        fse=f
        tse=t

        # Perform global fit. Set some common-sense limits on parameters
        fit_P=eval("pipe.fit_P_m{}".format(plM))
        fit_t0=eval("pipe.fit_t0_m{}".format(plM))
        fit_rp=eval("pipe.fit_rp_m{}".format(plM))
        fit_b=eval("pipe.fit_b_m{}".format(plM))
        fit_apl=eval("pipe.fit_apl_m{}".format(plM))
        fit_tdur=eval("pipe.fit_tdur_m{}".format(plM))

        if abs((pipe.fit_t0_se)+.5*fit_P-fit_t0)<=fit_P/10:
            tm = tval.TransitModel(fit_P, pipe.fit_t0_se - time_base, fit_rp*.1, fit_apl, fit_b, pipe.limb_parms[0], pipe.limb_parms[1], )
        else: 
            tm = tval.TransitModel(fit_P, fit_t0 - time_base+.5*fit_P, fit_rp*.1, fit_apl, fit_b, pipe.limb_parms[0], pipe.limb_parms[1], )   
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

        method = 'leastsq'
        # tm.lm_params.pretty_print()

        out = minimize(tm.residual, tm.lm_params, args=(tse, fse, ferr), method=method)

        # Store away best fit parameters
        par = out.params
        t0 = par['t0'].value + time_base
        pipe.update_header('fit_rp_se', par['rp'].value, "Best fit Rp/Rstar")
        pipe.update_header('fit_urp_se', par['rp'].stderr, "Uncertainty")
        pipe.update_header('fit_t0_se', par['t0'].value, "SE t0")


        tm_global = copy.deepcopy(tm)
        tm_global.lm_params = out.params
 
   
        if 0.1*fit_rp**2<pipe.fit_rp_se**2 and pipe.fit_urp_se!=None:
            if fit_b>=.9 and pipe.fit_rp_se/pipe.fit_urp_se>2:
                pipe.update_header('SE_found_m{}'.format(plM), True, "EB or hot Jup like")
                pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")
            else:
                pipe.update_header('SE_found_m{}'.format(plM), True, "EB or hot Jup like")
    else:
         pass
    
    return None            

def too_big_planet(pipe,plM):
    fit_rp=eval("pipe.fit_rp_m{}".format(plM))
    fit_b=eval("pipe.fit_b_m{}".format(plM))
    
    # if pipe.star_rad==1:
    #     if (fit_rp)>0.5:
    #         pipe.update_header('TB_planet_m{}'.format(plM), True, "Planet is too large")
    #         pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")
    #     else:
    #         pipe.update_header('TB_planet_m{}'.format(plM), True, "Planet is too large")
        
    if (fit_rp + fit_b)>1.04:
        pipe.update_header('TB_planet_m{}'.format(plM), True, "Planet is too large")
        pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")
    # elif (pipe.star_rad*fit_rp)>0.2:
    #     pipe.update_header('TB_planet_m{}'.format(plM), True, "Planet is too large")
    #     pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")
    elif (fit_rp)>0.3:
        pipe.update_header('TB_planet_m{}'.format(plM), True, "Planet is too large")
        pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")    
    else:
        pipe.update_header('TB_planet_m{}'.format(plM), False, "Planet is too large")
    return None    
  
def harmonic_test(pipe,plM,time_base):
    #####Check if harmonic!!!!
    
    fit_P=eval("pipe.fit_P_m{}".format(plM))
    fit_t0=eval("pipe.fit_t0_m{}".format(plM))
    fit_rp=eval("pipe.fit_rp_m{}".format(plM))
    fit_tdur=eval("pipe.fit_tdur_m{}".format(plM))
    mES=eval("pipe.grid_s2n_m{}".format(plM))
    
    if np.isnan(fit_tdur):
         fit_tdur=eval("pipe.grid_tdur_m{}".format(plM))
    
    if fit_P<5:
        
        lcdt = pipe.lc.copy()
        try:
            lcdt = lcdt[~lcdt.fmask].drop(['fdt_t_rollmed'],axis=1)
        except:
            lcdt = lcdt[~lcdt.fmask].drop(['ferr'],axis=1)    
      
        lcdt['t_shift'] = lcdt['t'] - time_base
        t = np.array(lcdt.t_shift)
        f = np.array(lcdt.f)/np.median(lcdt.f)
        t=t[~np.isnan(f)]
        f=f[~np.isnan(f)]
        
        f=f[~np.isnan(t)]
        t=t[~np.isnan(t)]
        
        
        
        ferr=f
           ####5-sigma clipping
        bg_err=pipe.phot_err
        
        
        sm = tval.CosModel(-.9*fit_rp**2, fit_t0-time_base , 1, 2*np.pi/(fit_P))
        sm.sin_params['amp'].max = -0.0001*fit_rp**2
        sm.sin_params['amp'].min = -3.0 * fit_rp**2
        sm.sin_params['offSet'].min = np.min(t)-time_base
        sm.sin_params['offSet'].max = np.min(t)+fit_P-time_base
        sm.sin_params['YoffSet'].min = .9
        sm.sin_params['YoffSet'].max = 1.1


        sm_initial = copy.deepcopy(sm)

        method = 'leastsq'
        #sm.sin_params.pretty_print()

        out = minimize(sm.residual, sm.sin_params, args=(t, f, ferr), method=method)

        # Store away best fit parameters
        par = out.params
        ampVal1=par['amp'].value
        ampStd1=par['amp'].stderr
        if ampStd1==None:
            ampStd1=ampVal1
        
        sm = tval.CosModel(-.9*fit_rp**2, fit_t0-time_base , 1, 2*np.pi/(fit_P/2))
        sm.sin_params['amp'].max = -0.0001*fit_rp**2
        sm.sin_params['amp'].min = -3.0 * fit_rp**2
        sm.sin_params['offSet'].min = np.min(t)-time_base
        sm.sin_params['offSet'].max = np.min(t)+fit_P-time_base
        sm.sin_params['YoffSet'].min = .9
        sm.sin_params['YoffSet'].max = 1.1


        sm_initial = copy.deepcopy(sm)

        method = 'leastsq'
        # sm.sin_params.pretty_print()

        out = minimize(sm.residual, sm.sin_params, args=(t, f, ferr), method=method)

        # Store away best fit parameters
        par = out.params
        ampVal2=par['amp'].value
        ampStd2=par['amp'].stderr
        if ampStd2==None:
            ampStd2=ampVal2
        
        sm = tval.CosModel(-.9*fit_rp**2, fit_t0-time_base , 1, 2*np.pi/(fit_P*2))
        sm.sin_params['amp'].max = -0.0001*fit_rp**2
        sm.sin_params['amp'].min = -3.0 * fit_rp**2
        sm.sin_params['offSet'].min = np.min(t)-time_base
        sm.sin_params['offSet'].max = np.min(t)+fit_P-time_base
        sm.sin_params['YoffSet'].min = .9
        sm.sin_params['YoffSet'].max = 1.1


        sm_initial = copy.deepcopy(sm)

        method = 'leastsq'
        # sm.sin_params.pretty_print()

        out = minimize(sm.residual, sm.sin_params, args=(t, f, ferr), method=method)

        # Store away best fit parameters
        par = out.params
        ampVal3=par['amp'].value
        ampStd3=par['amp'].stderr
        
        sm = tval.CosModel(-.9*fit_rp**2, fit_t0-time_base , 1, 2*np.pi/(fit_tdur*2))
        sm.sin_params['amp'].max = -0.0001*fit_rp**2
        sm.sin_params['amp'].min = -3.0 * fit_rp**2
        sm.sin_params['offSet'].min = np.min(t)-time_base
        sm.sin_params['offSet'].max = np.min(t)+fit_P-time_base
        sm.sin_params['YoffSet'].min = .9
        sm.sin_params['YoffSet'].max = 1.1


        sm_initial = copy.deepcopy(sm)

        method = 'leastsq'
        # sm.sin_params.pretty_print()

        out = minimize(sm.residual, sm.sin_params, args=(t, f, ferr), method=method)

        # Store away best fit parameters
        par = out.params
        ampVal4=par['amp'].value
        ampStd4=par['amp'].stderr
        
        sm = tval.CosModel(-.9*fit_rp**2, fit_t0-time_base , 1, 2*np.pi/(fit_tdur*4))
        sm.sin_params['amp'].max = -0.0001*fit_rp**2
        sm.sin_params['amp'].min = -3.0 * fit_rp**2
        sm.sin_params['offSet'].min = np.min(t)-time_base
        sm.sin_params['offSet'].max = np.min(t)+fit_P-time_base
        sm.sin_params['YoffSet'].min = .9
        sm.sin_params['YoffSet'].max = 1.1


        sm_initial = copy.deepcopy(sm)

        method = 'leastsq'
        # sm.sin_params.pretty_print()

        out = minimize(sm.residual, sm.sin_params, args=(t, f, ferr), method=method)

        # Store away best fit parameters
        par = out.params
        ampVal5=par['amp'].value
        ampStd5=par['amp'].stderr
        
        sm = tval.CosModel(-.9*fit_rp**2, fit_t0-time_base , 1, 2*np.pi/(fit_tdur))
        sm.sin_params['amp'].max = -0.0001*fit_rp**2
        sm.sin_params['amp'].min = -3.0 * fit_rp**2
        sm.sin_params['offSet'].min = np.min(t)-time_base
        sm.sin_params['offSet'].max = np.min(t)+fit_P-time_base
        sm.sin_params['YoffSet'].min = .9
        sm.sin_params['YoffSet'].max = 1.1


        sm_initial = copy.deepcopy(sm)

        method = 'leastsq'
        # sm.sin_params.pretty_print()

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
            pipe.update_header('Harmonic_m{}'.format(plM), True, "Sine wave fit(1)")
            pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")
        elif (abs(ampVal1)> fit_rp**2 and abs(ampVal1)>2*ampStd1) and mES<50:
            pipe.update_header('Harmonic_m{}'.format(plM), True, "Sine wave fit(2)")
            pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")
        elif (abs(ampVal2)> fit_rp**2 and abs(ampVal2)>2*ampStd2) and mES<50:
            pipe.update_header('Harmonic_m{}'.format(plM), True, "Sine wave fit(3)")
            pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")
        elif (abs(ampVal3)> fit_rp**2 and abs(ampVal3)>2*ampStd3) and mES<50:
            pipe.update_header('Harmonic_m{}'.format(plM), True, "Sine wave fit(4)")
            pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")
        elif (abs(ampVal4)> fit_rp**2 and abs(ampVal4)>2*ampStd4) and mES<50:
            pipe.update_header('Harmonic_m{}'.format(plM), True, "Sine wave fit(5)")
            pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")
        elif (abs(ampVal5)> fit_rp**2 and abs(ampVal5)>2*ampStd5) and mES<50:
            pipe.update_header('Harmonic_m{}'.format(plM), True, "Sine wave fit(6)")
            pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")
        elif (abs(ampVal6)> fit_rp**2 and abs(ampVal6)>2*ampStd6) and mES<50:
            pipe.update_header('Harmonic_m{}'.format(plM), True, "Sine wave fit(7)")
            pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")     
        else:
            pipe.update_header('Harmonic_m{}'.format(plM), False, "Sine wave fit")
    return None    

def individual_transits(pipe,plM,time_base):
    pipe.update_header('transit_mask_m{}'.format(plM), False, "Was a transit Masked")
    
    lcdt = pipe.lc.copy()
    lcdt = lcdt[~lcdt.fmask]
    lcdt['t_shift'] = lcdt['t'] - time_base
    fit_tdur=eval('pipe.fit_tdur_m{}'.format(plM))
    fit_P=eval("pipe.fit_P_m{}".format(plM))
    fit_t0=eval("pipe.fit_t0_m{}".format(plM))
    fit_rp=eval("pipe.fit_rp_m{}".format(plM))
    fit_b=eval("pipe.fit_b_m{}".format(plM))
    fit_apl=eval("pipe.fit_apl_m{}".format(plM))
    mES=eval("pipe.grid_s2n_m{}".format(plM))
    
    t = np.array(lcdt.t_shift)
    f = np.array(lcdt.f)/np.median(lcdt.f)
    ferr=pipe.phot_err
       ####5-sigma clipping
    bg_err=pipe.phot_err
    
    numTran=np.int(np.floor((np.max(t)-(fit_t0-time_base))/fit_P)+1)
    calcTrans=numTran
    sES=np.zeros(numTran)
    for idxx in range(numTran):
        
        tTran=t[(lcdt.t_shift>=(fit_t0)+idxx*fit_P-time_base-fit_P*.5) & (lcdt.t_shift<=(fit_t0)+idxx*fit_P-time_base+fit_P*.5)]
        fTran=f[(lcdt.t_shift>=(fit_t0)+idxx*fit_P-time_base-fit_P*.5) & (lcdt.t_shift<=(fit_t0)+idxx*fit_P-time_base+fit_P*.5)]
        ferr=pipe.phot_err
    
    
        tm = tval.TransitModel(fit_P, fit_t0-time_base, fit_rp, fit_apl, fit_b, pipe.limb_parms[0], pipe.limb_parms[1], )
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
            out = minimize(tm.residual, tm.lm_params, args=(tTran, fTran, ferr), method=method)

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
            out = minimize(sm.residual, sm.sin_params, args=(tTran, fTran, ferr), method=method)

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
            out = minimize(sm.residual, sm.sin_params, args=(tTran, fTran, ferr), method=method)

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
            out = minimize(sm.residual, sm.sin_params, args=(tTran, fTran, ferr), method=method)

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
            out = minimize(sm.residual, sm.sin_params, args=(tTran, fTran, ferr), method=method)

            # Store away best fit parameters
            par = out.params
            if BICmodel4 < BICmodelTran*.5: 
                BICmodel4=BICmodelTran
            else:
                BICmodel4=out.bic   

        except:
            BICmodel4=BICmodelTran    
        
        
        ############SES/MES ratio
        ####SES=MES*sqrt(N)-MES_Others*sqrt(N-1)
        ####SES/MES>0.8
        ##sqrt(N)-MES_Others/MES*sqrt(N-1)>0.8
        if fit_P>5:
            newT=t
            newF=f
            meddt=pipe.dt
            newMask=np.zeros((len(newF)), dtype=bool)
            newMask=np.where((lcdt.t_shift>=(fit_t0)+idxx*fit_P-time_base-fit_P*.5) & (lcdt.t_shift<=(fit_t0)+idxx*fit_P-time_base+fit_P*.5),True, False)
            resid=np.where((lcdt.t_shift>=(fit_t0)+idxx*fit_P-time_base-.5*fit_tdur) & (lcdt.t_shift<=(fit_t0)+idxx*fit_P-time_base+.5*fit_tdur),(1-newF), 0)
            resid=np.where(resid<0,-(resid)**2,resid**2)
            sumRes=np.sum(resid)**.5
            
            residTot=np.where((lcdt.t_shift%fit_P>=(fit_t0-time_base)%fit_P-.5*fit_tdur) & (lcdt.t_shift%fit_P<=(fit_t0-time_base)%fit_P+.5*fit_tdur),(1-newF), 0)
            residTot=np.where(residTot<0,-(residTot)**2,residTot**2)
            sumResTot=np.sum(residTot)**.5
            
            
            print("sum resid")
            print(sumRes)
            print(sumResTot)
            # newT=newT[~newMask]
        #     newF=newF[~newMask]
        #     newMask=newMask[~newMask]
        #
        #
        #     if (fit_t0+idxx*fit_P-time_base-fit_P*.5)>np.min(newT):
        #         offT=np.max(np.where(newT>=(fit_t0+idxx*fit_P-time_base-fit_P*.5),-1000000000,newT))
        #         if abs(offT-(fit_t0+idxx*fit_P-time_base-fit_P*.5))>meddt+.0001:
        #             offT=offT+np.floor(abs(offT-(fit_t0+idxx*fit_P-time_base-fit_P*.5))/meddt)*meddt
        #     else:
        #         offT=np.min(newT)
        #     if (fit_t0+idxx*fit_P-time_base+fit_P*.5)<np.max(newT):
        #         offTH=np.min(np.where(newT<=(fit_t0+idxx*fit_P-time_base+fit_P*.5),1000000000,newT))
        #         if abs(offTH-(fit_t0+idxx*fit_P-time_base+fit_P*.5))>meddt+.0001:
        #             offTH=offTH-np.floor(abs(offTH-(fit_t0+idxx*fit_P-time_base+fit_P*.5))/meddt)*meddt
        #     else:
        #         offTH=np.max(newT)
        #
        #     diffOff=offT-offTH
        #
        #     newT=np.where(newT>=(fit_t0+idxx*fit_P-time_base-fit_P*.5),newT+diffOff+meddt, newT)
        
        
        
        
        
        
        
            # dt = newT[1:] - newT[:-1]
 #            loca=np.where(dt>meddt+0.0001)[0]
 #
 #            while len(loca)>0:
 #                i=0
 #                newT=np.concatenate((newT[:loca[i]+1],newT[loca[i]+1]-meddt,newT[loca[i]+1:]), axis=None)
 #                newF=np.concatenate((newF[:loca[i]+1],1,newF[loca[i]+1:]), axis=None)
 #                newMask=np.concatenate((newMask[:loca[i]+1],True,newMask[loca[i]+1:]), axis=None)
 #                dt = newT[1:] - newT[:-1]
 #                meddt = np.median(dt)
 #                loca=np.where(dt>meddt+0.0001)[0]
            
        
            # fm = ma.masked_array(newF,mask=newMask, fill_value=0)
   #          fm -= ma.median(fm)
   #
   #
   #          grid = tfind.Grid(newT, fm)
   #          Pcad1 = fit_P / meddt - 1
   #          Pcad2 = fit_P / meddt + 1
   #          twd = fit_tdur/meddt
   #
   #
   #          pgram_params = [dict(Pcad1=Pcad1, Pcad2=Pcad2, twdG=[twd])]
            if np.isnan(sumRes):
                sES[idxx]=0
            if np.isnan(sumResTot):
                sES[idxx]=1    
            else:    
                sES[idxx]=sumRes/sumResTot
            # try:
  #               pgram = grid.periodogram(pgram_params,mode='max')
  #               row = pgram.sort_values('s2n').iloc[-1]
  #               if row.s2n>0 and ~np.isnan(row.s2n):
  #                   # sES[idxx]=np.sqrt(numTran)-row.s2n*np.sqrt(numTran-1)/mES
  #                   if row.s2n<mES:
  #                       sES[idxx]=(mES-row.s2n)/mES
  #                   else:
  #                       sES[idxx]=0
  #
  #               else:
  #                   sES[idxx]=0
  #           except:
  #               sES[idxx]=0
            print(sES[idxx])               



        
        
    
        numCad=len(lcdt.t_shift[(lcdt.t_shift>=(fit_t0)+idxx*fit_P-time_base-.5*fit_tdur) & (lcdt.t_shift<=(fit_t0)+idxx*fit_P-time_base+.5*fit_tdur)])
        numExpect=np.floor(fit_tdur/pipe.dt)
    


        if .6*numExpect>=numCad:
            isMask=(lcdt.t_shift>=(fit_t0)+idxx*fit_P-time_base-.75*fit_tdur) & (lcdt.t_shift<=(fit_t0)+idxx*fit_P-time_base+.75*fit_tdur)
            pipe.lc['fmask'] = pipe.lc['fmask'] | isMask
            pipe.update_header('transit_mask_m{}'.format(plM), True, "Was a transit Masked")
            calcTrans=calcTrans-1
        elif BICmodel1+10 < BICmodelTran:
            calcTrans=calcTrans-1
            isMask=(lcdt.t_shift>=(fit_t0)+idxx*fit_P-time_base-.75*fit_tdur) & (lcdt.t_shift<=(fit_t0)+idxx*fit_P-time_base+.75*fit_tdur)
            pipe.lc['fmask'] = pipe.lc['fmask'] | isMask
            pipe.update_header('transit_mask_m{}'.format(plM), True, "Was a transit Masked")
        elif BICmodel2+10 < BICmodelTran and mES/np.sqrt(numTran)>4:
            calcTrans=calcTrans-1
            isMask=(lcdt.t_shift>=(fit_t0)+idxx*fit_P-time_base-.75*fit_tdur) & (lcdt.t_shift<=(fit_t0)+idxx*fit_P-time_base+.75*fit_tdur)
            pipe.lc['fmask'] = pipe.lc['fmask'] | isMask
            pipe.update_header('transit_mask_m{}'.format(plM), True, "Was a transit Masked")
        elif BICmodel3+10 < BICmodelTran and mES/np.sqrt(numTran)>4:
            calcTrans=calcTrans-1
            isMask=(lcdt.t_shift>=(fit_t0)+idxx*fit_P-time_base-.75*fit_tdur) & (lcdt.t_shift<=(fit_t0)+idxx*fit_P-time_base+.75*fit_tdur)
            pipe.lc['fmask'] = pipe.lc['fmask'] | isMask
            pipe.update_header('transit_mask_m{}'.format(plM), True, "Was a transit Masked") 
        elif BICmodel4+10 < BICmodelTran and mES/np.sqrt(numTran)>4:
            calcTrans=calcTrans-1
            isMask=(lcdt.t_shift>=(fit_t0)+idxx*fit_P-time_base-.75*fit_tdur) & (lcdt.t_shift<=(fit_t0)+idxx*fit_P-time_base+.75*fit_tdur)
            pipe.lc['fmask'] = pipe.lc['fmask'] | isMask
            pipe.update_header('transit_mask_m{}'.format(plM), True, "Was a transit Masked")           
        else:
            pass
            
        if calcTrans<3:
            pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")
            pipe.update_header('numTran_m{}'.format(plM), True, "less than 3 transits?")
        else:
            pipe.update_header('numTran_m{}'.format(plM), False, "less than 3 transits?")

        
    if np.max(sES)>=0.8:
        pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")
        pipe.update_header('transit_domination{}'.format(plM), True, "Single Transit domination")
    else:
        pipe.update_header('transit_domination{}'.format(plM), False, "Single Transit domination")

    newMES=grid_search_NewMES(pipe, plM)
    
    if newMES<7.1:
        pipe.update_header('falsePos_m{}'.format(plM), True, "is it a FP?")
        pipe.update_header('MES_tooSmall_m{}'.format(plM), True, "did the MES drop below?")
    else:
        pipe.update_header('MES_tooSmall_m{}'.format(plM), False, "did the MES drop below?")
        pass                    
                
    return None    
           

def period_alias(pipe,plM,time_base): 
    ####likelihood p=p0###
    
    lcdt = pipe.lc.copy()
    try:
        lcdt = lcdt[~lcdt.fmask].drop(['fdt_t_rollmed'],axis=1)
    except:
        lcdt = lcdt[~lcdt.fmask].drop(['ferr'],axis=1)    
       
    lcdt['t_shift'] = lcdt['t'] - time_base
    t = np.array(lcdt.t_shift)
    f = np.array(lcdt.f)/np.median(lcdt.f)
    ferr=pipe.phot_err
       ####5-sigma clipping
    bg_err=pipe.phot_err
    
    fit_P=eval("pipe.fit_P_m{}".format(plM))
    fit_t0=eval("pipe.fit_t0_m{}".format(plM))
    fit_rp=eval("pipe.fit_rp_m{}".format(plM))
    fit_b=eval("pipe.fit_b_m{}".format(plM))
    fit_apl=eval("pipe.fit_apl_m{}".format(plM))

    bm_params = batman.TransitParams()
    bm_params.b = fit_b
    bm_params.per = fit_P
    bm_params.t0 = fit_t0-time_base
    bm_params.rp = fit_rp
    bm_params.u = [pipe.limb_parms[0], pipe.limb_parms[1]]  #orbital inclination (in degrees)
    bm_params.a = fit_apl
    bm_params.inc = np.arccos(bm_params.b/bm_params.a)*180/np.pi 
    bm_params.ecc = 0  
    bm_params.w = 90 
    bm_params.limb_dark = "quadratic" 
    m = batman.TransitModel(bm_params, t, transittype="primary")
    _model = m.light_curve(bm_params)
    
    likelihood_p0=np.mean(((f-_model)/bg_err)**2)
    
    # apl=((fit_P/2)**2*gravc*pipe.star_mass/(4*np.pi**2))**(1/3)/pipe.star_rad

    bm_params.per = fit_P/2
    bm_params.a = fit_apl/2
    bm_params.inc = np.arccos(bm_params.b/bm_params.a)*180/np.pi 
    m = batman.TransitModel(bm_params, t, transittype="primary")
    _model = m.light_curve(bm_params)
    
    likelihood_p0_half=np.mean(((f-_model)/bg_err)**2)
    
    # apl=((fit_P/3)**2*gravc*pipe.star_mass/(4*np.pi**2))**(1/3)/pipe.star_rad
    bm_params.per = fit_P/3
    bm_params.a = fit_apl/3
    bm_params.inc = np.arccos(bm_params.b/bm_params.a)*180/np.pi 
    
    m = batman.TransitModel(bm_params, t, transittype="primary")
    _model = m.light_curve(bm_params)
    
    likelihood_p0_third=np.mean(((f-_model)/bg_err)**2)
    
    # apl=((fit_P*2)**2*gravc*pipe.star_mass/(4*np.pi**2))**(1/3)/pipe.star_rad
    bm_params.per = fit_P*2
    bm_params.a = fit_apl*2
    bm_params.inc = np.arccos(bm_params.b/bm_params.a)*180/np.pi 
    
    m = batman.TransitModel(bm_params, t, transittype="primary")
    _model = m.light_curve(bm_params)
    
    likelihood_p0_double=np.mean(((f-_model)/bg_err)**2)
    
    # apl=((fit_P*3)**2*gravc*pipe.star_mass/(4*np.pi**2))**(1/3)/pipe.star_rad
    bm_params.per = fit_P*3
    bm_params.a = fit_apl*3
    bm_params.inc = np.arccos(bm_params.b/bm_params.a)*180/np.pi 

    m = batman.TransitModel(bm_params, t, transittype="primary")
    _model = m.light_curve(bm_params)
    
    likelihood_p0_trip=np.mean(((f-_model)/bg_err)**2)
    
    likeArray=np.array([likelihood_p0_trip+1,likelihood_p0_double+1,likelihood_p0_half+1,likelihood_p0_third+1,likelihood_p0])
    
    if likelihood_p0==np.min(likeArray):
        pipe.update_header('period_alias_m{}'.format(plM), False, "detected alias")
        pipe.update_header('period_alias_ratio_m{}'.format(plM), 1, "detected alias ratio")
        detection=False
    else:    
        pipe.update_header('period_alias_m{}'.format(plM), True, "detected alias")
        bolRatio=likeArray==np.min(likeArray)
        indBol=np.where(bolRatio==True)[0][0]
        detection=True
        if indBol==0:
            pipe.update_header('period_alias_ratio_m{}'.format(plM), 3, "detected alias ratio")
        elif indBol==1:
            pipe.update_header('period_alias_ratio_m{}'.format(plM), 2, "detected alias ratio") 
        elif indBol==2:
            pipe.update_header('period_alias_ratio_m{}'.format(plM), 0.5, "detected alias ratio")
        elif indBol==3:
            pipe.update_header('period_alias_ratio_m{}'.format(plM), 0.3333, "detected alias ratio")
        else:
            pipe.update_header('period_alias_ratio_m{}'.format(plM), 1, "detected alias ratio")        
    return None 

def grid_search_MES(pipe, P1=0.5, P2=None, periodogram_mode='max'): 
    """Run the grid based search

    Args:
        P1 (Optional[float]): Minimum period to search over. Default is 0.5
        P2 (Optional[float]): Maximum period to search over. Default is half 
            the time baseline
        **kwargs : passed to grid.periodogram

    Returns:
        None
    
    """
    phase_limit = 0.1
    lcdt = pipe.lc.copy()
    fit_apl=eval("pipe.fit_apl_m{}".format(plM))
    
    if type(P2) is type(None):
        P2 = 0.49 * pipe.lc.t.ptp()
   
    t = np.array(pipe.lc.t)
    dt = t[1:] - t[:-1]
    meddt = np.median(dt)
    loca=np.where(dt>meddt+0.0001)[0]

    while len(loca)>0:
        i=0
        var=list(pipe.lc.head())
        #print (var)
        try:
            line = pd.DataFrame({var[0]: pipe.lc[var[0]][loca[i]+1]-meddt,
             var[1]: pipe.lc[var[1]][loca[i]+1],
             var[2]: pipe.lc[var[2]][loca[i]+1],
             var[3]: pipe.lc[var[3]][loca[i]+1],
             var[4]: pipe.lc[var[4]][loca[i]+1],
             var[5]: pipe.lc[var[5]][loca[i]+1],
             var[6]: pipe.lc[var[6]][loca[i]+1],
             var[7]: pipe.lc[var[7]][loca[i]+1],
             var[8]: pipe.lc[var[8]][loca[i]+1],
             var[9]: pipe.lc[var[9]][loca[i]+1]} , index=[loca[i]+1])
        except:
            line = pd.DataFrame({var[0]: pipe.lc[var[0]][loca[i]+1]-meddt,
             var[1]: pipe.lc[var[1]][loca[i]+1],
             var[2]: pipe.lc[var[2]][loca[i]+1],
             var[3]: pipe.lc[var[3]][loca[i]+1],
             var[4]: pipe.lc[var[4]][loca[i]+1]}, index=[loca[i]+1])
                 
         # var[10]: pipe.lc[var[10]][loca[i]+1]}, index=[loca[i]+1])
        pipe.lc = pd.concat([pipe.lc[:loca[i]+1], line, pipe.lc[loca[i]+1:]], sort=False).reset_index(drop=True)
        t = np.array(pipe.lc.t)
        dt = t[1:] - t[:-1]
        meddt = np.median(dt)
        loca=np.where(dt>meddt+0.0001)[0]    
    

    
    fm = pipe._get_fm()

    grid = tfind.Grid(t, fm)
    Pcad1 = pipe.inject_per / meddt - 1
    Pcad2 = pipe.inject_per / meddt + 1
    try:
        twd = pipe.inject_per/np.pi*math.asin(np.sqrt((pipe.star_rad+pipe.inject_rp*pipe.star_rad)**2-(pipe.star_rad*pipe.inject_b)**2)/fit_apl)/meddt
    except:
        twd = pipe.inject_per/np.pi*np.sqrt((pipe.star_rad+pipe.inject_rp*pipe.star_rad)**2-(pipe.star_rad*pipe.inject_b)**2)/fit_apl/meddt    
    
    pgram_params = [dict(Pcad1=Pcad1, Pcad2=Pcad2, twdG=[twd])]
    pgram = grid.periodogram(pgram_params,mode='max')
    row = pgram.sort_values('s2n').iloc[-1]
    
    SNR=row.s2n
    print(SNR)
    pipe.update_header('inject_MES', SNR)   
    pipe.lc=lcdt.copy()
    return None
    
def grid_search_NewMES(pipe, plM): 
    """Run the grid based search

    Args:
        P1 (Optional[float]): Minimum period to search over. Default is 0.5
        P2 (Optional[float]): Maximum period to search over. Default is half 
            the time baseline
        **kwargs : passed to grid.periodogram

    Returns:
        None
    
    """
    fit_tdur=eval('pipe.grid_tdur_m{}'.format(plM))
    fit_P=eval("pipe.fit_P_m{}".format(plM))
    fit_apl=eval("pipe.fit_apl_m{}".format(plM))
    
    lcdt = pipe.lc.copy()
   
    t = np.array(pipe.lc.t)
    dt = t[1:] - t[:-1]
    meddt = np.median(dt)
    loca=np.where(dt>meddt+0.0001)[0]

    while len(loca)>0:
        i=0
        var=list(pipe.lc.head())
        #print (var)
        try:
            line = pd.DataFrame({var[0]: pipe.lc[var[0]][loca[i]+1]-meddt,
             var[1]: pipe.lc[var[1]][loca[i]+1],
             var[2]: pipe.lc[var[2]][loca[i]+1],
             var[3]: pipe.lc[var[3]][loca[i]+1],
             var[4]: pipe.lc[var[4]][loca[i]+1],
             var[5]: pipe.lc[var[5]][loca[i]+1],
             var[6]: pipe.lc[var[6]][loca[i]+1],
             var[7]: pipe.lc[var[7]][loca[i]+1],
             var[8]: pipe.lc[var[8]][loca[i]+1],
             var[9]: pipe.lc[var[9]][loca[i]+1]} , index=[loca[i]+1])
        except:
            line = pd.DataFrame({var[0]: pipe.lc[var[0]][loca[i]+1]-meddt,
             var[1]: pipe.lc[var[1]][loca[i]+1],
             var[2]: pipe.lc[var[2]][loca[i]+1],
             var[3]: pipe.lc[var[3]][loca[i]+1],
             var[4]: pipe.lc[var[4]][loca[i]+1]}, index=[loca[i]+1])
                 
         # var[10]: pipe.lc[var[10]][loca[i]+1]}, index=[loca[i]+1])
        pipe.lc = pd.concat([pipe.lc[:loca[i]+1], line, pipe.lc[loca[i]+1:]], sort=False).reset_index(drop=True)
        t = np.array(pipe.lc.t)
        dt = t[1:] - t[:-1]
        meddt = np.median(dt)
        loca=np.where(dt>meddt+0.0001)[0]    
    

    
    fm = pipe._get_fm()

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