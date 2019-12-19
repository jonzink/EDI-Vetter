import pandas as pd
try:
    import pipeline
except:
    import terra.pipeline
import numpy as np
import glob, os
from pathlib import Path
import batman
from scipy.stats import norm, uniform, beta, gamma, poisson, binom, binned_statistic
import math
import limbdark
import time
import multiprocessing
from multiprocessing import Pool
import random
from astropy.io import fits
from astropy.table import Table, unique
import everest
import k2plr



injection=False

gravc=6.674e-11*1.98855e30*(86400)**2/(6.957e8)**3

k2_data = Table.read('k2_dr2_20arcsec.fits', format='fits')

stellar_parms=pd.read_csv("k2_stellar.bar",sep="|", engine='python')
stellar_parms=stellar_parms.set_index("eid")

stellar_2mass=pd.read_csv("k2_mult_2mass.bar",sep="|", engine='python')
stellar_2mass=stellar_2mass.set_index("eid")

TCElist=list(np.array([211368318, 211730600, 211513786, 211511097, 212102971, 212008515,
       211969273, 211591078]))
#TCElist=list(np.array(pd.read_csv("TCElistSmall.csv",sep=",", engine='python').iloc[:,1]).astype(int))

path ='./lc_ex/' # use your path
allFiles = glob.glob(path + "/*.csv")
aper=1


random.shuffle(allFiles)
totLengthFile=len(allFiles)
counter=0
checkNum=0


for file_ in allFiles:
    perc=counter/totLengthFile
    file2 = open(file_, "r")
    print("{}Percent Complete".format(perc*100))
    counter=counter+1

    guy={"starname" : file_[8:17]}
    star=int(file_[8:17])
   # print(star)
    
    lc=pd.read_csv(file_, skiprows=1, engine='python')
    lc.columns = ["ind",'cad', 't',  'f',"fR","fR_err",  'fmask',  'aper', "flag"]
    aper=np.median(lc.aper)
    lc["ferr"]=lc.f
    if (np.sqrt(aper/np.pi)+1)*3.98>24:
        pass

    else:        
        try:
            if len(unique(k2_data[k2_data["epic_number"]==star],"k2_gaia_ang_dist"))==0:
               pass
            else:
                if abs(unique(k2_data[k2_data["epic_number"]==star],"k2_gaia_ang_dist")["k2_kepmag"][0]-unique(k2_data[k2_data["epic_number"]==star],"k2_gaia_ang_dist")["phot_g_mean_mag"][0])<=1:
                    pass
                elif abs(unique(k2_data[k2_data["epic_number"]==star],"k2_gaia_ang_dist")["k2_kepmag"][1]-unique(k2_data[k2_data["epic_number"]==star],"k2_gaia_ang_dist")["phot_g_mean_mag"][1])<=1:
                    pass

                else:
                    checkNum=checkNum+1
                    print(checkNum)
        except:
                pass
print(checkNum)                