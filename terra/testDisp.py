import pandas as pd
try:
    import dispoTest
except:
    import terra.dispoTest
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
from astropy.table import Table


injection=False

gravc=6.674e-11*1.98855e30*(86400)**2/(6.957e8)**3

k2_data = Table.read('k2_dr2_4arcsec.fits', format='fits')


path ='./lc_ex/' # use your path
allFiles = glob.glob(path + "/*.csv")
aper=1


random.shuffle(allFiles)
totLengthFile=len(allFiles)
counter=0



for file_ in allFiles:




##########EVEREST

    # perc=counter/totLengthFile
    # print("{}Percent Complete".format(perc*100))
    # counter=counter+1
    # lc=pd.DataFrame()
    # star=int(file_[28:37])
    # guy={"starname" : star}
    # lc=pd.read_csv(file_, skiprows=0, engine='python')
    # lc=lc.reset_index()
    # lc.columns = ["cad","cad2",'t',    'f', "fmask"]
    # lc["ferr"]=lc.f




# #####K2SFF
#     perc=counter/totLengthFile
#     print("{}Percent Complete".format(perc*100))
#     counter=counter+1
#     star=int(file_[33:42])
#     lc=pd.read_csv(file_, skiprows=0, engine='python')
#     lc=lc.reset_index()
#     lc.columns = ['t',    'f', "ferr"]
#     lc["ferr"]=lc.f
#     lc["fmask"]=False
#     star=int(file_[33:42])
#     guy={"starname" : star}



####K2Phot
    perc=counter/totLengthFile
    file2 = open(file_, "r")
    print("{}Percent Complete".format(perc*100))
    counter=counter+1
    ii=0
    for line in file2:
        if ii==0:
            guy={"starname" : int(line.split("-")[1])}
            star=int(line.split("-")[1])
        if ii==5:
            aper=float(line[32:37].strip())
        if ii>5:
            break
        ii+=1
    lc=pd.read_csv(file_, skiprows=39, engine='python')
    lc.columns = ['t',    'cad',    'fsap',    'ferr',    'fdt_t_rollmed',    'f',    'bgmask',    'thrustermask',    'fdtmask',    'fmask']

###########################  
    
    my_file = Path('./TCE/{}/header.csv'.format(star))
    my_file2 = Path('./noTCE/{}/header.csv'.format(star))
    if my_file2.is_file():
        pass
    elif my_file.is_file():
        indice=0
        while indice<30:
            try:
                star_details=pd.read_html("https://exofop.ipac.caltech.edu/k2/edit_target.php?id={}".format(star), header=0)[indice]
                if list(star_details)[0][0:12]=="Stellar Para":
                    break
            except:
                pass
            indice=indice+1
        indice=0
        while indice<30:
            try:
                star_location=pd.read_html("https://exofop.ipac.caltech.edu/k2/edit_target.php?id={}".format(star), header=0)[indice]
                if list(star_location)[0][0:5]=='2MASS':
                    break
            except:
                pass
            indice=indice+1    

        
        try:
            star_temp=np.float(k2_data[k2_data["epic_number"]==star]["teff_val"][0])
            star_utemp=(np.float(k2_data[k2_data["epic_number"]==star]["teff_percentile_upper"][0])-np.float(k2_data[k2_data["epic_number"]==star]["teff_percentile_lower"][0]))/2
            
            if np.isnan(star_temp):
                try:
                    star_temp=np.float(star_details.iloc[1,0].split("\xb1")[0])
                    star_utemp=np.float(star_details.iloc[1,0].split("\xb1")[1])

                except:
                    star_temp=5378
                    star_utemp=1000
                
        except:
            try:
                star_temp=np.float(star_details.iloc[1,0].split("\xb1")[0])
                star_utemp=np.float(star_details.iloc[1,0].split("\xb1")[1])

            except:
                star_temp=5378
                star_utemp=1000
        try:
            star_logg=np.float(star_details.iloc[1,1].split("\xb1")[0])
            star_ulogg=np.float(star_details.iloc[1,1].split("\xb1")[1])

        except:
            star_logg=4.59
            star_ulogg=2

        try:
            star_rad=np.float(k2_data[k2_data["epic_number"]==star]["radius_val"][0])
            star_urad=(np.float(k2_data[k2_data["epic_number"]==star]["radius_percentile_upper"][0])-np.float(k2_data[k2_data["epic_number"]==star]["radius_percentile_lower"][0]))/2
       
            if np.isnan(star_rad):
                try:
                    star_rad=np.float(star_details.iloc[1,2].split("\xb1")[0])
                    star_urad=np.float(star_details.iloc[1,2].split("\xb1")[1])
                except:
                    star_rad=1
                    star_urad=1
                
        except:
            try:
                star_rad=np.float(star_details.iloc[1,2].split("\xb1")[0])
                star_urad=np.float(star_details.iloc[1,2].split("\xb1")[1])
            except:
                star_rad=1
                star_urad=1
                
        try:
            star_mass=np.float(k2_data[k2_data["epic_number"]==star]["k2_mass"][0])
            star_umass=(abs(np.float(k2_data[k2_data["epic_number"]==star]["k2_masserr1"][0]))+abs(np.float(k2_data[k2_data["epic_number"]==star]["k2_masserr2"][0])))/2
            if np.isnan(star_mass):
                try:    
                    star_mass=np.float(star_details.iloc[1,8].split("\xb1")[0])
                    star_umass=np.float(star_details.iloc[1,8].split("\xb1")[1])
                except:
                    if star_logg>4:
                        if star_temp <10800 and star_temp >7400:
                            star_mass=1.4
                            star_umass=.3
                        elif star_temp <7400 and star_temp >7120:
                            star_mass=1.3
                            star_umass=.3
                        elif star_temp <7120 and star_temp >6840:
                            star_mass=1.2
                            star_umass=.2
                        elif star_temp <6840 and star_temp >6560:
                            star_mass=1.1
                            star_umass=.2
                        elif star_temp <6560 and star_temp >6140:
                            star_mass=1.0
                            star_umass=.2
                        elif star_temp <6140 and star_temp >5780:
                            star_mass=0.9
                            star_umass=.2
                        elif star_temp <5780 and star_temp >5340:
                            star_mass=0.8
                            star_umass=.2
                        elif star_temp <5340 and star_temp >4620:
                            star_mass=0.7
                            star_umass=.2
                        elif star_temp <4620 and star_temp >4200:
                            star_mass=0.6
                            star_umass=.2
                        elif star_temp <4620 and star_temp >4060:
                            star_mass=0.5
                            star_umass=.2
                        elif star_temp <4060 and star_temp >3500:
                            star_mass=0.3
                            star_umass=.2
                        elif star_temp <3500 and star_temp >2800:
                            star_mass=0.2
                            star_umass=.2
                        else:
                            star_mass=0.1
                            star_umass=.2
                    else:
                        star_mass=1
                        star_umass=1                                            
                
        
        except:
            try:    
                star_mass=np.float(star_details.iloc[1,8].split("\xb1")[0])
                star_umass=np.float(star_details.iloc[1,8].split("\xb1")[1])
            except:
                if star_logg>4:
                    if star_temp <10800 and star_temp >7400:
                        star_mass=1.4
                        star_umass=.3
                    elif star_temp <7400 and star_temp >7120:
                        star_mass=1.3
                        star_umass=.3
                    elif star_temp <7120 and star_temp >6840:
                        star_mass=1.2
                        star_umass=.2
                    elif star_temp <6840 and star_temp >6560:
                        star_mass=1.1
                        star_umass=.2
                    elif star_temp <6560 and star_temp >6140:
                        star_mass=1.0
                        star_umass=.2
                    elif star_temp <6140 and star_temp >5780:
                        star_mass=0.9
                        star_umass=.2
                    elif star_temp <5780 and star_temp >5340:
                        star_mass=0.8
                        star_umass=.2
                    elif star_temp <5340 and star_temp >4620:
                        star_mass=0.7
                        star_umass=.2
                    elif star_temp <4620 and star_temp >4200:
                        star_mass=0.6
                        star_umass=.2
                    elif star_temp <4620 and star_temp >4060:
                        star_mass=0.5
                        star_umass=.2
                    elif star_temp <4060 and star_temp >3500:
                        star_mass=0.3
                        star_umass=.2
                    elif star_temp <3500 and star_temp >2800:
                        star_mass=0.2
                        star_umass=.2
                    else:
                        star_mass=0.1
                        star_umass=.2
                else:
                    star_mass=1
                    star_umass=1                                            

        try:
            dist1=np.float(star_location.iloc[1,6])
            dist2=np.float(star_location.iloc[2,6])
            
            if dist1<=1:
                deltDist=dist2-dist1
            else:
                deltDist=dist1
        except:
            deltDist=np.inf
        try:
            mag1=np.float(star_location.iloc[1,3].split("\xb1")[0])
            mag2=np.float(star_location.iloc[2,3].split("\xb1")[0])
            if dist1<=1:
                deltaMag=mag2-mag1
            else:
                deltaMag=np.inf    
        except:
            deltaMag=np.inf
        if len(k2_data[k2_data["epic_number"]==star])==0:
            deltDistGaia=np.inf
            deltaMagGaia=np.inf
        elif len(k2_data[k2_data["epic_number"]==star])==1:    
            if abs(k2_data[k2_data["epic_number"]==star]["k2_kepmag"]-k2_data[k2_data["epic_number"]==star]["phot_g_mean_mag"])<=1:
                deltDistGaia=np.inf
                deltaMagGaia=np.inf
            else:
                deltaMagGaia=k2_data[k2_data["epic_number"]==star]["phot_g_mean_mag"]-k2_data[k2_data["epic_number"]==star]["k2_kepmag"] 
                deltDistGaia=k2_data[k2_data["epic_number"]==star]["k2_gaia_ang_dist"]
                try:
                    star_rad=np.float(star_details.iloc[1,2].split("\xb1")[0])
                except:
                    star_rad=1
        else:
            if abs(k2_data[k2_data["epic_number"]==star]["k2_kepmag"][0]-k2_data[k2_data["epic_number"]==star]["phot_g_mean_mag"][0])<=1:
                deltDistGaia=abs(k2_data[k2_data["epic_number"]==star]["k2_gaia_ang_dist"][0]-k2_data[k2_data["epic_number"]==star]["k2_gaia_ang_dist"][1])
                deltaMagGaia=k2_data[k2_data["epic_number"]==star]["phot_g_mean_mag"][1]-k2_data[k2_data["epic_number"]==star]["phot_g_mean_mag"][0]
            elif abs(k2_data[k2_data["epic_number"]==star]["k2_kepmag"][1]-k2_data[k2_data["epic_number"]==star]["phot_g_mean_mag"][1])<=1: 
                deltDistGaia=abs(k2_data[k2_data["epic_number"]==star]["k2_gaia_ang_dist"][0]-k2_data[k2_data["epic_number"]==star]["k2_gaia_ang_dist"][1])
                deltaMagGaia=k2_data[k2_data["epic_number"]==star]["phot_g_mean_mag"][0]-k2_data[k2_data["epic_number"]==star]["phot_g_mean_mag"][1]
                try:
                    star_rad=np.float(k2_data[k2_data["epic_number"]==star]["radius_val"][1])
                except:    
                    try:
                        star_rad=np.float(star_details.iloc[1,2].split("\xb1")[0])
                    except:
                        star_rad=1
            else:
                deltaMagGaia=k2_data[k2_data["epic_number"]==star]["phot_g_mean_mag"][0]-k2_data[k2_data["epic_number"]==star]["k2_kepmag"][0] 
                deltDistGaia=k2_data[k2_data["epic_number"]==star]["k2_gaia_ang_dist"][0]
                
        if star_rad==None:
            star_rad=1
            star_urad=1
        if star_mass==None:
            star_mass=1
            star_umass=1    
                         
                
        if deltDistGaia<0.0001:
            deltDistGaia=np.inf 
        if deltaMagGaia<0.0001:
            deltaMagGaia=np.inf       

        guy.update({"star_temp":np.float(star_temp)})
        guy.update({"star_utemp":np.float(star_utemp)})
        guy.update({"star_logg":np.float(star_logg)})
        guy.update({"star_ulogg":np.float(star_ulogg)})
        guy.update({"star_rad":np.float(star_rad)})
        guy.update({"star_urad":np.float(star_urad)})
        guy.update({"star_mass":np.float(star_mass)})
        guy.update({"star_umass":np.float(star_umass)})
        guy.update({"delta_dist":np.float(deltDist)})
        guy.update({"delta_mag":np.float(deltaMag)})
        guy.update({"delta_dist_Gaia":np.float(deltDistGaia)})
        guy.update({"delta_mag_Gaia":np.float(deltaMagGaia)})
        guy.update({"photo_ap":np.float(aper)})
        
        u1,u1e,u2,u2e=limbdark.claret(teff=star_temp, band="Kp", uteff=star_utemp,logg=star_logg,ulogg=star_ulogg,feh=0.16,ufeh=0.1, law="quadratic")
        
        guy.update({"limb_parms":[u1,u2]})
        
        if injection==True:

            ranPer=(np.max(lc.t)-np.min(lc.t))/2
            injectPer=np.random.uniform(.5,ranPer,1)
            if 3*injectPer>ranPer*2:
                injectT0=np.random.uniform(0,(ranPer*2-2*injectPer),1)+np.min(lc.t)
            else:
                injectT0=np.random.uniform(0,injectPer,1)+np.min(lc.t)
            injectRad=(np.random.uniform((.01/star_rad),(.05/star_rad),1))
            injectB=np.random.uniform(0,1,1)

            params = batman.TransitParams()

            b=injectB
            params.per = injectPer                #orbital period
            params.t0 = injectT0                     #time of inferior conjunction
            params.rp = injectRad                      #planet radius (in units of stellar radii)
            params.a = (params.per**2*star_mass*gravc/(4*math.pi**2))**(1/3)/star_rad       #semi-major axis (in units of stellar radii)
            try:
                params.inc =math.acos(b/params.a)*180/math.pi
            except:
                params.inc = 90
                      #orbital inclination (in degrees)
            params.ecc = 0                      #eccentricity
            params.w = 90                 #longitude of periastron (in degrees)
            params.limb_dark = "quadratic"
            u1,u1e,u2,u2e=limbdark.claret(teff=star_temp, band="Kp", uteff=star_utemp,logg=star_logg,ulogg=star_ulogg,feh=0.16,ufeh=0.1, law="quadratic")
            params.u = [u1,u2]       #limb darkening model

            m1 = batman.TransitModel(params, np.array(lc.t) ,transittype="primary")    #initializes model
            flux = m1.light_curve(params)-1
            lc.f=np.median(lc.f[lc.fmask==False])*flux+lc.f
            injectDur=injectPer/(params.a*np.pi)*(1-b**2)**.5
            bgerr=np.std(lc.f[lc.fmask==False]/np.median(lc.f[lc.fmask==False]))

            injectMES=injectRad**2/bgerr*(ranPer*2/injectPer)**.5*(injectDur/0.0204)**.5*245*144*10*141

            guy.update({"inject_MES":np.float(injectMES)})
            guy.update({"limb_parms":params.u})

            guy.update({"inject_per":np.float(injectPer)})
            guy.update({"inject_T0":np.float(injectT0)})
            guy.update({"inject_rp":np.float(injectRad)})
            guy.update({"inject_b":np.float(injectB)})
            guy.update({"inject_MES":np.float(injectB)})

        
        falsePosCount=0
        fpRun=100

        
        try:
            pipe=terra.dispoTest.Pipeline(lc=lc, header=guy)
            terra.dispoTest.preprocess(pipe)
            terra.dispoTest.gpfit(pipe)

            # terra.dispoTest.grid_search(pipe,P1=pipe.inject_per-1,P2=pipe.inject_per+1)
        except:
            pipe=dispoTest.Pipeline(lc=lc, header=guy)
            dispoTest.preprocess(pipe)
            dispoTest.gpfit(pipe)
        
        for idxxx in range(fpRun):
            print(idxxx)
            periodRan=10**np.random.uniform(np.log10(.5),np.log10(40),1)
            tdurRan=10**np.random.uniform(np.log10(.05),np.log10(0.5),1)    
            try:
                dispoTest.grid_search(pipe,P1=.75*periodRan,P2=1.25*periodRan,tdurRan=tdurRan)
                if pipe.number_TCE>0:
                    for kk in range(1, pipe.number_TCE+1):
                        try:
                            terra.dispoTest.find_PhotError(pipe)
                            terra.dispoTest.jon_fit(pipe,kk, ap=aper)
                        except NameError:
                            dispoTest.find_PhotError(pipe)
                            dispoTest.jon_fit(pipe,kk, ap=aper)   
                    # if not os.path.exists('./TCE/{}/'.format(pipe.starname)):
                    #     os.makedirs('./TCE/{}/'.format(pipe.starname))
                        if eval("pipe.falsePos_m{}".format(plM))=="False":
                            falsePosCount=falsePosCount+1
            except:
                pass                
                        


        fwrite = open('./TCE/{}/dispTest.txt'.format(pipe.starname), 'w')
        fwrite.write(str(falsePosCount/fpRun))  # python will convert \n to os.linesep
        fwrite.close()
# guy={"starname": 210777017}
# lc=pd.read_csv("./lc_ex/210777017.csv",skiprows=39)
# lc.columns = ['t',    'cad',    'fsap',    'ferr',    'fdt_t_rollmed',    'f',    'bgmask',    'thrustermask',    'fdtmask',    'fmask']
# aper=9
#
# pipe=terra.pipeline.Pipeline(lc=lc, header=guy)
# terra.pipeline.preprocess(pipe)
# terra.pipeline.grid_search(pipe)
# for kk in range(1, pipe.number_TCE+1):
#     terra.pipeline.jon_fit(pipe,kk, ap=aper)
# pipe.header.to_csv('./TCE/{}/header.csv'.format(pipe.starname))
#
# for iii in range(0,len(allFiles)):
#     # print(iii)
#     p = Pool(processes = 1)#multiprocessing.cpu_count()-1)
#    # print(file_)
#     async_result = p.map(TerraProcess,range(0,len(allFiles)))
#     p.close()
#     p.join()
#     print("Complete")


#
# for file_ in allFiles:

