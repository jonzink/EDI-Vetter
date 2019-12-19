# EDI-Vetter
This is a program meant identify false positive transit signal in the K2 data set. This program has been simplified to test single transiting planet signals. Systems with multiple signals require additional testing, which will be made available in a later iteration.   

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for research, development, and testing purposes. EDI-Vetter was written in Python 3.4 

### Prerequisites

Several python packages are required to run this software. Here are a few:
Pandas
Numpy
emcee
scipy
lmfit
batman
astropy




## Running EDI-Vetter in Python

Here we provide quick example using the light curve provided.

Begin by opening Python in the appropriate directory. 
```
$ python
```
Now import the necessary packages
```
>>> import pandas as pd
>>> import EDI_Vetter
```
Import the light curve file
```
>>> lc=pd.read_csv("K2_138.csv")
```
Now you can set up the EDI-Vetter parameters object with the appropriate transit signal parameters 
```
>>> params=EDI_Vetter.parameters(per=8.26144,t0=2907.6451,radRatio=0.0349,tdur=0.128,lc=lc)
```
It is essential that EDII-Vetter re-fits the light curve to measure changes from the transit detection.
```
>>> params=EDI_Vetter.MCfit(params)
```
Now you can run all of the vetting metrics on the signal
```
>>> params=EDI_Vetter.Go(params,delta_mag=10,delta_dist=1000, photoAp=41)
```
