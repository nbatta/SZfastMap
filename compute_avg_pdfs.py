import numpy as np
from astropy.io import fits
from enlib import mapdata, clusters
import szfastmap as szfm
from pixell import enplot, enmap
from scipy import ndimage

np.seterr(divide = 'ignore') 

version = '3d'

indir = "./output/"

Nsamps = 400
reals = 10

nbins = 50
ylolim = -1.
yhilim = 3.

savepdfs = []

for j in range(Nsamps):
    avgh = 0.
    avgh = np.zeros(nbins)
    for i in range (reals):
        f = indir + "ymap"+version+"_par"+str(j)+"_r"+str(i)+".fits"
        imap = enmap.read_map(f)
        h1, be1 = np.histogram(np.log10(-1*imap[0]),bins=nbins,range = [ylolim,yhilim])
        avgh += h1
        imap = 0

    print(j)
    b1 = (be1[:-1] + be1[1:])/2.
    avgh /= nbins
    savepdfs = np.append(savepdfs,avgh)
    
savebins = b1
np.savez(indir+"avgpdfs"+version+"_"+str(nbins)+".npz",savebins,savepdfs)
