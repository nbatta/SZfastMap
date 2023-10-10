import numpy as np
from astropy.io import fits
from enlib import mapdata, clusters
import szfastmap as szfm
from pixell import enplot, enmap
from scipy import ndimage

version = '3D'

indir = "./output/"

nbins = 25

savepdfs = []

Nsamps = 400
reals = 10

for j in range(Nsamps):
    avgh = 0.
    avgh = np.zeros(nbins)
    for i in range (reals):
        f = indir + "ymap"+version+"_par"+str(j)+"_r"+str(i)+".fits"
        imap = enmap.read_map(f)
        h1, be1 = np.histogram(np.log10(-1*imap[0]),bins=nbins,range = [-2.5,2.5])
        avgh += h1
        imap = 0

    b1 = (be1[:-1] + be1[1:])/2.
    avgh /= nbins
    savepdfs = np.append(savepdfs,avgh)
    
savebins = b1
np.savez(indir+"avgpdfs"+version+".npz",savebins,savepdfs)