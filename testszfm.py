import szfastmap as szfm
import numpy as np

cosmo = szfm.cosmology.cosmology()

hmf = szfm.hmf.hmf(cosmo)

lc = szfm.lightcone.lightcone(cosmo=cosmo,fsky=0.1,Mmin=2e14)