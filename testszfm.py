import szfastmap as szfm
import numpy as np
from pixell import enmap, utils, bunch, pointsrcs
from enlib import clusters
import time

dtype   = np.float32

inputdir = "/Users/nab/Desktop/desktop_stuff/ysims/input/"
#mapfile = inputdir + "cmb_daynight_tot_f150_coadd.zip" #"fullsky_actbeam_f150.zip"
#mapfile = inputdir + "fullsky_actbeam_f090.zip"
geometry = inputdir + 'my_geometry.fits'

beam = inputdir + "cmb_daynight_tot_f150_coadd/beam.txt"
freq = 150 * 1e9

ofile = "Test_ymap1.fits"

start = time.time()

cosmo = szfm.cosmology.cosmology()

Cosmo4enlib = {'Omega_m':cosmo.omegam,'h':cosmo.h,'Omega_b':cosmo.omegab}

hmf = szfm.hmf.hmf(cosmo)

lc = szfm.lightcone.lightcone(cosmo=cosmo,fsky=0.02,Mmin=2e14)

lc.populate(hmf.dndmofmz)

# write halos in pksc halo format used in Websky
catfile = 'test_cat.pksc'
lc.write_pksc(catfile=catfile)

#bsize   = 100000.
margin  = 100
beta_range = [-14,-3]
vmin    = 0.001

print ("nhalo start",time.time() - start)
nhalo   = clusters.websky_pkcs_nhalo(catfile)
print ("nhalo = ",nhalo)
prof_builder= clusters.ProfileBattagliaFast(cosmology=Cosmo4enlib, beta_range=beta_range)
mass_interp = clusters.MdeltaTranslator(Cosmo4enlib)
print ("websky read",time.time() - start)
data  = clusters.websky_pkcs_read(catfile)

#meta        = mapdata.read_meta(mapfile)
shape, wcs  = enmap.read_map_geometry(geometry)
omap        = enmap.zeros(shape[-2:], wcs, dtype)
rht         = utils.RadialFourierTransform()
#lbeam       = np.interp(rht.l, np.arange(len(meta.beam)), meta.beam)
# Read the beam from one of the two formats
try:
	sigma = float(beam)*utils.fwhm*utils.arcmin
	lbeam = np.exp(-0.5*rht.l**2*sigma**2)
except ValueError:
	l, bl = np.loadtxt(beam, usecols=(0,1), ndmin=2).T
	lbeam = np.interp(rht.l, l, bl)

#fullsky = enmap.area(shape, wcs)/(4*np.pi) > 0.8

#if not fullsky:
#	pixs  = enmap.sky2pix(shape, wcs, utils.rect2ang(data.T[:3])[::-1])
#	good  = np.all((pixs >= -margin) & (pixs < np.array(shape[-2:])[:,None]+margin), 0)
#	data  = data[good]

print ("cat",time.time() - start)

cat    = clusters.websky_decode(data, Cosmo4enlib, mass_interp); del data

print ("prof builder",time.time() - start)

rprofs  = prof_builder.y(cat.m200[:,None], cat.z[:,None], rht.r)
lprofs  = rht.real2harm(rprofs)
lprofs  *= lbeam
rprofs  = rht.harm2real(lprofs)
r, rprofs = rht.unpad(rht.r, rprofs)
# and factor out peak value
yamps   = rprofs[:,0].copy()
rprofs /= yamps[:,None]
# Prepare for painting
#amps   = (yamps * utils.tsz_spectrum(meta.freq*1e9) / utils.dplanck(meta.freq*1e9) * 1e6).astype(dtype)
amps   = (yamps * utils.tsz_spectrum(freq) / utils.dplanck(freq) * 1e6).astype(dtype)
poss   = np.array([cat.dec,cat.ra]).astype(dtype)
profiles = [np.array([r,prof]).astype(dtype) for prof in rprofs]; del rprofs
prof_ids = np.arange(len(profiles)).astype(np.int32)

print ("sim obs",time.time() - start)

pointsrcs.sim_objects(shape, wcs, poss, amps, profiles, prof_ids=prof_ids, omap=omap, vmin=vmin)

print ("write map",time.time() - start)

enmap.write_map(ofile, omap)

print ("end",time.time() - start)

#print (poss)
