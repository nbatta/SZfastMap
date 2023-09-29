import szfastmap as szfm
import numpy as np
from pixell import enmap, utils, bunch, pointsrcs
from enlib import clusters
import time, sys
from scipy.stats import qmc


dtype   = np.float32

inputdir  = "/Users/nab/Desktop/desktop_stuff/ysims/input/"
geometry  = inputdir + 'my_geometry2.fits'
outputdir = "/Users/nab/Repos/SZFastMap/sim_output/"

beam = inputdir + "cmb_daynight_tot_f150_coadd/beam.txt"
freq = 150 * 1e9

Nsamps = 200

sampler = qmc.LatinHypercube(d=4)
sample = sampler.random(n=Nsamps)

l_bounds = [0.26, 1.5e-9,0.8,0.5]
u_bounds = [0.28, 2.5e-9,1.2,1.5]
sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

sys.exit()

np.savez(outputdir + "params_4d.npz", sample_scaled)

#print (np.shape(sample_scaled))
#print (sample_scaled[0,:])

rht = utils.RadialFourierTransform()

cosmo_fid = szfm.cosmology.cosmology(Omega_c=0.27,As=2e-9)

try:
	sigma = float(beam)*utils.fwhm*utils.arcmin
	lbeam = np.exp(-0.5*rht.l**2*sigma**2)
except ValueError:
	l, bl = np.loadtxt(beam, usecols=(0,1), ndmin=2).T
	lbeam = np.interp(rht.l, l, bl)

for j in range(Nsamps):

	start = time.time()

	cosmo = szfm.cosmology.cosmology(Omega_c=sample_scaled[j,0],As=sample_scaled[j,1])
	Cosmo4enlib = {'Omega_m':cosmo_fid.omegam,'h':cosmo_fid.h,'Omega_b':cosmo_fid.omegab}
	Astro4enlib = {'beta':sample_scaled[j,2], 'P0':sample_scaled[j,3]}
	for i in range(10):
		ofile 		= outputdir + "ymap4d_par"+str(j)+"_r"+str(i)+".fits"
		shape, wcs  = enmap.read_map_geometry(geometry)
		omap  		= enmap.zeros(shape[-2:], wcs, dtype)
		print(ofile)

		hmf = szfm.hmf.hmf(cosmo,newtable=True)

		lc = szfm.lightcone.lightcone(cosmo=cosmo,fsky=0.039,Mmin=4e14)

		lc.populate(hmf.dndmofmz)

		# write halos in pksc halo format used in Websky
		catfile = 'test_cat.pksc'
		lc.write_pksc(catfile=catfile)
		#bsize   = 100000.
		margin  = 100
		beta_range = [-14,-3]
		vmin    = 0.001

		nhalo   = clusters.websky_pkcs_nhalo(catfile)
		prof_builder= clusters.ProfileBattagliaFast(cosmology=Cosmo4enlib, astropars=Astro4enlib, beta_range=beta_range)
		mass_interp = clusters.MdeltaTranslator(Cosmo4enlib)
		data  = clusters.websky_pkcs_read(catfile)

		cat    = clusters.websky_decode(data, Cosmo4enlib, mass_interp); del data

		rprofs  = prof_builder.y(cat.m200[:,None], cat.z[:,None], rht.r)
		lprofs  = rht.real2harm(rprofs)
		lprofs  *= lbeam
		rprofs  = rht.harm2real(lprofs)
		r, rprofs = rht.unpad(rht.r, rprofs)
		yamps   = rprofs[:,0].copy()
		rprofs /= yamps[:,None]
		amps   = (yamps * utils.tsz_spectrum(freq) / utils.dplanck(freq) * 1e6).astype(dtype)
		poss   = np.array([cat.dec,cat.ra]).astype(dtype)
		profiles = [np.array([r,prof]).astype(dtype) for prof in rprofs]; del rprofs
		prof_ids = np.arange(len(profiles)).astype(np.int32)

		pointsrcs.sim_objects(shape, wcs, poss, amps, profiles, prof_ids=prof_ids, omap=omap, vmin=vmin)

		enmap.write_map(ofile, omap)

#print ("end",time.time() - start)

#print (poss)
