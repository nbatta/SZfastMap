import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.optimize import curve_fit
import scipy.integrate

import sys
import os
from os import path

import camb
from camb import model, initialpower

# routines to get dndm using the Tinker (2008) fitting function

class hmf:
    '''Halo mass function'''

    def __init__(self, cosmo, **kwargs):

        self.mmin = kwargs.get('mmin',1e13)
        self.mmax = kwargs.get('mmin',1e16)
        self.zmin = kwargs.get('zmin',0.0)
        self.zmax = kwargs.get('zmax',4.5)
        self.newt = kwargs.get('newtable',False)

        mmin = self.mmin
        mmax = self.mmax
        zmin = self.zmin
        zmax = self.zmax
        newt = self.newt

        self.h      = cosmo.h
        self.omegab = cosmo.omegab
        self.omegam = cosmo.omegam
        self.As = cosmo.As
        self.ns     = cosmo.ns
        self.omegal = 1-self.omegam

        self.rho_mean    = 2.775e11 * self.omegam * h**2

        if not path.exists('dndmtab.npz') or newt:

            print("\n creating table")
            dlogm = 0.05
            dz    = 0.1

            nm = int((np.log10(mmax)-np.log10(mmin))/dlogm)+1
            nz = int((zmax-zmin)/dz)+1

            m = np.logspace(np.log10(mmin),np.log10(mmax),nm)
            z = np.linspace(zmin,zmax,nz)

            '''Initialize power spectrum'''
            kmax = 340.
            pars = camb.CAMBparams()
            pars.set_cosmology(H0=self.h*100, ombh2=self.omegab*self.h**2, omch2=self.omegam*self.h**2)
            pars.InitPower.set_params(ns=self.ns,As=self.As)
            pars.set_matter_power(redshifts=[0.], kmax=kmax)
            pars.NonLinear = model.NonLinear_none
            results = camb.get_results(pars)
            self.k, _, pktemp = results.get_linear_matter_power_spectrum(hubble_units=False,k_hunit = False)
            self.pk = pktemp[0,:]

            dndmofmz = np.zeros((nz,nm))

            iz=0
            for zv in z:
                print(" z: ","{:4.2f}".format(zv),
                  end="\r", flush=True)
                m_t,dndm_t, f = dndmofm_tinker(mmin,mmax,zv) # M, dndM in Msun, 1/Mpc^3/Msun
                dndmofmz[iz,:] = np.log10(f(m))
                iz += 1
            np.savez('dndmtab.npz',m=m,z=z,dndmofmz=dndmofmz)

        else:

            data = np.load('dndmtab.npz')
            m = data['m']
            z = data['z']
            dndmofmz = data['dndmofmz']

        dndmofmzfunc_log = interp2d(np.log10(m),z,dndmofmz)
        dndmofmzfunc     = lambda m,z: 10**dndmofmzfunc_log(np.log10(m),z)

        self.dndmofmz = dndmofmzfunc

    def mass_to_radius(m):
        """
        Returns the radius of a sphere of uniform denstiy rho and mass m
        """
        r = (3 * m / 4. / np.pi / self.rho_mean)**(1./3.)
        
        return r

    def windowfunction(x):
        """
        Computes the window function in Fourier space (top hat in real space).
        """
        W = (3. / x**3) * (np.sin(x) - x * np.cos(x))

        return W

    def M_to_sigma(k, pk, M):
        """
        Returns mass and sigma(mass)
        """

        # Normalization

        Anorm    = 1./(2.*np.pi**2)

        # Now compute sigma(M)

        sigma = np.zeros(M.shape[0])

        for i in range(sigma.shape[0]):

            radius = mass_to_radius(M[i])

            x = self.k * radius
            y = self.pk * (self.k * windowfunction(x))**2

            sigma[i] = np.sqrt(Anorm * sp.integrate.simps(y, self.k, even="avg"))

        return sigma

    def dlnsigmainv_dM(M, sigma):
        lnsigmainv = np.log(1/sigma)

        diff_sig = np.diff(lnsigmainv)
        diff_M   = np.diff(M)

        dlnsigmainvdM_int = diff_sig / diff_M
        Mint              = M[:-1] + diff_M / 2

        f = interp1d(Mint,dlnsigmainvdM_int,fill_value="extrapolate")

        return f(M)

    def growth_factor(z):
        """
        Returns growth factor using fitting formulae from Carrol, Press & Turner (1992)
        """
        # returns growth factor using fitting formulae from Carrol, Press & Turner (1992)

        omegaM = self.omegam
        omegaL = self.omegal

        w = -1.
        x = 1.+z
        x2 = x**2
        x3 = x**3
        x3w = x**(3*w)

        #calculate omegaM(z) and omegaL(z)
        denom = omegaM*x3 + (1-omegaM-omegaL)*x2 + omegaL*x3*x3w
        omega = omegaM*x3/denom
        lamb = omegaL*x3*x3w/denom

        #fitting formulae for w=-1 universe
        g = 2.5*omega/(omega**(4.0/7.0) - lamb + (1+(omega/2))*(1+(lamb/70)))
        g0 = 2.5*omegaM/(omegaM**(4.0/7.0) - omegaL + (1+(omegaM/2))*(1+(omegaL/70)))

        D = (g/x)/g0

        return D

    def tinker_func(x, z):
        """
        Uses fitting coefficients from Table 2 of Tinker et al. (2008) for Delta = 200 and
        redshift evolution equations 5 through 8.
        """

        A_hmfcalc = 1.858659e-01
        a_hmfcalc = 1.466904
        b_hmfcalc = 2.571104
        c_hmfcalc = 1.193958

        A =  0.186
        a = 1.47
        b = 2.57
        c = 1.19

        A =  A_hmfcalc
        a =  a_hmfcalc
        b =  b_hmfcalc
        c =  c_hmfcalc

        amp = A * (1. + z)**-0.14
        a = a * (1. + z)**-0.06
        alpha = 10**(-1. * (0.75 / np.log10(200. / 75.))**1.2)

        b = b * (1. + z)**-alpha
        c = c

        f = amp * ((x / b)**(-a) + 1.) * np.exp(-c / x**2)

        return f

    def dndmofm_tinker(Mmin, Mmax, redshift):
        """
        Returns dn/dm for Tinker et al. (2008) mass function using Eqs. (3, 5-8) of their paper.
        Assumes k, pk, omegam, omegal, h, and rho_mean have been defined already
        units: m [Msun], dndm [1/Msun/Mpc^3]
        """

        dlog = 0.001
        n    = int((np.log10(Mmax)-np.log10(Mmin))/dlog)
        M    = np.logspace(np.log10(Mmin),np.log10(Mmax),n)

        sigma  = M_to_sigma(self.k, self.pk, M)
        D      = growth_factor(redshift)
        sigma *= D

        fsigma = tinker_func(sigma, redshift)

        dlnsigmainv = dlnsigmainv_dM(M, sigma)

        dndm = tinker_func(sigma, redshift) * self.rho_mean / M * dlnsigmainv

        dndmofm = interp1d(M,dndm,fill_value='extrapolate')

        return M,dndm,dndmofm