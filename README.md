# SZFastMap

Fast, random thermal Sunyaev–Zel'dovich (tSZ) maps of galaxy clusters.

SZFastMap skips the N-body simulation. It draws a halo catalogue directly from an analytic halo mass function on a light cone, then paints a beam-convolved pressure profile onto each halo. One 40° × 40° map with 0.5′ pixels takes under a minute on a laptop. That makes it practical to generate thousands of realisations across a grid of cosmological and astrophysical parameters, for example to emulate or fit the one-point PDF of a tSZ map.

Authors: Marcelo Alvarez, Sigurd Naess, Nick Battaglia. MIT licence.

## How it works

```
cosmology ──► hmf ──► lightcone ──► halos.pksc ──► enlib.clusters + pixell ──► ΔT map (FITS)
(χ(z))       (Tinker08    (Poisson       (Websky       (Battaglia profile,         (μK at 150 GHz)
              dn/dM)       halos)         format)       beam, painting)
```

1. **`szfastmap.cosmology`** defines a flat ΛCDM cosmology (Ω_b, Ω_c, h, n_s, A_s) and tabulates the comoving distance χ(z) and its inverse z(χ) for 0 < z < 6.
2. **`szfastmap.hmf`** computes the linear matter power spectrum with CAMB, then the [Tinker et al. (2008)](https://arxiv.org/abs/0803.2706) Δ = 200 (mean) halo mass function dn/dM(M, z). The amplitude is set by `As`; `sigma8` is not used.
3. **`szfastmap.lightcone`** splits a square sky patch of area `fsky` into redshift shells (Δz = 0.1). In each shell it Poisson-samples the number of halos above `Mmin`, draws their masses from N(>M), and places them uniformly in comoving volume. The catalogue can be written in the [Websky](https://mocks.cita.utoronto.ca/data/websky/v0.0/) `.pksc` format.
4. **Map painting** happens in the scripts, not the package. It uses [`enlib.clusters`](https://github.com/amaurea/enlib) to build [Battaglia et al. (2012)](https://arxiv.org/abs/1109.3711) pressure profiles for each halo, convolves them with the instrument beam, and paints them onto a CAR map with [`pixell.pointsrcs.sim_objects`](https://github.com/simonsobs/pixell).

**Map units.** The output maps are the beam-convolved tSZ temperature decrement **ΔT in μK_CMB at 150 GHz**, not Compton-y. Values are ≤ 0. Some filenames still say `ymap`. To get another frequency, change `freq`. To get Compton-y instead, drop the `utils.tsz_spectrum(freq) / utils.dplanck(freq) * 1e6` factor.

## Installation

Dependencies: `numpy`, `scipy`, `matplotlib`, `astropy`, [`camb`](https://camb.readthedocs.io), [`pixell`](https://github.com/simonsobs/pixell), and [`enlib`](https://github.com/amaurea/enlib). Only the map-making scripts need `enlib`; the `szfastmap` package itself does not.

```bash
# a conda environment is recommended
conda create -n szfastmap python=3.10 numpy scipy matplotlib astropy
conda activate szfastmap
pip install camb pixell

# enlib is not on PyPI: clone it and put the directory *containing* it on your PYTHONPATH
git clone https://github.com/amaurea/enlib.git ~/Repos/enlib
export PYTHONPATH=~/Repos:$PYTHONPATH

# SZFastMap itself
git clone git@github.com:nbatta/SZfastMap.git
cd SZfastMap
pip install -e .
```

## Quick start: a halo catalogue

```python
import szfastmap as szfm

cosmo = szfm.cosmology.cosmology(Omega_c=0.27, As=2e-9)   # Omega_b=0.049, h=0.68, ns=0.965 by default

# Tinker08 mass function; tabulated once and cached in ./dndmtab.npz
hmf = szfm.hmf.hmf(cosmo, newtable=True)

# ~1600 deg^2 patch, halos with M200m > 4e14 Msun, 0 < z < 5
lc = szfm.lightcone.lightcone(cosmo=cosmo, fsky=0.039, Mmin=4e14)
lc.populate(hmf.dndmofmz)

print(lc.Nobj)              # number of halos (~1600 for this setup)
lc.M, lc.x, lc.y, lc.z      # masses [Msun] and comoving positions [Mpc]

lc.write_pksc(catfile="halos.pksc")
```

### Options

| Object | Keyword | Default | Meaning |
|---|---|---|---|
| `cosmology` | `Omega_b`, `Omega_c`, `h`, `ns`, `As` | 0.049, 0.261, 0.68, 0.965, 2e-9 | Flat ΛCDM parameters |
| `hmf` | `mmin`, `mmax` | 1e13, 1e16 | Mass range of the tabulated dn/dM [Msun] |
| | `zmin`, `zmax` | 0.0, 4.5 | Redshift range of the table. Outside it, the edge value is used |
| | `newtable` | `False` | Recompute the table. **Set this to `True` whenever you change cosmology** (see [Known issues](#known-issues)) |
| `lightcone` | `cosmo` | default `cosmology()` | Cosmology object |
| | `fsky` | 0.01 | Sky fraction (must be ≤ 0.5, or exactly 1 for the full sky) |
| | `Mmin`, `Mmax` | 1e13, 1e16 | Halo mass range [Msun] |
| | `zmin`, `zmax`, `dz` | 0.0, 5.0, 0.1 | Light-cone redshift range and shell width |

**Geometry.** The patch is a square of side `fov` (in `lc.fov`, degrees), centred on the +x axis, i.e. RA = 0, Dec = 0. Make the map geometry match it. For example, `fsky=0.039` ↔ 40° × 40°, and `fsky=0.0097` ↔ 20° × 20°.

**Catalogue format.** `write_pksc` writes an int32 header `[Nhalo, 0, 0]`, followed by 10 float32 values per halo: `x, y, z, vx, vy, vz, R, xL, yL, zL`. Positions are comoving Mpc (no h), and `R = (3M / 4πρ̄_m)^{1/3}` encodes the mass. Velocities and Lagrangian positions are **placeholders** (copies of the position), so these catalogues are not suitable for kSZ.

## Making a tSZ map

This is a condensed version of [`mksimobs.py`](mksimobs.py) and [`testszfm.py`](testszfm.py). Run it from the repository root so that the `input/` paths resolve.

```python
import numpy as np
import szfastmap as szfm
from pixell import enmap, utils, pointsrcs
from enlib import clusters

dtype = np.float32
freq  = 150e9                                             # observing frequency [Hz]
geometry = "input/my_geometry.fits"                       # 20 x 20 deg CAR, 0.5' pixels
beam     = "input/cmb_daynight_tot_f150_coadd/beam.txt"   # ell, B_ell (or a FWHM in arcmin, e.g. "1.4")

# 1. halo catalogue
cosmo = szfm.cosmology.cosmology(Omega_c=0.27, As=2e-9)
hmf   = szfm.hmf.hmf(cosmo, newtable=True)
lc    = szfm.lightcone.lightcone(cosmo=cosmo, fsky=0.0097, Mmin=4e14)
lc.populate(hmf.dndmofmz)
catfile = "halos.pksc"
lc.write_pksc(catfile=catfile)

# 2. Battaglia pressure profiles, convolved with the beam
cosmo_enlib = {'Omega_m': cosmo.omegam, 'h': cosmo.h, 'Omega_b': cosmo.omegab}
astro_enlib = {'beta': 1., 'P0': 1.}          # P0 rescales the pressure-profile amplitude
prof_builder = clusters.ProfileBattagliaFast(cosmology=cosmo_enlib, astropars=astro_enlib, beta_range=[-14, -3])
mass_interp  = clusters.MdeltaTranslator(cosmo_enlib)
cat = clusters.websky_decode(clusters.websky_pkcs_read(catfile), cosmo_enlib, mass_interp)

rht = utils.RadialFourierTransform()
try:
    sigma = float(beam)*utils.fwhm*utils.arcmin
    lbeam = np.exp(-0.5*rht.l**2*sigma**2)
except ValueError:
    l, bl = np.loadtxt(beam, usecols=(0, 1), ndmin=2).T
    lbeam = np.interp(rht.l, l, bl)

rprofs = prof_builder.y(cat.m200[:, None], cat.z[:, None], rht.r)
rprofs = rht.harm2real(rht.real2harm(rprofs) * lbeam)
r, rprofs = rht.unpad(rht.r, rprofs)
yamps   = rprofs[:, 0].copy()
rprofs /= yamps[:, None]

# 3. paint onto the map, converting y -> Delta T [uK] at freq
shape, wcs = enmap.read_map_geometry(geometry)
omap  = enmap.zeros(shape[-2:], wcs, dtype)
amps  = (yamps * utils.tsz_spectrum(freq) / utils.dplanck(freq) * 1e6).astype(dtype)
poss  = np.array([cat.dec, cat.ra]).astype(dtype)
profiles = [np.array([r, p]).astype(dtype) for p in rprofs]
pointsrcs.sim_objects(shape, wcs, poss, amps, profiles,
                      prof_ids=np.arange(len(profiles)).astype(np.int32), omap=omap, vmin=0.001)
enmap.write_map("tsz_150GHz.fits", omap)
```

This produces a 2400 × 2400 map containing about 400 clusters, with ΔT ranging down to several hundred μK. To view it:

```python
from pixell import enplot
enplot.write("tsz_150GHz", enplot.plot(-omap, color="gray", min=0, max=50))
```

## Scripts

All scripts read from `./input/` and write to `./output/`. Run them from the repository root and create the output directory first with `mkdir -p output`.

| Script | What it does | Output |
|---|---|---|
| [`testszfm.py`](testszfm.py) | Makes one end-to-end test map: fiducial cosmology, 40° × 40°, f150 beam | `output/astropartest12.fits`, `test_cat.pksc` |
| [`mksimobs.py`](mksimobs.py) | Makes a mock "observed" map at the fiducial cosmology (Ω_c = 0.27, A_s = 2e-9, P0 = 1) and its tSZ PDF (histogram of log10(−ΔT), 50 bins over [−1, 3]; with `scales=2` it also histograms a 4× downsampled map and appends the result) | `output/simobs{version}.fits`, `output/simobspdf{version}_{nbins}[_2scl].npz` |
| [`szfm_lhcsample.py`](szfm_lhcsample.py) | Draws a Latin hypercube in (Ω_c ∈ [0.26, 0.28], A_s ∈ [1.5, 2.5] × 10⁻⁹), 1000 points × 10 random realisations each | `output/params_2dv3.npz` (sampled parameters, key `arr_0`), `output/ymap2dv3_par{j}_r{i}.fits` |
| [`compute_avg_pdfs.py`](compute_avg_pdfs.py) | For each parameter point, averages the tSZ PDF over its realisations (same binning and `scales` option as `mksimobs.py`) | `output/avgpdfs{version}_{nbins}[_2scl].npz` (`arr_0` = bin centres, `arr_1` = flattened PDFs, `Nsamps × nbins·scales`) |

Together they form a simulation-based-inference workflow. `szfm_lhcsample.py` + `compute_avg_pdfs.py` build a training set of PDFs versus parameters, and `mksimobs.py` builds a mock data vector to compare it with.

Settings (`version`, `Nsamps`, `nbins`, `scales`, parameter bounds) are set as variables at the top of each script. When you change `version` in one script, change it in the others as well.

**Cost.** Each 40° × 40° map takes about 45 s on a laptop, and the HMF table is recomputed for every sample. The full 1000 × 10 hypercube is therefore expensive and produces about 10⁴ maps of roughly 90 MB each at 4800², so run it on a cluster.

## Input files

| File | Description |
|---|---|
| `input/my_geometry.fits` | CAR geometry, 2400 × 2400 pixels at 0.5′ (20° × 20°), centred on RA = Dec = 0. Only the header is read |
| `input/my_geometry2.fits` | CAR geometry, 4800 × 4800 pixels at 0.5′ (40° × 40°), matches `fsky=0.039` |
| `input/cmb_daynight_tot_f150_coadd/beam.txt` | ACT f150 beam transfer function: columns `ell`, `B_ell` |
| `input/cmb_daynight_tot_f150_coadd/info.txt` | Beam metadata (frequency, gain) |

To make maps for another instrument, use your own geometry (for example from `pixell.enmap.geometry(...)`) and either a beam file or a Gaussian FWHM in arcmin.

## Known issues

- **Stale mass-function cache.** `hmf` saves its table as `dndmtab.npz` in the *current working directory* and does not record which cosmology produced it. If you change cosmology without `newtable=True`, you silently get the old mass function. All the scripts pass `newtable=True`.
- **Profiles use a fixed cosmology in the hypercube.** `szfm_lhcsample.py` builds pressure profiles with the *fiducial* cosmology, so the sampled Ω_c changes halo abundances but not the profiles.
- **`version` mismatch.** `szfm_lhcsample.py` writes `ymap2dv3_*` files, but `compute_avg_pdfs.py` reads `version='3dbv2'`. Edit one to match the other.
- **PDF uses one row of the map.** The maps are 2D, so `imap[0]` in `compute_avg_pdfs.py` and `mksimobs.py` selects the first row of pixels, not the whole map. Use `imap` to histogram the full map.
- **`fsky` limits.** `0.5 < fsky < 1` is not supported.
