import os
import logging
import functools
import collections

import ngmix
import numpy as np
import fitsio
import galsim

from numpy import log, sqrt

# beta for fixed moffat psfs
FIXED_MOFFAT_BETA = 2.5

RAND_PSF_FWHM_STD = 0.1
RAND_PSF_FWHM_MIN = 0.6
RAND_PSF_FWHM_MAX = 1.3
RAND_PSF_E_STD = 0.01
RAND_PSF_E_MAX = 0.10


def _make_rand_psf(rng, med_fwhm):
    fwhm = _get_fwhm(rng, med_fwhm)
    e1, e2 = _get_e1e2(rng)
    psf = galsim.Moffat(fwhm=fwhm, beta=FIXED_MOFFAT_BETA)
    psf = psf.shear(e1=e1, e2=e2)
    return psf


def _get_fwhm(rng, med_fwhm):
    ln_mean = log(
        med_fwhm**2 / sqrt(med_fwhm**2 + med_fwhm**2)
    )  # noqa
    ln_sigma = sqrt(log(1+(RAND_PSF_FWHM_STD/med_fwhm)**2))

    while True:
        fwhm = rng.lognormal(
            mean=ln_mean,
            sigma=ln_sigma,
        )
        if RAND_PSF_FWHM_MIN < fwhm < RAND_PSF_FWHM_MAX:
            break

    return fwhm


def _get_e1e2(rng):
    while True:
        e1, e2 = rng.normal(scale=RAND_PSF_E_STD, size=2)
        e = sqrt(e1**2 + e2**2)
        if e < RAND_PSF_E_MAX:
            break

    return e1, e2


LOGGER = logging.getLogger(__name__)
WLDeblendData = collections.namedtuple(
    'WLDeblendData',
    [
        'cat', 'survey_name', 'bands', 'surveys',
        'builders', 'total_sky', 'noise', 'ngal_per_arcmin2',
        'psf_fwhm', 'pixel_scale',
    ],
)


@functools.lru_cache(maxsize=8)
def _cached_catalog_read():
    fname = os.path.join(
        os.environ.get('CATSIM_DIR', '.'),
        'OneDegSq.fits',
    )
    return fitsio.read(fname)


@functools.lru_cache(maxsize=8)
def init_wldeblend(*, survey_bands):
    """Initialize weak lensing deblending survey data.

    Parameters
    ----------
    survey_bands : str
        The name of the survey followed by the bands like 'des-riz', 'lsst-iz', etc.

    Returns
    -------
    data : WLDeblendData
        Namedtuple with data for making galaxies via the weak lesning
        deblending package.
    """
    survey_name, bands = survey_bands.split("-")
    bands = [b for b in bands]
    LOGGER.info('simulating survey: %s', survey_name)
    LOGGER.info('simulating bands: %s', bands)

    if survey_name not in ["des", "lsst"]:
        raise RuntimeError(
            "Survey for wldeblend must be one of 'des' or 'lsst'"
            " - got %s!" % survey_name
        )

    if survey_name == "lsst":
        scale = 0.2
    elif survey_name == "des":
        scale = 0.263

    # guard the import here
    import descwl

    # set the exposure times
    if survey_name == 'des':
        exptime = 90 * 10
    else:
        exptime = None

    wldeblend_cat = _cached_catalog_read()
    if "survey_bands" == "lsst-r":
        wldeblend_cat = wldeblend_cat[wldeblend_cat["r_ab"] < 26]

    surveys = []
    builders = []
    total_sky = 0.0
    for iband, band in enumerate(bands):
        # make the survey and code to build galaxies from it
        pars = descwl.survey.Survey.get_defaults(
            survey_name=survey_name.upper(),
            filter_band=band)

        pars['survey_name'] = survey_name
        pars['filter_band'] = band
        pars['pixel_scale'] = scale

        # note in the way we call the descwl package, the image width
        # and height is not actually used
        pars['image_width'] = 100
        pars['image_height'] = 100

        # reset the exposure times if we want
        if exptime is not None:
            pars['exposure_time'] = exptime

        # some versions take in the PSF and will complain if it is not
        # given
        try:
            _svy = descwl.survey.Survey(**pars)
        except Exception:
            pars['psf_model'] = None
            _svy = descwl.survey.Survey(**pars)

        surveys.append(_svy)
        builders.append(descwl.model.GalaxyBuilder(
            survey=surveys[iband],
            no_disk=False,
            no_bulge=False,
            no_agn=False,
            verbose_model=False))

        total_sky += surveys[iband].mean_sky_level

    noise = np.sqrt(total_sky)

    if survey_name == "lsst":
        psf_fwhm = 0.85
    elif survey_name == "des":
        psf_fwhm = 1.1

    # when we sample from the catalog, we need to pull the right number
    # of objects. Since the default catalog is one square degree
    # and we fill a fraction of the image, we need to set the
    # base source density `ngal`. This is in units of number per
    # square arcminute.
    ngal_per_arcmin2 = wldeblend_cat.size / (60 * 60)

    LOGGER.info('catalog density: %f per sqr arcmin', ngal_per_arcmin2)

    return WLDeblendData(
        wldeblend_cat, survey_name, bands, surveys,
        builders, total_sky, noise, ngal_per_arcmin2,
        psf_fwhm, scale,
    )


def get_gal_wldeblend(*, rng, data, vary_psf=False):
    """Draw a galaxy from the weak lensing deblending package.

    Parameters
    ----------
    rng : np.random.RandomState
        An RNG to use for making galaxies.
    data : WLDeblendData
        Namedtuple with data for making galaxies via the weak lesning
        deblending package.

    Returns
    -------
    gal : galsim Object
        The galaxy as a galsim object.
    psf : galsim Object
        The PSF as a galsim object.
    """
    rind = rng.choice(data.cat.size)
    angle = rng.uniform() * 360
    pa_angle = rng.uniform() * 360

    data.cat['pa_disk'][rind] = pa_angle
    data.cat['pa_bulge'][rind] = pa_angle

    if vary_psf:
        psf = _make_rand_psf(rng, data.psf_fwhm)
    else:
        psf = galsim.Moffat(fwhm=data.psf_fwhm, beta=FIXED_MOFFAT_BETA)

    return (
        galsim.Sum([
            data.builders[band].from_catalog(
                data.cat[rind], 0, 0,
                data.surveys[band].filter_band).model.rotate(
                    angle * galsim.degrees)
            for band in range(len(data.builders))
        ]),
        psf,
        data.cat["redshift"][rind],
    )


def make_ngmix_obs(*, gal, psf, nse, pixel_scale, rng, n=101):
    xoff, yoff = rng.uniform(size=2, low=-0.5, high=0.5)
    im = galsim.Convolve([gal, psf]).drawImage(
        nx=n, ny=n, scale=pixel_scale, offset=(xoff, yoff),
    ).array
    psf_im = psf.drawImage(nx=n, ny=n, scale=pixel_scale).array
    cen = (n-1)/2

    _im = im + rng.normal(size=im.shape, scale=nse)

    obs = ngmix.Observation(
        image=_im,
        weight=np.ones_like(im)/nse**2,
        jacobian=ngmix.DiagonalJacobian(scale=pixel_scale, row=cen+yoff, col=cen+xoff),
        psf=ngmix.Observation(
            image=psf_im,
            weight=np.ones_like(im),
            jacobian=ngmix.DiagonalJacobian(scale=pixel_scale, row=cen, col=cen),
        ),
    )

    obs_nn = ngmix.Observation(
        image=im,
        weight=np.ones_like(im)/nse**2,
        jacobian=ngmix.DiagonalJacobian(scale=pixel_scale, row=cen+yoff, col=cen+xoff),
        psf=ngmix.Observation(
            image=psf_im,
            weight=np.ones_like(im),
            jacobian=ngmix.DiagonalJacobian(scale=pixel_scale, row=cen, col=cen),
        ),
    )

    return obs, np.sum(im), obs_nn
