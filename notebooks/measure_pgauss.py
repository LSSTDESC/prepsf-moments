import os
import logging
import functools
import collections
import sys

import ngmix
import numpy as np
import fitsio
import galsim
import joblib
import tqdm
from ngmix.prepsfmom import PGaussMom


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


def get_gal_wldeblend(*, rng, data):
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

    return (
        galsim.Sum([
            data.builders[band].from_catalog(
                data.cat[rind], 0, 0,
                data.surveys[band].filter_band).model.rotate(
                    angle * galsim.degrees)
            for band in range(len(data.builders))
        ]),
        galsim.Kolmogorov(fwhm=data.psf_fwhm),
        data.cat["redshift"][rind],
    )


def _make_obs(gal, psf, nse, rng, n=101):
    im = galsim.Convolve([gal, psf]).drawImage(
        nx=n, ny=n, scale=0.2,
    ).array
    psf_im = psf.drawImage(nx=n, ny=n, scale=0.2).array
    cen = (n-1)/2

    im += rng.normal(size=im.shape, scale=nse)

    obs = ngmix.Observation(
        image=im,
        weight=np.ones_like(im)/nse**2,
        jacobian=ngmix.DiagonalJacobian(scale=0.2, row=cen, col=cen),
        psf=ngmix.Observation(
            image=psf_im,
            weight=np.ones_like(im),
            jacobian=ngmix.DiagonalJacobian(scale=0.2, row=cen, col=cen),
        ),
    )
    return obs


def _meas(gal, psf, redshift, nse, aps, seed):
    rng = np.random.RandomState(seed=seed)
    obs = _make_obs(
        gal,
        psf,
        nse,
        rng,
    )

    s2ns = []
    g1s = []
    ts = []
    trs = []
    flags = []
    g1errs = []
    redshifts = []
    for ap in aps:
        mom = PGaussMom(ap).go(obs)
        psf_mom = PGaussMom(ap).go(obs.psf, no_psf=True)
        if psf_mom["flags"] == 0:
            psf_mom_t = psf_mom["T"]
        else:
            psf_mom_t = np.nan

        flags.append(mom["flags"] | psf_mom["flags"])
        s2ns.append(mom["s2n"])
        g1s.append(mom["e1"])
        g1errs.append(mom["e_err"][0])
        ts.append(mom["T"])
        trs.append(mom["T"]/psf_mom_t)
        redshifts.append(redshift)

    return s2ns, g1s, flags, ts, trs, g1errs, redshifts


def main():
    n_per_chunk = 100
    n_chunks = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    seed = np.random.randint(low=1, high=2**29)
    rng = np.random.RandomState(seed=seed)

    os.makedirs("./results", exist_ok=True)

    wldeblend_data = init_wldeblend(survey_bands="lsst-r")

    aps = np.linspace(1.25, 2.75, 25)
    outputs = []
    with joblib.Parallel(n_jobs=-1, verbose=10, batch_size=2) as par:
        for chunk in tqdm.trange(n_chunks):
            if False:
                for i in tqdm.trange(n_per_chunk):
                    gal, psf, redshift = get_gal_wldeblend(rng=rng, data=wldeblend_data)
                    outputs.append(_meas(
                        gal, psf, redshift, wldeblend_data.noise,
                        aps, rng.randint(low=1, high=2**29)
                    ))
            else:
                jobs = []
                for i in range(n_per_chunk):
                    gal, psf, redshift = get_gal_wldeblend(rng=rng, data=wldeblend_data)
                    jobs.append(joblib.delayed(_meas)(
                        gal, psf, redshift, wldeblend_data.noise,
                        aps, rng.randint(low=1, high=2**29))
                    )

                outputs.extend(par(jobs))

            d = np.zeros(len(outputs), dtype=[
                ("s2n", "f4", (len(aps),)),
                ("e1", "f4", (len(aps),)),
                ("T", "f4", (len(aps),)),
                ("Tratio", "f4", (len(aps),)),
                ("flags", "i4", (len(aps),)),
                ("e1_err", "i4", (len(aps),)),
                ("redshift", "f4", (len(aps),)),
            ])
            _o = np.array(outputs)
            d["s2n"] = _o[:, 0]
            d["e1"] = _o[:, 1]
            d["flags"] = _o[:, 2]
            d["T"] = _o[:, 3]
            d["Tratio"] = _o[:, 4]
            d["e1_err"] = _o[:, 5]
            d["redshift"] = _o[:, 6]

            fitsio.write(
                "./results/meas_seed%d.fits" % seed,
                d, extname="data", clobber=True)
            fitsio.write("./results/meas_seed%d.fits" % seed, aps, extname="aps")


if __name__ == "__main__":
    main()
