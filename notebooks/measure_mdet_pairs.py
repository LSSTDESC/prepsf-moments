import os
import logging
import collections
import sys

import ngmix
import numpy as np
import fitsio
import galsim
import tqdm
import yaml
import joblib

from mattspy import BNLCondorParallel
from shear_meas import meas_m_c
from metadetect.metadetect import do_metadetect


BACKEND = "bnl"
USE_EXP = True
FLUX_FAC = 1
MIN_FLUX = 1.2e5  # about S/N ~ 20

MDET_CFG = yaml.safe_load("""\
model: pgauss

metacal:
  psf: fitgauss
  types: [noshear, 1p, 1m, 2p, 2m]
  use_noise_image: True

sx: null

nodet_flags: 33554432  # 2**25 is GAIA stars

bmask_flags: 1610612736  # 2**29 | 2**30 edge in either MEDS of pizza cutter

mfrac_fwhm: 1.5  # arcsec
mask_region: 1

weight:
  fwhm: 1.5  # arcsec

meds:
  box_padding: 2
  box_type: iso_radius
  max_box_size: 48
  min_box_size: 48
  rad_fac: 2
  rad_min: 4
""")


LOGGER = logging.getLogger(__name__)
WLDeblendData = collections.namedtuple(
    'WLDeblendData',
    [
        'cat', 'survey_name', 'bands', 'surveys',
        'builders', 'total_sky', 'noise', 'ngal_per_arcmin2',
        'psf_fwhm', 'pixel_scale',
    ],
)


def _cached_catalog_read():
    fname = os.path.join(
        os.environ.get('CATSIM_DIR', '.'),
        'OneDegSq.fits',
    )
    return fitsio.read(fname)


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
    if USE_EXP:
        gal = galsim.Exponential(half_light_radius=0.5) * MIN_FLUX
    else:
        while True:
            rind = rng.choice(data.cat.size)
            angle = rng.uniform() * 360
            pa_angle = rng.uniform() * 360

            data.cat['pa_disk'][rind] = pa_angle
            data.cat['pa_bulge'][rind] = pa_angle

            gal = galsim.Sum([
                data.builders[band].from_catalog(
                    data.cat[rind], 0, 0,
                    data.surveys[band].filter_band).model.rotate(
                        angle * galsim.degrees)
                for band in range(len(data.builders))
            ])
            if gal.flux > MIN_FLUX:
                break

    return (
        gal * FLUX_FAC,
        galsim.Kolmogorov(fwhm=data.psf_fwhm),
    )


def _make_obs(gal1, gal2, psf, nse, rng, sep, g1, n=201):
    pixel_scale = 0.2
    xoff1, yoff1 = rng.uniform(size=2, low=-0.5, high=0.5) * pixel_scale
    xoff2, yoff2 = rng.uniform(size=2, low=-0.5, high=0.5) * pixel_scale

    ang = rng.uniform(low=0, high=np.pi)
    sep_2 = sep/2
    cosa = np.cos(ang)
    sina = np.sin(ang)
    xa = sep_2*cosa
    ya = sep_2*sina

    xoff1 += xa
    yoff1 += ya

    xoff2 -= xa
    yoff2 -= ya

    im = galsim.Convolve([
        (gal1.shift(xoff1, yoff1) + gal2.shift(xoff2, yoff2)).shear(
            galsim.Shear(g1=g1, g2=0)
        ),
        psf
    ]).drawImage(
        nx=n, ny=n, scale=pixel_scale,
    ).array
    psf_im = psf.drawImage(nx=n, ny=n, scale=pixel_scale).array
    cen = (n-1)/2

    _im = im + rng.normal(size=im.shape, scale=nse)

    obs = ngmix.Observation(
        image=_im,
        weight=np.ones_like(im)/nse**2,
        bmask=np.zeros_like(im, dtype="i4"),
        ormask=np.zeros_like(im, dtype="i4"),
        mfrac=np.zeros_like(im, dtype="f4"),
        noise=rng.normal(size=im.shape, scale=nse),
        jacobian=ngmix.DiagonalJacobian(scale=pixel_scale, row=cen, col=cen),
        psf=ngmix.Observation(
            image=psf_im,
            weight=np.ones_like(im),
            jacobian=ngmix.DiagonalJacobian(scale=pixel_scale, row=cen, col=cen),
        ),
    )

    mbobs = ngmix.MultiBandObsList()
    obslist = ngmix.ObsList()
    obslist.append(obs)
    mbobs.append(obslist)
    return mbobs


def _is_none(res):
    if res is None:
        return True

    if any(res[k] is None for k in res):
        return True

    return False


def _make_mask(d):
    return (
        (d["flags"] == 0)
        & (d["pgauss_s2n"] > 10)
        & (d["pgauss_T_ratio"] > 0.5)
    )


def _meas_one(gal1, gal2, psf, nse, seed, sep):
    # run the sims
    rng = np.random.RandomState(seed=seed)
    pmbobs = _make_obs(
        gal1, gal2, psf, nse, rng, sep, 0.02
    )
    pmdet_res = do_metadetect(MDET_CFG, pmbobs, rng)

    rng = np.random.RandomState(seed=seed)
    mmbobs = _make_obs(
        gal1, gal2, psf, nse, rng, sep, -0.02
    )
    mmdet_res = do_metadetect(MDET_CFG, mmbobs, rng)

    # failures are tragic
    if _is_none(pmdet_res) or _is_none(mmdet_res):
        return None

    # for now require at least one detection above threshold for every image
    msks = {"p": {}, "m": {}}
    for key in ["noshear", "1p", "1m", "2p", "2m"]:
        pmsk = _make_mask(pmdet_res[key])
        if not np.any(pmsk):
            return None
        else:
            msks["p"][key] = pmsk

        mmsk = _make_mask(mmdet_res[key])
        if not np.any(mmsk):
            return None
        else:
            msks["m"][key] = mmsk

    # now we cancel noise
    p_g1 = np.mean(pmdet_res["noshear"]["pgauss_g"][msks["p"]["noshear"], 0])
    m_g1 = np.mean(mmdet_res["noshear"]["pgauss_g"][msks["m"]["noshear"], 0])
    g1 = (p_g1 - m_g1)/2

    p_g2 = np.mean(pmdet_res["noshear"]["pgauss_g"][msks["p"]["noshear"], 1])
    m_g2 = np.mean(mmdet_res["noshear"]["pgauss_g"][msks["m"]["noshear"], 1])
    g2 = (p_g2 + m_g2)/2

    # and measure average R
    p_R11 = (
        np.mean(pmdet_res["1p"]["pgauss_g"][msks["p"]["1p"], 0])
        -
        np.mean(pmdet_res["1m"]["pgauss_g"][msks["p"]["1m"], 0])
    ) / 0.02
    m_R11 = (
        np.mean(mmdet_res["1p"]["pgauss_g"][msks["m"]["1p"], 0])
        -
        np.mean(mmdet_res["1m"]["pgauss_g"][msks["m"]["1m"], 0])
    ) / 0.02
    R11 = (p_R11 + m_R11)/2

    p_R22 = (
        np.mean(pmdet_res["2p"]["pgauss_g"][msks["p"]["2p"], 1])
        -
        np.mean(pmdet_res["2m"]["pgauss_g"][msks["p"]["2m"], 1])
    ) / 0.02
    m_R22 = (
        np.mean(mmdet_res["2p"]["pgauss_g"][msks["m"]["2p"], 1])
        -
        np.mean(mmdet_res["2m"]["pgauss_g"][msks["m"]["2m"], 1])
    ) / 0.02
    R22 = (p_R22 + m_R22)/2

    num_det = (
        np.sum(msks["p"]["noshear"])
        + np.sum(msks["m"]["noshear"])
    ) / 2

    return (g1, R11, g2, R22, num_det)


def _meas_many(seed, n_per_chunk, sep):
    rng = np.random.RandomState(seed=seed)
    seeds = rng.randint(low=1, high=2**31, size=n_per_chunk)

    wldeblend_data = init_wldeblend(survey_bands="lsst-r")

    output = []
    for seed in tqdm.tqdm(seeds, ncols=79, desc="pair loop"):
        gal1, psf = get_gal_wldeblend(rng=rng, data=wldeblend_data)
        gal2, _ = get_gal_wldeblend(rng=rng, data=wldeblend_data)
        res = _meas_one(gal1, gal2, psf, wldeblend_data.noise, seed, sep)
        if res is not None:
            output.append(res)

    return output


def _process_outputs(outputs, sep, seed):
    os.makedirs("./mdet_results", exist_ok=True)

    d = np.array(outputs, dtype=[
        ("g1", "f8"),
        ("R11", "f8"),
        ("g2", "f8"),
        ("R22", "f8"),
        ("n_det", "f8"),
    ])

    m, msd, c, csd = meas_m_c(d)
    msg = """\
# of sims: {n_sims}
sep: {sep}
# of detections: {ndet}
noise cancel m   : {m: f} +/- {msd: f} [1e-3, 3-sigma]
noise cancel c   : {c: f} +/- {csd: f} [1e-5, 3-sigma]""".format(
                n_sims=len(d),
                sep=sep,
                ndet=np.mean(d["n_det"]),
                m=m/1e-3,
                msd=msd/1e-3 * 3,
                c=c/1e-5,
                csd=csd/1e-5 * 3,
    )
    print("\n" + msg, flush=True)

    if USE_EXP:
        fn = (
            "./mdet_results/meas_exp_"
            "fluxfac%0.1f_sep%0.3f_seed%d.fits" % (
                FLUX_FAC, sep, seed
            )
        )
    else:
        fn = "./mdet_results/meas_fluxfac%0.1f_sep%0.3f_seed%d.fits" % (
            FLUX_FAC, sep, seed
        )

    fitsio.write(fn, d, clobber=True)


def _run_sep(sep, n_chunks):
    seed = np.random.randint(low=1, high=2**29)
    rng = np.random.RandomState(seed=seed)

    outputs = []
    if BACKEND == "local":
        n_per_chunk = 10
        n_chunks = 1
        for chunk in tqdm.trange(n_chunks, ncols=79, desc="chunks loop"):
            outputs.extend(
                _meas_many(rng.randint(low=1, high=2**29), n_per_chunk, sep)
            )
        import pprint
        pprint.pprint(outputs)
    elif BACKEND == "joblib":
        n_per_chunk = 10
        with joblib.Parallel(n_jobs=-1, verbose=100) as par:
            jobs = [
                joblib.delayed(_meas_many)(
                    rng.randint(low=1, high=2**29), n_per_chunk, sep
                )
                for chunk in range(n_chunks)
            ]
            _outputs = par(jobs)
        for _o in _outputs:
            outputs.extend(_o)
    else:
        n_per_chunk = 1000
        jobs = [
            joblib.delayed(_meas_many)(
                rng.randint(low=1, high=2**29), n_per_chunk, sep
            )
            for chunk in range(n_chunks)
        ]
        outputs = []
        n_done = 0
        with BNLCondorParallel(verbose=0, n_jobs=n_chunks) as exc:
            for pr in tqdm.tqdm(
                exc(jobs), ncols=79, total=n_chunks, desc="running jobs"
            ):
                try:
                    res = pr.result()
                except Exception as e:
                    print(f"failure: {repr(e)}", flush=True)
                else:
                    outputs.extend(res)
                    n_done += 1
                    if n_done % 20 == 0:
                        _process_outputs(outputs, sep, seed)

    _process_outputs(outputs, sep, seed)


def main():
    global FLUX_FAC

    sep = float(sys.argv[1])
    n_chunks = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    ff = float(sys.argv[3]) if len(sys.argv) > 3 else 1
    if ff != 1:
        FLUX_FAC = ff

    if sep <= 0:
        for sep in np.linspace(1, 4, 13).tolist():
            _run_sep(sep, n_chunks)
    else:
        _run_sep(sep, n_chunks)


if __name__ == "__main__":
    main()
