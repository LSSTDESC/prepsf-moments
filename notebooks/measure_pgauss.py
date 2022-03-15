import os
import logging
import sys

import numpy as np
import fitsio
import joblib
import tqdm
from ngmix.prepsfmom import PGaussMom
from wldeblend_sim import init_wldeblend, get_gal_wldeblend, make_ngmix_obs

LOGGER = logging.getLogger(__name__)


def _meas(gal, psf, redshift, nse, pixel_scale, aps, seed):
    rng = np.random.RandomState(seed=seed)
    obs, true_flux, obs_nn = make_ngmix_obs(
        gal=gal, psf=psf, nse=nse, pixel_scale=pixel_scale, rng=rng,
    )

    s2ns = []
    g1s = []
    ts = []
    trs = []
    flags = []
    g1errs = []
    redshifts = []
    fluxes = []
    flux_errs = []
    terr = []
    tflux = []
    tapflux = []
    ts2ns = []
    fflags = []
    for ap in aps:
        mom = PGaussMom(ap).go(obs)
        mom_nn = PGaussMom(ap).go(obs_nn)
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
        fluxes.append(mom["flux"])
        flux_errs.append(mom["flux_err"])
        terr.append(mom["T_err"])
        tflux.append(true_flux)
        tapflux.append(mom_nn["flux"])
        ts2ns.append(mom_nn["flux"]/mom_nn["flux_err"])
        fflags.append(mom["flux_flags"])

    return (
        s2ns, g1s, flags, ts, trs, g1errs, redshifts, fluxes, flux_errs, terr,
        tflux, tapflux, ts2ns, fflags,
    )


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
                        wldeblend_data.pixel_scale,
                        aps, rng.randint(low=1, high=2**29)
                    ))
            else:
                jobs = []
                for i in range(n_per_chunk):
                    gal, psf, redshift = get_gal_wldeblend(rng=rng, data=wldeblend_data)
                    jobs.append(joblib.delayed(_meas)(
                        gal, psf, redshift, wldeblend_data.noise,
                        wldeblend_data.pixel_scale,
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
                ("flux", "f4", (len(aps),)),
                ("flux_err", "f4", (len(aps),)),
                ("T_err", "f4", (len(aps),)),
                ("true_flux", "f4", (len(aps),)),
                ("true_ap_flux", "f4", (len(aps),)),
                ("true_s2n", "f4", (len(aps),)),
                ("flux_flags", "i4", (len(aps),)),
            ])
            _o = np.array(outputs)
            d["s2n"] = _o[:, 0]
            d["e1"] = _o[:, 1]
            d["flags"] = _o[:, 2]
            d["T"] = _o[:, 3]
            d["Tratio"] = _o[:, 4]
            d["e1_err"] = _o[:, 5]
            d["redshift"] = _o[:, 6]
            d["flux"] = _o[:, 7]
            d["flux_err"] = _o[:, 8]
            d["T_err"] = _o[:, 9]
            d["true_flux"] = _o[:, 10]
            d["true_ap_flux"] = _o[:, 11]
            d["true_s2n"] = _o[:, 12]
            d["flux_flags"] = _o[:, 13]

            fitsio.write(
                "./results/meas_seed%d.fits" % seed,
                d, extname="data", clobber=True)
            fitsio.write("./results/meas_seed%d.fits" % seed, aps, extname="aps")


if __name__ == "__main__":
    main()
