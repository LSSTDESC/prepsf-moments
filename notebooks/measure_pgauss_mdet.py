import os
import logging
import sys

import numpy as np
import fitsio
import joblib
import tqdm
from ngmix.metacal import get_all_metacal
from ngmix.prepsfmom import PGaussMom
from wldeblend_sim import init_wldeblend, get_gal_wldeblend, make_ngmix_obs
from ngmix_maxlike import run_maxlike

LOGGER = logging.getLogger(__name__)


def _meas(gal, psf, redshift, nse, pixel_scale, aps, seed):
    rng = np.random.RandomState(seed=seed)
    obs, true_flux, obs_nn = make_ngmix_obs(
        gal=gal, psf=psf, nse=nse, pixel_scale=pixel_scale, rng=rng,
    )

    try:
        mcal_res = get_all_metacal(
            obs,
            psf='fitgauss',
            fixnoise=True,
            rng=rng,
            types=["noshear", "1p", "1m"],
        )
        for k, v in mcal_res.items():
            if v is None:
                raise RuntimeError("bad mcal result!")

        s2ns = []
        g1s = []
        g1errs = []
        trs = []
        flags = []
        redshifts = []
        maps = []
        msteps = []
        kinds = []
        for ap in aps:
            for step, mcal_obs in mcal_res.items():
                mom = PGaussMom(ap).go(mcal_obs)
                psf_mom = PGaussMom(ap).go(mcal_obs.psf, no_psf=True)
                if psf_mom["flags"] == 0:
                    psf_mom_t = psf_mom["T"]
                else:
                    psf_mom_t = np.nan

                flags.append(mom["flags"] | psf_mom["flags"])
                s2ns.append(mom["s2n"])
                g1s.append(mom["e1"])
                g1errs.append(mom["e_err"][0])
                trs.append(mom["T"]/psf_mom_t)
                redshifts.append(redshift)
                maps.append(ap)
                msteps.append(step)
                kinds.append("pgauss")

        for step, mcal_obs in mcal_res.items():
            mom = run_maxlike(mcal_obs, rng=rng)
            psf_mom = mcal_obs.psf.meta["result"]
            mom["e1"] = mom["g"][0]
            mom["e_err"] = mom["g_err"]

            if psf_mom["flags"] == 0:
                psf_mom_t = psf_mom["T"]
            else:
                psf_mom_t = np.nan

            flags.append(mom["flags"] | psf_mom["flags"])
            s2ns.append(mom["s2n"])
            g1s.append(mom["e1"])
            g1errs.append(mom["e_err"][0])
            trs.append(mom["T"]/psf_mom_t)
            redshifts.append(redshift)
            maps.append(-1)
            msteps.append(step)
            kinds.append("mgauss")

        for i in range(2):
            if i == 0:
                dtype = []
            else:
                md = np.zeros(len(flags), dtype=dtype)
            for cname, arr in [
                ("flags", flags),
                ("s2n", s2ns),
                ("e1", g1s),
                ("e1_err", g1errs),
                ("Tratio", trs),
                ("redshift", redshifts),
            ]:
                if i == 0:
                    dtype.append((cname, "f4"))
                else:
                    md[cname] = np.array(arr)

            if i == 0:
                dtype.append(("mdet_step", "U7"))
                dtype.append(("ap", "f4"))
                dtype.append(("kind", "U7"))
            else:
                md["mdet_step"] = msteps
                md["ap"] = maps
                md["kind"] = kinds

        return md

    except Exception as e:
        print("ERROR: " + repr(e), flush=True)
        return None


def main():
    n_per_chunk = 100
    n_chunks = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    seed = np.random.randint(low=1, high=2**29)
    rng = np.random.RandomState(seed=seed)

    os.makedirs("./results_pgauss_mdet", exist_ok=True)

    wldeblend_data = init_wldeblend(survey_bands="lsst-r")

    aps = np.linspace(1.25, 2.75, 25)
    outputs = []
    with joblib.Parallel(n_jobs=-1, verbose=10, batch_size=2) as par:
        for chunk in tqdm.trange(n_chunks):
            jobs = []
            for i in range(n_per_chunk):
                gal, psf, redshift = get_gal_wldeblend(rng=rng, data=wldeblend_data)
                jobs.append(joblib.delayed(_meas)(
                    gal, psf, redshift, wldeblend_data.noise,
                    wldeblend_data.pixel_scale,
                    aps, rng.randint(low=1, high=2**29))
                )

            outputs.extend([_d for _d in par(jobs) if _d is not None])

            d = np.concatenate(outputs, axis=0)
            fitsio.write(
                "./results_pgauss_mdet/meas_seed%d.fits" % seed,
                d, extname="data", clobber=True)
            fitsio.write(
                "./results_pgauss_mdet/meas_seed%d.fits" % seed,
                aps, extname="aps")


if __name__ == "__main__":
    main()
