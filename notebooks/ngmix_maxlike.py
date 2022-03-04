import numpy as np

import ngmix
from ngmix.joint_prior import PriorSimpleSep
from ngmix.runners import Runner, PSFRunner
from ngmix.guessers import SimplePSFGuesser, TFluxAndPriorGuesser
from ngmix.fitting import Fitter
from ngmix.gaussmom import GaussMom


def _make_prior(rng):
    """Match this from the old ngmix config:

        'priors': {
            'cen': {
                'type': 'normal2d',
                'sigma': 0.263
            },

            'g': {
                'type': 'ba',
                'sigma': 0.2
            },

            'T': {
                'type': 'two-sided-erf',
                'pars': [-1.0, 0.1, 1.0e+06, 1.0e+05]
            },

            'flux': {
                'type': 'two-sided-erf',
                'pars': [-100.0, 1.0, 1.0e+09, 1.0e+08]
            }
        }
    """
    g_prior = ngmix.priors.GPriorBA(0.2, rng=rng)
    size_prior = ngmix.priors.TwoSidedErf(
        -1.0, 0.1, 1.0e+06, 1.0e+05,
        rng=rng,
    )
    flux_prior = ngmix.priors.TwoSidedErf(
        -100.0, 1.0, 1.0e+09, 1.0e+08,
        rng=rng,
    )
    cen_prior = ngmix.priors.CenPrior(
        0.0,
        0.0,
        0.2,
        0.2,
        rng=rng,
    )

    return PriorSimpleSep(
        cen_prior,
        g_prior,
        size_prior,
        [flux_prior],
    )


def run_maxlike(obs, rng=None):
    if rng is None:
        rng = np.random.RandomState()

    metacal_prior = _make_prior(rng)

    gm = GaussMom(1.2).go(obs)
    if gm['flags'] == 0:
        flux_guess = gm['flux']
        Tguess = gm['T']
    else:
        gm = GaussMom(1.2).go(obs.psf)
        if gm['flags'] == 0:
            Tguess = 2 * gm['T']
        else:
            Tguess = 2
        flux_guess = np.sum(obs.image)

    guesser = TFluxAndPriorGuesser(
        rng=rng, T=Tguess, flux=flux_guess, prior=metacal_prior,
    )
    psf_guesser = SimplePSFGuesser(rng=rng, guess_from_moms=True)

    fitter = Fitter(
        model="gauss",
        fit_pars={
            'maxfev': 2000,
            'xtol': 5.0e-5,
            'ftol': 5.0e-5,
        },
        prior=metacal_prior,
    )
    psf_fitter = Fitter(
        model='gauss',
        fit_pars={
            'maxfev': 2000,
            'ftol': 1.0e-5,
            'xtol': 1.0e-5,
        },
    )
    psf_runner = PSFRunner(
        fitter=psf_fitter,
        guesser=psf_guesser,
        ntry=2,
    )
    runner = Runner(
        fitter=fitter,
        guesser=guesser,
        ntry=2,
    )

    return ngmix.bootstrap.bootstrap(
        obs,
        runner,
        psf_runner=psf_runner,
        ignore_failed_psf=False,
    )
