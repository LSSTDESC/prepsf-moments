import numpy as np

import ngmix
from ngmix.joint_prior import PriorSimpleSep
from ngmix.runners import Runner, PSFRunner
from ngmix.guessers import SimplePSFGuesser, TFluxAndPriorGuesser
from ngmix.fitting import Fitter
from ngmix.gaussmom import GaussMom
from ngmix.moments import fwhm_to_T, make_mom_result
import ngmix.flags


def regularize_mom_shapes(res, fwhm_reg):
    """Apply regularization to the shapes computed from moments sums.

    This routine transforms the shapes as
        e_{1,2} = M_{1,2}/(T + T_reg)
    where T_reg is the T value equivalent to fwhm_reg, T is the original T value
    from the moments, and M_{1,2} are the moments for shapes e_{1,2}.
    This form of regularization is equivalent, for Gaussians, to convolving the Gaussian
    with an isotropic Gaussian smoothing kernel of size fwhm_reg.

    Parameters
    ----------
    res : dict
        The original moments result before regularization.
    fwhm_reg : float
        The regularization FWHM value. Typically this should be of order the size of
        the PSF for a pre-PSF moment.

    Returns
    -------
    res_reg : dict
        The regularized moments result. The size and flux are unchanged.
    """
    if fwhm_reg > 0:
        raw_mom = res["sums"]
        raw_mom_cov = res["sums_cov"]

        T_reg = fwhm_to_T(fwhm_reg)

        # the moments are not normalized and are sums, so convert T_reg to a sum using
        # the flux sum first via T_reg -> T_reg * raw_mom[5]
        amat = np.eye(6)
        amat[4, 5] = T_reg

        # the pre-PSF fitters do not fill out the centroid moments so hack around that
        raw_mom_orig = raw_mom.copy()
        if np.isnan(raw_mom_orig[0]):
            raw_mom[0] = 0
        if np.isnan(raw_mom_orig[1]):
            raw_mom[1] = 0
        reg_mom = np.dot(amat, raw_mom)
        if np.isnan(raw_mom_orig[0]):
            raw_mom[0] = np.nan
            reg_mom[0] = np.nan
        if np.isnan(raw_mom_orig[1]):
            raw_mom[1] = np.nan
            reg_mom[1] = np.nan

        reg_mom_cov = np.dot(amat, np.dot(raw_mom_cov, amat.T))
        momres = make_mom_result(reg_mom, reg_mom_cov)

        # use old T
        for col in ["T", "T_err", "T_flags", "T_flagstr"]:
            momres[col] = res[col]

        momres["flags"] |= res["flags"]
        momres["flagstr"] = ngmix.flags.get_flags_str(momres["flags"])

        return momres
    else:
        return res


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
