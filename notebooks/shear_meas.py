import sys
import numpy as np
import tqdm


def _run_boostrap(x1, y1, x2, y2, wgts, silent):
    rng = np.random.RandomState(seed=100)
    mvals = []
    cvals = []
    if silent:
        itrl = range(500)
    else:
        itrl = tqdm.trange(
            500, leave=False, desc='running bootstrap', ncols=79,
            file=sys.stderr,
        )
    for _ in itrl:
        ind = rng.choice(len(y1), replace=True, size=len(y1))
        _wgts = wgts[ind].copy()
        _wgts /= np.sum(_wgts)
        mvals.append(np.mean(y1[ind] * _wgts) / np.mean(x1[ind] * _wgts) - 1)
        cvals.append(np.mean(y2[ind] * _wgts) / np.mean(x2[ind] * _wgts))

    return (
        np.mean(y1 * wgts) / np.mean(x1 * wgts) - 1, np.std(mvals),
        np.mean(y2 * wgts) / np.mean(x2 * wgts), np.std(cvals))


def _run_jackknife(x1, y1, x2, y2, wgts, jackknife):
    n_per = x1.shape[0] // jackknife
    n = n_per * jackknife
    x1j = np.zeros(jackknife)
    y1j = np.zeros(jackknife)
    x2j = np.zeros(jackknife)
    y2j = np.zeros(jackknife)
    wgtsj = np.zeros(jackknife)

    loc = 0
    for i in range(jackknife):
        wgtsj[i] = np.sum(wgts[loc:loc+n_per])
        x1j[i] = np.sum(x1[loc:loc+n_per] * wgts[loc:loc+n_per]) / wgtsj[i]
        y1j[i] = np.sum(y1[loc:loc+n_per] * wgts[loc:loc+n_per]) / wgtsj[i]
        x2j[i] = np.sum(x2[loc:loc+n_per] * wgts[loc:loc+n_per]) / wgtsj[i]
        y2j[i] = np.sum(y2[loc:loc+n_per] * wgts[loc:loc+n_per]) / wgtsj[i]

        loc += n_per

    # weighted jackknife from Busing et al. 1999, Statistics and Computing, 9, 3-8
    mhat = np.mean(y1[:n] * wgts[:n]) / np.mean(x1[:n] * wgts[:n]) - 1
    chat = np.mean(y2[:n] * wgts[:n]) / np.mean(x2[:n] * wgts[:n])
    mhatj = np.zeros(jackknife)
    chatj = np.zeros(jackknife)
    for i in range(jackknife):
        _wgts = np.delete(wgtsj, i)
        mhatj[i] = (
            np.sum(np.delete(y1j, i) * _wgts) / np.sum(np.delete(x1j, i) * _wgts)
            - 1
        )
        chatj[i] = (
            np.sum(np.delete(y2j, i) * _wgts) / np.sum(np.delete(x2j, i) * _wgts)
        )

    tot_wgt = np.sum(wgtsj)
    mbar = jackknife * mhat - np.sum((1.0 - wgtsj/tot_wgt) * mhatj)
    cbar = jackknife * chat - np.sum((1.0 - wgtsj/tot_wgt) * chatj)

    hj = tot_wgt / wgtsj
    mtildej = hj * mhat - (hj - 1) * mhatj
    ctildej = hj * chat - (hj - 1) * chatj

    mvarj = np.sum((mtildej - mbar)**2 / (hj-1)) / jackknife
    cvarj = np.sum((ctildej - cbar)**2 / (hj-1)) / jackknife

    return (
        mbar,
        np.sqrt(mvarj),
        cbar,
        np.sqrt(cvarj),
    )


def meas_m_c(d):
    wgts = np.ones_like(d["g1"])
    if d.shape[0] > 10000:
        return _run_jackknife(
            d["R11"], d["g1"]/0.02, d["R22"], d["g2"], wgts, 200,
        )
    else:
        return _run_boostrap(
            d["R11"], d["g1"]/0.02, d["R22"], d["g2"], wgts, True,
        )
