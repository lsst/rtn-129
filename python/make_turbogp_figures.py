"""Figures of Sect. 3 of RTN-129 illustrating TURBO-GP on a real LSSTCam visit.

The visit (2025052000348, z band, 163 661 sources; data/rubin_visit_2025052000348.npz:
float32 u, v in degrees on the tangent plane, du, dv and their errors du_err, dv_err
in mas) is the one used by the interactive slides https://pfleget.github.io/treegp/.
One component of the residual field is modeled with treegp exactly as in the paper
(10 000 random training sources, seed 1234; TURBO-GP: 20" pixels, +-0.3 deg, Hann
window of radius 0.278 deg with the "auto" anisotropy, S/N threshold 2.5; L21:
anisotropic von Karman kernel fitted to the 21 x 21 correlation map) and the
intermediate products are drawn:

  figures/turbogp_kernel.pdf   the kernel: measured 2-pcf, L21 fit and its residual
                               (top), raw 20" map x window, Fourier power with the
                               kept modes, cleaned map = the TURBO-GP kernel (bottom)
  figures/turbogp_matvec.pdf   the four steps of C y (Eq. 12) on the whole visit
  figures/turbogp_cg.pdf       the conjugate-gradient solve: relative residual with
                               and without the Jacobi preconditioner, and the mean
                               field after 1, 3, 10, 30 iterations and at convergence

The L21 fit is slow (10-25 min per component, bootstrap); its result is cached in
data/cache_l21.pkl (delete it to refit, or --no-l21 to leave the top row of the
kernel figure out). Everything else runs in about a minute.

Usage: python3 make_turbogp_figures.py [--data PATH] [--component u|v]
                                       [--no-l21] [--out DIR]
"""

import argparse
import os
import pickle
import time
import warnings

import matplotlib
import numpy as np
import treegp
from matplotlib import pyplot as plt
from scipy.sparse.linalg import LinearOperator, cg
from treegp.empirical_2pcf import _apod, _corr2power, _power2corr, _threshold

matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "data", "rubin_visit_2025052000348.npz")
CACHE_L21 = os.path.join(HERE, "data", "cache_l21.pkl")
FIG_DIR = os.path.join(HERE, "..", "figures")
DATA_KEYS = ("u", "v", "du", "dv", "du_err", "dv_err")

N_TRAIN = 10000
SEED = 1234
PIXEL = 20.0 / 3600.0  # deg
MAX_SEP = 0.3  # deg
APOD_RADIUS = 0.278  # deg
WINDOW = "hann"
THRESHOLD = 2.5
CG_SNAPSHOTS = [1, 3, 10, 30]  # plus the final iterate
DS = 8  # block averaging of the full-grid images for display
ZOOM_DEG = 0.2  # side of the zoom on the grid in the matvec figure

BLUE = "#1f5fbf"
ORANGE = "#d97a00"
CMAP = "seismic"  # diverging, white at zero

plt.rcParams.update(
    {
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.linewidth": 0.6,
    }
)


def load_visit(path):
    """Read the visit from the compact npz or from the original pickle."""
    if path.endswith(".npz"):
        with np.load(path) as z:
            D = {k: z[k] for k in z.files}
    else:
        with open(path, "rb") as f:
            D = pickle.load(f)
    out = {k: np.asarray(D[k], float) for k in DATA_KEYS}
    out["visit"] = str(D["visit"])
    out["band"] = str(D["band"])
    return out


def fit_l21(X, y, yerr, train):
    """L21 anisotropic von Karman fit on the training sample (slow)."""
    gpl = treegp.GPInterpolation(
        kernel="15**2 * AnisotropicVonKarman(invLam=array([[1./0.8**2,0],[0,1./0.8**2]]))",
        optimizer="anisotropic",
        normalize=True,
        nbins=21,
        min_sep=None,
        max_sep=MAX_SEP,
        p0=[0.3, 0.0, 0.0],
    )
    gpl.initialize(X[train], y[train], y_err=yerr[train])
    t0 = time.time()
    gpl.solve()
    t_fit = time.time() - t0
    o2 = gpl._optimizer
    n2 = int(np.sqrt(len(o2._2pcf)))
    vk = dict(
        n=n2,
        lag=((np.arange(n2) - n2 // 2) * (2 * MAX_SEP / n2)).tolist(),
        measured=np.asarray(o2._2pcf).reshape(n2, n2).tolist(),
        fit=np.asarray(o2._2pcf_fit).reshape(n2, n2).tolist(),
        kernel=str(gpl.kernel),
        t_fit=t_fit,
    )
    return dict(vk=vk, vk_info={})


def cg_history(eng, y, yerr, precond=True):
    """Re-run the engine's conjugate gradient, recording the relative
    residual at every iteration and the iterates at a few of them."""
    w = eng._w_train
    noise = yerr**2
    n = len(y)

    def matvec(v):
        field = (w.T @ v).reshape(eng._ny, eng._nx)
        return w @ eng._convolve(field).ravel() + noise * v

    op = LinearOperator((n, n), matvec=matvec, dtype=float)
    M = None
    if precond:
        d = 1.0 / (eng.xi0_eff + noise)
        M = LinearOperator((n, n), matvec=lambda v: d * v, dtype=float)
    yn = np.linalg.norm(y)
    hist = [1.0]
    snaps = {}

    def callback(xk):
        k = len(hist)
        hist.append(float(np.linalg.norm(y - matvec(xk)) / yn))
        if k in CG_SNAPSHOTS:
            snaps[k] = xk.copy()

    alpha, _ = cg(
        op,
        y,
        rtol=eng.cg_rtol,
        atol=0.0,
        maxiter=eng.cg_maxiter,
        M=M,
        callback=callback,
    )
    snaps[len(hist) - 1] = alpha.copy()
    return np.array(hist), snaps


def block_mean(img, ds):
    """Average an image over ds x ds blocks (the remainder is dropped)."""
    ny, nx = ((s // ds) * ds for s in img.shape)
    return img[:ny, :nx].reshape(ny // ds, ds, nx // ds, ds).mean(axis=(1, 3))


def heat(ax, img, extent, zmax, title, cmap=CMAP, cbar_label=None, zmin=None):
    """Image with the origin at the bottom, symmetric diverging scale by default."""
    im = ax.imshow(
        img,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=-zmax if zmin is None else zmin,
        vmax=zmax,
        interpolation="nearest",
        rasterized=True,
        aspect="equal",
    )
    ax.set_title(title)
    if cbar_label is not None:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(cbar_label)
        cb.ax.tick_params(labelsize=6)
    return im


def lag_extent(lag):
    d = lag[1] - lag[0]
    return [lag[0] - d / 2, lag[-1] + d / 2, lag[0] - d / 2, lag[-1] + d / 2]


def fig_kernel(opt, l21, comp, out):
    """The kernel: measured 2-pcf, L21 fit, residual; raw map x window, power, cleaned."""
    npix, pix = opt.npix, opt.pixel_size
    lag = (np.arange(npix) - npix // 2) * pix
    raw, clean = opt._xi, opt._xi_clean
    g1, g2 = opt._apod_g_applied
    window = _apod(raw, r_max=APOD_RADIUS / pix, window=WINDOW, g1=g1, g2=g2)
    pk = _corr2power(raw * window)
    pk_kept = _threshold(pk, n_sigma=THRESHOLD)
    kept = pk_kept > 0
    assert np.allclose(_power2corr(pk_kept), clean, atol=1e-6 * np.abs(clean).max())
    nz = pk[pk != 0]
    hml = np.percentile(nz, (16.0, 50.0, 84.0))
    thresh = hml[1] + THRESHOLD * 0.5 * (hml[2] - hml[0])
    zmax = 1.1 * np.abs(clean).max()  # the raw map has noisier pixels
    print(
        f"kernel figure ({comp}): raw xi(0) {raw[npix // 2, npix // 2]:.1f}, cleaned xi(0) "
        f"{clean[npix // 2, npix // 2]:.1f} mas^2; window g = ({g1:.3f}, {g2:.3f}); "
        f"threshold {thresh:.1f} mas^2 deg^2 (median {hml[1]:.1f}); kept {kept.sum()} of "
        f"{pk.size} modes of the half spectrum ({npix} x {pk.shape[1]})"
    )

    nrow = 2 if l21 else 1
    fig, axes = plt.subplots(
        nrow, 3, figsize=(7.2, 2.35 * nrow), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    unit = r"$\xi$ (mas$^2$)"
    if l21:
        vk = l21["vk"]
        lag21 = np.asarray(vk["lag"])
        meas, fit = np.asarray(vk["measured"]), np.asarray(vk["fit"])
        ext21 = lag_extent(lag21)
        z21 = np.abs(meas).max()
        heat(
            axes[0, 0],
            meas,
            ext21,
            z21,
            r"(a) measured $\xi$, $21\times21$ map (L21)",
            cbar_label=unit,
        )
        heat(
            axes[0, 1],
            fit,
            ext21,
            z21,
            "(b) L21: anisotropic von Kármán fit",
            cbar_label=unit,
        )
        heat(
            axes[0, 2],
            meas - fit,
            ext21,
            z21 / 3,
            "(c) measured $-$ fit",
            cbar_label=unit,
        )
        print(
            f"  L21 kernel: {vk['kernel']}; rms of measured - fit {np.std(meas - fit):.1f} mas^2, "
            f"max |measured - fit| {np.abs(meas - fit).max():.1f} mas^2 (fit {vk['t_fit']:.0f} s)"
        )
    ext = lag_extent(lag)
    r = nrow - 1
    heat(
        axes[r, 0],
        raw * window,
        ext,
        zmax,
        "(d) measured $\\xi$, $20''$ pixels, $\\times$ window",
        cbar_label=unit,
    )
    # low-frequency part of the half spectrum (k_u >= 0): log power, kept modes outlined
    q = npix // 4
    rows = slice(npix // 2 - q, npix // 2 + q)
    lp = np.log10(1.0 + np.abs(pk[rows, :q]))
    ku = np.arange(q) / (npix * pix)
    kv = (np.arange(-q, q)) / (npix * pix)
    dk = 1.0 / (npix * pix)
    ext_k = [ku[0] - dk / 2, ku[-1] + dk / 2, kv[0] - dk / 2, kv[-1] + dk / 2]
    im = axes[r, 1].imshow(
        lp,
        origin="lower",
        extent=ext_k,
        cmap="Blues",
        vmin=0,
        vmax=lp.max(),
        aspect="auto",
        interpolation="nearest",
        rasterized=True,
    )
    axes[r, 1].contour(
        ku,
        kv,
        kept[rows, :q].astype(float),
        levels=[0.5],
        colors=ORANGE,
        linewidths=0.9,
    )
    axes[r, 1].set_title(f"(e) Fourier power, {kept.sum()} modes kept (outlined)")
    axes[r, 1].set_xlabel(r"$k_u$ (deg$^{-1}$)")
    axes[r, 1].set_ylabel(r"$k_v$ (deg$^{-1}$)")
    cb = plt.colorbar(im, ax=axes[r, 1], fraction=0.046, pad=0.03)
    cb.set_label(r"$\log_{10}(1 + |\hat P_k|)$ (mas$^2$ deg$^2$)")
    cb.ax.tick_params(labelsize=6)
    heat(
        axes[r, 2],
        clean,
        ext,
        zmax,
        r"(f) cleaned $\xi$ = the TURBO-GP kernel",
        cbar_label=unit,
    )
    for ax in axes.ravel():
        if ax is not axes[r, 1]:
            ax.set_xlabel(r"$\Delta u$ (deg)")
            ax.set_ylabel(r"$\Delta v$ (deg)")
    path = os.path.join(out, "turbogp_kernel.pdf")
    fig.savefig(path)
    plt.close(fig)
    print("  wrote", path)


def fig_matvec(eng, X, y, sig, comp, out):
    """The four steps of C y: spread on the grid, convolve, read back, add the noise."""
    w = eng._w_train
    t0 = time.time()
    img = (w.T @ y).reshape(eng._ny, eng._nx)
    fld = eng._convolve(img)
    Ky = w @ fld.ravel()
    Cy = Ky + sig**2 * y
    dt = time.time() - t0
    h, x0, y0 = eng.h, eng._x0, eng._y0
    ks = eng._kernel_image.shape[0]
    print(
        f"matvec figure ({comp}): grid {eng._nx} x {eng._ny} nodes of {h * 3600:.0f}'', padded FFT "
        f"{eng._lx} x {eng._ly}, kernel image {ks} x {ks}; C y in {dt * 1e3:.0f} ms; "
        f"|y| max {np.abs(y).max():.0f} mas, |K y| max {np.abs(Ky).max():.3g} mas^3, "
        f"|D y| max {np.abs(sig**2 * y).max():.3g} mas^3"
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), constrained_layout=True)
    # (a) zoom on the grid around the field centre: W^T y and the stars
    nz = int(round(ZOOM_DEG / h))
    cx, cy = 0.5 * (X[:, 0].min() + X[:, 0].max()), 0.5 * (
        X[:, 1].min() + X[:, 1].max()
    )
    ix0 = int(round((cx - x0) / h)) - nz // 2
    iy0 = int(round((cy - y0) / h)) - nz // 2
    zoom = img[iy0 : iy0 + nz, ix0 : ix0 + nz]
    ext = [
        x0 + (ix0 - 0.5) * h,
        x0 + (ix0 + nz - 0.5) * h,
        y0 + (iy0 - 0.5) * h,
        y0 + (iy0 + nz - 0.5) * h,
    ]
    inside = (
        (X[:, 0] > ext[0])
        & (X[:, 0] < ext[1])
        & (X[:, 1] > ext[2])
        & (X[:, 1] < ext[3])
    )
    zv = np.percentile(np.abs(y), 99)
    heat(
        axes[0, 0],
        zoom,
        ext,
        zv,
        f"(a) step 1: $\\mathbf{{W}}^T\\mathbf{{y}}$ on the grid, {ZOOM_DEG}° zoom ({inside.sum()} stars)",
        cbar_label=r"$\mathbf{W}^T\mathbf{y}$ (mas)",
    )
    axes[0, 0].scatter(
        X[inside, 0],
        X[inside, 1],
        s=9,
        facecolors="none",
        edgecolors="k",
        linewidths=0.4,
        rasterized=True,
    )
    axes[0, 0].set_xlabel("u (deg)")
    axes[0, 0].set_ylabel("v (deg)")
    print(
        f"  zoom: {nz} x {nz} nodes, {inside.sum()} training stars, {np.count_nonzero(zoom)} non-zero nodes"
    )
    # (b) the kernel image
    kimg = eng._kernel_image
    extk = [
        -(ks // 2 + 0.5) * h,
        (ks // 2 + 0.5) * h,
        -(ks // 2 + 0.5) * h,
        (ks // 2 + 0.5) * h,
    ]
    heat(
        axes[0, 1],
        kimg,
        extk,
        np.abs(kimg).max(),
        f"(b) the kernel image $\\kappa$ ({ks}$\\times${ks} nodes of {h * 3600:.0f}$''$)",
        cbar_label=r"$\kappa$ (mas$^2$)",
    )
    axes[0, 1].set_xlabel(r"$\Delta u$ (deg)")
    axes[0, 1].set_ylabel(r"$\Delta v$ (deg)")
    # (c) the convolution over the whole field
    disp = block_mean(fld, DS) / 1e3
    extf = [
        x0 - h / 2,
        x0 + (disp.shape[1] * DS - 0.5) * h,
        y0 - h / 2,
        y0 + (disp.shape[0] * DS - 0.5) * h,
    ]
    zf = np.percentile(np.abs(Ky), 99.5) / 1e3
    heat(
        axes[1, 0],
        disp,
        extf,
        zf,
        f"(c) step 2: $\\kappa * \\mathbf{{W}}^T\\mathbf{{y}}$, whole grid (mean over {DS}$\\times${DS} nodes)",
        cbar_label=r"$\kappa * \mathbf{W}^T\mathbf{y}$ ($10^3$ mas$^3$)",
    )
    axes[1, 0].set_xlabel("u (deg)")
    axes[1, 0].set_ylabel("v (deg)")
    axes[1, 0].add_patch(
        plt.Rectangle(
            (ext[0], ext[2]),
            ext[1] - ext[0],
            ext[3] - ext[2],
            fill=False,
            ec="k",
            lw=0.6,
        )
    )
    # (d) read back at the stars and add the noise term
    sc = axes[1, 1].scatter(
        X[:, 0],
        X[:, 1],
        c=Cy / 1e3,
        s=1.2,
        cmap=CMAP,
        vmin=-zf,
        vmax=zf,
        linewidths=0,
        rasterized=True,
    )
    axes[1, 1].set_title(
        r"(d) steps 3-4: read back at the stars + noise: $\mathbf{C}\mathbf{y}$"
    )
    axes[1, 1].set_xlabel("u (deg)")
    axes[1, 1].set_ylabel("v (deg)")
    axes[1, 1].set_aspect("equal")
    axes[1, 1].set_xlim(extf[0], extf[1])
    axes[1, 1].set_ylim(extf[2], extf[3])
    cb = plt.colorbar(sc, ax=axes[1, 1], fraction=0.046, pad=0.03)
    cb.set_label(r"$\mathbf{C}\mathbf{y}$ ($10^3$ mas$^3$)")
    cb.ax.tick_params(labelsize=6)
    path = os.path.join(out, "turbogp_matvec.pdf")
    fig.savefig(path)
    plt.close(fig)
    print("  wrote", path)


def fig_cg(eng, y, sig, comp, out):
    """Conjugate gradient: residual history and mean-field snapshots."""
    t0 = time.time()
    hist_pre, snaps = cg_history(eng, y, sig, precond=True)
    t_pre = time.time() - t0
    hist_nopre, _ = cg_history(eng, y, sig, precond=False)
    n_pre, n_nopre = len(hist_pre) - 1, len(hist_nopre) - 1
    print(
        f"cg figure ({comp}): {n_pre} iterations with the Jacobi preconditioner ({t_pre:.1f} s, "
        f"{t_pre / n_pre / 2 * 1e3:.0f} ms per iteration once the extra product recorded "
        f"per iteration is discounted), {n_nopre} without; rtol {eng.cg_rtol:g}; "
        f"treegp's own solve took {eng.n_iterations} iterations"
    )
    h, x0, y0 = eng.h, eng._x0, eng._y0
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), constrained_layout=True)
    ax = axes[0, 0]
    ax.semilogy(
        np.arange(len(hist_pre)),
        hist_pre,
        color=BLUE,
        lw=1.2,
        label=f"Jacobi preconditioner ({n_pre} it.)",
    )
    ax.semilogy(
        np.arange(len(hist_nopre)),
        hist_nopre,
        color=ORANGE,
        lw=1.2,
        label=f"no preconditioner ({n_nopre} it.)",
    )
    ax.axhline(eng.cg_rtol, color="k", lw=0.5, ls="--")
    for k in CG_SNAPSHOTS:
        ax.plot(k, hist_pre[k], "o", color="k", ms=3.5, mec="w", mew=0.5)
    ax.set_xlabel("iteration $k$")
    ax.set_ylabel(r"$\|\mathbf{r}_k\| / \|\mathbf{y}\|$")
    ax.set_title("(a) relative residual of the solve")
    ax.legend(loc="upper right")
    ax.set_ylim(0.3 * eng.cg_rtol, 2)
    ks = sorted(snaps)
    zmax = 40.0
    panels = axes.ravel()[1:]
    last = None
    for ax, k in zip(panels, ks):
        field = eng._convolve((eng._w_train.T @ snaps[k]).reshape(eng._ny, eng._nx))
        disp = block_mean(field, DS)
        extf = [
            x0 - h / 2,
            x0 + (disp.shape[1] * DS - 0.5) * h,
            y0 - h / 2,
            y0 + (disp.shape[0] * DS - 0.5) * h,
        ]
        label = chr(ord("b") + ks.index(k))
        what = ", converged" if k == ks[-1] else ", mean field"
        last = heat(
            ax,
            disp,
            extf,
            zmax,
            f"({label}) $k$ = {k}{what}",
        )
        ax.set_xlabel("u (deg)")
        ax.set_ylabel("v (deg)")
        rms, vmax = np.sqrt(np.mean(field**2)), np.abs(field).max()
        print(
            f"  snapshot k = {k}: field rms {rms:.1f} mas, max |field| {vmax:.1f} mas"
        )
    cb = fig.colorbar(
        last, ax=axes[:, 2].tolist(), fraction=0.046, pad=0.03, shrink=0.9
    )
    cb.set_label(r"$\hat y$ (mas)")
    cb.ax.tick_params(labelsize=6)
    path = os.path.join(out, "turbogp_cg.pdf")
    fig.savefig(path)
    plt.close(fig)
    print("  wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--data", default=NPZ, help="the visit, .npz (default) or the original .pkl"
    )
    ap.add_argument(
        "--component", default="u", choices=("u", "v"), help="residual component shown"
    )
    ap.add_argument(
        "--no-l21", action="store_true", help="skip the L21 fit if it is not cached"
    )
    ap.add_argument("--out", default=FIG_DIR, help="directory receiving the three PDFs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    D = load_visit(args.data)
    comp = args.component
    X = np.vstack([D["u"], D["v"]]).T
    y_all, yerr_all = D["d" + comp], D["d" + comp + "_err"]
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X))
    train = perm[:N_TRAIN]
    print(
        f"visit {D['visit']} ({D['band']} band): {len(X)} sources, {N_TRAIN} training sources (seed {SEED}), "
        f"component d{comp}"
    )

    gp = treegp.GPInterpolation(
        optimizer="empirical-2pcf",
        normalize=True,
        max_sep=MAX_SEP,
        pixel_size=PIXEL,
        power_threshold=THRESHOLD,
        apodize=True,
        apod_window=WINDOW,
        apod_radius=APOD_RADIUS,
        apod_anisotropy="auto",
        apod_g_scale=1.0,
        white_noise=0.0,
    )
    gp.initialize(X[train], y_all[train], y_err=yerr_all[train])
    t0 = time.time()
    gp.solve()
    t_solve = time.time() - t0
    t0 = time.time()
    gp.predict(X[train[:10]])  # builds the mean field
    t_pred = time.time() - t0
    opt, eng = gp._optimizer, gp._engine
    y = gp._y - gp._mean - gp._spatial_average  # the centered values the solver sees
    sig = np.asarray(gp._y_err, float)
    print(
        f"TURBO-GP: solve {t_solve:.1f} s, mean field {t_pred:.1f} s; {eng.n_iterations} CG iterations; "
        f"xi(0) raw {opt._xi[opt.npix // 2, opt.npix // 2]:.1f}, cleaned "
        f"{opt._xi_clean[opt.npix // 2, opt.npix // 2]:.1f}, effective {eng.xi0_eff:.1f} mas^2; "
        f"median noise {np.median(sig):.1f} mas"
    )

    cache = {}
    if os.path.exists(CACHE_L21):
        with open(CACHE_L21, "rb") as f:
            cache = pickle.load(f)
    l21 = cache.get(comp)
    if l21 is None and not args.no_l21:
        print("L21 anisotropic fit (slow, 10-25 min)...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            l21 = fit_l21(X, y_all, yerr_all, train)
        cache[comp] = l21
        with open(CACHE_L21, "wb") as f:
            pickle.dump(cache, f)

    fig_kernel(opt, l21, comp, args.out)
    fig_matvec(eng, X[train], y, sig, comp, args.out)
    fig_cg(eng, y, sig, comp, args.out)


if __name__ == "__main__":
    main()
