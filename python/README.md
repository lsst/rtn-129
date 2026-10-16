# Figures of Sect. 3 (TURBO-GP on a real visit)

`make_turbogp_figures.py` produces the three figures of Sect. 3 of RTN-129 that
illustrate the TURBO-GP method on one real LSSTCam visit, the same visit as the
interactive slides at https://pfleget.github.io/treegp/ :

| figure | content |
|---|---|
| `figures/turbogp_kernel.pdf` | the kernel: measured 21x21 correlation map, L21 von Karman fit and residual (top); 20" map x window, Fourier power with the kept modes, cleaned map = the TURBO-GP kernel (bottom) |
| `figures/turbogp_matvec.pdf` | the four steps of the product C y (Eq. 12) on the whole visit: deposit on the grid (zoom), kernel image, convolution over the whole grid, read back at the stars + noise |
| `figures/turbogp_cg.pdf` | the conjugate-gradient solve: relative residual with and without the Jacobi preconditioner, mean field after 1, 3, 10, 30 iterations and at convergence |

## Data

`data/rubin_visit_2025052000348.npz` (3.4 MB): visit 2025052000348, z band,
163 661 sources; float32 arrays `u`, `v` (tangent-plane position, degrees),
`du`, `dv` (astrometric residuals, mas), `du_err`, `dv_err` (their errors, mas),
plus `visit` and `band`. It is the compact version of the example pickle
distributed by the Gomes et al. (2025) authors with their code, and is also
served at https://pfleget.github.io/treegp/data/rubin_visit_2025052000348.npz .

`data/cache_l21.pkl` (126 kB): the cached L21 anisotropic fit of the du and dv
components (measured 21x21 map and fitted kernel), used for the top row of the
kernel figure. Delete it to refit (a few tens of seconds for the fit itself).

## Run

```
cd rtn-129
python3 python/make_turbogp_figures.py            # du component, ~30 s
python3 python/make_turbogp_figures.py --component v
python3 python/make_turbogp_figures.py --no-l21   # kernel figure without the L21 row
```

The script prints the numbers quoted in the captions (iterations, timings,
kept Fourier modes, ...). Requirements: treegp >= 1.5.0 (with its
`empirical-2pcf` optimizer and `spectral` solver), treecorr, numpy, scipy,
matplotlib. Configuration identical to the paper: 10 000 random training
sources (seed 1234), 20" pixels out to +-0.3 deg, Hann window of radius
0.278 deg with the "auto" anisotropy, S/N threshold 2.5, grid spacing 10",
conjugate gradient to a relative residual of 1e-7.

Style: `black -l 88 python/` and
`flake8 --ignore=E203,W503 --max-line-length=120 python/`.
