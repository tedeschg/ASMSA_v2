"""Post-processing a metadynamics run: COLVAR/HILLS parsing, FES, convergence.

Two independent routes to the free-energy surface are provided, which is
deliberate — agreeing results from both is the strongest evidence a run has
converged:

``sum_hills``
    Sum the deposited Gaussians (requires the ``plumed`` executable). This is
    the bias-potential estimate of the FES.
``reweight_fes``
    Reweight the biased trajectory back to the unbiased ensemble using the
    recorded bias and the time-dependent offset c(t), then histogram. Pure
    NumPy, no PLUMED needed, and it can reweight onto *any* observable, not
    only the biased CVs.

The reweighting estimator is the standard one for well-tempered
metadynamics: a frame at CV value s and time t carries weight
``exp(beta * (V(s, t) - c(t)))`` (Tiwary & Parrinello, J. Phys. Chem. B 119,
736 (2015)); ``c(t)`` is what ``CALC_RCT`` writes to COLVAR as ``metad.rct``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Boltzmann constant in PLUMED/GROMACS default units (kJ/mol/K).
KB_KJ_PER_MOL = 0.008314462618


def kT(temperature: float = 300.0) -> float:
    """Thermal energy in kJ/mol."""
    return KB_KJ_PER_MOL * float(temperature)


def _parse_fields(path: Path) -> list[str] | None:
    """Read PLUMED's ``#! FIELDS`` header, which names the columns."""
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith("#! FIELDS"):
                return line.split()[2:]
            if not line.startswith("#"):
                break
    return None


def read_plumed_file(path: str | Path):
    """Read a PLUMED output file (COLVAR, HILLS, fes.dat) into a DataFrame.

    Column names come from the file's own ``#! FIELDS`` line, so there is no
    hard-coded assumption about how many CVs or bias terms were printed.
    """
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. PLUMED writes it during the biased run; "
            f"if the simulation has not started yet there is nothing to read."
        )
    fields = _parse_fields(path)
    has_data = any(
        line.strip() and not line.lstrip().startswith(("#", "@"))
        for line in path.read_text(errors="replace").splitlines()
    )
    if not has_data:
        raise ValueError(
            f"{path} contains no data rows yet (header only). PLUMED appends "
            f"rows as the simulation progresses."
        )
    data = np.loadtxt(path, comments=["#", "@"], ndmin=2)
    if fields and len(fields) == data.shape[1]:
        return pd.DataFrame(data, columns=fields)
    return pd.DataFrame(
        data, columns=[f"col{i}" for i in range(data.shape[1])]
    )


read_colvar = read_plumed_file
read_hills = read_plumed_file


def reweight_weights(
    colvar,
    temperature: float = 300.0,
    bias_col: str | None = None,
    rct_col: str | None = None,
    discard_fraction: float = 0.0,
) -> np.ndarray:
    """Per-frame reweighting weights from a COLVAR DataFrame.

    Parameters
    ----------
    colvar : DataFrame
        From :func:`read_colvar`.
    temperature : float
        In kelvin; must match the simulation's ``ref_t``.
    bias_col, rct_col : str, optional
        Column names. Auto-detected (``*.bias``, ``*.rct``) if omitted.
    discard_fraction : float
        Drop this leading fraction of the trajectory as transient. In
        well-tempered metadynamics the early bias is far from converged, so
        discarding the first 10-20% is routine.

    Returns
    -------
    array
        Weights normalized to sum to 1. If no ``rct`` column is present a
        warning is issued and ``c(t)`` is approximated by the running maximum
        of the bias, which is cruder; prefer re-running with ``CALC_RCT``.
    """
    import warnings

    if bias_col is None:
        cands = [c for c in colvar.columns if c.endswith(".bias")]
        if not cands:
            raise ValueError(
                f"no bias column found in {list(colvar.columns)}; the COLVAR "
                "file must print the metadynamics bias to be reweightable"
            )
        bias_col = cands[0]
    bias = colvar[bias_col].to_numpy(dtype=np.float64)

    if rct_col is None:
        cands = [c for c in colvar.columns if c.endswith(".rct")]
        rct_col = cands[0] if cands else None

    if rct_col is not None:
        ct = colvar[rct_col].to_numpy(dtype=np.float64)
    else:
        warnings.warn(
            "COLVAR has no 'metad.rct' column, so c(t) is approximated by the "
            "running maximum of the bias. Add CALC_RCT to the METAD line "
            "(asmsa_v2 does so by default) for a correct reweighting.",
            stacklevel=2,
        )
        ct = np.maximum.accumulate(bias)

    n = len(bias)
    start = int(n * float(discard_fraction))
    w = np.zeros(n, dtype=np.float64)
    arg = (bias[start:] - ct[start:]) / kT(temperature)
    arg -= arg.max()                      # stabilize the exponential
    w[start:] = np.exp(arg)
    total = w.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError("reweighting produced degenerate weights")
    return w / total


def reweight_fes(
    cv_values,
    weights,
    bins: int = 80,
    temperature: float = 300.0,
    max_fe: float | None = None,
    in_kT: bool = False,
    ranges=None,
):
    """Free-energy surface by weighted histogram of 1 or 2 CVs.

    Parameters
    ----------
    cv_values : ``[n_frames]`` or ``[n_frames, 2]``
    weights : ``[n_frames]``
        From :func:`reweight_weights`.
    bins : int
    temperature : float
    max_fe : float, optional
        Clip the surface for readability. Empty bins are set to NaN rather
        than clipped, so unsampled regions read as blank instead of as a
        plateau at the cap.
    in_kT : bool
        Report in units of kT rather than kJ/mol.

    Returns
    -------
    tuple
        1-D: ``(centers, fe)``. 2-D: ``(x_centers, y_centers, fe)`` with
        ``fe`` shaped ``[ny, nx]``, ready for ``pcolormesh``/``contourf``.
    """
    cv = np.asarray(cv_values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    # FE = -kT ln P; expressed in units of kT that is simply -ln P
    factor = 1.0 if in_kT else kT(temperature)

    if cv.ndim == 1:
        H, edges = np.histogram(cv, bins=bins, weights=w, range=ranges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        with np.errstate(divide="ignore"):
            fe = -factor * np.log(H)
        fe = _shift_clip(fe, max_fe)
        return centers, fe

    if cv.ndim != 2 or cv.shape[1] != 2:
        raise ValueError(f"expected [n,1] or [n,2] CV values, got {cv.shape}")
    H, xe, ye = np.histogram2d(cv[:, 0], cv[:, 1], bins=bins, weights=w,
                               range=ranges)
    with np.errstate(divide="ignore"):
        fe = -factor * np.log(H.T)        # transpose -> [ny, nx]
    fe = _shift_clip(fe, max_fe)
    return 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]), fe


def _shift_clip(fe: np.ndarray, max_fe: float | None) -> np.ndarray:
    fe = np.where(np.isfinite(fe), fe, np.nan)
    if np.all(np.isnan(fe)):
        return fe
    fe = fe - np.nanmin(fe)
    if max_fe is not None:
        fe = np.where(fe > max_fe, np.nan, fe)
    return fe


def sum_hills(
    hills: str | Path = "HILLS",
    out_dir: str | Path = ".",
    runner=None,
    plumed: str = "plumed",
    stride: int | None = None,
    bins: int | None = None,
    mintozero: bool = True,
    kt_value: float | None = None,
    extra_args: list[str] | None = None,
):
    """Run ``plumed sum_hills`` and read the resulting free-energy surface.

    Parameters
    ----------
    stride : int, optional
        Emit an intermediate FES every ``stride`` hills, producing
        ``fes_0.dat``, ``fes_1.dat``, ... — this is what
        :func:`fes_convergence` consumes.
    kt_value : float, optional
        ``--kt`` in kJ/mol; required by ``sum_hills`` to un-temper a
        well-tempered run.

    Returns
    -------
    list[Path]
        The ``fes*.dat`` files written, in order.
    """
    from .runner import LocalRunner

    hills = Path(hills)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not hills.exists():
        raise FileNotFoundError(
            f"{hills} not found. It is written by METAD during the biased run."
        )
    runner = runner or LocalRunner(out_dir, echo=True)
    if not runner.available(plumed):
        raise FileNotFoundError(
            f"{plumed!r} not on PATH, so sum_hills cannot run. Use "
            "reweight_fes() on the COLVAR file instead -- it needs no PLUMED "
            "executable -- or install PLUMED."
        )

    args = [plumed, "sum_hills", "--hills", str(hills.resolve())]
    if stride:
        args += ["--stride", str(int(stride))]
    if bins:
        args += ["--bin", ",".join([str(int(bins))] * 2)]
    if mintozero:
        args += ["--mintozero"]
    if kt_value is not None:
        args += ["--kt", f"{kt_value}"]
    args += extra_args or []

    runner.run(args, cwd=out_dir).check()
    files = sorted(out_dir.glob("fes*.dat"),
                   key=lambda p: (len(p.stem), p.stem))
    if not files:
        raise RuntimeError(f"sum_hills produced no fes*.dat in {out_dir}")
    return files


def read_fes_grid(path: str | Path):
    """Read a ``sum_hills`` output into ``(x, y, fe)`` or ``(x, fe)``.

    ``sum_hills`` writes the grid as a flat list with blank-line
    separators; this reshapes it back.
    """
    df = read_plumed_file(path)
    cols = list(df.columns)
    if len(cols) >= 3 and "file.free" in cols:
        xcol, ycol = cols[0], cols[1]
        x = np.unique(df[xcol].to_numpy())
        y = np.unique(df[ycol].to_numpy())
        if x.size * y.size == len(df):
            fe = df["file.free"].to_numpy().reshape(y.size, x.size, order="F")
            return x, y, fe
    if "file.free" in cols:
        return df[cols[0]].to_numpy(), df["file.free"].to_numpy()
    raise ValueError(
        f"{path} does not look like sum_hills output (columns: {cols})"
    )


def fes_convergence(fes_files, reference: int = -1) -> np.ndarray:
    """Mean absolute FES deviation of each intermediate surface from a reference.

    The standard metadynamics convergence check: run ``sum_hills`` with a
    ``--stride``, then confirm this curve flattens near zero. A curve still
    descending at the end means the run is not converged, however smooth the
    trajectory looks.
    """
    grids = []
    for f in fes_files:
        g = read_fes_grid(f)
        grids.append(g[-1])
    ref = grids[reference]
    out = []
    for g in grids:
        if g.shape != ref.shape:
            out.append(np.nan)
            continue
        d = g - ref
        d = d - np.nanmean(d)            # FES is defined up to a constant
        out.append(float(np.nanmean(np.abs(d))))
    return np.asarray(out)


def hill_height_decay(hills) -> np.ndarray:
    """Deposited Gaussian heights over time.

    In well-tempered metadynamics the height decays as the bias fills the
    basins, so a height series that is still near its initial value is a
    direct sign the run has not explored enough.
    """
    import pandas as pd

    if isinstance(hills, (str, Path)):
        hills = read_hills(hills)
    cands = [c for c in hills.columns if "height" in c.lower()]
    if not cands:
        raise ValueError(f"no height column in {list(hills.columns)}")
    return hills[cands[0]].to_numpy(dtype=np.float64)
