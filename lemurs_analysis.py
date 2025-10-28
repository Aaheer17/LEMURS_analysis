
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import h5py  # for .h5/.hdf5 loading
import xml.etree.ElementTree as ET

# ============================ Utilities ============================

def to_numpy(x):
    """Return a NumPy array from possibly-NumPy or torch tensors (torch optional)."""
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def corrcoef_safe(X2d: np.ndarray) -> np.ndarray:
    """Pearson correlation across columns of X2d (N x K).
    NaNs from zero-variance columns are replaced with 0; diag=1.
    """
    C = np.corrcoef(X2d, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)
    return C


def _tickify(ax, k: int, prefix: str, step: int | None = None):
    """Nice ticks: L0, L5, ... or ϕ0, ϕ2, ..."""
    if step is None:
        step = 5 if k >= 25 else max(1, k // 10)
    ticks = np.arange(0, k, step)
    ax.set_xticks(ticks + 0.5)
    ax.set_yticks(ticks + 0.5)
    ax.set_xticklabels([f"{prefix}{i}" for i in ticks], rotation=0, fontsize=8)
    ax.set_yticklabels([f"{prefix}{i}" for i in ticks], rotation=0, fontsize=8)

    
def dims_from_xml(xml_path: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    layers = root.findall(".//Layer")
    assert len(layers) > 0, "No <Layer> elements found."
    L = len(layers)

    # check consistency across layers
    phi_vals = []
    rbin_vals = []
    for ly in layers:
        n_phi = int(ly.attrib["n_bin_alpha"])
        phi_vals.append(n_phi)
        edges = [e.strip() for e in ly.attrib["r_edges"].split(",") if e.strip() != ""]
        rbin_vals.append(len(edges) - 1)

    # ensure all layers use the same binning
    assert len(set(phi_vals)) == 1, f"Inconsistent n_bin_alpha across layers: {set(phi_vals)}"
    assert len(set(rbin_vals)) == 1, f"Inconsistent r bin counts across layers: {set(rbin_vals)}"
    P = phi_vals[0]
    R = rbin_vals[0]
    return L, P, R

def reshape_flat_using_xml(X_flat: np.ndarray, xml_path: str, order="LPR"):
    """
    X_flat: [N, 6480]
    order: the flattening order used originally (default assumes [L, P, R] contiguous).
           Valid strings are permutations of 'L','P','R'.
    Returns X with shape [N, 45, 16, 9] based on the XML.
    """
    assert X_flat.ndim == 2, f"Expected [N, D], got {X_flat.shape}"
    L, P, R = dims_from_xml(xml_path)
    D = L * P * R
    assert X_flat.shape[1] == D, f"Feature dim {X_flat.shape[1]} != L*P*R={D} from XML."

    # reshape into the original order, then permute to L,P,R
    shape_in = tuple({"L": L, "P": P, "R": R}[c] for c in order)  # e.g. (L,P,R)
    X = X_flat.reshape(-1, *shape_in)

    # permute to L,P,R
    pos = {c: i for i, c in enumerate(order)}
    X = X.transpose((0, 1 + order.index("L"), 1 + order.index("P"), 1 + order.index("R")))
    return X  # [N,45,16,9] for your XML
def _has_angles(theta, phi, n_theta_bins: int, n_phi_bins: int) -> bool:
    """True only if angles are provided AND you asked for nonzero bins."""
    return (theta is not None) and (phi is not None) and (n_theta_bins > 0) and (n_phi_bins > 0)

def _stratify_ids(E_inc, theta, phi, n_energy_bins: int, n_theta_bins: int, n_phi_bins: int,
                  percentile_bins: bool = False):
    """
    Returns (bin_ids, strat_label, strat_kind)
      - bin_ids: np.array shape [N] with integers 0..K-1
      - strat_label: short text for figure titles
      - strat_kind: 'angles' or 'energy'
    If angles are missing or bins set to 0, falls back to energy bins.
    """
    

    if _has_angles(theta, phi, n_theta_bins, n_phi_bins):
        # θ/φ binning (your current logic here)
        tmin, tmax = float(np.nanmin(theta)), float(np.nanmax(theta))
        pmin, pmax = -np.pi, np.pi  # or your stored range
        t_edges = np.linspace(tmin, tmax, n_theta_bins + 1)
        p_edges = np.linspace(pmin, pmax, n_phi_bins + 1)
        t_idx = np.digitize(theta, t_edges) - 1
        p_idx = np.digitize(phi,   p_edges) - 1
        # combine to a single id  (θ major)
        bin_ids = (t_idx * n_phi_bins + p_idx).astype(int)
        label = f"θ×φ bins ({n_theta_bins}×{n_phi_bins})"
        return bin_ids, label, 'angles'
    else:
        # Energy binning (fallback when no angles OR bins==0)
        assert E_inc is not None, "E_inc required for energy stratification"
        if percentile_bins:
            qs = np.linspace(0, 100, n_energy_bins + 1)
            edges = np.percentile(E_inc, qs)
            # ensure strictly increasing to avoid edge collisions
            edges = np.unique(edges)
            # if heavy ties reduce bins, ensure at least 2 edges
            if edges.size < 2:
                edges = np.array([E_inc.min(), E_inc.max()])
        else:
            emin, emax = float(np.nanmin(E_inc)), float(np.nanmax(E_inc))
            edges = np.linspace(emin, emax, n_energy_bins + 1)
        # guard if n_energy_bins might be 0
        if len(edges) < 2:
            # put everything into a single bin
            bin_ids = np.zeros_like(E_inc, dtype=int)
            label = "all data (no strat)"
        else:
            bin_ids = np.digitize(E_inc, edges) - 1
            label = f"energy bins ({len(edges)-1})"
        return bin_ids, label, 'energy'

def plot_grid(
    mats: List[np.ndarray],
    titles: List[str],
    xlabel: str,
    ylabel: str,
    tick_prefix: str,
    annotate: bool = False,
    ncols: int = 3,
    figsize_per: Tuple[float, float] = (4.0, 3.6),
) -> plt.Figure:
    """Lay out multiple heatmaps in a grid with ONE shared colorbar."""
    n = len(mats)
    if n == 0:
        return plt.figure()

    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    mappable = None

    for i, C in enumerate(mats):
        ax = axes[i]
        hm = sns.heatmap(
            C, vmin=-1, vmax=1, center=0, cmap="coolwarm",
            square=True, cbar=False, annot=annotate,
            fmt=".2f" if annotate else "", ax=ax
        )
        ax.set_title(titles[i], fontsize=9)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        _tickify(ax, C.shape[0], tick_prefix)
        if mappable is None:
            mappable = hm.collections[0]

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")  # type: ignore[index]

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes[:n], shrink=0.9, pad=0.01)
        cbar.set_label("Pearson r")
    return fig


def _corr_adjacent(U_2d: np.ndarray) -> np.ndarray:
    """Adjacent-bin correlation across columns of U_2d (N x K) -> (K-1,)."""
    C = np.corrcoef(U_2d.T)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)
    return np.diag(C, k=1)


# ============================ Data Loaders ============================

def _load_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    X  = data.get('X') or data.get('showers') or data.get('shower')
    E  = data.get('E_inc') or data.get('incident_energy') or data.get('E')
    th = data.get('theta') or data.get('incident_theta')
    ph = data.get('phi')   or data.get('incident_phi')
    if X is None or E is None or th is None or ph is None:
        raise ValueError(
            ".npz must contain X (or showers), E_inc (or incident_energy), theta (or incident_theta), phi (or incident_phi)"
        )
    return to_numpy(X), to_numpy(E), to_numpy(th), to_numpy(ph)


def _load_h5(h5_path: Path, dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """HDF5 datasets:
        showers          : (N, R, Phi, Z)  -> transpose to (N, Z, Phi, R) = (N,45,16,9)
        incident_energy  : (N,)
        incident_theta   : (N,)
        incident_phi     : (N,)
    """
    if dataset=='lemurs':
        with h5py.File(h5_path, "r") as f:
            showers = f["showers"][:]               # (N, R, Phi, Z)
            E_inc   = f["incident_energy"][:]
            theta   = f["incident_theta"][:]
            phi     = f["incident_phi"][:]
        X = np.transpose(showers, (0, 3, 2, 1)).copy()  # -> (N, Z, Phi, R) = (N,45,16,9)
    else: 
        with h5py.File(h5_path, "r") as f:
            showers = f["showers"][:]               # (N, R, Phi, Z)
            E_inc   = f["incident_energies"][:]
            theta=None
            phi=None
        X=showers
    return X, E_inc, theta, phi


def _load_any(path: Path, dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    suf = path.suffix.lower()
    if suf == ".npz":
        return _load_npz(path)
    if suf in (".h5", ".hdf5"):
        return _load_h5(path, dataset)
    raise ValueError(f"Unsupported input: {path} (expected .npz, .h5, .hdf5)")


# ============================ Analyses ============================

def compute_stratified_correlations(
    shower: np.ndarray,
    E_inc: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    n_energy_bins: int = 3,
    n_theta_bins: int = 3,
    n_phi_bins: int = 3,
    min_samples: int = 50,
    save_path: str = 'stratified_correlations.pdf',
    annotate: bool = False,
    ncols: int = 3,
    normalize_per_event: bool = False,   
    use_log1p: bool = False,             
    use_spearman: bool = False,          
    eps: float = 1e-12                  
) -> Dict[str, list]:
    """Stratified correlation matrices across Layers/φ/r; writes a multi-page PDF."""
    #X = to_numpy(shower); E = to_numpy(E_inc); 
    X = to_numpy(shower)
    E = np.asarray(E_inc).reshape(-1)      # <-- ensure (N,)
    assert X.ndim == 4 and X.shape[1:] == (45, 16, 9), f"Expected (N,45,16,9), got {X.shape}"

    E_bins  = np.percentile(E, np.linspace(0, 100, n_energy_bins + 1))
    if _has_angles(theta, phi, n_theta_bins, n_phi_bins):
        th = to_numpy(theta); ph = to_numpy(phi)
        th_bins = np.linspace(th.min(), th.max(), n_theta_bins + 1)
        ph_bins = np.linspace(ph.min(), ph.max(), n_phi_bins + 1)
    else: 
        th=None; ph=None
        th_bins=0
        ph_bins=0
    results = {'energy': [], 'theta': [], 'phi': []}
    def _maybe_normalize_eventwise(X_block: np.ndarray) -> np.ndarray:
        # shape (Nb, 45, 16, 9) -> normalise each event by its total deposited energy
        if not normalize_per_event:
            return X_block
        totals = X_block.sum(axis=(1,2,3), keepdims=True)
        return X_block / (totals + eps)

    def _prep_features(F: np.ndarray) -> np.ndarray:
        """
        F is (Nb, K). Apply log1p and/or rank transform before correlation.
        - use_log1p: Pearson on log1p(F) (variance-stabilised).
        - use_spearman: Pearson on ranks(F) (Spearman).
        If both flags are True, ranks are taken on log1p(F).
        """
        G = F
        if use_log1p:
            G = np.log1p(np.maximum(G, 0.0))  # energies are >=0; guard anyway
        if use_spearman:
            # rank each column (average ranks for ties) without SciPy:
            # argsort twice trick -> dense ranks; for ties, average via grouping
            # For simplicity and robustness, if you have SciPy available, use rankdata.
            try:
                from scipy.stats import rankdata
                G = np.apply_along_axis(rankdata, 0, G, method='average')
            except Exception:
                # Fallback dense ranks (ties get same integer rank)
                order = np.argsort(G, axis=0)
                ranks = np.empty_like(order, dtype=float)
                # assign ranks column-wise
                for k in range(G.shape[1]):
                    ranks[order[:,k], k] = np.arange(1, G.shape[0]+1)
                G = ranks
        return G

    with PdfPages(save_path) as pdf:
        # ---- Energy strata ----
        mats, titles = [], []
        for i in range(n_energy_bins):
            lo, hi = E_bins[i], E_bins[i + 1]
            mask = (E >= lo) & ((E <= hi) if i == n_energy_bins - 1 else (E < hi))
            mask = mask.ravel()   
            #mask = (E >= lo) & ((E <= hi) if i == n_energy_bins - 1 else (E < hi))
            if mask.sum() < min_samples:
                continue
            #Xb = X[mask]
            Xb = _maybe_normalize_eventwise(X[mask])
            feat_layers = Xb.sum(axis=(2, 3))
            feat_phi    = Xb.sum(axis=(1, 3))
            feat_r      = Xb.sum(axis=(1, 2))

            feat_layers = _prep_features(feat_layers)
            feat_phi    = _prep_features(feat_phi)
            feat_r      = _prep_features(feat_r)

            C_layers = corrcoef_safe(feat_layers)
            C_phi    = corrcoef_safe(feat_phi)
            C_r      = corrcoef_safe(feat_r)
            results['energy'].append({'range': (lo, hi), 'N': int(mask.sum()),
                                      'C_layers': C_layers, 'C_phi': C_phi, 'C_r': C_r})
            titles.append(f"E ∈ [{lo:.0f}, {hi:.0f}] MeV\nN={mask.sum():,}")
            mats.append(C_layers)
        if mats:
            fig = plot_grid(mats, titles, "Layer j", "Layer i", "L", annotate, ncols)
            fig.suptitle('Layer–Layer Correlations Stratified by Incident Energy', fontsize=12, fontweight='bold')
            pdf.savefig(fig); plt.close(fig)

        # ---- Theta strata ----
        mats, titles = [], []
        for i in range(n_theta_bins):
            lo, hi = th_bins[i], th_bins[i + 1]
            mask = (th >= lo) & ((th <= hi) if i == n_theta_bins - 1 else (th < hi))
            if mask.sum() < min_samples:
                continue
            Xb = _maybe_normalize_eventwise(X[mask])
            feat_layers = Xb.sum(axis=(2, 3))
            feat_phi    = Xb.sum(axis=(1, 3))
            feat_r      = Xb.sum(axis=(1, 2))

            feat_layers = _prep_features(feat_layers)
            feat_phi    = _prep_features(feat_phi)
            feat_r      = _prep_features(feat_r)

            C_layers = corrcoef_safe(feat_layers)
            C_phi    = corrcoef_safe(feat_phi)
            C_r      = corrcoef_safe(feat_r)
            results['theta'].append({'range': (lo, hi), 'N': int(mask.sum()),
                                     'C_layers': C_layers, 'C_phi': C_phi, 'C_r': C_r})
            titles.append(f"θ_inc ∈ [{np.degrees(lo):.1f}°, {np.degrees(hi):.1f}°]\nN={mask.sum():,}")
            mats.append(C_layers)
        if mats:
            fig = plot_grid(mats, titles, "Layer j", "Layer i", "L", annotate, ncols)
            fig.suptitle('Layer–Layer Correlations Stratified by Incident θ', fontsize=12, fontweight='bold')
            pdf.savefig(fig); plt.close(fig)

        # ---- Phi strata ----
        mats, titles = [], []
        for i in range(n_phi_bins):
            lo, hi = ph_bins[i], ph_bins[i + 1]
            mask = (ph >= lo) & ((ph <= hi) if i == n_phi_bins - 1 else (ph < hi))
            if mask.sum() < min_samples:
                continue
            Xb = _maybe_normalize_eventwise(X[mask])
            feat_layers = Xb.sum(axis=(2, 3))
            feat_phi    = Xb.sum(axis=(1, 3))
            feat_r      = Xb.sum(axis=(1, 2))

            feat_layers = _prep_features(feat_layers)
            feat_phi    = _prep_features(feat_phi)
            feat_r      = _prep_features(feat_r)

            C_layers = corrcoef_safe(feat_layers)
            C_phi    = corrcoef_safe(feat_phi)
            C_r      = corrcoef_safe(feat_r)
            results['phi'].append({'range': (lo, hi), 'N': int(mask.sum()),
                                   'C_layers': C_layers, 'C_phi': C_phi, 'C_r': C_r})
            titles.append(f"φ_inc ∈ [{np.degrees(lo):.1f}°, {np.degrees(hi):.1f}°]\nN={mask.sum():,}")
            mats.append(C_layers)
        if mats:
            fig = plot_grid(mats, titles, "Layer j", "Layer i", "L", annotate, ncols)
            fig.suptitle('Layer–Layer Correlations Stratified by Incident φ', fontsize=12, fontweight='bold')
            pdf.savefig(fig); plt.close(fig)

        # ---- Angular- & radial-bin correlations by strata ----
        if _has_angles(theta, phi, n_theta_bins, n_phi_bins):
            for key, label, xlabel, tick_prefix in [
                ('C_phi', 'Angular-Bin (ϕ) Correlations', 'ϕ bin j', 'ϕ'),
                ('C_r',   'Radial-Bin (r) Correlations',  'r bin j', 'r'),
            ]:
                for section, stitle in [('energy', 'Incident Energy'),
                                        ('theta',  'Incident θ'),
                                        ('phi',    'Incident φ')]:
                    if results[section]:
                        mats = [res[key] for res in results[section]]
                        titles = [
                            f"{stitle}: " +
                            (f"E ∈ [{lo:.0f}, {hi:.0f}] MeV" if section == 'energy'
                             else f"θ_inc ∈ [{np.degrees(lo):.1f}°, {np.degrees(hi):.1f}°]" if section == 'theta'
                             else f"φ_inc ∈ [{np.degrees(lo):.1f}°, {np.degrees(hi):.1f}°]") +
                            f"\nN={res['N']:,}"
                            for res in results[section] for (lo, hi) in [res['range']]
                        ]
                        fig = plot_grid(mats, titles, xlabel, xlabel.replace('j', 'i'), tick_prefix, annotate, ncols)
                        fig.suptitle(f"{label} Stratified by {stitle}", fontsize=12, fontweight='bold')
                        pdf.savefig(fig); plt.close(fig)

    print(f"[write] {save_path}")
    return results


def compute_and_plot_mean_profiles(
    shower: np.ndarray,
    E_inc: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    n_energy_bins: int = 3,
    n_theta_bins: int = 3,
    n_phi_bins: int = 3,
    min_samples: int = 50,
    save_path: str = 'mean_profiles_stratified.pdf',
    use_percentile_bins: bool = True,
    normalize_per_event: bool = False,
    ncols: int = 3,
    show_ci: bool = True
) -> Dict[str, list]:
    """Stratified mean ± SEM profiles for Layers/ϕ/r; writes a multi-page PDF."""
    X = to_numpy(shower); E = to_numpy(E_inc); #th = to_numpy(theta); ph = to_numpy(phi)
    assert X.ndim == 4 and X.shape[1:] == (45, 16, 9), f"Expected (N,45,16,9), got {X.shape}"

    def _bin_edges(values, n_bins, pct=False):
        return (np.percentile(values, np.linspace(0, 100, n_bins + 1)) if pct
                else np.linspace(values.min(), values.max(), n_bins + 1))

    def _bin_masks(values, edges):
        out = []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            mask = (values >= lo) & ((values <= hi) if i == len(edges) - 2 else (values < hi))
            out.append((mask, (lo, hi)))
        return out

    def _profiles_from_X(X_bin, normalize=False, eps=1e-12):
        Nb = X_bin.shape[0]
        Xn = X_bin / (X_bin.sum(axis=(1, 2, 3), keepdims=True) + eps) if normalize else X_bin
        per_layers = Xn.sum(axis=(2, 3))
        per_phi    = Xn.sum(axis=(1, 3))
        per_r      = Xn.sum(axis=(1, 2))

        def _ms(x):
            mean = x.mean(axis=0)
            sem  = x.std(axis=0, ddof=1) / np.sqrt(max(1, Nb))
            return mean, sem

        mL, sL = _ms(per_layers)
        mP, sP = _ms(per_phi)
        mR, sR = _ms(per_r)
        return dict(mean_layers=mL, sem_layers=sL, mean_phi=mP, sem_phi=sP, mean_r=mR, sem_r=sR, N=Nb)

    def _plot_profile_grid(profiles, titles, x_label, y_label, tick_prefix, ncols=3, figsize_per=(4.2, 3.4), x_tick_step=None, show_ci=True):
        n = len(profiles)
        if n == 0:
            return None
        ncols_local = max(1, min(ncols, n)); nrows = math.ceil(n / ncols_local)
        fig, axes = plt.subplots(nrows, ncols_local, figsize=(figsize_per[0]*ncols_local, figsize_per[1]*nrows), constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()

        K = len(next(iter(profiles[0].values())))
        xs = np.arange(K)
        if x_tick_step is None:
            x_tick_step = 5 if K >= 25 else max(1, K // 10)
        tick_idx = np.arange(0, K, x_tick_step)

        for i, (p, t) in enumerate(zip(profiles, titles)):
            ax = axes[i]
            key_mean = [k for k in p.keys() if k.startswith('mean_')][0]
            base = key_mean.replace('mean_', '')
            key_sem = f'sem_{base}'
            y = p[key_mean]
            ax.plot(xs, y, marker='o', markersize=2, linewidth=1)
            if show_ci and key_sem in p:
                se = p[key_sem]
                ax.fill_between(xs, y - se, y + se, alpha=0.25, linewidth=0)
            ax.set_title(t, fontsize=9)
            ax.set_xlabel(x_label, fontsize=9)
            ax.set_ylabel(y_label, fontsize=9)
            ax.set_xticks(tick_idx)
            ax.set_xticklabels([f"{tick_prefix}{i}" for i in tick_idx], fontsize=8)

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        return fig

    # Binning
    E_edges  = _bin_edges(E, n_energy_bins, pct=use_percentile_bins)
    has_angles= _has_angles(theta, phi, n_theta_bins, n_phi_bins)
    if has_angles:
        th = to_numpy(theta); ph = to_numpy(phi)
        th_edges = _bin_edges(th, n_theta_bins, pct=False)
        ph_edges = _bin_edges(ph, n_phi_bins, pct=False)
    else: 
        th=None; ph=None
        th_edges=0
        ph_edges=0
   

    results = {"energy": [], "theta": [], "phi": []}

    with PdfPages(save_path) as pdf:
        # Energy
        e_titles_L, e_prof_L = [], []
        e_titles_P, e_prof_P = [], []
        e_titles_R, e_prof_R = [], []
        for mask, (lo, hi) in _bin_masks(E, E_edges):
            Nb = int(mask.sum())
            if Nb < min_samples:
                continue
            mask = mask.ravel()
            Xb = X[mask]
            prof = _profiles_from_X(Xb, normalize=normalize_per_event)
            results['energy'].append({'range': (lo, hi), **prof})
            t = f"E ∈ [{lo:.0f}, {hi:.0f}] MeV\nN={Nb:,}"
            e_titles_L.append(t); e_prof_L.append({'mean_layers': prof['mean_layers'], 'sem_layers': prof['sem_layers']})
            e_titles_P.append(t); e_prof_P.append({'mean_phi':    prof['mean_phi'],    'sem_phi':    prof['sem_phi']})
            e_titles_R.append(t); e_prof_R.append({'mean_r':      prof['mean_r'],      'sem_r':      prof['sem_r']})
        if e_prof_L:
            fig = _plot_profile_grid(e_prof_L, e_titles_L, 'Layer index', 'Mean deposited energy', 'L', ncols=ncols, show_ci=show_ci)
            if not normalize_per_event:
                for ax in np.atleast_1d(fig.axes): 
                    try: ax.set_yscale('log')
                    except Exception: pass
            fig.suptitle('Mean Layer Energy — Stratified by Incident Energy', fontsize=12, fontweight='bold'); pdf.savefig(fig); plt.close(fig)

            fig = _plot_profile_grid(e_prof_P, e_titles_P, 'ϕ-bin index', 'Mean deposited energy', 'ϕ', ncols=ncols, show_ci=show_ci, x_tick_step=2)
            if not normalize_per_event:
                for ax in np.atleast_1d(fig.axes): 
                    try: ax.set_yscale('log')
                    except Exception: pass
            fig.suptitle('Mean Angular (ϕ-bin) Energy — Stratified by Incident Energy', fontsize=12, fontweight='bold'); pdf.savefig(fig); plt.close(fig)

            fig = _plot_profile_grid(e_prof_R, e_titles_R, 'r-bin index', 'Mean deposited energy', 'r', ncols=ncols, show_ci=show_ci, x_tick_step=1)
            if not normalize_per_event:
                for ax in np.atleast_1d(fig.axes): 
                    try: ax.set_yscale('log')
                    except Exception: pass
            fig.suptitle('Mean Radial (r-bin) Energy — Stratified by Incident Energy', fontsize=12, fontweight='bold'); pdf.savefig(fig); plt.close(fig)
        if has_angles:
            # Theta
            t_titles_L, t_prof_L = [], []
            t_titles_P, t_prof_P = [], []
            t_titles_R, t_prof_R = [], []
            for mask, (lo, hi) in _bin_masks(th, th_edges):
                Nb = int(mask.sum())
                if Nb < min_samples:
                    continue
                Xb = X[mask]
                prof = _profiles_from_X(Xb, normalize=normalize_per_event)
                results['theta'].append({'range': (lo, hi), **prof})
                t = f"θ_inc ∈ [{np.degrees(lo):.1f}°, {np.degrees(hi):.1f}°]\nN={Nb:,}"
                t_titles_L.append(t); t_prof_L.append({'mean_layers': prof['mean_layers'], 'sem_layers': prof['sem_layers']})
                t_titles_P.append(t); t_prof_P.append({'mean_phi':    prof['mean_phi'],    'sem_phi':    prof['sem_phi']})
                t_titles_R.append(t); t_prof_R.append({'mean_r':      prof['mean_r'],      'sem_r':      prof['sem_r']})
            if t_prof_L:
                fig = _plot_profile_grid(t_prof_L, t_titles_L, 'Layer index', 'Mean deposited energy', 'L', ncols=ncols, show_ci=show_ci)
                if not normalize_per_event:
                    for ax in np.atleast_1d(fig.axes): 
                        try: ax.set_yscale('log')
                        except Exception: pass
                fig.suptitle('Mean Layer Energy — Stratified by Incident θ', fontsize=12, fontweight='bold'); pdf.savefig(fig); plt.close(fig)
    
                fig = _plot_profile_grid(t_prof_P, t_titles_P, 'ϕ-bin index', 'Mean deposited energy', 'ϕ', ncols=ncols, show_ci=show_ci, x_tick_step=2)
                if not normalize_per_event:
                    for ax in np.atleast_1d(fig.axes): 
                        try: ax.set_yscale('log')
                        except Exception: pass
                fig.suptitle('Mean Angular (ϕ-bin) Energy — Stratified by Incident θ', fontsize=12, fontweight='bold'); pdf.savefig(fig); plt.close(fig)
    
                fig = _plot_profile_grid(t_prof_R, t_titles_R, 'r-bin index', 'Mean deposited energy', 'r', ncols=ncols, show_ci=show_ci, x_tick_step=1)
                if not normalize_per_event:
                    for ax in np.atleast_1d(fig.axes): 
                        try: ax.set_yscale('log')
                        except Exception: pass
                fig.suptitle('Mean Radial (r-bin) Energy — Stratified by Incident θ', fontsize=12, fontweight='bold'); pdf.savefig(fig); plt.close(fig)
    
            # Phi
            p_titles_L, p_prof_L = [], []
            p_titles_P, p_prof_P = [], []
            p_titles_R, p_prof_R = [], []
            for mask, (lo, hi) in _bin_masks(ph, ph_edges):
                Nb = int(mask.sum())
                if Nb < min_samples:
                    continue
                Xb = X[mask]
                prof = _profiles_from_X(Xb, normalize=normalize_per_event)
                results['phi'].append({'range': (lo, hi), **prof})
                t = f"φ_inc ∈ [{np.degrees(lo):.1f}°, {np.degrees(hi):.1f}°]\nN={Nb:,}"
                p_titles_L.append(t); p_prof_L.append({'mean_layers': prof['mean_layers'], 'sem_layers': prof['sem_layers']})
                p_titles_P.append(t); p_prof_P.append({'mean_phi':    prof['mean_phi'],    'sem_phi':    prof['sem_phi']})
                p_titles_R.append(t); p_prof_R.append({'mean_r':      prof['mean_r'],      'sem_r':      prof['sem_r']})
            if p_prof_L:
                fig = _plot_profile_grid(p_prof_L, p_titles_L, 'Layer index', 'Mean deposited energy', 'L', ncols=ncols, show_ci=show_ci)
                if not normalize_per_event:
                    for ax in np.atleast_1d(fig.axes): 
                        try: ax.set_yscale('log')
                        except Exception: pass
                fig.suptitle('Mean Layer Energy — Stratified by Incident φ', fontsize=12, fontweight='bold'); pdf.savefig(fig); plt.close(fig)
    
                fig = _plot_profile_grid(p_prof_P, p_titles_P, 'ϕ-bin index', 'Mean deposited energy', 'ϕ', ncols=ncols, show_ci=show_ci, x_tick_step=2)
                if not normalize_per_event:
                    for ax in np.atleast_1d(fig.axes): 
                        try: ax.set_yscale('log')
                        except Exception: pass
                fig.suptitle('Mean Angular (ϕ-bin) Energy — Stratified by Incident φ', fontsize=12, fontweight='bold'); pdf.savefig(fig); plt.close(fig)
    
                fig = _plot_profile_grid(p_prof_R, p_titles_R, 'r-bin index', 'Mean deposited energy', 'r', ncols=ncols, show_ci=show_ci, x_tick_step=1)
                if not normalize_per_event:
                    for ax in np.atleast_1d(fig.axes): 
                        try: ax.set_yscale('log')
                        except Exception: pass
                fig.suptitle('Mean Radial (r-bin) Energy — Stratified by Incident φ', fontsize=12, fontweight='bold'); pdf.savefig(fig); plt.close(fig)

    print(f"[write] {save_path}")
    return results


def compute_global_correlations(
    shower: np.ndarray,
    save_path: str = "global_correlations.pdf",
    annotate: bool = False,
) -> dict:
    """All-data (non-stratified) correlation matrices for Layers/φ/r; writes a compact PDF."""
    X = to_numpy(shower)
    assert X.ndim == 4 and X.shape[1:] == (45, 16, 9), f"Expected (N,45,16,9), got {X.shape}"

    feat_layers = X.sum(axis=(2, 3))  # (N,45)
    feat_phi    = X.sum(axis=(1, 3))  # (N,16)
    feat_r      = X.sum(axis=(1, 2))  # (N,9)

    C_layers = corrcoef_safe(feat_layers)
    C_phi    = corrcoef_safe(feat_phi)
    C_r      = corrcoef_safe(feat_r)

    with PdfPages(save_path) as pdf:
        fig = plot_grid([C_layers], [f"Global Layer–Layer Correlation  (N={X.shape[0]:,})"],
                        "Layer j", "Layer i", "L", annotate, 1)
        fig.suptitle("All-Data Correlations — Layers", fontsize=12, fontweight="bold")
        pdf.savefig(fig); plt.close(fig)

        fig = plot_grid([C_phi], ["Global ϕ-bin Correlation"], "ϕ bin j", "ϕ bin i", "ϕ", annotate, 1)
        fig.suptitle("All-Data Correlations — Angular (ϕ) Bins", fontsize=12, fontweight="bold")
        pdf.savefig(fig); plt.close(fig)

        fig = plot_grid([C_r], ["Global r-bin Correlation"], "r bin j", "r bin i", "r", annotate, 1)
        fig.suptitle("All-Data Correlations — Radial (r) Bins", fontsize=12, fontweight="bold")
        pdf.savefig(fig); plt.close(fig)

    print(f"[write] {save_path}")
    return {"C_layers": C_layers, "C_phi": C_phi, "C_r": C_r}


def analyze_layer_shifts(
    U_or_X: np.ndarray,
    theta_rad: np.ndarray,
    phi_rad: np.ndarray,
    n_theta_bins: int = 10,
    n_phi_bins: int = 4,
    min_samples: int = 50,
    save_path: str = "layer_shifts_analysis.pdf",
    log_y: bool = True,
) -> dict:
    """Layer-wise angular dependence : means/stds and adjacent-layer correlation shifts."""
    U = np.asarray(U_or_X)
    if U.ndim == 4 and U.shape[1:] == (45, 16, 9):
        U = U.sum(axis=(2, 3))  # -> (N,45)
    assert U.ndim == 2 and U.shape[1] == 45, f"Expected (N,45), got {U.shape}"

    th = np.asarray(theta_rad); ph = np.asarray(phi_rad)

    th_edges = np.linspace(th.min(), th.max(), n_theta_bins + 1)
    ph_edges = np.linspace(ph.min(), ph.max(), n_phi_bins + 1)

    def _iter_bins(values, edges):
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            mask = (values >= lo) & ((values <= hi) if i == len(edges) - 2 else (values < hi))
            yield mask, (lo, hi)

    def _collect(U2d, values, edges, labelfn):
        labels, means, stds, adjs = [], [], [], []
        for mask, (lo, hi) in _iter_bins(values, edges):
            n = int(mask.sum())
            if n < min_samples:
                continue
            Ub = U2d[mask]
            means.append(Ub.mean(axis=0))
            stds.append(Ub.std(axis=0))
            C = np.corrcoef(Ub.T); C = np.nan_to_num(C, nan=0.0); np.fill_diagonal(C, 1.0)
            adjs.append(np.diag(C, k=1))
            labels.append(labelfn(lo, hi))
        return np.array(labels), np.array(means), np.array(stds), np.array(adjs)

    th_labels, th_means, th_stds, th_adj = _collect(U, th, th_edges, lambda lo, hi: f"{np.degrees(lo):.1f}°-{np.degrees(hi):.1f}°")
    ph_labels, ph_means, ph_stds, ph_adj = _collect(U, ph, ph_edges, lambda lo, hi: f"{np.degrees(lo):.1f}°-{np.degrees(hi):.1f}°")

    def _shift(arr): return None if arr.size == 0 else (arr.max(axis=0) - arr.min(axis=0))
    mean_shift_theta = _shift(th_means); std_shift_theta = _shift(th_stds); corr_shift_theta = _shift(th_adj)
    mean_shift_phi   = _shift(ph_means); std_shift_phi   = _shift(ph_stds); corr_shift_phi   = _shift(ph_adj)

    with PdfPages(save_path) as pdf:
        # Page 1
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("Angular Dependence of Layer Energies (Layer-wise)", fontsize=14, fontweight="bold")

        for mean, label in zip(th_means, th_labels):
            axes[0, 0].plot(mean, label=label, linewidth=2)
        axes[0, 0].set_xlabel("Layer"); axes[0, 0].set_ylabel("Mean Energy (MeV)")
        if log_y: axes[0, 0].set_yscale("log")
        axes[0, 0].set_title("Mean Profile vs θ"); axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True, alpha=0.3)

        for mean, label in zip(ph_means, ph_labels):
            axes[0, 1].plot(mean, label=label, linewidth=2)
        axes[0, 1].set_xlabel("Layer"); axes[0, 1].set_ylabel("Mean Energy (MeV)")
        if log_y: axes[0, 1].set_yscale("log")
        axes[0, 1].set_title("Mean Profile vs φ"); axes[0, 1].legend(fontsize=8); axes[0, 1].grid(True, alpha=0.3)

        if mean_shift_theta is not None and mean_shift_phi is not None:
            axes[1, 0].plot(mean_shift_theta, "o-", label="θ", linewidth=2)
            axes[1, 0].plot(mean_shift_phi,   "s-", label="φ", linewidth=2)
        axes[1, 0].set_xlabel("Layer"); axes[1, 0].set_ylabel("Max - Min Mean (MeV)")
        axes[1, 0].set_title("Mean Energy Shifts"); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

        if std_shift_theta is not None and std_shift_phi is not None:
            axes[1, 1].plot(std_shift_theta, "o-", label="θ", linewidth=2)
            axes[1, 1].plot(std_shift_phi,   "s-", label="φ", linewidth=2)
        axes[1, 1].set_xlabel("Layer"); axes[1, 1].set_ylabel("Max - Min Std (MeV)")
        axes[1, 1].set_title("Std Shifts"); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 2
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("Angular Dependence of Adjacent Layer Correlations", fontsize=14, fontweight="bold")

        for adj, label in zip(th_adj, th_labels):
            axes[0, 0].plot(adj, linewidth=2, label=label)
        axes[0, 0].set_xlabel("Layer Pair"); axes[0, 0].set_ylabel("Adjacent Correlation")
        axes[0, 0].set_title("Adjacent Correlations vs θ"); axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True, alpha=0.3)

        for adj, label in zip(ph_adj, ph_labels):
            axes[0, 1].plot(adj, linewidth=2, label=label)
        axes[0, 1].set_xlabel("Layer Pair"); axes[0, 1].set_ylabel("Adjacent Correlation")
        axes[0, 1].set_title("Adjacent Correlations vs φ"); axes[0, 1].legend(fontsize=8); axes[0, 1].grid(True, alpha=0.3)

        if corr_shift_theta is not None and corr_shift_phi is not None:
            axes[1, 0].plot(corr_shift_theta, "o-", label="θ", linewidth=2)
            axes[1, 0].plot(corr_shift_phi,   "s-", label="φ", linewidth=2)
        axes[1, 0].set_xlabel("Layer Pair"); axes[1, 0].set_ylabel("Max - Min Correlation")
        axes[1, 0].set_title("Correlation Shifts"); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].axis("off")
        plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    print(f"[write] {save_path}")
    return {
        "theta": {"labels": th_labels, "means": th_means, "stds": th_stds, "adj": th_adj},
        "phi":   {"labels": ph_labels, "means": ph_means, "stds": ph_stds, "adj": ph_adj},
        "shifts": {
            "mean_theta": mean_shift_theta, "std_theta": std_shift_theta, "adj_theta": corr_shift_theta,
            "mean_phi":   mean_shift_phi,   "std_phi":   std_shift_phi,   "adj_phi":   corr_shift_phi,
        }
    }


def analyze_phi_r_shifts(
    shower: np.ndarray,
    theta_rad: np.ndarray,
    phi_rad: np.ndarray,
    n_theta_bins: int = 10,
    n_phi_bins: int = 4,
    min_samples: int = 50,
    save_path_phi: str = 'angular_bin_shifts_analysis.pdf',
    save_path_r:   str = 'radial_bin_shifts_analysis.pdf',
    normalize_per_event: bool = False
) -> Dict[str, dict]:
    """'Layer-like' shift analysis for ϕ-bins (16) and r-bins (9)."""
    X = np.asarray(shower); th = np.asarray(theta_rad); ph = np.asarray(phi_rad)
    assert X.ndim == 4 and X.shape[1:] == (45, 16, 9), f"Expected (N,45,16,9), got {X.shape}"

    if normalize_per_event:
        totals = X.sum(axis=(1, 2, 3), keepdims=True) + 1e-12
        X = X / totals

    U_phi = X.sum(axis=(1, 3))  # (N,16)
    U_r   = X.sum(axis=(1, 2))  # (N,9)

    def _edges(values, n_bins): return np.linspace(values.min(), values.max(), n_bins + 1)

    def _iter(values, edges):
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            mask = (values >= lo) & ((values <= hi) if i == len(edges) - 2 else (values < hi))
            yield mask, (lo, hi)

    def _profile_stats(U, theta, phi, n_theta_bins, n_phi_bins, min_samples):
        out = {}
        # theta
        th_edges = _edges(theta, n_theta_bins)
        th_labels, th_means, th_stds, th_adj = [], [], [], []
        for mask, (lo, hi) in _iter(theta, th_edges):
            n = int(mask.sum())
            if n < min_samples: continue
            Ub = U[mask]
            th_means.append(Ub.mean(axis=0))
            th_stds.append(Ub.std(axis=0))
            th_adj.append(_corr_adjacent(Ub))
            th_labels.append(f"{np.degrees(lo):.1f}°-{np.degrees(hi):.1f}°")
        th_means = np.array(th_means); th_stds = np.array(th_stds); th_adj = np.array(th_adj)
        # phi
        ph_edges = _edges(phi, n_phi_bins)
        ph_labels, ph_means, ph_stds, ph_adj = [], [], [], []
        for mask, (lo, hi) in _iter(phi, ph_edges):
            n = int(mask.sum())
            if n < min_samples: continue
            Ub = U[mask]
            ph_means.append(Ub.mean(axis=0))
            ph_stds.append(Ub.std(axis=0))
            ph_adj.append(_corr_adjacent(Ub))
            ph_labels.append(f"{np.degrees(lo):.1f}°-{np.degrees(hi):.1f}°")
        ph_means = np.array(ph_means); ph_stds = np.array(ph_stds); ph_adj = np.array(ph_adj)
        # shifts
        def _sh(arr): return None if arr.size == 0 else (arr.max(axis=0) - arr.min(axis=0))
        shifts = {
            'mean_theta': _sh(th_means), 'std_theta': _sh(th_stds), 'adj_theta': _sh(th_adj),
            'mean_phi':   _sh(ph_means), 'std_phi':   _sh(ph_stds), 'adj_phi':   _sh(ph_adj),
        }
        out['theta'] = {'labels': th_labels, 'means': th_means, 'stds': th_stds, 'adj': th_adj}
        out['phi']   = {'labels': ph_labels, 'means': ph_means, 'stds': ph_stds, 'adj': ph_adj}
        out['shifts']= shifts
        return out

    def _plot_pages(stats, K, thing_name, x_label, save_path, logy=True):
        th = stats['theta']; phs = stats['phi']; sh = stats['shifts']
        with PdfPages(save_path) as pdf:
            # Page 1
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            fig.suptitle(f"Angular Dependence of {thing_name.capitalize()} Energies", fontsize=14, fontweight='bold')

            for mean, lbl in zip(th['means'], th['labels']):
                axes[0, 0].plot(np.arange(K), mean, linewidth=2, label=lbl)
            axes[0, 0].set_xlabel(x_label); axes[0, 0].set_ylabel('Mean Energy (MeV)')
            if logy: axes[0, 0].set_yscale('log')
            axes[0, 0].set_title(f"Mean {thing_name} Profile vs Theta"); axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True, alpha=0.3)

            for mean, lbl in zip(phs['means'], phs['labels']):
                axes[0, 1].plot(np.arange(K), mean, linewidth=2, label=lbl)
            axes[0, 1].set_xlabel(x_label); axes[0, 1].set_ylabel('Mean Energy (MeV)')
            if logy: axes[0, 1].set_yscale('log')
            axes[0, 1].set_title(f"Mean {thing_name} Profile vs Phi"); axes[0, 1].legend(fontsize=8); axes[0, 1].grid(True, alpha=0.3)

            if sh['mean_theta'] is not None and sh['mean_phi'] is not None:
                axes[1, 0].plot(sh['mean_theta'], 'o-', label='Theta', linewidth=2)
                axes[1, 0].plot(sh['mean_phi'],  's-', label='Phi',   linewidth=2)
            axes[1, 0].set_xlabel(x_label); axes[1, 0].set_ylabel('Max - Min Mean (MeV)')
            axes[1, 0].set_title('Mean Energy Shifts Across θ/φ Bins'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

            if sh['std_theta'] is not None and sh['std_phi'] is not None:
                axes[1, 1].plot(sh['std_theta'], 'o-', label='Theta', linewidth=2)
                axes[1, 1].plot(sh['std_phi'],  's-', label='Phi',   linewidth=2)
            axes[1, 1].set_xlabel(x_label); axes[1, 1].set_ylabel('Max - Min Std (MeV)')
            axes[1, 1].set_title('Std Shifts Across θ/φ Bins'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout(); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

            # Page 2
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            fig.suptitle(f"Angular Dependence of Adjacent {thing_name.capitalize()} Correlations", fontsize=14, fontweight='bold')

            for adj, lbl in zip(th['adj'], th['labels']):
                axes[0, 0].plot(np.arange(K-1), adj, linewidth=2, label=lbl)
            axes[0, 0].set_xlabel(f"{thing_name.capitalize()} Pair"); axes[0, 0].set_ylabel('Adjacent Correlation')
            axes[0, 0].set_title(f"Adjacent {thing_name} Correlations vs Theta"); axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True, alpha=0.3)

            for adj, lbl in zip(phs['adj'], phs['labels']):
                axes[0, 1].plot(np.arange(K-1), adj, linewidth=2, label=lbl)
            axes[0, 1].set_xlabel(f"{thing_name.capitalize()} Pair"); axes[0, 1].set_ylabel('Adjacent Correlation')
            axes[0, 1].set_title(f"Adjacent {thing_name} Correlations vs Phi"); axes[0, 1].legend(fontsize=8); axes[0, 1].grid(True, alpha=0.3)

            if sh['adj_theta'] is not None and sh['adj_phi'] is not None:
                axes[1, 0].plot(sh['adj_theta'], 'o-', label='Theta', linewidth=2)
                axes[1, 0].plot(sh['adj_phi'],  's-', label='Phi',   linewidth=2)
            axes[1, 0].set_xlabel(f"{thing_name.capitalize()} Pair"); axes[1, 0].set_ylabel('Max - Min Correlation')
            axes[1, 0].set_title('Correlation Shifts Across θ/φ Bins'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

            axes[1, 1].axis('off')
            plt.tight_layout(); pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

    stats_phi = _profile_stats(U_phi, th, ph, n_theta_bins, n_phi_bins, min_samples)
    _plot_pages(stats_phi, K=16, thing_name='ϕ-bin', x_label='ϕ-bin', save_path=save_path_phi, logy=not normalize_per_event)

    stats_r = _profile_stats(U_r, th, ph, n_theta_bins, n_phi_bins, min_samples)
    _plot_pages(stats_r, K=9, thing_name='r-bin', x_label='r-bin', save_path=save_path_r, logy=not normalize_per_event)

    print(f"[write] {save_path_phi}")
    print(f"[write] {save_path_r}")
    return {'phi_bins': stats_phi, 'r_bins': stats_r}


# ============================ CLI ============================

def main():
    p = argparse.ArgumentParser(description="LEMURS correlation & profile analysis")

    p.add_argument('--dataset', choices=['lemurs', 'cc2'], required=True,
                   help='Which dataset: LEMURS has angles; CC2 does not.')
    sub = p.add_subparsers(dest='cmd', required=True)
    
    pc = sub.add_parser('corr', help='Stratified correlation matrices (Layers/ϕ/r) -> PDF')
    pc.add_argument('--data', type=str, required=True, help='Path to .npz or .h5/.hdf5 with X/E/theta/phi')
    pc.add_argument('--n-energy-bins', type=int, default=10)
    pc.add_argument('--n-theta-bins', type=int, default=10)
    pc.add_argument('--n-phi-bins', type=int, default=10)
    pc.add_argument('--min-samples', type=int, default=50)
    pc.add_argument('--save', type=str, default='stratified_correlations.pdf')
    pc.add_argument('--annotate', action='store_true')
    pc.add_argument('--ncols', type=int, default=3)
    pc.add_argument('--normalize-per-event', action='store_true',
                help='Divide each event by its total deposited energy (shape-only correlations)')
    pc.add_argument('--log1p', action='store_true',
                    help='Compute correlations on log1p-transformed features')
    pc.add_argument('--spearman', action='store_true',
                    help='Use Spearman (Pearson on ranks). If used with --log1p, ranks are of log1p(features).')


    pm = sub.add_parser('means', help='Stratified mean profiles (Layers/ϕ/r) -> PDF')
    pm.add_argument('--data', type=str, required=True)
    pm.add_argument('--n-energy-bins', type=int, default=10)
    pm.add_argument('--n-theta-bins', type=int, default=10)
    pm.add_argument('--n-phi-bins', type=int, default=10)
    pm.add_argument('--min-samples', type=int, default=50)
    pm.add_argument('--save', type=str, default='mean_profiles_stratified.pdf')
    pm.add_argument('--percentile-bins', action='store_true', help='Use percentile bins for energy')
    pm.add_argument('--normalize-per-event', action='store_true', help='Normalize each event by its total energy')
    pm.add_argument('--ncols', type=int, default=3)
    pm.add_argument('--no-ci', action='store_true', help='Disable SEM shading')

    pr = sub.add_parser('phi-r', help='ϕ/r-bin shift analyses (mean/std + adjacent-corr) -> PDFs')
    pr.add_argument('--data', type=str, required=True)
    pr.add_argument('--n-theta-bins', type=int, default=10)
    pr.add_argument('--n-phi-bins', type=int, default=4)
    pr.add_argument('--min-samples', type=int, default=50)
    pr.add_argument('--save-phi', type=str, default='angular_bin_shifts_analysis.pdf')
    pr.add_argument('--save-r',   type=str, default='radial_bin_shifts_analysis.pdf')
    pr.add_argument('--normalize-per-event', action='store_true')

    pg = sub.add_parser('global-corr', help='All-data correlations for Layers/ϕ/r (no strat) -> PDF')
    pg.add_argument('--data', type=str, required=True)
    pg.add_argument('--save', type=str, default='global_correlations.pdf')
    pg.add_argument('--annotate', action='store_true')

    pls = sub.add_parser('layer-shifts', help='Layer-wise mean/std + adjacent-corr shifts (θ/φ) -> PDF')
    pls.add_argument('--data', type=str, required=True)
    pls.add_argument('--n-theta-bins', type=int, default=10)
    pls.add_argument('--n-phi-bins', type=int, default=4)
    pls.add_argument('--min-samples', type=int, default=50)
    pls.add_argument('--save', type=str, default='layer_shifts_analysis.pdf')
    pls.add_argument('--no-logy', action='store_true', help='Disable log scale on mean energy plots')

    args = p.parse_args()
    path = Path(args.data)
    X, E, th, ph = _load_any(path, args.dataset)
    if args.dataset == 'cc2': 
        bin_file_path='/project/biocomplexity/fa7sa/calo_dreamer/src/challenge_files/binning_dataset_2.xml'
        X = reshape_flat_using_xml(X, bin_file_path, order="LPR")
        
       

        
    if args.cmd == 'corr':
        #X, E, th, ph = _load_any(path)
        compute_stratified_correlations(
            shower=X, E_inc=E, theta=th, phi=ph,
            n_energy_bins=args.n_energy_bins, n_theta_bins=args.n_theta_bins, n_phi_bins=args.n_phi_bins,
            min_samples=args.min_samples, save_path=args.save, annotate=args.annotate, ncols=args.ncols,
            normalize_per_event=args.normalize_per_event, use_log1p=args.log1p, use_spearman=args.spearman
        )

    elif args.cmd == 'means':
        #X, E, th, ph = _load_any(path)
        compute_and_plot_mean_profiles(
            shower=X, E_inc=E, theta=th, phi=ph,
            n_energy_bins=args.n_energy_bins, n_theta_bins=args.n_theta_bins, n_phi_bins=args.n_phi_bins,
            min_samples=args.min_samples, save_path=args.save, use_percentile_bins=args.percentile_bins,
            normalize_per_event=args.normalize_per_event, ncols=args.ncols, show_ci=not args.no_ci
        )

    elif args.cmd == 'phi-r':
        #X, E, th, ph = _load_any(path)
        analyze_phi_r_shifts(
            shower=X, theta_rad=th, phi_rad=ph,
            n_theta_bins=args.n_theta_bins, n_phi_bins=args.n_phi_bins,
            min_samples=args.min_samples, save_path_phi=args.save_phi, save_path_r=args.save_r,
            normalize_per_event=args.normalize_per_event
        )

    elif args.cmd == 'global-corr':
        #X, E, th, ph = _load_any(path)
        compute_global_correlations(shower=X, save_path=args.save, annotate=args.annotate)

    elif args.cmd == 'layer-shifts':
        # If it's an .npz and contains U_true, prefer it; otherwise derive layers from X
        if path.suffix.lower() == '.npz':
            data = np.load(path)
            if 'U_true' in data:
                U = data['U_true']
                th = data.get('theta') or data.get('incident_theta')
                ph = data.get('phi')   or data.get('incident_phi')
                if th is None or ph is None:
                    _, _, th, ph = _load_any(path)
                analyze_layer_shifts(
                    U_or_X=U, theta_rad=th, phi_rad=ph,
                    n_theta_bins=args.n_theta_bins, n_phi_bins=args.n_phi_bins,
                    min_samples=args.min_samples, save_path=args.save, log_y=not args.no_logy
                )
            else:
                X, _, th, ph = _load_any(path)
                analyze_layer_shifts(
                    U_or_X=X, theta_rad=th, phi_rad=ph,
                    n_theta_bins=args.n_theta_bins, n_phi_bins=args.n_phi_bins,
                    min_samples=args.min_samples, save_path=args.save, log_y=not args.no_logy
                )
        else:
            X, _, th, ph = _load_any(path)
            analyze_layer_shifts(
                U_or_X=X, theta_rad=th, phi_rad=ph,
                n_theta_bins=args.n_theta_bins, n_phi_bins=args.n_phi_bins,
                min_samples=args.min_samples, save_path=args.save, log_y=not args.no_logy
            )

    else:
        p.error('unknown subcommand')


if __name__ == '__main__':
    main()
