import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.io import loadmat


# ============================================================
# USER SETTINGS
# ============================================================
FC = 77e9
C = 299792458.0
LAMBDA = C / FC
D_MIN = LAMBDA / 2.0

M = 16
TARGET_AOA_DEG = np.arange(-80, 81, 20, dtype=float)

# ------------------------------------------------------------
# Beam-dependent PSL sector
# ------------------------------------------------------------
# True  -> for a beam steered to phi0, PSL is evaluated only in
#          [phi0 - PSL_SECTOR_HALF_WIDTH_DEG, phi0 + PSL_SECTOR_HALF_WIDTH_DEG]
# False -> sidelobes are evaluated over the whole 360 deg
USE_BEAM_CENTERED_PSL_SECTOR = False
PSL_SECTOR_HALF_WIDTH_DEG = 90.0

# ------------------------------------------------------------
# Mainbeam evaluation mode
# ------------------------------------------------------------
# False -> use automatic null-to-null mainbeam bounds
# True  -> use a fixed mainbeam width around each steering angle
USE_FIXED_MAINBEAM_WIDTH = False

# Full mainbeam width in degrees
FIXED_MAINBEAM_WIDTH_DEG = 21.0

# ------------------------------------------------------------
# Alternating optimization settings
# ------------------------------------------------------------
MAX_AO_ITER = 20
AO_STOP_TOL_DB = 1e-4

# ------------------------------------------------------------
# Initial alpha
# ------------------------------------------------------------
ALPHA_MIN_DEG = 1.0
ALPHA_MAX_DEG = 359.0
ALPHA0_DEG = 180.0

# ------------------------------------------------------------
# Alpha-step
# ------------------------------------------------------------
MAX_ITER_ALPHA = 25
N_SAMPLES_ALPHA = 9
STOP_TOL_ALPHA_DEG = 1e-5
MAX_BACKTRACK_ALPHA = 18

MU_ALPHA0_DEG = 12.0
MU_ALPHA_MIN_DEG = 0.02
MU_ALPHA_MAX_DEG = 35.0
TRUST_SHRINK_ALPHA = 0.5
TRUST_GROW_ALPHA = 1.25

# ------------------------------------------------------------
# Beta-step
# ------------------------------------------------------------
MAX_ITER_BETA = 25
N_SAMPLES_BETA = 9
MU_BETA_FACTOR = 0.10
MU_BETA_MIN_FACTOR = 0.01
MU_BETA_MAX_FACTOR = 0.25
TRUST_SHRINK_BETA = 0.5
TRUST_GROW_BETA = 1.2

MAX_BACKTRACK_BETA = 18
STOP_COUNT_LIMIT_BETA = 6
STEP_NORM_TOL_BETA = 1e-7

# ------------------------------------------------------------
# Coordinate refining after beta-block
# ------------------------------------------------------------
ENABLE_BETA_POLISH = True
MAX_POLISH_SWEEPS = 3
N_POLISH_SAMPLES = 11

# Parallel workers
N_JOBS = -1

# ------------------------------------------------------------
# Pattern file
# ------------------------------------------------------------
PATTERN_MAT_FILE = "xMSPatchPattern10.mat"
PATTERN_VAR_NAME = "xMSPatchPattern10"




# ============================================================
# LOAD ELEMENT PATTERN
# ============================================================
mat = loadmat(PATTERN_MAT_FILE)
pattern_dB = np.asarray(mat[PATTERN_VAR_NAME]).reshape(-1)
elemE = 10.0 ** (pattern_dB / 20.0)

# ------------------------------------------------------------
# FULL 360-DEGREE GRID
# ------------------------------------------------------------
phi_plot_deg = np.arange(-180.0, 180.0, 1.0)
phi_full_deg = np.mod(phi_plot_deg, 360.0)
phi_deg = phi_full_deg.copy()


# ============================================================
# HELPERS
# ============================================================
def rotate_pattern(E: np.ndarray, angle_deg: float) -> np.ndarray:
    shift = int(np.round(angle_deg)) % 360
    return np.roll(E, shift)


def alpha_to_R(alpha_deg: float, d: float, M: int) -> float:
    alpha_rad = np.deg2rad(alpha_deg)
    denom = 2.0 * np.sin(alpha_rad / (2.0 * (M - 1)))
    denom = max(denom, 1e-12)
    return d / denom


def build_geometry(beta: np.ndarray, R: float):
    x_elem = -R + R * np.cos(beta)
    y_elem = R * np.sin(beta)
    z_elem = np.zeros_like(beta)
    phi_elem_deg = np.rad2deg(beta)
    return x_elem, y_elem, z_elem, phi_elem_deg


def enforce_centering(beta: np.ndarray) -> np.ndarray:
    beta = np.asarray(beta, dtype=float).copy()
    shift = 0.5 * (beta[0] + beta[-1])
    return beta - shift


def compute_delta_beta_min(R: float, d_min: float) -> float:
    arg = d_min / (2.0 * R)
    if arg > 1.0:
        raise ValueError("Invalid geometry: d_min/(2R) > 1.")
    return 2.0 * np.arcsin(arg)


def project_beta_feasible(beta: np.ndarray, R: float, d_min: float) -> np.ndarray:
    beta = np.sort(np.asarray(beta, dtype=float).copy())
    delta_beta_min = compute_delta_beta_min(R, d_min)

    for k in range(1, beta.size):
        beta[k] = max(beta[k], beta[k - 1] + delta_beta_min)

    beta = enforce_centering(beta)
    return beta


def get_shape_eta(beta: np.ndarray) -> np.ndarray:
    beta = np.asarray(beta, dtype=float).copy()
    span = beta[-1] - beta[0]
    if span <= 1e-14:
        raise ValueError("beta span is too small.")
    return beta / span


def beta_from_eta_alpha(eta: np.ndarray, alpha_deg: float, R: float, d_min: float) -> np.ndarray:
    alpha_rad = np.deg2rad(alpha_deg)
    beta = eta * alpha_rad
    beta = project_beta_feasible(beta, R, d_min)
    return beta


def compute_af_on_grid_exact_multibeam(
    phi_deg_grid: np.ndarray,
    beta: np.ndarray,
    R: float,
    c: float,
    wavelength: float,
    elemE: np.ndarray,
    target_aoa_deg: np.ndarray,
) -> np.ndarray:
    phi_deg_grid = np.asarray(phi_deg_grid).reshape(-1)
    target_aoa_deg = np.asarray(target_aoa_deg).reshape(-1)

    nphi = phi_deg_grid.size
    nbeams = target_aoa_deg.size
    m = beta.size

    x_elem, y_elem, z_elem, phi_elem_deg = build_geometry(beta, R)

    idx = np.mod(np.round(phi_deg_grid).astype(int), 360)
    elem_pattern_rot = np.zeros((nphi, m), dtype=float)

    for k in range(m):
        Eshift = rotate_pattern(elemE, phi_elem_deg[k])
        elem_pattern_rot[:, k] = Eshift[idx]

    W = np.zeros((m, nbeams), dtype=np.complex128)
    for ib, aoa in enumerate(target_aoa_deg):
        r_tgt = np.array([np.cos(np.deg2rad(aoa)), np.sin(np.deg2rad(aoa)), 0.0])
        tau_tgt = -(x_elem * r_tgt[0] + y_elem * r_tgt[1] + z_elem * r_tgt[2]) / c
        W[:, ib] = np.exp(1j * (2.0 * np.pi / wavelength) * c * tau_tgt)

    r_obs_x = np.cos(np.deg2rad(phi_deg_grid))[:, None]
    r_obs_y = np.sin(np.deg2rad(phi_deg_grid))[:, None]
    tau_obs = -(r_obs_x * x_elem[None, :] + r_obs_y * y_elem[None, :]) / c

    a = elem_pattern_rot * np.exp(1j * (2.0 * np.pi / wavelength) * c * tau_obs)
    AF = np.conjugate(a) @ W
    return AF


def find_mainbeam_bounds_null_to_null(
    P_dB_beam: np.ndarray,
    phi_plot_deg: np.ndarray,
    target_aoa_deg: float,
    peak_search_half_window: int = 8,
):
    P_dB_beam = np.asarray(P_dB_beam).reshape(-1)
    phi_plot_deg = np.asarray(phi_plot_deg).reshape(-1)

    visible_mask = (phi_plot_deg >= -90.0) & (phi_plot_deg <= 90.0)
    vis_idx = np.where(visible_mask)[0]

    if vis_idx.size == 0:
        raise ValueError("No visible-sector samples found in phi_plot_deg.")

    idx0 = int(np.argmin(np.abs(phi_plot_deg - target_aoa_deg)))
    idx0 = min(max(idx0, vis_idx[0]), vis_idx[-1])

    i1 = max(vis_idx[0], idx0 - peak_search_half_window)
    i2 = min(vis_idx[-1] + 1, idx0 + peak_search_half_window + 1)
    i_peak = i1 + int(np.argmax(P_dB_beam[i1:i2]))

    i_left = i_peak
    while i_left > vis_idx[0]:
        if P_dB_beam[i_left - 1] <= P_dB_beam[i_left]:
            i_left -= 1
        else:
            break

    i_right = i_peak
    while i_right < vis_idx[-1]:
        if P_dB_beam[i_right + 1] <= P_dB_beam[i_right]:
            i_right += 1
        else:
            break

    return i_left, i_peak, i_right


def find_mainbeam_bounds_fixed_width(
    phi_plot_deg: np.ndarray,
    target_aoa_deg: float,
    mainbeam_width_deg: float,
):
    phi_plot_deg = np.asarray(phi_plot_deg).reshape(-1)

    half_bw = 0.5 * mainbeam_width_deg
    left_deg = target_aoa_deg - half_bw
    right_deg = target_aoa_deg + half_bw

    i_peak = int(np.argmin(np.abs(phi_plot_deg - target_aoa_deg)))
    i_left = int(np.argmin(np.abs(phi_plot_deg - left_deg)))
    i_right = int(np.argmin(np.abs(phi_plot_deg - right_deg)))

    if i_left > i_right:
        i_left, i_right = i_right, i_left

    return i_left, i_peak, i_right


def build_beam_centered_sector_mask(
    phi_plot_deg: np.ndarray,
    center_deg: float,
    half_width_deg: float,
) -> np.ndarray:
    left_deg = center_deg - half_width_deg
    right_deg = center_deg + half_width_deg
    return (phi_plot_deg >= left_deg) & (phi_plot_deg <= right_deg)


def evaluate_exact_af_and_psl(
    beta: np.ndarray,
    R: float,
    phi_deg_grid: np.ndarray,
    phi_plot_deg: np.ndarray,
    target_aoa_deg: np.ndarray,
):
    AF = compute_af_on_grid_exact_multibeam(
        phi_deg_grid, beta, R, C, LAMBDA, elemE, target_aoa_deg
    )

    P_dB = 20.0 * np.log10(np.abs(AF) + np.finfo(float).eps)

    nphi, nbeams = P_dB.shape
    side_masks = np.zeros((nphi, nbeams), dtype=bool)
    psl_per_beam = np.zeros(nbeams, dtype=float)
    mainbeam_bounds_deg = []

    for ib, aoa in enumerate(target_aoa_deg):
        if USE_FIXED_MAINBEAM_WIDTH:
            i_left, i_peak, i_right = find_mainbeam_bounds_fixed_width(
                phi_plot_deg, aoa, FIXED_MAINBEAM_WIDTH_DEG
            )
        else:
            i_left, i_peak, i_right = find_mainbeam_bounds_null_to_null(
                P_dB[:, ib], phi_plot_deg, aoa
            )

        if USE_BEAM_CENTERED_PSL_SECTOR:
            sector_mask = build_beam_centered_sector_mask(
                phi_plot_deg, aoa, PSL_SECTOR_HALF_WIDTH_DEG
            )

            side_mask = np.zeros(nphi, dtype=bool)
            left_mask = sector_mask & (phi_plot_deg < phi_plot_deg[i_left])
            right_mask = sector_mask & (phi_plot_deg > phi_plot_deg[i_right])
            side_mask[left_mask] = True
            side_mask[right_mask] = True
        else:
            side_mask = np.ones(nphi, dtype=bool)
            side_mask[i_left:i_right + 1] = False

        side_masks[:, ib] = side_mask

        peak_val = np.max(P_dB[:, ib])

        if np.any(side_mask):
            sidelobe_peak = np.max(P_dB[side_mask, ib])
            psl_per_beam[ib] = sidelobe_peak - peak_val
        else:
            psl_per_beam[ib] = -np.inf

        mainbeam_bounds_deg.append((
            float(phi_plot_deg[i_left]),
            float(phi_plot_deg[i_peak]),
            float(phi_plot_deg[i_right]),
            float(phi_plot_deg[i_right] - phi_plot_deg[i_left]),
        ))

    PSL = float(np.max(psl_per_beam))
    return AF, P_dB, PSL, psl_per_beam, side_masks, mainbeam_bounds_deg


def build_sidelobe_stack(AF0: np.ndarray, Jfull: np.ndarray, side_masks: np.ndarray):
    nphi, nbeams = AF0.shape
    A0_list = []
    A_list = []

    scalar_case = (Jfull.ndim == 1)

    for ib in range(nbeams):
        rows_beam = np.arange(ib * nphi, (ib + 1) * nphi)
        mask = side_masks[:, ib]
        A0_list.append(AF0[mask, ib])

        if scalar_case:
            A_list.append(Jfull[rows_beam[mask]])
        else:
            A_list.append(Jfull[rows_beam[mask], :])

    A0_stack = np.concatenate(A0_list, axis=0)

    if scalar_case:
        A_stack = np.concatenate(A_list, axis=0)
    else:
        A_stack = np.vstack(A_list)

    return A0_stack, A_stack


def recover_R_and_arcspan_from_xy(x_elem: np.ndarray, y_elem: np.ndarray):
    x_elem = np.asarray(x_elem).reshape(-1)
    y_elem = np.asarray(y_elem).reshape(-1)

    idx_valid = np.abs(x_elem) > 1e-12
    R_each = -(x_elem[idx_valid] ** 2 + y_elem[idx_valid] ** 2) / (2.0 * x_elem[idx_valid])
    R_est = np.mean(R_each)

    beta_est = np.arctan2(y_elem, x_elem + R_est)
    beta_est = np.sort(beta_est)

    arc_span_rad = beta_est[-1] - beta_est[0]
    arc_span_deg = np.rad2deg(arc_span_rad)
    return R_est, arc_span_rad, arc_span_deg, beta_est


# ============================================================
# SYMMETRIC LS JACOBIANS
# ============================================================
def symmetric_ls_scalar_jacobian(
    beta_ref: np.ndarray,
    R_ref: float,
    alpha_current_deg: float,
    eta: np.ndarray,
    mu_alpha_deg: float,
):
    AF0 = compute_af_on_grid_exact_multibeam(
        phi_full_deg, beta_ref, R_ref, C, LAMBDA, elemE, TARGET_AOA_DEG
    )
    AF0_vec = AF0.reshape(-1, order="F")

    t_samples = np.linspace(-mu_alpha_deg, mu_alpha_deg, N_SAMPLES_ALPHA)
    t_samples = t_samples[np.abs(t_samples) >= 1e-14]

    sigma = max(0.45 * mu_alpha_deg, 1e-12)
    w_ls = np.exp(-(t_samples ** 2) / (2.0 * sigma ** 2))

    denom = np.sum(w_ls * (t_samples ** 2))
    if denom < 1e-20:
        return AF0, np.zeros_like(AF0_vec)

    num = np.zeros(AF0_vec.size, dtype=np.complex128)

    for q, t in enumerate(t_samples):
        alpha_tmp = float(np.clip(alpha_current_deg + t, ALPHA_MIN_DEG, ALPHA_MAX_DEG))
        R_tmp = alpha_to_R(alpha_tmp, D_MIN, M)
        beta_tmp = beta_from_eta_alpha(eta, alpha_tmp, R_tmp, D_MIN)

        AFtmp = compute_af_on_grid_exact_multibeam(
            phi_full_deg, beta_tmp, R_tmp, C, LAMBDA, elemE, TARGET_AOA_DEG
        ).reshape(-1, order="F")

        num += w_ls[q] * t * (AFtmp - AF0_vec)

    Jalpha = num / denom
    return AF0, Jalpha

def symmetric_ls_beta_column_worker(
    k: int,
    beta_bar: np.ndarray,
    AF0_vec: np.ndarray,
    mu_beta_local: float,
    R: float,
):
    t_samples = np.linspace(-mu_beta_local, mu_beta_local, N_SAMPLES_BETA)
    t_samples = t_samples[np.abs(t_samples) >= 1e-14]

    sigma = max(0.45 * mu_beta_local, 1e-16)
    w_ls = np.exp(-(t_samples ** 2) / (2.0 * sigma ** 2))

    denom = np.sum(w_ls * (t_samples ** 2))
    if denom < 1e-20:
        return np.zeros(AF0_vec.size, dtype=np.complex128)

    num = np.zeros(AF0_vec.size, dtype=np.complex128)

    for iq, t in enumerate(t_samples):
        beta_tmp = beta_bar.copy()
        beta_tmp[k] += t

        AF_tmp = compute_af_on_grid_exact_multibeam(
            phi_full_deg, beta_tmp, R, C, LAMBDA, elemE, TARGET_AOA_DEG
        ).reshape(-1, order="F")

        num += w_ls[iq] * t * (AF_tmp - AF0_vec)

    return num / denom


# ============================================================
# REFINING STAGE
# ============================================================
def compute_single_index_bounds(beta: np.ndarray, R: float, k: int, mu_beta: float):
    beta = np.asarray(beta, dtype=float)
    delta_min = compute_delta_beta_min(R, D_MIN)

    low = beta[k] - mu_beta
    high = beta[k] + mu_beta

    if k > 0:
        low = max(low, beta[k - 1] + delta_min)
    if k < beta.size - 1:
        high = min(high, beta[k + 1] - delta_min)

    return low, high


def polish_beta_coordinates(beta_init: np.ndarray, R: float):
    beta_best = beta_init.copy()
    _, _, psl_best, _, _, _ = evaluate_exact_af_and_psl(
        beta_best, R, phi_full_deg, phi_plot_deg, TARGET_AOA_DEG
    )

    mu_beta = MU_BETA_FACTOR * LAMBDA / (4.0 * np.pi * R)
    improved_global = False

    for _ in range(MAX_POLISH_SWEEPS):
        improved_sweep = False

        for k in range(M):
            low, high = compute_single_index_bounds(beta_best, R, k, mu_beta)
            if high <= low:
                continue

            center = beta_best[k]
            samples = np.linspace(low, high, N_POLISH_SAMPLES)

            order = np.argsort(np.abs(samples - center))
            samples = samples[order]

            local_best_beta = beta_best.copy()
            local_best_psl = psl_best

            for val in samples:
                beta_try = beta_best.copy()
                beta_try[k] = val
                beta_try = project_beta_feasible(beta_try, R, D_MIN)

                _, _, psl_try, _, _, _ = evaluate_exact_af_and_psl(
                    beta_try, R, phi_full_deg, phi_plot_deg, TARGET_AOA_DEG
                )

                if psl_try < local_best_psl - 1e-10:
                    local_best_psl = psl_try
                    local_best_beta = beta_try.copy()

            if local_best_psl < psl_best - 1e-10:
                beta_best = local_best_beta.copy()
                psl_best = local_best_psl
                improved_sweep = True
                improved_global = True

        if not improved_sweep:
            break

    return beta_best, improved_global


# ============================================================
# ALPHA BLOCK
# ============================================================
def optimize_alpha_block(beta_init: np.ndarray, R_init: float, mu_alpha_deg: float):
    beta_current = np.asarray(beta_init, dtype=float).copy()
    R_current = float(R_init)

    _, _, PSL_current, _, _, _ = evaluate_exact_af_and_psl(
        beta_current, R_current, phi_full_deg, phi_plot_deg, TARGET_AOA_DEG
    )

    best_beta = beta_current.copy()
    best_R = R_current
    best_PSL = PSL_current

    alpha_current_deg = np.rad2deg(beta_current[-1] - beta_current[0])
    alpha_current_deg = float(np.clip(alpha_current_deg, ALPHA_MIN_DEG, ALPHA_MAX_DEG))
    best_alpha_deg = alpha_current_deg

    improved_any = False
    hist = {"psl": [], "alpha": [], "R": [], "mu": []}

    for _ in range(MAX_ITER_ALPHA):
        eta = get_shape_eta(beta_current)

        AF0, Jalpha = symmetric_ls_scalar_jacobian(
            beta_current, R_current, alpha_current_deg, eta, mu_alpha_deg
        )

        _, _, _, _, side_masks, _ = evaluate_exact_af_and_psl(
            beta_current, R_current, phi_full_deg, phi_plot_deg, TARGET_AOA_DEG
        )

        A0_stack, J_stack = build_sidelobe_stack(AF0, Jalpha, side_masks)

        DeltaAlpha = cp.Variable()
        rho = cp.Variable(nonneg=True)

        constraints = [
            alpha_current_deg + DeltaAlpha >= ALPHA_MIN_DEG,
            alpha_current_deg + DeltaAlpha <= ALPHA_MAX_DEG,
            DeltaAlpha >= -mu_alpha_deg,
            DeltaAlpha <= +mu_alpha_deg,
        ]

        for k in range(A0_stack.size):
            constraints.append(cp.abs(A0_stack[k] + J_stack[k] * DeltaAlpha) <= rho)

        problem = cp.Problem(cp.Minimize(rho), constraints)

        solved = False
        for solver in [cp.CLARABEL, cp.SCS, cp.ECOS]:
            try:
                problem.solve(solver=solver, verbose=False)
                if DeltaAlpha.value is not None:
                    solved = True
                    break
            except Exception:
                pass

        if not solved or DeltaAlpha.value is None:
            break

        delta_alpha_val = float(DeltaAlpha.value)
        if not np.isfinite(delta_alpha_val):
            break

        step_scale = 1.0
        accepted = False

        for _ in range(MAX_BACKTRACK_ALPHA):
            alpha_trial = float(
                np.clip(alpha_current_deg + step_scale * delta_alpha_val,
                        ALPHA_MIN_DEG, ALPHA_MAX_DEG)
            )
            R_trial = alpha_to_R(alpha_trial, D_MIN, M)
            beta_trial = beta_from_eta_alpha(eta, alpha_trial, R_trial, D_MIN)

            _, _, PSL_trial, _, _, _ = evaluate_exact_af_and_psl(
                beta_trial, R_trial, phi_full_deg, phi_plot_deg, TARGET_AOA_DEG
            )

            if PSL_trial <= best_PSL - 1e-10:
                accepted = True
                break

            step_scale *= 0.5

        if accepted:
            beta_current = beta_trial.copy()
            R_current = float(R_trial)
            alpha_current_deg = alpha_trial

            best_beta = beta_current.copy()
            best_R = R_current
            best_alpha_deg = alpha_current_deg
            best_PSL = PSL_trial
            improved_any = True

            hist["psl"].append(best_PSL)
            hist["alpha"].append(best_alpha_deg)
            hist["R"].append(best_R)
            hist["mu"].append(mu_alpha_deg)

            mu_alpha_deg = min(mu_alpha_deg * TRUST_GROW_ALPHA, MU_ALPHA_MAX_DEG)

            if abs(step_scale * delta_alpha_val) < STOP_TOL_ALPHA_DEG:
                break
        else:
            mu_alpha_deg = max(mu_alpha_deg * TRUST_SHRINK_ALPHA, MU_ALPHA_MIN_DEG)
            if mu_alpha_deg <= MU_ALPHA_MIN_DEG + 1e-12:
                break

    if not improved_any:
        return beta_init.copy(), float(R_init), alpha_current_deg, mu_alpha_deg, False, hist

    return best_beta, best_R, best_alpha_deg, mu_alpha_deg, True, hist


# ============================================================
# BETA BLOCK
# ============================================================
def optimize_beta_block(beta_init: np.ndarray, R: float):
    beta_bar = project_beta_feasible(beta_init, R, D_MIN)

    _, _, PSL_best, _, _, _ = evaluate_exact_af_and_psl(
        beta_bar, R, phi_full_deg, phi_plot_deg, TARGET_AOA_DEG
    )
    best_beta = beta_bar.copy()

    mu_beta_nom = MU_BETA_FACTOR * LAMBDA / (4.0 * np.pi * R)
    mu_beta_min = MU_BETA_MIN_FACTOR * LAMBDA / (4.0 * np.pi * R)
    mu_beta_max = MU_BETA_MAX_FACTOR * LAMBDA / (4.0 * np.pi * R)
    mu_beta = np.clip(mu_beta_nom, mu_beta_min, mu_beta_max)

    delta_beta_min = compute_delta_beta_min(R, D_MIN)

    improved_any = False
    hist = {"psl": [], "step": [], "scale": [], "mu": []}
    stop_cnt = 0

    for _ in range(MAX_ITER_BETA):
        AF0, _, PSL_now, _, side_masks, _ = evaluate_exact_af_and_psl(
            beta_bar, R, phi_full_deg, phi_plot_deg, TARGET_AOA_DEG
        )
        AF0_vec = AF0.reshape(-1, order="F")

        J_cols = Parallel(n_jobs=N_JOBS, verbose=0)(
            delayed(symmetric_ls_beta_column_worker)(k, beta_bar, AF0_vec, mu_beta, R)
            for k in range(M)
        )
        Jfull = np.column_stack(J_cols)

        if (not np.all(np.isfinite(Jfull))) or (not np.all(np.isfinite(AF0_vec))):
            break

        A0_stack, A_stack = build_sidelobe_stack(AF0, Jfull, side_masks)

        DeltaBeta = cp.Variable(M)
        rho = cp.Variable(nonneg=True)

        beta_new = beta_bar + DeltaBeta
        constraints = []

        for k in range(A0_stack.size):
            constraints.append(cp.abs(A0_stack[k] + A_stack[k, :] @ DeltaBeta) <= rho)

        for k in range(M - 1):
            constraints.append(beta_new[k + 1] - beta_new[k] >= delta_beta_min)

        constraints.append(DeltaBeta >= -mu_beta)
        constraints.append(DeltaBeta <= +mu_beta)

        prob = cp.Problem(cp.Minimize(rho), constraints)

        solved = False
        for solver in [cp.CLARABEL, cp.SCS, cp.ECOS]:
            try:
                prob.solve(solver=solver, verbose=False)
                if DeltaBeta.value is not None:
                    solved = True
                    break
            except Exception:
                pass

        if not solved or DeltaBeta.value is None:
            break

        delta_beta_val = np.asarray(DeltaBeta.value).reshape(-1)
        if not np.all(np.isfinite(delta_beta_val)):
            break

        step_scale = 1.0
        accepted = False

        for _ in range(MAX_BACKTRACK_BETA):
            beta_trial = beta_bar + step_scale * delta_beta_val
            beta_trial = project_beta_feasible(beta_trial, R, D_MIN)

            _, _, PSL_trial, _, _, _ = evaluate_exact_af_and_psl(
                beta_trial, R, phi_full_deg, phi_plot_deg, TARGET_AOA_DEG
            )

            if PSL_trial <= PSL_best - 1e-10:
                accepted = True
                break

            step_scale *= 0.5

        if accepted:
            beta_bar = beta_trial.copy()
            best_beta = beta_trial.copy()
            PSL_best = PSL_trial
            step_norm = np.linalg.norm(step_scale * delta_beta_val)
            improved_any = True

            hist["psl"].append(PSL_trial)
            hist["step"].append(step_norm)
            hist["scale"].append(step_scale)
            hist["mu"].append(mu_beta)

            stop_cnt = 0
            mu_beta = min(mu_beta * TRUST_GROW_BETA, mu_beta_max)

            if step_norm < STEP_NORM_TOL_BETA:
                break
        else:
            hist["psl"].append(PSL_now)
            hist["step"].append(0.0)
            hist["scale"].append(0.0)
            hist["mu"].append(mu_beta)

            stop_cnt += 1
            mu_beta = max(mu_beta * TRUST_SHRINK_BETA, mu_beta_min)

            if stop_cnt >= STOP_COUNT_LIMIT_BETA:
                break

    if ENABLE_BETA_POLISH:
        beta_polished, polish_improved = polish_beta_coordinates(best_beta, R)
        if polish_improved:
            best_beta = beta_polished.copy()
            improved_any = True

    if not improved_any:
        return beta_init.copy(), False, hist

    return best_beta, True, hist


# ============================================================
# INITIALIZATION
# ============================================================
alpha_deg = float(np.clip(ALPHA0_DEG, ALPHA_MIN_DEG, ALPHA_MAX_DEG))
R = alpha_to_R(alpha_deg, D_MIN, M)

beta = np.linspace(-np.deg2rad(alpha_deg) / 2.0, +np.deg2rad(alpha_deg) / 2.0, M)
beta = project_beta_feasible(beta, R, D_MIN)

mu_alpha_deg = MU_ALPHA0_DEG

ao_psl_hist = []
ao_alpha_hist = []
ao_R_hist = []
ao_span_hist = []
ao_psl_per_beam_hist = []

print("====================================================")
print("Improved alternating optimization started")
print(f"Initial alpha = {alpha_deg:.6f} deg")
print(f"Initial R     = {R:.6e} m ({R / LAMBDA:.4f} λ)")
if USE_BEAM_CENTERED_PSL_SECTOR:
    print(
        f"PSL region = beam-dependent sector [phi0 - {PSL_SECTOR_HALF_WIDTH_DEG:.1f}, "
        f"phi0 + {PSL_SECTOR_HALF_WIDTH_DEG:.1f}] deg"
    )
else:
    print("PSL region = full 360 deg outside mainbeam")
if USE_FIXED_MAINBEAM_WIDTH:
    print(f"Mainbeam mode = fixed width ({FIXED_MAINBEAM_WIDTH_DEG:.4f} deg)")
else:
    print("Mainbeam mode = automatic null-to-null")
print("====================================================")

AF_init, _, PSL_init, psl_init_per_beam, _, mb_init = evaluate_exact_af_and_psl(
    beta, R, phi_full_deg, phi_plot_deg, TARGET_AOA_DEG
)

ao_psl_per_beam_hist.append(psl_init_per_beam.copy())

print(f"Initial worst-case sidelobe level = {PSL_init:.6f} dB")
print("Initial per-beam sidelobe levels =", " ".join(f"{v:.4f}" for v in psl_init_per_beam))

current_psl = PSL_init


# ============================================================
# ALTERNATING OPTIMIZATION LOOP
# ============================================================
for ao_it in range(1, MAX_AO_ITER + 1):
    print("\n----------------------------------------------------")
    print(f"AO iteration {ao_it:2d}")
    print("----------------------------------------------------")

    psl_before = current_psl

    beta, R, alpha_deg, mu_alpha_deg, alpha_improved, _ = optimize_alpha_block(
        beta, R, mu_alpha_deg
    )

    _, _, current_psl, psl_tmp_per_beam, _, _ = evaluate_exact_af_and_psl(
        beta, R, phi_full_deg, phi_plot_deg, TARGET_AOA_DEG
    )

    print(
        f"After alpha-block: alpha = {alpha_deg:.6f} deg | "
        f"R/λ = {R / LAMBDA:.6f} | sidelobe level = {current_psl:.6f} dB"
    )

    beta, beta_improved, _ = optimize_beta_block(beta, R)

    _, AF_dB_tmp, current_psl, psl_tmp_per_beam, _, mb_info = evaluate_exact_af_and_psl(
        beta, R, phi_full_deg, phi_plot_deg, TARGET_AOA_DEG
    )
    
    ao_psl_per_beam_hist.append(psl_tmp_per_beam.copy())

    span_deg = np.rad2deg(beta[-1] - beta[0])

    print(
        f"After beta-block : span = {span_deg:.6f} deg | "
        f"R/λ = {R / LAMBDA:.6f} | sidelobe level = {current_psl:.6f} dB"
    )
    print("Per-beam sidelobe :", " ".join(f"{v:.4f}" for v in psl_tmp_per_beam))

    ao_psl_hist.append(current_psl)
    ao_alpha_hist.append(alpha_deg)
    ao_R_hist.append(R)
    ao_span_hist.append(span_deg)

    psl_improvement = psl_before - current_psl

    if current_psl > psl_before + 1e-10:
        print("Warning: sidelobe level got worse, which should not happen now.")

    if (not alpha_improved) and (not beta_improved):
        print("Stopping AO: neither alpha-block nor beta-block improved.")
        break

    if psl_improvement < AO_STOP_TOL_DB:
        print(f"Stopping AO: improvement {psl_improvement:.6e} dB < {AO_STOP_TOL_DB:.1e}")
        break


# ============================================================
# FINAL EVALUATION
# ============================================================
AF_final, AF_dB, PSL_dB, psl_per_beam_final, side_masks_final, mb_final = evaluate_exact_af_and_psl(
    beta, R, phi_full_deg, phi_plot_deg, TARGET_AOA_DEG
)

final_span_deg = np.rad2deg(beta[-1] - beta[0])

print("\n====================================================")
print("FINAL RESULTS")
print("====================================================")
print(f"Final worst-case sidelobe level = {PSL_dB:.6f} dB")
print("Final per-beam sidelobe levels =", " ".join(f"{v:.6f}" for v in psl_per_beam_final))
print(f"Final effective span            = {final_span_deg:.6f} deg")
if USE_FIXED_MAINBEAM_WIDTH:
    print(f"Fixed mainbeam width used       = {FIXED_MAINBEAM_WIDTH_DEG:.6f} deg")
if USE_BEAM_CENTERED_PSL_SECTOR:
    print(f"PSL sector half-width used      = {PSL_SECTOR_HALF_WIDTH_DEG:.6f} deg")
print(f"Final R (m)                     = {R:.12e}")
print(f"Final R/λ                  = {R / LAMBDA:.6f}")

chord_dist = 2.0 * R * np.sin(np.diff(beta) / 2.0)
print(f"Minimum exact chord spacing     = {np.min(chord_dist):.12e} m")
print(f"Target d_min                    = {D_MIN:.12e} m")


# ============================================================
# FINAL GEOMETRY
# ============================================================
x_elem, y_elem, _, _ = build_geometry(beta, R)

th = np.linspace(np.min(beta) - 0.1, np.max(beta) + 0.1, 500)
x_arc = -R + R * np.cos(th)
y_arc = R * np.sin(th)


# ============================================================
# PLOTS
# ============================================================
plt.figure(figsize=(10, 5))
plt.plot(phi_plot_deg, AF_dB, linewidth=1.5)

plt.grid(True, which="both")
plt.xlim([-180, 180])
plt.xticks(np.arange(-180, 181, 60))

ymax = np.max(AF_dB)
ymin = np.min(AF_dB)
plt.ylim([max(ymin, ymax - 80), ymax + 2])

plt.xlabel("Azimuth Angle (deg)")
plt.ylabel("Beampattern Response (dB)")
plt.title(f"Unnormalized Beampattern, worst-case PSL = {PSL_dB:.4f} dB")
plt.tight_layout()

plt.figure(figsize=(8, 4))
plt.stem(np.arange(1, M + 1), np.rad2deg(beta), basefmt=" ")
plt.grid(True, which="both")
plt.xlabel("Element Index")
plt.ylabel(r"$\beta_m$ (deg)")
plt.title("Optimized Element Angles")
plt.tight_layout()

plt.figure(figsize=(6, 6))
plt.plot(x_arc, y_arc, "k--", linewidth=1.0)
plt.plot(x_elem, y_elem, "ro", markersize=6)
plt.plot(x_elem[0], y_elem[0], "bs", markersize=8)
plt.plot(x_elem[-1], y_elem[-1], "gs", markersize=8)

for m in range(len(x_elem) - 1):
    d_m = np.sqrt((x_elem[m + 1] - x_elem[m]) ** 2 + (y_elem[m + 1] - y_elem[m]) ** 2)
    d_lambda = d_m / LAMBDA

    xm = 0.5 * (x_elem[m + 1] + x_elem[m])
    ym = 0.5 * (y_elem[m + 1] + y_elem[m])

    dx = x_elem[m + 1] - x_elem[m]
    dy = y_elem[m + 1] - y_elem[m]

    nvec = np.array([-dy, dx], dtype=float)
    nvec /= (np.linalg.norm(nvec) + 1e-12)

    offset = 0.01 * LAMBDA
    plt.text(
        xm + offset * nvec[0],
        ym + offset * nvec[1],
        f"{d_lambda:.3f} λ",
        fontsize=9,
        ha="center",
        bbox=dict(facecolor="white", edgecolor="none", pad=0.4),
    )

plt.axis("equal")
plt.grid(True, which="both")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Optimized Arc Geometry")
plt.tight_layout()

if len(ao_psl_hist) > 0:
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(ao_psl_hist) + 1), ao_psl_hist, "-o", linewidth=1.5)
    plt.grid(True, which="both")
    plt.xlabel("AO Iteration")
    plt.ylabel("Worst-Case Sidelobe Level (dB)")
    plt.title("Alternating Optimization Sidelobe History")
    plt.tight_layout()

if len(ao_alpha_hist) > 0:
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(ao_alpha_hist) + 1), ao_alpha_hist, "-o", linewidth=1.5)
    plt.grid(True, which="both")
    plt.xlabel("AO Iteration")
    plt.ylabel("Alpha / Span (deg)")
    plt.title("Effective Arc Span per AO Iteration")
    plt.tight_layout()

if len(ao_R_hist) > 0:
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(ao_R_hist) + 1), np.array(ao_R_hist) / LAMBDA, "-o", linewidth=1.5)
    plt.grid(True, which="both")
    plt.xlabel("AO Iteration")
    plt.ylabel(r"R / $\lambda$")
    plt.title("Radius per AO Iteration")
    plt.tight_layout()

plt.show()


# ============================================================
# Recover R and arc span from final x,y if needed
# ============================================================
R_est, arc_span_rad, arc_span_deg, beta_est = recover_R_and_arcspan_from_xy(x_elem, y_elem)
print(f"\nEstimated R from x,y     = {R_est:.6e} m")
print(f"Estimated arc span       = {arc_span_rad:.6f} rad")
print(f"Estimated arc span       = {arc_span_deg:.6f} deg")