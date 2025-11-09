import numpy as np
import time

# -------------------------
# GF(2)
# -------------------------
def symplectic_J(k: int) -> np.ndarray:
    """Genera la matriz simpléctica estándar J."""
    I = np.eye(k, dtype=np.uint8)
    Z = np.zeros((k, k), dtype=np.uint8)
    return np.block([[Z, I], [I, Z]]).astype(np.uint8)

def pump_S_from_w(w_bits: np.ndarray) -> np.ndarray:
    """Calcula la matriz de transvección S = I + w(Jw)^T desde un vector w."""
    w = np.array(w_bits, dtype=np.uint8).reshape(-1, 1)  # 2k×1
    k2 = w.shape[0]
    assert k2 % 2 == 0, "w length must be even (=2k)"
    k = k2 // 2
    J = symplectic_J(k)
    Jw = (J @ w) % 2
    # S = I + w (Jw)^T  over GF(2)
    S = (np.eye(2*k, dtype=np.uint8) ^ (w @ Jw.T)) % 2
    return S

def is_symplectic(S: np.ndarray) -> bool:
    """Verifica si una matriz S es simpléctica (S^T J S = J mod 2)."""
    k2 = S.shape[0]
    assert S.shape[0] == S.shape[1] and k2 % 2 == 0
    k = k2 // 2
    J = symplectic_J(k)
    SJ = (S.T @ J) % 2
    SJ = (SJ @ S) % 2
    return np.array_equal(SJ, J)

def labels_XZ(k: int) -> list[str]:
    """Genera etiquetas [X1..Xk, Z1..Zk]."""
    return [*(f"X{i+1}" for i in range(k)), *(f"Z{i+1}" for i in range(k))]

def pretty_mapping_from_S(S: np.ndarray, lbls: list[str]) -> list[str]:
    """Genera una lista legible de las asignaciones de S (P_j -> ...)."""
    outs = []
    for j, name in enumerate(lbls):
        col = S[:, j].astype(int).tolist()
        terms = [lbls[i] for i, b in enumerate(col) if b]
        rhs = "I" if len(terms) == 0 else " ".join(terms)
        outs.append(f"{name} -> {rhs}")
    return outs

def changed_columns(S: np.ndarray, lbls: list[str]) -> list[tuple[int, str]]:
    """Encuentra qué columnas de S difieren de la identidad."""
    I = np.eye(S.shape[0], dtype=np.uint8)
    mask = np.any(S != I, axis=0)
    idxs = np.where(mask)[0].tolist()
    return [(i, lbls[i]) for i in idxs]

def weight_bits(v: np.ndarray) -> int:
    """Calcula el peso de Hamming (recuento de no ceros)."""
    return int(np.count_nonzero(v))

def verify_transvection_equivalence(S: np.ndarray, w: np.ndarray) -> bool:
    """Verifica S e_j = e_j ^ w si (Jw)_j=1."""
    k2 = S.shape[0]
    assert w.shape[0] == k2
    k = k2 // 2
    J = symplectic_J(k)
    Jw = (J @ w.reshape(-1, 1)) % 2  # (2k,1)
    ok = True
    bad_js = []
    I = np.eye(k2, dtype=np.uint8)
    for j in range(k2):
        ej = I[:, j]
        pred = ej ^ (w if Jw[j, 0] == 1 else np.zeros_like(w))
        if not np.array_equal(S[:, j], pred):
            ok = False
            bad_js.append(j)
            if len(bad_js) >= 8:
                break
    if not ok:
        print(f"[Check] Transvection equivalence failed for columns: {bad_js}")
    return ok

# -------------------------
# Lógica de Actualización de Frame (Benchmark)
# -------------------------
def rand_sparse_vec(length: int, weight: int, rng: np.random.Generator) -> np.ndarray:
    """Crea un vector GF(2) disperso aleatorio."""
    v = np.zeros(length, dtype=np.uint8)
    weight = max(0, min(weight, length))
    if weight > 0:
        idx = rng.choice(length, size=weight, replace=False)
        v[idx] = 1
    return v

def apply_transvection_matrix_inplace(V: np.ndarray, u: np.ndarray, Ju: np.ndarray):
    """
    Aplica la actualización de transvección V -> V S in-place, donde S=I+u(Ju)^T.
    V: (d, r), u: (d,), Ju: (d,)
    """
    # s_j = parity( (Ju & V[:, j]) )
    # V[:, j] ^= u if s_j == 1
    # Vectorizado: calcula todas las paridades s, luego aplica XOR con broadcast
    s = (Ju[:, None] & V).sum(axis=0) & 1  # (r,)
    if np.any(s):
        V ^= (u[:, None] & s[None, :])  # Broadcast, solo efectivo en columnas donde s=1

def bench_frame_update(n_phys: int, k: int, w_logical: np.ndarray,
                       r_phys: int, r_log: int, loops: int,
                       u_weight: int | None, seed: int):
    """
    Ejecuta el micro-benchmark comparando las actualizaciones de frames físicos vs. lógicos.
    Imprime los resultados en la consola.
    """
    rng = np.random.default_rng(seed)

    # Dimensiones: físico d_p = 2 n_phys; lógico d_l = 2 k
    d_p = 2 * n_phys
    d_l = 2 * k

    # Vector u físico (simulando bomba a lo largo de un camino)
    if u_weight is None:
        u_weight = max(64, int(0.015 * d_p)) # Default ~1.5% d_p
    u_phys = rand_sparse_vec(d_p, u_weight, rng)
    Jp = symplectic_J(n_phys)
    Ju_phys = (Jp @ u_phys.reshape(-1, 1)) % 2
    Ju_phys = Ju_phys.ravel()

    # Vector w lógico y Jw
    wl = np.array(w_logical, dtype=np.uint8).ravel()
    Jl = symplectic_J(k)
    Jw_log = (Jl @ wl.reshape(-1, 1)) % 2
    Jw_log = Jw_log.ravel()

    # Frames aleatorios: físico y lógico
    Vp = rng.integers(0, 2, size=(d_p, r_phys), dtype=np.uint8)
    Vl = rng.integers(0, 2, size=(d_l, r_log), dtype=np.uint8)

    # Calentamiento
    apply_transvection_matrix_inplace(Vp, u_phys, Ju_phys)
    apply_transvection_matrix_inplace(Vl, wl, Jw_log)

    # Timing: Capa Física
    t0 = time.perf_counter()
    for _ in range(loops):
        apply_transvection_matrix_inplace(Vp, u_phys, Ju_phys)
    t1 = time.perf_counter()

    # Timing: Capa Lógica (Equivalente Clifford)
    t2 = time.perf_counter()
    for _ in range(loops):
        apply_transvection_matrix_inplace(Vl, wl, Jw_log)
    t3 = time.perf_counter()

    # Tiempo por vector por actualización (nanosegundos)
    phys_ns = (t1 - t0) * 1e9 / (loops * max(1, r_phys))
    logi_ns = (t3 - t2) * 1e9 / (loops * max(1, r_log))
    speedup = phys_ns / max(1e-9, logi_ns)

    # Contar operaciones XOR reales (usando la s de la última ronda)
    s_phys = (Ju_phys[:, None] & Vp).sum(axis=0) & 1
    s_log = (Jw_log[:, None] & Vl).sum(axis=0) & 1
    xor_bits_phys = int(s_phys.sum()) * int(weight_bits(u_phys))
    xor_bits_log = int(s_log.sum()) * int(weight_bits(wl))

    print("=== Frame update micro-benchmark ===")
    print(f"Physical: n={n_phys}, d_p=2n={d_p}, r_phys={r_phys}, u_weight={weight_bits(u_phys)}")
    print(f"Logical:  k={k}, d_l=2k={d_l}, r_log={r_log},  w_weight={weight_bits(wl)}")
    print(f"Loops: {loops}")
    print(f"Time per vector per update: physical ~ {phys_ns:.1f} ns, logical ~ {logi_ns:.1f} ns")
    print(f"Speed-up (physical/logical): ×{speedup:.1f}")
    print(f"XOR bit ops (one round, measured): physical ≈ {xor_bits_phys}, logical ≈ {xor_bits_log}")
    
    # Verificaciones de correctitud
    S_log = pump_S_from_w(wl)
    print(f"Correctness: symplectic(S_log)={is_symplectic(S_log)}, "
          f"equiv(S_log = I + w (Jw)^T)={verify_transvection_equivalence(S_log, wl)}")
    print()