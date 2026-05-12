from itertools import product
import math

from lighthouse.schedule.xegpu.mlp_schedule import (
    PFETCH_MAX_COLS,
    PFETCH_MAX_ROWS,
    PFETCH_MIN_COLS,
    PFETCH_MIN_ROWS,
    LOAD_MAX_COLS,
    LOAD_MAX_ROWS,
    MAX_NB_SG_THREADS,
    MIN_NB_THREADS,
    DPAS,
)


def check_constraints(params: dict, verbose: bool = False) -> bool:
    """Check that the given tile size configuration is valid."""

    # FIXME generalize and refactor, e.g. re-use check_prefetch_tile_a/b
    def print_reason(msg):
        if verbose:
            print(f"  Invalid: {msg}")

    M = params["m"]
    N = params["n"]
    wg_tile_m = params["wg_m"]
    wg_tile_n = params["wg_n"]
    sg_tile_m = params["sg_m"]
    sg_tile_n = params["sg_n"]
    load_tile_a_m = params["load_a_m"]
    load_tile_a_k = params["load_a_k"]
    load_tile_b_k = params["load_b_k"]
    load_tile_b_n = params["load_b_n"]
    prefetch_tile_a_m = params["prefetch_a_m"]
    prefetch_tile_a_k = params["prefetch_a_k"]
    prefetch_tile_b_k = params["prefetch_b_k"]
    prefetch_tile_b_n = params["prefetch_b_n"]
    k_tile = params["k_tile"]

    if M % wg_tile_m != 0:
        print_reason("wg_tile_m does not divide M")
        return False
    if N % wg_tile_n != 0:
        print_reason("wg_tile_n does not divide N")
        return False
    if wg_tile_m % sg_tile_m != 0:
        print_reason("sg_tile_m does not divide wg_tile_m")
        return False
    if wg_tile_n % sg_tile_n != 0:
        print_reason("sg_tile_n does not divide wg_tile_n")
        return False
    if sg_tile_m % DPAS.M != 0:
        print_reason("sg_tile_m not multiple of dpas_m")
        return False
    if sg_tile_n % DPAS.N != 0:
        print_reason("sg_tile_n not multiple of dpas_n")
        return False
    if k_tile % DPAS.K != 0:
        print_reason("k_tile not multiple of dpas_k")
        return False

    # SG level thread layout: [nb_sg_threads_m, nb_sg_threads_n]
    nb_sg_threads_m = wg_tile_m // sg_tile_m
    nb_sg_threads_n = wg_tile_n // sg_tile_n
    nb_sg_threads = nb_sg_threads_m * nb_sg_threads_n
    if nb_sg_threads > MAX_NB_SG_THREADS:
        print_reason("too many sg threads")
        return False
    if nb_sg_threads < MIN_NB_THREADS:
        print_reason("too few sg threads")
        return False

    if sg_tile_m % load_tile_a_m != 0:
        print_reason("load_tile_a_m does not divide sg_tile_m")
        return False
    if k_tile % load_tile_a_k != 0:
        print_reason("load_tile_a_k does not divide k_tile")
        return False
    if k_tile % load_tile_b_k != 0:
        print_reason("load_tile_b_k does not divide k_tile")
        return False
    if sg_tile_n % load_tile_b_n != 0:
        print_reason("load_tile_b_n does not divide sg_tile_n")
        return False
    if load_tile_a_m > LOAD_MAX_ROWS:
        print_reason("too large load_tile_a_m")
        return False
    if load_tile_a_k > LOAD_MAX_COLS:
        print_reason("too large load_tile_a_k")
        return False
    if load_tile_b_k > LOAD_MAX_ROWS:
        print_reason("too large load_tile_b_k")
        return False
    if load_tile_b_n > LOAD_MAX_COLS:
        print_reason("too large load_tile_b_n")
        return False
    if sg_tile_m % prefetch_tile_a_m != 0:
        print_reason("prefetch_tile_a_m does not divide sg_tile_m")
        return False
    if k_tile % prefetch_tile_a_k != 0:
        print_reason("prefetch_tile_a_k does not divide k_tile")
        return False
    if k_tile % prefetch_tile_b_k != 0:
        print_reason("prefetch_tile_b_k does not divide k_tile")
        return False
    if sg_tile_n % prefetch_tile_b_n != 0:
        print_reason("prefetch_tile_b_n does not divide sg_tile_n")
        return False
    if prefetch_tile_a_m > PFETCH_MAX_ROWS:
        print_reason("too large prefetch_tile_a_m")
        return False
    if prefetch_tile_a_k > PFETCH_MAX_COLS:
        print_reason("too large prefetch_tile_a_k")
        return False
    if prefetch_tile_b_k > PFETCH_MAX_ROWS:
        print_reason("too large prefetch_tile_b_k")
        return False
    if prefetch_tile_b_n > PFETCH_MAX_COLS:
        print_reason("too large prefetch_tile_b_n")
        return False
    if prefetch_tile_a_m < PFETCH_MIN_ROWS:
        print_reason("too small prefetch_tile_a_m")
        return False
    if prefetch_tile_a_k < PFETCH_MIN_COLS:
        print_reason("too small prefetch_tile_a_k")
        return False
    if prefetch_tile_b_k < PFETCH_MIN_ROWS:
        print_reason("too small prefetch_tile_b_k")
        return False
    if prefetch_tile_b_n < PFETCH_MIN_COLS:
        print_reason("too small prefetch_tile_b_n")
        return False
    if load_tile_a_m % DPAS.M != 0:
        print_reason("load_tile_a_m not multiple of dpas_m")
        return False
    if load_tile_a_k % DPAS.K != 0:
        print_reason("load_tile_a_k not multiple of dpas_k")
        return False
    if load_tile_b_k % DPAS.K != 0:
        print_reason("load_tile_b_k not multiple of dpas_k")
        return False
    if load_tile_b_n % DPAS.N != 0:
        print_reason("load_tile_b_n not multiple of dpas_n")
        return False

    # prefetch A layout
    nb_prefetch_a_m = wg_tile_m // prefetch_tile_a_m
    nb_prefetch_a_k = k_tile // prefetch_tile_a_k
    if nb_prefetch_a_m * nb_prefetch_a_k > MAX_NB_SG_THREADS:
        print_reason("too many prefetch A tiles")
        return False
    if nb_prefetch_a_m * nb_prefetch_a_k < MIN_NB_THREADS:
        print_reason("too few prefetch A threads")
        return False

    # prefetch B layout
    nb_prefetch_b_k = k_tile // prefetch_tile_b_k
    nb_prefetch_b_n = wg_tile_n // prefetch_tile_b_n
    if nb_prefetch_b_k * nb_prefetch_b_n > MAX_NB_SG_THREADS:
        print_reason("too many prefetch B tiles")
        return False
    if nb_prefetch_b_k * nb_prefetch_b_n < MIN_NB_THREADS:
        print_reason("too few prefetch B threads")
        return False

    return True


def get_int_factors(n):
    "Return a and b such that n = a * b."
    factors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            a = i
            b = n // i
            factors.append((a, b))
            factors.append((b, a))
    # sort by how close to square the factors are
    factors.sort(key=lambda x: abs(x[0] - x[1]))
    return factors


def check_prefetch_tile(tile, data_shape, gpu_specs, name="A", verbose=False):
    shape = data_shape
    if tile[0] < PFETCH_MIN_ROWS:
        raise ValueError(
            f"Prefetch tile {name} {tile} has too few rows (min {PFETCH_MIN_ROWS})."
        )
    if tile[0] > PFETCH_MAX_ROWS:
        raise ValueError(
            f"Prefetch tile {name} {tile} has too many rows (max {PFETCH_MAX_ROWS})."
        )
    if tile[1] < PFETCH_MIN_COLS:
        raise ValueError(
            f"Prefetch tile {name} {tile} has too few cols (min {PFETCH_MIN_COLS})."
        )
    if tile[1] > PFETCH_MAX_COLS:
        raise ValueError(
            f"Prefetch tile {name} {tile} has too many cols (max {PFETCH_MAX_COLS})."
        )
    if shape[0] % tile[0] != 0 or shape[1] % tile[1] != 0:
        raise ValueError(
            f"Prefetch tile {name} {tile} does not divide the parent shape {shape}."
        )
    rows = int(shape[0] / tile[0])
    cols = int(shape[1] / tile[1])
    nb_threads = int(rows * cols)
    if verbose:
        print(f"=== Prefetch {name} ===")
        print(f"tile size {tile}, grid size ({rows}, {cols}), {nb_threads} threads")
    if nb_threads > gpu_specs["max_nb_threads"]:
        raise ValueError(
            f"Number of threads for {name} prefetch ({nb_threads}) exceeds max threads ({gpu_specs['max_nb_threads']})."
        )
    return rows, cols


def check_prefetch_tile_a(tile, wg_tile, k_tile, gpu_specs, verbose=False):
    data_shape = (wg_tile[0], k_tile)
    return check_prefetch_tile(tile, data_shape, gpu_specs, name="A", verbose=verbose)


def check_prefetch_tile_b(tile, wg_tile, k_tile, gpu_specs, verbose=False):
    data_shape = (k_tile, wg_tile[1])
    return check_prefetch_tile(tile, data_shape, gpu_specs, name="B", verbose=verbose)


def generate_prefetch_tiles(wg_tile, k_tile, gpu_specs, n=None):
    """Generates valid prefetch tile sizes for A and B.

    Candidates are sorted by number of threads (descending) and then by how
    balanced the thread grid is (descending).
    """

    def gridsearch(check_fn):
        tiles = []
        for rows in range(PFETCH_MIN_ROWS, PFETCH_MAX_ROWS + 1):
            for cols in range(PFETCH_MIN_COLS, PFETCH_MAX_COLS + 1):
                tile = (rows, cols)
                try:
                    grid = check_fn(tile, wg_tile, k_tile, gpu_specs)
                    nb_threads = int(grid[0] * grid[1])
                    tiles.append((tile, nb_threads, grid))
                except ValueError:
                    pass
        # sort by number of threads and then by how balanced the thread grid is
        tiles.sort(key=lambda x: (x[1], -abs(x[2][0] - x[2][1])), reverse=True)
        tiles = [t[0] for t in tiles]
        return tiles

    prefetch_tiles_a = gridsearch(check_prefetch_tile_a)
    prefetch_tiles_b = gridsearch(check_prefetch_tile_b)
    if n is not None:
        if n == 1:
            prefetch_tiles_a = prefetch_tiles_a[0]
            prefetch_tiles_b = prefetch_tiles_b[0]
        else:
            prefetch_tiles_a = prefetch_tiles_a[:n]
            prefetch_tiles_b = prefetch_tiles_b[:n]

    return prefetch_tiles_a, prefetch_tiles_b


def generate_load_tiles(min_rows=8, min_cols=8):
    # FIXME load tile must divide sg_tile and k tile
    load_elems = [8, 16, 32]
    load_tiles = []
    for a, b in product(load_elems, load_elems):
        if (
            a >= min_rows
            and a <= LOAD_MAX_ROWS
            and b >= min_cols
            and b <= LOAD_MAX_COLS
        ):
            load_tiles.append((a, b))
    return load_tiles


def generate_load_tiles_a():
    return generate_load_tiles(min_rows=DPAS.A_TILE[0], min_cols=DPAS.A_TILE[1])


def generate_load_tiles_b():
    return generate_load_tiles(min_rows=DPAS.B_TILE[0], min_cols=DPAS.B_TILE[1])


def estimate_perf(
    M,
    N,
    K,
    wg_tile,
    sg_tile,
    k_tile,
    gpu_specs,
    prefetch_tile_a=None,
    prefetch_tile_b=None,
    verbose=True,
):
    """Estimate the performance of the given tile size configuration."""
    # NOTE this is basically just constraint checking with the roofline model...
    if verbose:
        print("=== Global Level ===")
        print(f"Matrix sizes: M={M}, N={N}, K={K}")

    # TODO generalize
    ab_dtype_size = 2  # bytes for f16
    c_dtype_size = 4  # bytes for f32

    # WG
    if verbose:
        print("=== Workgroup Level ===")
    roofline_threshold = (
        gpu_specs["peak_flops"] / gpu_specs["bw_global_mem"]
    )  # in FLOPs/Byte

    # TODO refactor to get_wg_grid util with constraint checks
    wg_grid = (M // wg_tile[0], N // wg_tile[1])
    nb_wgs = wg_grid[0] * wg_grid[1]
    if verbose:
        print(f"Workgroup tile size: {wg_tile}, grid size: {wg_grid}, nb WGs: {nb_wgs}")
        print(f"K tile size: {k_tile}")

    A_wg_shape = (wg_tile[0], k_tile)
    B_wg_shape = (k_tile, wg_tile[1])
    C_wg_shape = (wg_tile[0], wg_tile[1])

    A_footprint = A_wg_shape[0] * A_wg_shape[1] * ab_dtype_size
    B_footprint = B_wg_shape[0] * B_wg_shape[1] * ab_dtype_size
    C_footprint = C_wg_shape[0] * C_wg_shape[1] * c_dtype_size

    if verbose:
        print(f"A: shape={A_wg_shape}, footprint={A_footprint / 1024:.2f} KB")
        print(f"B: shape={B_wg_shape}, footprint={B_footprint / 1024:.2f} KB")
        print(f"C: shape={C_wg_shape}, footprint={C_footprint / 1024:.2f} KB")

    total_footprint = A_footprint + B_footprint
    if verbose:
        print(
            f"Total SLM footprint: {total_footprint / 1024:.1f} / "
            f"{gpu_specs['local_mem_size'] / 1024:.1f} KB"
        )
    # TODO check that A,B,C fit in shared local memory

    # arithmetic intensity
    f = (wg_tile[0] * wg_tile[1]) / (wg_tile[0] + wg_tile[1])
    ai = f * ab_dtype_size
    if verbose:
        print(f"Arithmetic intensity: {ai:.2f} FLOPs/Byte")
        print(f"Roofline threshold:   {roofline_threshold:.2f} FLOPs/Byte")

    if verbose:
        if ai < roofline_threshold:
            print(" => Bandwidth-bound regime")
        else:
            print(" => Compute-bound regime")

    xe_core_utilization = min(nb_wgs / gpu_specs["nb_xe_cores"], 1.0)
    if verbose:
        print(f"XE core utilization: {xe_core_utilization:.2f}")

    # predict flops
    peak_flops = (
        gpu_specs["peak_flops"] * xe_core_utilization
    )  # possible under-utilization
    predicted_throughput = min(peak_flops, ai * gpu_specs["bw_global_mem"])
    if verbose:
        print(f"Predicted throughput: {predicted_throughput / 1e12:.2f} TFLOPS")

    # SG
    if verbose:
        print("=== Subgroup Level ===")

    # TODO refactor to compute_sg_grid util with constraint checks
    sg_grid = (wg_tile[0] // sg_tile[0], wg_tile[1] // sg_tile[1])
    nb_sgs = sg_grid[0] * sg_grid[1]
    if verbose:
        print(
            f"Subgroup tile size: {sg_tile}, grid size: {sg_grid}, nb SGs per WG: {nb_sgs}"
        )

    if nb_sgs > gpu_specs["max_nb_threads"]:
        raise ValueError(
            f"Number of SGs ({nb_sgs}) exceeds max threads ({gpu_specs['max_nb_threads']})."
        )

    A_sg_shape = (sg_tile[0], k_tile)
    B_sg_shape = (k_tile, sg_tile[1])
    C_sg_shape = (sg_tile[0], sg_tile[1])

    A_footprint = A_sg_shape[0] * A_sg_shape[1] * ab_dtype_size
    B_footprint = B_sg_shape[0] * B_sg_shape[1] * ab_dtype_size
    C_footprint = C_sg_shape[0] * C_sg_shape[1] * c_dtype_size

    total_footprint = A_footprint + B_footprint + C_footprint
    if verbose:
        print(f"A: shape={A_sg_shape}, footprint={A_footprint / 1024:.2f} KB")
        print(f"B: shape={B_sg_shape}, footprint={B_footprint / 1024:.2f} KB")
        print(f"C: shape={C_sg_shape}, footprint={C_footprint / 1024:.2f} KB")
        print(f"Total register footprint: {total_footprint / 1024:.2f} KB")

    nb_parallel_dpas = (sg_tile[0] // DPAS.M) * (sg_tile[1] // DPAS.N)
    if verbose:
        print(f"Number of DPAS threads: {nb_parallel_dpas}")
    nb_dpas_ops = nb_parallel_dpas * (k_tile // DPAS.K)
    if verbose:
        print(f"Number of total DPAS ops: {nb_dpas_ops}")

    if nb_parallel_dpas > gpu_specs["dpas_exec_size"]:
        raise ValueError(
            f"Number of parallel DPAS ops ({nb_parallel_dpas}) exceeds hardware execution size ({gpu_specs['dpas_exec_size']})."
        )

    # estimate number of used registers
    reg_size = 64  # bytes per register
    nb_reg = int((A_footprint + B_footprint + C_footprint) / reg_size)
    if verbose:
        print(f"Number of registers: {nb_reg}")

    if nb_reg > gpu_specs["nb_registers"]:
        raise ValueError(
            f"Number of registers ({nb_reg}) exceeds hardware register file size ({gpu_specs['nb_registers']})."
        )

    if prefetch_tile_a:
        # check that prefetch tile is suitable for WG-k tile
        check_prefetch_tile_a(
            prefetch_tile_a, wg_tile, k_tile, gpu_specs, verbose=verbose
        )

    if prefetch_tile_b:
        # check that prefetch tile is suitable for WG-k tile
        check_prefetch_tile_b(
            prefetch_tile_b, wg_tile, k_tile, gpu_specs, verbose=verbose
        )

    return predicted_throughput


def generate_configs(
    M,
    N,
    K,
    gpu_specs,
    perf_threshold=None,
    load_strategy="dpas",
    pf_strategy="best",
    max_nb_configs=None,
):
    """Generate valid tile size configurations based on the selection strategy.

    perf_threshold: if set, only return configurations with
    estimated_perf >= perf_threshold * max_found_estimated_perf.

    load_strategy: sets the load tile selection strategy
    - "large": use the largest supported load tile
    - "dpas": use dpas op A/B tile size as load tile

    pf_strategy: sets the prefetch tile selection strategy
    - "best": only the best prefetch tile for A and B based on number of threads
    - "all": append all valid prefetch tiles for A and B

    Returns:
    A list of (perf_estimate, params_dict) tuples sorted by perf_estimate (descending).
    """
    # TODO add data types as variables
    # TODO dpas tile sizes should be in gpu_specs

    def tuple_to_param_dict(M, N, K, config):
        wg_tile, sg_tile, k_tile, ld_a, ld_b, pf_a, pf_b = config
        return {
            "m": M,
            "n": N,
            "k": K,
            "wg_m": wg_tile[0],
            "wg_n": wg_tile[1],
            "sg_m": sg_tile[0],
            "sg_n": sg_tile[1],
            "k_tile": k_tile,
            "load_a_m": ld_a[0],
            "load_a_k": ld_a[1],
            "load_b_k": ld_b[0],
            "load_b_n": ld_b[1],
            "prefetch_a_m": pf_a[0],
            "prefetch_a_k": pf_a[1],
            "prefetch_b_k": pf_b[0],
            "prefetch_b_n": pf_b[1],
            "prefetch_a_nb": 1,
            "prefetch_b_nb": 1,
        }

    # define search space
    wg_options = [64, 128, 256]
    sg_options = [32, 64, 128]
    k_tile_options = [16, 32, 64]

    wg_tiles = product(wg_options, wg_options)
    sg_tiles = product(sg_options, sg_options)

    # grid search
    valid_configs = []
    for config in product(wg_tiles, sg_tiles, k_tile_options):
        wg_tile, sg_tile, k_tile = config
        try:
            perf = estimate_perf(
                M, N, K, wg_tile, sg_tile, k_tile, gpu_specs, verbose=False
            )
            if pf_strategy == "best":
                pf_a, pf_b = generate_prefetch_tiles(wg_tile, k_tile, gpu_specs, n=1)
                pf_a_list = [pf_a]
                pf_b_list = [pf_b]
            else:
                pf_a_list, pf_b_list = generate_prefetch_tiles(
                    wg_tile, k_tile, gpu_specs
                )
            if load_strategy == "large":
                load_a_list = [(32, 32)]
                load_b_list = [(32, 32)]
            elif load_strategy == "double-rows":
                load_a_list = [(DPAS.A_TILE[0] * 2, DPAS.A_TILE[1])]
                load_b_list = [(DPAS.B_TILE[0] * 2, DPAS.B_TILE[1])]
            else:
                load_a_list = [DPAS.A_TILE]
                load_b_list = [DPAS.B_TILE]
            for la, lb, pa, pb in product(
                load_a_list, load_b_list, pf_a_list, pf_b_list
            ):
                c = (wg_tile, sg_tile, k_tile, la, lb, pa, pb)
                params = tuple_to_param_dict(M, N, K, c)
                if check_constraints(params, verbose=False):
                    valid_configs.append((perf, params))
        except ValueError:
            pass

    # sort by performance (descending)
    valid_configs.sort(key=lambda x: x[0], reverse=True)

    if perf_threshold is not None:
        assert 0 < perf_threshold <= 1, "perf_threshold must be in (0, 1]"
        max_perf = valid_configs[0][0]
        valid_configs = [c for c in valid_configs if c[0] >= perf_threshold * max_perf]

    if max_nb_configs is not None:
        valid_configs = valid_configs[:max_nb_configs]

    return valid_configs


def expand_configs_with_load_tiles(
    param_list, load_strategy="dpas", exclude_duplicates=False
):
    """Expand the parameter configs with different load tile options."""
    expanded_configs = []
    for params in param_list:
        if load_strategy == "all":
            load_a_list = generate_load_tiles_a()
            load_b_list = generate_load_tiles_b()
        else:
            load_a_list = [DPAS.A_TILE]
            load_b_list = [DPAS.B_TILE]

        for la, lb in product(load_a_list, load_b_list):
            new_params = params.copy()
            new_params["load_a_m"] = la[0]
            new_params["load_a_k"] = la[1]
            new_params["load_b_k"] = lb[0]
            new_params["load_b_n"] = lb[1]
            if (
                check_constraints(new_params, verbose=False)
                and new_params not in expanded_configs
                and (not exclude_duplicates or new_params not in param_list)
            ):
                expanded_configs.append(new_params)

    return expanded_configs


def expand_configs_with_prefetch_depth(
    param_list, max_depth=2, exclude_duplicates=False
):
    """Expand the parameter configs with different prefetch depth options."""
    pf_depth_list = list(range(1, max_depth + 1))

    expanded_configs = []
    for params in param_list:
        for a, b in product(pf_depth_list, pf_depth_list):
            new_params = params.copy()
            new_params["prefetch_a_nb"] = a
            new_params["prefetch_b_nb"] = b
            if (
                check_constraints(new_params, verbose=False)
                and new_params not in expanded_configs
                and (not exclude_duplicates or new_params not in param_list)
            ):
                expanded_configs.append(new_params)

    return expanded_configs
