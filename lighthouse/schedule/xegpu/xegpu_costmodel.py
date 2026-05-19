from itertools import product
import math

from lighthouse.schedule.xegpu.mlp_schedule import (
    PFETCH_MAX_COLS,
    PFETCH_MAX_ROWS,
    PFETCH_MIN_COLS,
    PFETCH_MIN_ROWS,
    LOAD_MAX_COLS,
    LOAD_MAX_ROWS,
    MIN_NB_THREADS,
    DPAS,
)


def check_wg_tile(M: int, N: int, wg_tile: tuple[int, int]):
    if M % wg_tile[0] != 0:
        raise ValueError("wg_tile_m does not divide M")
    if N % wg_tile[1] != 0:
        raise ValueError("wg_tile_n does not divide N")


def check_sg_tile(
    wg_tile: tuple[int, int],
    sg_tile: tuple[int, int],
    gpu_specs: dict,
    min_nb_threads=None,
):
    if wg_tile[0] % sg_tile[0] != 0:
        raise ValueError("sg_tile_m does not divide wg_tile_m")
    if wg_tile[1] % sg_tile[1] != 0:
        raise ValueError("sg_tile_n does not divide wg_tile_n")
    if sg_tile[0] % DPAS.M != 0:
        raise ValueError("sg_tile_m not multiple of dpas_m")
    if sg_tile[1] % DPAS.N != 0:
        raise ValueError("sg_tile_n not multiple of dpas_n")
    nb_sg_threads_m = wg_tile[0] // sg_tile[0]
    nb_sg_threads_n = wg_tile[1] // sg_tile[1]
    nb_sg_threads = nb_sg_threads_m * nb_sg_threads_n
    if nb_sg_threads > gpu_specs["max_nb_threads"]:
        raise ValueError("too many sg threads")
    if min_nb_threads is not None and nb_sg_threads < min_nb_threads:
        raise ValueError("too few sg threads")


def check_k_tile(K: int, k_tile: int):
    if K % k_tile != 0:
        raise ValueError("k_tile does not divide K")
    if k_tile % DPAS.K != 0:
        raise ValueError("k_tile not multiple of dpas_k")


def check_load_tile(tile, parent_shape, child_shape, name="A"):
    if parent_shape[0] % tile[0] != 0 or parent_shape[1] % tile[1] != 0:
        raise ValueError(
            f"Load tile {name} {tile} does not divide the parent shape {parent_shape}."
        )
    if tile[0] % child_shape[0] != 0 or tile[1] % child_shape[1] != 0:
        raise ValueError(
            f"Load tile {name} {tile} does not divide the child shape {child_shape}."
        )
    if tile[0] < child_shape[0]:
        raise ValueError(f"Load tile {name} {tile} has too few rows.")
    if tile[1] < child_shape[1]:
        raise ValueError(f"Load tile {name} {tile} has too few cols.")
    if tile[0] > LOAD_MAX_ROWS:
        raise ValueError(f"Load tile {name} {tile} has too many rows.")
    if tile[1] > LOAD_MAX_COLS:
        raise ValueError(f"Load tile {name} {tile} has too many cols.")


def check_load_tile_a(tile, sg_tile, k_tile):
    data_shape = (sg_tile[0], k_tile)
    child_shape = DPAS.A_TILE
    return check_load_tile(tile, data_shape, child_shape, name="A")


def check_load_tile_b(tile, sg_tile, k_tile):
    data_shape = (k_tile, sg_tile[1])
    child_shape = DPAS.B_TILE
    return check_load_tile(tile, data_shape, child_shape, name="B")


def check_prefetch_tile(
    tile, data_shape, gpu_specs, name="A", min_nb_threads=None, verbose=False
):
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
    if data_shape[0] % tile[0] != 0 or data_shape[1] % tile[1] != 0:
        raise ValueError(
            f"Prefetch tile {name} {tile} does not divide the parent shape {data_shape}."
        )
    rows = int(data_shape[0] / tile[0])
    cols = int(data_shape[1] / tile[1])
    nb_threads = int(rows * cols)
    if verbose:
        print(f"=== Prefetch {name} ===")
        print(f"tile size {tile}, grid size ({rows}, {cols}), {nb_threads} threads")
    if nb_threads > gpu_specs["max_nb_threads"]:
        raise ValueError(
            f"Number of threads for {name} prefetch ({nb_threads}) exceeds max threads ({gpu_specs['max_nb_threads']})."
        )
    if min_nb_threads is not None and nb_threads < min_nb_threads:
        raise ValueError(
            f"Number of threads for {name} prefetch ({nb_threads}) is less than minimum threads ({min_nb_threads})."
        )
    return rows, cols


def check_prefetch_tile_a(
    tile, wg_tile, k_tile, gpu_specs, min_nb_threads=None, verbose=False
):
    data_shape = (wg_tile[0], k_tile)
    return check_prefetch_tile(
        tile,
        data_shape,
        gpu_specs,
        name="A",
        min_nb_threads=min_nb_threads,
        verbose=verbose,
    )


def check_prefetch_tile_b(
    tile, wg_tile, k_tile, gpu_specs, min_nb_threads=None, verbose=False
):
    data_shape = (k_tile, wg_tile[1])
    return check_prefetch_tile(
        tile,
        data_shape,
        gpu_specs,
        name="B",
        min_nb_threads=min_nb_threads,
        verbose=verbose,
    )


def check_constraints(params: dict, gpu_specs: dict, verbose: bool = False) -> bool:
    """Check that the given tile size configuration is valid."""

    M = params["m"]
    N = params["n"]
    K = params["k"]
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

    wg_tile = (wg_tile_m, wg_tile_n)
    sg_tile = (sg_tile_m, sg_tile_n)
    load_tile_a = (load_tile_a_m, load_tile_a_k)
    load_tile_b = (load_tile_b_k, load_tile_b_n)
    prefetch_tile_a = (prefetch_tile_a_m, prefetch_tile_a_k)
    prefetch_tile_b = (prefetch_tile_b_k, prefetch_tile_b_n)

    try:
        check_wg_tile(M, N, wg_tile)
        check_sg_tile(wg_tile, sg_tile, gpu_specs, min_nb_threads=MIN_NB_THREADS)
        check_k_tile(K, k_tile)
        check_load_tile_a(load_tile_a, sg_tile, k_tile)
        check_load_tile_b(load_tile_b, sg_tile, k_tile)
        check_prefetch_tile_a(
            prefetch_tile_a,
            wg_tile,
            k_tile,
            gpu_specs,
            min_nb_threads=MIN_NB_THREADS,
            verbose=verbose,
        )
        check_prefetch_tile_b(
            prefetch_tile_b,
            wg_tile,
            k_tile,
            gpu_specs,
            min_nb_threads=MIN_NB_THREADS,
            verbose=verbose,
        )
    except ValueError as e:
        if verbose:
            print(f"Invalid configuration: {e}")
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


def generate_load_tiles(check_func: callable, sg_tile, k_tile):
    load_elems = [8, 16, 32]
    load_tiles = []
    for a, b in product(load_elems, load_elems):
        tile = (a, b)
        try:
            check_func(tile, sg_tile, k_tile)
            load_tiles.append(tile)
        except ValueError:
            pass

    return load_tiles


def generate_load_tiles_a(sg_tile, k_tile):
    return generate_load_tiles(check_load_tile_a, sg_tile, k_tile)


def generate_load_tiles_b(sg_tile, k_tile):
    return generate_load_tiles(check_load_tile_b, sg_tile, k_tile)


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
) -> list[tuple[float, dict]]:
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
                if check_constraints(params, gpu_specs, verbose=False):
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
    param_list, gpu_specs, load_strategy="dpas", exclude_duplicates=False
):
    """Expand the parameter configs with different load tile options."""
    expanded_configs = []
    for params in param_list:
        sg_tile = (params["sg_m"], params["sg_n"])
        k_tile = params["k_tile"]
        if load_strategy == "all":
            load_a_list = generate_load_tiles_a(sg_tile, k_tile)
            load_b_list = generate_load_tiles_b(sg_tile, k_tile)
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
                check_constraints(new_params, gpu_specs, verbose=False)
                and new_params not in expanded_configs
                and (not exclude_duplicates or new_params not in param_list)
            ):
                expanded_configs.append(new_params)

    return expanded_configs


def expand_configs_with_prefetch_depth(
    param_list, gpu_specs, max_depth=2, exclude_duplicates=False
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
                check_constraints(new_params, gpu_specs, verbose=False)
                and new_params not in expanded_configs
                and (not exclude_duplicates or new_params not in param_list)
            ):
                expanded_configs.append(new_params)

    return expanded_configs
