from itertools import product
import math

from lighthouse.schedule.xegpu.mlp_schedule import (
    PFETCH_MAX_COLS,
    PFETCH_MAX_ROWS,
    PFETCH_MIN_COLS,
    PFETCH_MIN_ROWS,
    LOAD_MAX_COLS,
    LOAD_MAX_ROWS,
    DPAS,
)
from tune_matmul_gridsearch import check_constraints

# GPU specs

# Intel Arc B70
gpu_specs_B70 = {
    "name": "Intel Arc B70",
    "nb_xe_cores": 32,
    "peak_flops": 155000e9,  # float16
    "bw_global_mem": 608e9,  # GB/s
    "bw_local_mem": 3600e9,  # GB/s ???
    "local_mem_size": 128 * 1024,  # B; both SLM by ecore and max SLM per workgroup
    "max_nb_threads": 32,  # large register file
    # "max_nb_threads": 64  # small register file
    "dpas_exec_size": 16,  # number of parallel dpas ops in sg
    "nb_registers": 256,  # large register file
}

# Intel Arc B580
gpu_specs_B580 = {
    "name": "Intel Arc B580",
    "nb_xe_cores": 20,
    "peak_flops": 100000e9,  # float16
    "bw_global_mem": 456e9,  # GB/s
    "bw_local_mem": 3600e9,  # GB/s ???
    "local_mem_size": 128 * 1024,  # B; both SLM by ecore and max SLM per workgroup
    "max_nb_threads": 32,  # large register file
    # "max_nb_threads": 64,  # small register file
    "dpas_exec_size": 16,  # number of parallel dpas ops in sg
    "nb_registers": 256,  # large register file
}

gpu_specs_db = {
    "B70": gpu_specs_B70,
    "B580": gpu_specs_B580,
}


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

    if total_footprint > gpu_specs["local_mem_size"]:
        raise ValueError("SLM footprint exceeds local memory size.")

    # arithmetic intensity
    f = (wg_tile[0] * wg_tile[1]) / (wg_tile[0] + wg_tile[1])
    ai = f * ab_dtype_size
    if verbose:
        print(f"Arithmetic intensity: {ai:.2f} FLOPs/Byte")
        print(f"Roofline threshold:   {roofline_threshold:.2f} FLOPs/Byte")

    # is this compute or bandwidth bound?
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
    roofline_threshold = (
        gpu_specs["peak_flops"] / gpu_specs["bw_local_mem"]
    )  # in FLOPs/Byte

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

    # # arithmetic intensity
    # f = (sg_tile[0] * sg_tile[1]) / (sg_tile[0] + sg_tile[1])
    # ai = f * ab_dtype_size
    # print(f"Arithmetic intensity: {ai:.2f} FLOPs/Byte")
    # print(f"Roofline threshold:   {roofline_threshold:.2f} FLOPs/Byte")

    # # is this compute or bandwidth bound?
    # if ai < roofline_threshold:
    #     print(" => Bandwidth-bound regime")
    # else:
    #     print(" => Compute-bound regime")

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

    # TODO add tile size strategy "best-equal": best wg/sg/k cases with equal perf
    # TODO add "heuristic" load tile selection: [16, 32]**2
    # TODO default heuristic: "best-equal", "heuristic" load and "best" prefetch
    #      for each wg/sg/k case, 4x load tiles and 1 prefetch tile
    # TODO add nb_prefetch strategy ?? [1, 2]**2 is 4 configs => 16x total
    # TODO check if prefetch layout actually affects the performance (same nb threads)
    # TODO add data type as variable, dpas tiles should be in gpu_specs ?

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
    """Expand the configs with different load tile options."""
    expanded_configs = []
    for params in param_list:
        if load_strategy == "all":
            load_a_list = generate_load_tiles_a()
            load_b_list = generate_load_tiles_b()
        elif load_strategy == "large":
            load_a_list = [(32, 32)]
            load_b_list = [(32, 32)]
        elif load_strategy == "double-rows":
            load_a_list = [(DPAS.A_TILE[0] * 2, DPAS.A_TILE[1])]
            load_b_list = [(DPAS.B_TILE[0] * 2, DPAS.B_TILE[1])]
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
    """Expand the configs with different prefetch depth options."""
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


if __name__ == "__main__":
    # M = 4096
    # N = 4096
    # K = 4096

    M = 128
    N = 8192
    K = 16384

    device = "B70"

    # TODO add data types as inputs

    gpu_specs = gpu_specs_db[device]

    print(f"Matrix multiply M={M}, N={N}, K={K} on {gpu_specs['name']}")

    # tile sizes
    # wg_tile = [256, 256]  # M, N
    # k_tile = 32
    # sg_tile = [64, 32]  # M, N
    # prefetch_tile_a = [16, 16]
    # prefetch_tile_b = [16, 32]

    # wg_tile = [128, 256]  # M, N
    # k_tile = 32
    # sg_tile = [64, 32]  # M, N
    # # prefetch_tile_a = [16, 16]
    # # prefetch_tile_a = [16, 16]
    # prefetch_tile_a = [8, 16]
    # prefetch_tile_b = [8, 32]
    # perf = estimate_perf(M, N, K, wg_tile, sg_tile, k_tile, gpu_specs, verbose=True,
    #                      prefetch_tile_a=prefetch_tile_a, prefetch_tile_b=prefetch_tile_b)
    # print(f"Estimated performance: {perf/1e12:.2f} TFLOPS")

    # # pf_a_list, pf_b_list = generate_prefetch_tiles(wg_tile, k_tile)
    # # print("Valid prefetch tiles for A:")
    # # for pf_a in pf_a_list:
    # #     print(pf_a)
    # # print("Valid prefetch tiles for B:")
    # # for pf_b in pf_b_list:
    # #     print(pf_b)
    # exit(0)

    configs = generate_configs(M, N, K, gpu_specs)

    print(f"Found {len(configs)} valid configurations.")

    nprint = 50
    nprint = min(nprint, len(configs))
    print(f"\nBest {nprint} configurations:")
    for perf, params in configs[:nprint]:
        wg_tile = (params["wg_m"], params["wg_n"])
        sg_tile = (params["sg_m"], params["sg_n"])
        k_tile = params["k_tile"]
        ld_a = (params["load_a_m"], params["load_a_k"])
        ld_b = (params["load_b_k"], params["load_b_n"])
        pf_a = (params["prefetch_a_m"], params["prefetch_a_k"])
        pf_b = (params["prefetch_b_k"], params["prefetch_b_n"])

        def to_str(iterable):
            return " ".join(str(x) for x in iterable)

        config = "  ".join(
            [
                f"WG: {to_str(wg_tile):7s}",
                f"SG: {to_str(sg_tile):5s}",
                f"K: {k_tile:2d}",
                f"LD_A: {to_str(ld_a):5s}",
                f"LD_B: {to_str(ld_b):5s}",
                f"PF_A: {to_str(pf_a):5s}",
                f"PF_B: {to_str(pf_b):5s}",
            ]
        )
        print(f"{config} => {perf / 1e12:.2f} TFLOPS")
