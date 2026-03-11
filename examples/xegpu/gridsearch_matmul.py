from time import perf_counter
import multiprocessing
from multiprocessing.sharedctypes import Value
from ctypes import c_double
from datetime import timedelta
from itertools import product
import numpy
import os
import sys
from csv_logger import CSVLogger

from mlir import ir

from lighthouse.workload import benchmark
from lighthouse.schedule.xegpu.mlp_schedule import DPAS_TILE

from matmul import XeGPUMatMul, cli_parser
from genetic_algorithm import (
    Variable,
    VariableSet,
)


def run_experiment(
    ab_type="f16",
    c_type="f32",
    nruns=None,
    nwarmup=None,
    check_result=False,
    has_bias=False,
    has_relu=False,
    accumulate_c=True,
    **params,
):
    M = params.pop("M")
    N = params.pop("N")
    K = params.pop("K")

    with ir.Context(), ir.Location.unknown():
        wload = XeGPUMatMul(
            M=M,
            N=N,
            K=K,
            ab_type=ab_type,
            c_type=c_type,
            has_bias=has_bias,
            has_relu=has_relu,
            accumulate_c=accumulate_c,
        )
        if nruns is None and nwarmup is None:
            # first run to estimate cost
            times = benchmark(
                wload,
                nruns=10,
                nwarmup=10,
                schedule_parameters=params,
                check_correctness=False,
                verbose=0,
            )
            # estimate number of runs
            cost = times.mean()
            warmup_target = 0.25
            nwarmup = max(int(warmup_target / cost), 10)
            nruns = 3 * nwarmup
            print(f"{nwarmup=} {nruns=}")
        # benchmark
        times = benchmark(
            wload,
            nruns=nruns,
            nwarmup=nwarmup,
            schedule_parameters=params,
            check_correctness=check_result,
            verbose=0,
        )

    times *= 1e6  # convert to microseconds
    elapsed = numpy.mean(times)
    flop_count = wload.get_complexity()[0]
    gflops = flop_count / (elapsed * 1e-6) / 1e9

    return elapsed, gflops


def run_with_timeout(*args, timeout=20, **kwargs):
    """
    Wrapper to execute the experiment with a new thread and a timeout.

    Experiments must be run in a new process to ensure reliable timings.

    Sends kill signal if timeout is reached.
    """
    # wrap return values
    timing = Value(c_double, 0.0)
    gflops = Value(c_double, 0.0)

    def wrapped(timing, gflops, *args, **kwargs):
        res = run_experiment(*args, **kwargs)
        timing.value = res[0]
        gflops.value = res[1]

    all_args = tuple([timing, gflops] + list(args))
    proc = multiprocessing.Process(target=wrapped, args=all_args, kwargs=kwargs)
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        print("TIMEOUT")
        proc.kill()
        proc.join()
        return 0, 0
    proc.close()
    return timing.value, gflops.value


def execute_and_log(
    csv_logger,
    nruns,
    nwarmup,
    check_result,
    params,
    ab_type,
    c_type,
    has_bias,
    has_relu,
    accumulate_c,
    timeout=20,
):
    try:
        tic = perf_counter()
        entry = params.copy()
        elapsed, gflops = run_with_timeout(
            ab_type=ab_type,
            c_type=c_type,
            nruns=nruns,
            nwarmup=nwarmup,
            check_result=check_result,
            timeout=timeout,
            has_bias=has_bias,
            has_relu=has_relu,
            accumulate_c=accumulate_c,
            **params,
        )
        duration = perf_counter() - tic
        entry["time (ms)"] = elapsed
        entry["GFLOPS/s"] = gflops
        csv_logger.log(entry)
        duration_str = f"Duration: {duration:.3f} s GFLOP/s: {gflops:.2f}"
        print(duration_str)
    except Exception as e:
        print("FAILED")
        print(entry)
        print(f"  Error: {e}")
    sys.stdout.flush()
    return elapsed, gflops


def counted(func):
    def wrapper(*args, **kwargs):
        wrapper.call_count += 1
        return func(*args, **kwargs)

    wrapper.call_count = 0
    return wrapper


@counted
def check_constraints(params, verbose=False):
    def print_reason(msg):
        if verbose:
            print(f"  Invalid: {msg}")

    # hardware constraints
    max_nb_sg_threads = 64
    load_max_rows = 32
    load_max_cols = 16
    pfetch_min_rows = 8
    pfetch_max_rows = 32
    pfetch_min_cols = 16
    pfetch_max_cols = 32

    # heuristics: skip likely suboptimal configurations
    min_nb_threads = 16

    M = params["M"]
    N = params["N"]
    wg_tile_m = params["wg_m"]
    wg_tile_n = params["wg_n"]
    sg_tile_m = params["sg_m"]
    sg_tile_n = params["sg_n"]
    load_tile_a_m = params["load_a_m"]
    load_tile_a_k = params["load_a_k"]
    load_tile_b_k = params["load_b_k"]
    load_tile_b_n = params["load_b_n"]
    prefetch_tile_a_m = params["pf_a_m"]
    prefetch_tile_a_k = params["pf_a_k"]
    prefetch_tile_b_k = params["pf_b_k"]
    prefetch_tile_b_n = params["pf_b_n"]
    k_tile = params["k"]

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
    if sg_tile_m % DPAS_TILE[0] != 0:
        print_reason("sg_tile_m not multiple of dpas_m")
        return False
    if sg_tile_n % DPAS_TILE[1] != 0:
        print_reason("sg_tile_n not multiple of dpas_n")
        return False
    if k_tile % DPAS_TILE[2] != 0:
        print_reason("k_tile not multiple of dpas_k")
        return False

    # SG level thread layout: [nb_sg_threads_m, nb_sg_threads_n]
    nb_sg_threads_m = wg_tile_m // sg_tile_m
    nb_sg_threads_n = wg_tile_n // sg_tile_n
    nb_sg_threads = nb_sg_threads_m * nb_sg_threads_n
    if nb_sg_threads > max_nb_sg_threads:
        print_reason("too many sg threads")
        return False
    if nb_sg_threads < min_nb_threads:
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
    if load_tile_a_m > load_max_rows:
        print_reason("too large load_tile_a_m")
        return False
    if load_tile_a_k > load_max_cols:
        print_reason("too large load_tile_a_k")
        return False
    if load_tile_b_k > load_max_rows:
        print_reason("too large load_tile_b_k")
        return False
    if load_tile_b_n > load_max_cols:
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
    if prefetch_tile_a_m > pfetch_max_rows:
        print_reason("too large prefetch_tile_a_m")
        return False
    if prefetch_tile_a_k > pfetch_max_cols:
        print_reason("too large prefetch_tile_a_k")
        return False
    if prefetch_tile_b_k > pfetch_max_rows:
        print_reason("too large prefetch_tile_b_k")
        return False
    if prefetch_tile_b_n > pfetch_max_cols:
        print_reason("too large prefetch_tile_b_n")
        return False
    if prefetch_tile_a_m < pfetch_min_rows:
        print_reason("too small prefetch_tile_a_m")
        return False
    if prefetch_tile_a_k < pfetch_min_cols:
        print_reason("too small prefetch_tile_a_k")
        return False
    if prefetch_tile_b_k < pfetch_min_rows:
        print_reason("too small prefetch_tile_b_k")
        return False
    if prefetch_tile_b_n < pfetch_min_cols:
        print_reason("too small prefetch_tile_b_n")
        return False
    if load_tile_a_m % DPAS_TILE[0] != 0:
        print_reason("load_tile_a_m not multiple of dpas_m")
        return False
    if load_tile_a_k % DPAS_TILE[2] != 0:
        print_reason("load_tile_a_k not multiple of dpas_k")
        return False
    if load_tile_b_k % DPAS_TILE[2] != 0:
        print_reason("load_tile_b_k not multiple of dpas_k")
        return False
    if load_tile_b_n % DPAS_TILE[1] != 0:
        print_reason("load_tile_b_n not multiple of dpas_n")
        return False

    nb_load_b_n = load_tile_b_n // DPAS_TILE[1]
    if nb_load_b_n > 1:
        # unsupported VNNI layout, loaded tile can only be row-sliced for vnni
        # NOTE this can plausibly be relaxed
        print_reason("invalid load_tile_b_n for VNNI")
        return False

    # prefetch A layout
    nb_prefetch_a_m = sg_tile_m // prefetch_tile_a_m
    nb_prefetch_a_k = k_tile // prefetch_tile_a_k
    if nb_prefetch_a_m * nb_prefetch_a_k > max_nb_sg_threads:
        print_reason("too many prefetch A tiles")
        return False
    if nb_prefetch_a_m * nb_prefetch_a_k < min_nb_threads:
        print_reason("too few prefetch A threads")
        return False

    # prefetch B layout
    nb_prefetch_b_k = k_tile // prefetch_tile_b_k
    nb_prefetch_b_n = sg_tile_n // prefetch_tile_b_n
    if nb_prefetch_b_k * nb_prefetch_b_n > max_nb_sg_threads:
        print_reason("too many prefetch B tiles")
        return False
    if nb_prefetch_b_k * nb_prefetch_b_n < min_nb_threads:
        print_reason("too few prefetch B threads")
        return False

    return True


def get_divisors(n, min_tile=32, max_tile=256):
    p = numpy.ceil(n / max_tile)
    q = n // min_tile
    candidates = n / numpy.arange(max(p, 1), q + 1)
    candidates = [int(v) for v in candidates if int(v) == v]
    return candidates[::-1]


def divisible_by(a_list, b):
    return [a for a in a_list if a % b == 0]


def construct_search_space(M, N, K):
    wg_tile_lim_m = min(max(M // 4, 16), 64), min(M, 256)
    wg_tile_lim_n = min(max(N // 4, 16), 64), min(N, 256)
    sg_tile_lim_m = min(max(M // 8, 16), 32), min(M, 128)
    sg_tile_lim_n = min(max(N // 8, 16), 32), min(N, 128)

    wg_tiles_m = divisible_by(get_divisors(M, *wg_tile_lim_m), DPAS_TILE[0])
    wg_tiles_n = divisible_by(get_divisors(N, *wg_tile_lim_n), DPAS_TILE[1])
    sg_tiles_m = divisible_by(get_divisors(M, *sg_tile_lim_m), DPAS_TILE[0])
    sg_tiles_n = divisible_by(get_divisors(N, *sg_tile_lim_n), DPAS_TILE[1])
    k_tiles = divisible_by(get_divisors(K, 16, min(K, 256)), DPAS_TILE[2])
    load_tiles = [8, 16, 32]
    prefetches = [1]

    def sample_is_valid(sample_params, verbose=False):
        params = {"M": M, "N": N, "K": K}
        params.update(sample_params)
        return check_constraints(params, verbose=verbose)

    var_set = VariableSet(
        [
            Variable("wg_m", wg_tiles_m),
            Variable("wg_n", wg_tiles_n),
            Variable("sg_m", sg_tiles_m),
            Variable("sg_n", sg_tiles_n),
            Variable("k", k_tiles),
            Variable("load_a_m", load_tiles),
            Variable("load_a_k", load_tiles),
            Variable("load_b_k", load_tiles),
            Variable("load_b_n", load_tiles),
            Variable("pf_a_m", load_tiles),
            Variable("pf_a_k", load_tiles),
            Variable("pf_b_k", load_tiles),
            Variable("pf_b_n", load_tiles),
            Variable("pf_nb", prefetches),
        ],
        is_valid_fn=sample_is_valid,
    )

    def sample_to_dict(sample: list) -> dict:
        res = {"M": M, "N": N, "K": K}
        res.update(var_set.sample_to_dict(sample))
        return res

    return var_set, sample_to_dict


if __name__ == "__main__":
    parser = cli_parser(
        description="Optimize matmul kernel parameters using a exhaustive search."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check validity of combinations but do not execute kernels.",
    )
    args = parser.parse_args()

    M, N, K = args.sizes
    has_bias = args.bias
    has_relu = args.relu
    accumulate_c = not args.no_accumulate_c
    ab_type = "f16"
    c_type = "f32"
    verbose = True
    check_result = args.check_result
    dry_run = args.dry_run

    nwarmup = None
    nruns = None
    timeout = 60

    # env
    os.environ["NEO_CACHE_PERSISTENT"] = "0"  # disable compiler cache

    if not dry_run:
        csv_file = "out_gridsearch.csv"
        csv_logger = CSVLogger(csv_file)

    var_set, sample_to_dict = construct_search_space(M, N, K)
    print(f"Matmul problem size: {M=} {N=} {K=}")
    print(f"{ab_type=}")
    print(f"{c_type=}")
    print(f"{has_bias=}")
    print(f"{has_relu=}")
    print(f"{accumulate_c=}")
    print(f"{nwarmup=}")
    print(f"{nruns=}")
    var_set.print()

    i = 0
    tic = perf_counter()
    for sample in product(*var_set.iterables()):
        params = sample_to_dict(sample)
        if not check_constraints(params, verbose=False):
            continue

        i += 1
        if dry_run:
            continue
        execute_and_log(
            csv_logger,
            nruns,
            nwarmup,
            check_result,
            params,
            timeout=timeout,
            ab_type=ab_type,
            c_type=c_type,
            has_bias=has_bias,
            has_relu=has_relu,
            accumulate_c=accumulate_c,
        )

    duration = perf_counter() - tic
    print(f"Number of executed configurations: {i}")
    print(f"Total duration: {timedelta(seconds=duration)}")
