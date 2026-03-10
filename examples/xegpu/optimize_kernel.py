"""
Genetic algorithm-based optimization of kernel parameters.
"""

import argparse
from functools import cache
import sys
import numpy as np
from gridsearch_matmul import (
    get_divisors,
    divisible_by,
    check_constraints,
)
from genetic_algorithm import (
    Variable,
    VariableSet,
    init_random_population,
    GeneticAlgorithm,
    load_experiment_data,
)
from gridsearch_matmul import execute_kernel
from gridsearch_matmul import CSVLogger

nb_new_evaluations = 0


def optimize_kernel(
    sizes: list[int],
    has_bias: bool,
    has_relu: bool,
    accumulate_c: bool,
    ab_type: str,
    c_type: str,
    dry_run: bool = False,
    check_result: bool = True,
):
    # set random seed for reproducibility
    seed = 2
    np.random.seed(seed)

    M, N, K = sizes
    timeout = 50
    dpas_tile = [8, 16, 16]

    # Estimate required number of iterations
    # NOTE For large problems compile time can also be significant, > 1 s
    complexity = 2 * M * N * K  # floating point ops
    throughput = 40e12  # typical value, flops
    time_estimate = complexity / throughput  # seconds
    duration = 0.8  # desired warm-up duration in seconds
    iterations = int(max(50, int(duration / time_estimate)))
    nwarmup = iterations
    nruns = int(1.5 * iterations)

    wg_tiles_m = divisible_by(get_divisors(M, 16, 256), dpas_tile[0])
    wg_tiles_n = divisible_by(get_divisors(N, 64, 256), dpas_tile[1])
    sg_tiles_m = divisible_by(get_divisors(M, 16, 128), dpas_tile[0])
    sg_tiles_n = divisible_by(get_divisors(N, 32, 128), dpas_tile[1])
    k_tiles = divisible_by(get_divisors(K, 16, 256), dpas_tile[2])
    load_tiles = [8, 16, 32]
    nb_prefetch = [1]

    # genetic algorithm parameters
    npop = 14
    ngenerations = 30
    recombination_rate = 0.5
    mutation_rate = 0.001
    fertility_rate = 1.0

    print(f"Matmul problem size: {sizes}")
    print(f"{ab_type=}")
    print(f"{c_type=}")
    print(f"{has_bias=}")
    print(f"{has_relu=}")
    print(f"{accumulate_c=}")
    print(f"{wg_tiles_m=}")
    print(f"{wg_tiles_n=}")
    print(f"{sg_tiles_m=}")
    print(f"{sg_tiles_n=}")
    print(f"{k_tiles=}")
    print(f"{load_tiles=}")
    print(f"{nb_prefetch=}")

    print(f"{nwarmup=}")
    print(f"{nruns=}")

    wg_m = Variable("wg_m", wg_tiles_m)
    wg_n = Variable("wg_n", wg_tiles_n)
    sg_m = Variable("sg_m", sg_tiles_m)
    sg_n = Variable("sg_n", sg_tiles_n)
    k_tile = Variable("k", k_tiles)
    load_tile_a_m = Variable("load_a_m", load_tiles)
    load_tile_a_k = Variable("load_a_k", load_tiles)
    load_tile_b_k = Variable("load_b_k", load_tiles)
    load_tile_b_n = Variable("load_b_n", load_tiles)
    prefetch_tile_a_m = Variable("pf_a_m", load_tiles)
    prefetch_tile_a_k = Variable("pf_a_k", load_tiles)
    prefetch_tile_b_k = Variable("pf_b_k", load_tiles)
    prefetch_tile_b_n = Variable("pf_b_n", load_tiles)
    nb_prefetch = Variable("pf_nb", nb_prefetch)

    def params_to_dict(sample: list) -> dict:
        return {
            "M": M,
            "N": N,
            "K": K,
            "wg_m": sample[0],
            "wg_n": sample[1],
            "sg_m": sample[2],
            "sg_n": sample[3],
            "k": sample[4],
            "load_a_m": sample[5],
            "load_a_k": sample[6],
            "load_b_k": sample[7],
            "load_b_n": sample[8],
            "pf_a_m": sample[9],
            "pf_a_k": sample[10],
            "pf_b_k": sample[11],
            "pf_b_n": sample[12],
            "pf_nb": sample[13],
        }

    def is_valid_fn(sample: list) -> bool:
        params = params_to_dict(sample)
        return check_constraints(params, verbose=False)

    var_set = VariableSet(
        [
            wg_m,
            wg_n,
            sg_m,
            sg_n,
            k_tile,
            load_tile_a_m,
            load_tile_a_k,
            load_tile_b_k,
            load_tile_b_n,
            prefetch_tile_a_m,
            prefetch_tile_a_k,
            prefetch_tile_b_k,
            prefetch_tile_b_n,
            nb_prefetch,
        ],
        is_valid_fn=is_valid_fn,
    )
    print(f"Total search space size: {var_set.complexity()}")
    sys.stdout.flush()

    if dry_run:
        # load experiment data from csv file
        csv_file = "out_gridsearch.csv"
        experiment_data = load_experiment_data(
            csv_file,
            var_set,
            cost_param="time (ms)",
        )

        all_times = np.array(list(experiment_data.values()))
        best_exp = np.argmin(all_times)
        print(f"Best recorded time: {all_times[best_exp]:.6f} ms")
        print(f"Best configuration: {list(list(experiment_data.keys())[best_exp])}")
    else:
        csv_file = "out_genetic_algorithm.csv"
        csv_logger = CSVLogger(csv_file)

    gflops_cache = {}
    global nb_new_evaluations
    nb_new_evaluations = 0

    @cache
    def evaluate_fitness(*parameters) -> float:
        global nb_new_evaluations
        nb_new_evaluations += 1
        if dry_run:
            key = tuple(parameters)
            if key in experiment_data:
                return experiment_data[key]
            else:
                return float("inf")
        else:
            elapsed, gflops = execute_kernel(
                csv_logger,
                nruns,
                nwarmup,
                check_result,
                params_to_dict(parameters),
                timeout=timeout,
                ab_type=ab_type,
                c_type=c_type,
                has_bias=has_bias,
                has_relu=has_relu,
                accumulate_c=accumulate_c,
            )
            gflops_cache[tuple(parameters)] = gflops
            return elapsed if elapsed != 0.0 else float("inf")

    pop = init_random_population(npop, var_set)
    ga_optimizer = GeneticAlgorithm(
        population=pop,
        recombination_rate=recombination_rate,
        mutation_rate=mutation_rate,
        fertility_rate=fertility_rate,
        evaluate_fitness=evaluate_fitness,
    )

    ga_optimizer.initialize()
    pop.print()

    ga_optimizer.optimize(ngen=ngenerations, verbose=1)
    pop.sort()

    print("Best configurations found:")
    for params, time in zip(pop.individuals, pop.fitness_scores):
        gflops = gflops_cache.get(tuple(params), 0.0)
        print(f" Time: {time:.2f} us, GFLOPS: {gflops:.2f}: {params}")
    print(f"\nNumber of cost function evaluations: {nb_new_evaluations}")
    print(f"Number of constraint checks: {check_constraints.call_count}")


def parse_cli():
    parser = argparse.ArgumentParser(
        description="Optimize matmul kernel parameters using a genetic algorithm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs=3,
        default=[4096, 4096, 4096],
        help="M,N,K matrix sizes (A=MxK, B=KxN, C=MxN).",
    )
    parser.add_argument(
        "--bias",
        action="store_true",
        help="Add bias op after the matrix multiplication.",
    )
    parser.add_argument(
        "--relu",
        action="store_true",
        help="Add relu op after the matrix multiplication (and bias if any).",
    )
    parser.add_argument(
        "--no-accumulate-c",
        action="store_true",
        help="Compute plain matrix-multiply C=A*B instead of matrix-multiply-accumulate C+=A*B.",
    )
    parser.add_argument(
        "--check-result",
        action="store_true",
        help="Check the result of the matrix multiplication.",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_cli()

    ab_type = "f16"
    c_type = "f32"
    check_result = True

    # do not execute kernels, look up timings from experiment data
    dry_run = False

    optimize_kernel(
        args.sizes,
        args.bias,
        args.relu,
        not args.no_accumulate_c,
        ab_type,
        c_type,
        dry_run=dry_run,
        check_result=check_result,
    )
