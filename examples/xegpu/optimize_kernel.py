"""
Genetic algorithm-based optimization of kernel parameters.
"""

from functools import cache
import sys
from typing import Optional
import random
from matmul import cli_parser
from gridsearch_matmul import (
    check_constraints,
    construct_search_space,
)
from genetic_algorithm import (
    init_random_population,
    GeneticAlgorithm,
)
from gridsearch_matmul import execute_and_log
from csv_logger import CSVLogger

# count the number of executed kernels
nb_new_evaluations = 0


def optimize_kernel(
    sizes: list[int],
    has_bias: bool,
    has_relu: bool,
    accumulate_c: bool,
    ab_type: str = "f16",
    c_type: str = "f32",
    check_result: bool = True,
    random_seed: Optional[int] = None,
):
    if random_seed:
        # set random seed for reproducibility
        random.seed(random_seed)

    M, N, K = sizes
    timeout = 50

    nwarmup = None
    nruns = None

    # genetic algorithm parameters
    npop = 14
    ngenerations = 30
    recombination_rate = 0.5
    mutation_rate = 0.001
    fertility_rate = 1.0

    var_set, sample_to_dict = construct_search_space(M, N, K)
    print(f"Matmul problem size: {sizes}")
    print(f"{ab_type=}")
    print(f"{c_type=}")
    print(f"{has_bias=}")
    print(f"{has_relu=}")
    print(f"{accumulate_c=}")
    print(f"{nwarmup=}")
    print(f"{nruns=}")
    var_set.print()
    sys.stdout.flush()

    csv_file = "out_genetic_algorithm.csv"
    csv_logger = CSVLogger(csv_file)

    global nb_new_evaluations
    nb_new_evaluations = 0

    perf_cache = {}

    @cache
    def evaluate_fitness(*parameters) -> float:
        global nb_new_evaluations
        nb_new_evaluations += 1
        elapsed, gflops = execute_and_log(
            csv_logger,
            nruns,
            nwarmup,
            check_result,
            sample_to_dict(parameters),
            timeout=timeout,
            ab_type=ab_type,
            c_type=c_type,
            has_bias=has_bias,
            has_relu=has_relu,
            accumulate_c=accumulate_c,
        )
        perf_cache[tuple(parameters)] = elapsed, gflops
        return gflops

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

    print("Best configurations found:")
    for params in pop.individuals:
        time, gflops = perf_cache.get(tuple(params), (0.0, 0.0))
        print(f" Time: {time:.2f} us, GFLOPS: {gflops:.2f}: {params}")
    print(f"Number of constraint checks: {check_constraints.call_count}")
    print(f"\nNumber of kernel evaluations: {nb_new_evaluations}")


if __name__ == "__main__":
    parser = cli_parser(
        description="Optimize matmul kernel parameters using a genetic algorithm."
    )
    args = parser.parse_args()

    optimize_kernel(
        args.sizes,
        args.bias,
        args.relu,
        not args.no_accumulate_c,
        check_result=args.check_result,
        random_seed=2,
    )
