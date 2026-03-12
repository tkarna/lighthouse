"""
Genetic algorithm-based optimization of kernel parameters.
"""

from functools import cache
import sys
import os
import json
from typing import Optional
import random
from matmul import cli_parser
from tune_matmul_gridsearch import (
    construct_search_space,
    execute_and_log,
)
from genetic_algorithm import (
    init_random_population,
    GeneticAlgorithm,
)
from csv_logger import CSVLogger


def optimize_kernel(
    sizes: list[int],
    has_bias: bool,
    has_relu: bool,
    accumulate_c: bool,
    ab_type: str = "f16",
    c_type: str = "f32",
    check_result: bool = True,
    npopulation: int = 14,
    ngenerations: int = 30,
    mutation_rate: float = 0.001,
    dump_json: int = 0,
    random_seed: Optional[int] = None,
):
    if random_seed:
        # set random seed for reproducibility
        random.seed(random_seed)

    # timeout for kernel execution in seconds
    timeout = 50

    # number of iterations in kernel timing is chosen adaptively
    nwarmup = None
    nruns = None

    # disable IGC compiler cache
    os.environ["NEO_CACHE_PERSISTENT"] = "0"

    var_set, sample_to_dict = construct_search_space(*sizes)
    print(f"Matmul problem size: {sizes}")
    print(f"{ab_type=}")
    print(f"{c_type=}")
    print(f"{has_bias=}")
    print(f"{has_relu=}")
    print(f"{accumulate_c=}")
    var_set.print()
    sys.stdout.flush()

    csv_file = "out_genetic_algorithm.csv"
    csv_logger = CSVLogger(csv_file)

    @cache
    def evaluate_fitness(*parameters) -> float:
        elapsed, gflops = execute_and_log(
            csv_logger,
            nruns,
            nwarmup,
            sample_to_dict(parameters),
            check_result,
            timeout=timeout,
            ab_type=ab_type,
            c_type=c_type,
            has_bias=has_bias,
            has_relu=has_relu,
            accumulate_c=accumulate_c,
        )
        return gflops

    pop = init_random_population(npopulation, var_set)
    ga_optimizer = GeneticAlgorithm(
        population=pop,
        mutation_rate=mutation_rate,
        evaluate_fitness=evaluate_fitness,
    )

    ga_optimizer.initialize()
    pop.print()
    ga_optimizer.optimize(ngen=ngenerations, verbose=1)

    nb_kernel_evals = evaluate_fitness.cache_info().currsize
    print("Best configurations found:")
    for params, gflops in zip(pop.individuals, pop.fitness_scores):
        print(f" GFLOPS: {gflops:.2f}: {params}")
    print(f"Number of kernel evaluations: {nb_kernel_evals}")

    if dump_json > 0:
        print("")
        for i in range(min(dump_json, len(pop.individuals))):
            filename = f"matmul_params_{i:02d}.json"
            with open(filename, "w") as f:
                json.dump(sample_to_dict(pop.individuals[i]), f, indent=4)
            print(f"Saved config to {filename}")


if __name__ == "__main__":
    parser = cli_parser(
        description="Optimize matmul kernel parameters using a genetic algorithm."
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=30,
        help="Number of generations for the genetic algorithm.",
    )
    parser.add_argument(
        "--dump-json",
        dest="n_dump_json",
        type=int,
        default=0,
        help="Dump the best n configurations as JSON files.",
    )

    args = parser.parse_args()

    optimize_kernel(
        args.sizes,
        args.bias,
        args.relu,
        not args.no_accumulate_c,
        check_result=True,
        ngenerations=args.generations,
        dump_json=args.n_dump_json,
        random_seed=2,
    )
