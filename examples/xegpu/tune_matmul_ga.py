"""
Genetic algorithm-based optimization of kernel parameters.
"""

from functools import cache
import sys
import os
from typing import Optional
import random
from matmul import cli_parser
from tune_matmul_gridsearch import construct_search_space, run_experiment
from tune_utils import dump_configs_json, execute_and_log

from genetic_algorithm import (
    init_random_population,
    GeneticAlgorithm,
    Population,
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
    init_candidates: Optional[list[dict]] = None,
    random_seed: Optional[int] = None,
):
    if random_seed is not None:
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
            run_experiment,
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

    pop = Population(variable_set=var_set)
    if init_candidates is not None:
        # add given initial candidates to the population
        for candidate in init_candidates:
            params_tuple = list(candidate[param.name] for param in var_set.variables)
            if not var_set.is_valid(params_tuple):
                raise ValueError(f"Invalid initial candidate: {candidate}")
            if params_tuple not in pop.individuals:
                pop.individuals.append(params_tuple)
    pop = init_random_population(npopulation, var_set, population=pop)

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
        configs = [sample_to_dict(p) for p in pop.individuals[:dump_json]]
        sizes_str = "-".join(str(s) for s in sizes)
        relu_str = "_relu" if has_relu else ""
        bias_str = "_bias" if has_bias else ""
        acc_str = "_acc" if accumulate_c else ""
        prefix = (
            f"matmul_params_{sizes_str}_{ab_type}-{c_type}{bias_str}{relu_str}{acc_str}"
        )
        dump_configs_json(configs, filename_prefix=prefix)


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
        "--population-size",
        type=int,
        default=11,
        help="Number of individuals in the population for the genetic algorithm.",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.01,
        help="Mutation rate for the genetic algorithm.",
    )
    parser.add_argument(
        "--dump-json",
        dest="n_dump_json",
        type=int,
        default=0,
        help="Dump the best n configurations as JSON files.",
    )
    parser.add_argument(
        "--no-check-result",
        action="store_true",
        help="Skip correctness check.",
    )

    args = parser.parse_args()

    # Always include these configurations in the initial population
    init_candidates = [
        {
            "wg_m": 256,
            "wg_n": 256,
            "sg_m": 64,
            "sg_n": 32,
            "k_tile": 32,
            "load_a_m": 32,
            "load_a_k": 32,
            "load_b_k": 32,
            "load_b_n": 32,
            "prefetch_a_m": 16,
            "prefetch_a_k": 16,
            "prefetch_b_k": 16,
            "prefetch_b_n": 32,
            "prefetch_a_nb": 1,
            "prefetch_b_nb": 2,
        },
        {
            "wg_m": 256,
            "wg_n": 256,
            "sg_m": 32,
            "sg_n": 64,
            "k_tile": 32,
            "load_a_m": 32,
            "load_a_k": 32,
            "load_b_k": 32,
            "load_b_n": 32,
            "prefetch_a_m": 16,
            "prefetch_a_k": 32,
            "prefetch_b_k": 16,
            "prefetch_b_n": 16,
            "prefetch_a_nb": 1,
            "prefetch_b_nb": 1,
        },
        {
            "wg_m": 128,
            "wg_n": 128,
            "sg_m": 32,
            "sg_n": 32,
            "k_tile": 32,
            "load_a_m": 16,
            "load_a_k": 16,
            "load_b_k": 16,
            "load_b_n": 16,
            "prefetch_a_m": 16,
            "prefetch_a_k": 16,
            "prefetch_b_k": 16,
            "prefetch_b_n": 16,
            "prefetch_a_nb": 1,
            "prefetch_b_nb": 1,
        },
    ]

    optimize_kernel(
        args.sizes,
        args.bias,
        args.relu,
        not args.no_accumulate_c,
        check_result=not args.no_check_result,
        ngenerations=args.generations,
        mutation_rate=args.mutation_rate,
        npopulation=args.population_size,
        dump_json=args.n_dump_json,
        init_candidates=init_candidates,
        random_seed=2,
    )
