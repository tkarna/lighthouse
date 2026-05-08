# TODO add lit tests or skip

from time import perf_counter
from datetime import timedelta
import os
import sys
from csv_logger import CSVLogger

from matmul import cli_parser
from tune_utils import dump_configs_json, execute_and_log
from tilesize_selector import (
    generate_configs,
    gpu_specs_db,
    expand_configs_with_load_tiles,
    expand_configs_with_prefetch_depth,
)
from tune_matmul_gridsearch import check_constraints, run_experiment


if __name__ == "__main__":
    parser = cli_parser(
        description="Optimize matmul kernel parameters using a cost model search."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check validity of combinations but do not execute kernels.",
    )
    parser.add_argument(
        "--target",
        choices=["B70", "B580"],
        default="B70",
        help="Target GPU device.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        help="Maximum number of executed configurations.",
    )
    parser.add_argument(
        "--no-check-result",
        action="store_true",
        help="Skip correctness check.",
    )
    parser.add_argument(
        "--dump-json",
        dest="n_dump_json",
        type=int,
        default=0,
        help="Dump the best n configurations as JSON files.",
    )
    args = parser.parse_args()

    sizes = args.sizes
    has_bias = args.bias
    has_relu = args.relu
    accumulate_c = not args.no_accumulate_c
    ab_type = "f16"
    c_type = "f32"

    # timeout for kernel execution in seconds
    timeout = 50

    # number of iterations in kernel timing is chosen adaptively
    nwarmup = None
    nruns = None

    # disable IGC compiler cache
    os.environ["NEO_CACHE_PERSISTENT"] = "0"

    if not args.dry_run:
        csv_file = "out_costmodel.csv"
        csv_logger = CSVLogger(csv_file)

    gpu_specs = gpu_specs_db[args.target]

    print(f"Matmul problem size: {sizes}")
    print(f"device={gpu_specs['name']}")
    print(f"{ab_type=}")
    print(f"{c_type=}")
    print(f"{has_bias=}")
    print(f"{has_relu=}")
    print(f"{accumulate_c=}")
    sys.stdout.flush()

    load_strategy = "dpas"
    prefetch_strategy = "all"
    perf_threshold = 0.8  # skip config if perf_estimate < th * best_perf_estimate
    max_nb_configs = None
    nb_select_load_tune = 1  # number of top configs to select for load tile tuning
    nb_select_pfnb_tune = 4  # number of top configs to select for prefetch depth

    print(f"{load_strategy=}")
    print(f"{prefetch_strategy=}")
    configs = generate_configs(
        *sizes,
        gpu_specs,
        perf_threshold=perf_threshold,
        load_strategy=load_strategy,
        pf_strategy=prefetch_strategy,
        max_nb_configs=max_nb_configs,
    )
    configs = [params for _, params in configs]

    def eval_configs(configs):
        print(f"Total complexity: {len(configs)} configurations")
        i = 0
        executed_configs = []
        tic = perf_counter()
        for params in configs:
            if not check_constraints(params, verbose=True):
                print(f"Skipping invalid configuration: {params}")
                continue

            i += 1
            if args.max_iters is not None and i >= args.max_iters:
                print(f"Reached maximum number of iterations: {args.max_iters}")
                break
            if args.dry_run:
                continue
            time, gflops = execute_and_log(
                run_experiment,
                csv_logger,
                nruns,
                nwarmup,
                params,
                check_result=not args.no_check_result,
                timeout=timeout,
                ab_type=ab_type,
                c_type=c_type,
                has_bias=has_bias,
                has_relu=has_relu,
                accumulate_c=accumulate_c,
            )
            executed_configs.append((gflops, params))

        duration = perf_counter() - tic

        return executed_configs, duration, i

    executed_configs, time, iters = eval_configs(configs)

    if nb_select_load_tune is not None and nb_select_load_tune > 0:
        time1 = time
        iters1 = iters

        print(f"Time spent in tuning: {timedelta(seconds=time1)}")
        print(f"Number of executed configurations: {iters1}")

        # Summary of first phase
        n_print = 10
        executed_configs.sort(key=lambda x: x[0], reverse=True)
        best_configs = [c for c in executed_configs[:n_print]]
        print("Best configurations found in first tuning phase:")
        for gflops, params in best_configs:
            print(f" GFLOPS: {gflops:.2f}: {list(params.values())}")

        # take n best configs and tune load tile sizes
        print(f"Tuning load tiles for best {nb_select_load_tune} configurations")
        configs = [c[1] for c in executed_configs[:nb_select_load_tune]]
        new_configs = expand_configs_with_load_tiles(
            configs, load_strategy="all", exclude_duplicates=True
        )

        executed_configs2, time2, iters2 = eval_configs(new_configs)
        executed_configs.extend(executed_configs2)

        print(f"Time spent in tuning: {timedelta(seconds=time2)}")
        print(f"Number of executed configurations: {iters2}")

        # Summary of second phase
        executed_configs.sort(key=lambda x: x[0], reverse=True)
        best_configs = [c for c in executed_configs[:n_print]]
        print("Best configurations found after load tile tuning:")
        for gflops, params in best_configs:
            print(f" GFLOPS: {gflops:.2f}: {list(params.values())}")

        time = time1 + time2
        iters = iters1 + iters2

    if nb_select_pfnb_tune is not None and nb_select_pfnb_tune > 0:
        time1 = time
        iters1 = iters

        # take n best configs and tune prefetch depth
        print(f"Tuning prefetch depth for best {nb_select_pfnb_tune} configurations")
        configs = [c[1] for c in executed_configs[:nb_select_pfnb_tune]]
        new_configs = expand_configs_with_prefetch_depth(
            configs, max_depth=2, exclude_duplicates=True
        )

        executed_configs2, time2, iters2 = eval_configs(new_configs)
        executed_configs.extend(executed_configs2)

        print(f"Time spent in tuning: {timedelta(seconds=time2)}")
        print(f"Number of executed configurations: {iters2}")

        # Summary of second phase
        executed_configs.sort(key=lambda x: x[0], reverse=True)
        best_configs = [c for c in executed_configs[:n_print]]
        print("Best configurations found after prefetch depth tuning:")
        for gflops, params in best_configs:
            print(f" GFLOPS: {gflops:.2f}: {list(params.values())}")

        time = time1 + time2
        iters = iters1 + iters2

    print(f"Total duration: {timedelta(seconds=time)}")
    print(f"Number of executed configurations: {iters}")

    if args.n_dump_json > 0 and not args.dry_run:
        best_configs = [c for c in executed_configs[: args.n_dump_json]]
        print("Best configurations found:")
        for gflops, params in best_configs:
            print(f" GFLOPS: {gflops:.2f}: {list(params.values())}")
        sizes_str = "-".join(str(s) for s in sizes)
        relu_str = "_relu" if has_relu else ""
        bias_str = "_bias" if has_bias else ""
        acc_str = "_acc" if accumulate_c else ""
        prefix = (
            f"matmul_params_{sizes_str}_{ab_type}-{c_type}{bias_str}{relu_str}{acc_str}"
        )
        dump_configs_json([p for _, p in best_configs], filename_prefix=prefix)
