__all__ = ["gpu_specs_db"]


# GPU specs

gpu_bmg_common = {
    "dpas_exec_size": 16,  # number of parallel dpas ops in sg
    "max_nb_threads": 32,  # large register file
    "nb_registers": 256,  # large register file
    # "max_nb_threads": 64  # small register file
    # "nb_registers": 128,  # small register file
}

# Intel Arc B70
gpu_specs_B70 = {
    "name": "Intel Arc B70",
    "nb_xe_cores": 32,
    "peak_flops": 155000e9,  # float16
    "bw_global_mem": 608e9,  # GB/s
    **gpu_bmg_common,
}

# Intel Arc B580
gpu_specs_B580 = {
    "name": "Intel Arc B580",
    "nb_xe_cores": 20,
    "peak_flops": 100000e9,  # float16
    "bw_global_mem": 456e9,  # GB/s
    **gpu_bmg_common,
}

gpu_specs_db = {
    "B70": gpu_specs_B70,
    "B580": gpu_specs_B580,
}
