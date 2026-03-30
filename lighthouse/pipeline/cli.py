import argparse


def opt_cli_parser(description: str = None) -> argparse.ArgumentParser:
    """Commanline arg parser for lh-opt tool."""
    if description is None:
        description = "Lighthouse Optimization Pipeline: Applies a series of transformations to an input MLIR module, and produces an optimized MLIR module as output. The transformations are applied in argument order. The names of the passes are registered by the driver."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "payload_module", type=str, help="Path to the payload MLIR module to optimize."
    )
    parser.add_argument(
        "--stage",
        action="append",
        required=True,
        help="List of transformations to apply to the input module.",
    )
    return parser
