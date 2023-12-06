# training entry point for both diffusion and the VAE
import argparse
import logging
import os
import sys
import torch
import time
import yaml
import wandb

from pathlib import Path
from typing import Optional, Sequence, Mapping, Any
from distutils.util import strtobool

from common.utils import (
    create_instance,
    parse_spec_overrides,
    print_spec,
    setup_logging
)

logger = logging.getLogger(__name__)


def parse_command_line(
    args: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    """Parses command-line flags passed to the training script.

    Args:
        args: A sequence of strings used as command-line arguments.
            If None, sys.argv will be used.

    Returns:
        A namespace with members for all parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-e",
        "--experiment_spec_file",
        required=True,
        type=Path,
        help="Path to training spec.",
    )
    parser.add_argument(
        "-r",
        "--results_dir",
        required=True,
        type=Path,
        help="Path to a folder under which experiment results will be created. ",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Flag to debug the pipeline. "
        "In this case, num_workers is set to 0 and multiprocessing_context is None.",
    )
    parser.add_argument(
        "-o",
        "--spec_overrides",
        action='append',
        default=None,
        help="Parses spec settings to override ones in the given spec file. "
        "Must start with spec as prefix, for example: spec.trainer.config.batch_size=8.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resumes training either from latest available checkpoint or provided file. "
        "If provided path is a directory then tries to load the latest (last) checkpoint.",
    )
    parser.add_argument(
        "--use_wandb",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Enable wandb experiment tracking"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="policy_diffusion"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="default_run" + time.strftime("_%Y%m%d_%H%M%S")
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="The group to which this run belongs to. Runs can be grouped together"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default='qdrl',
        help="The team/owner of this project"
    )
    parser.add_argument(
        "--wandb_tag",
        type=str,
        default='',
        help="Tags to help identify this run. Runs can be later filtered by tags"
    )

    return parser.parse_args(args)


def parse_spec_file(args: argparse.Namespace) -> Mapping[str, Any]:
    """Parses and post-processes spec file.

    Args:
        args: Parsed command-line args.

    Returns:
        A processed spec file, populated with overrides.
    """
    with open(args.experiment_spec_file, "r", encoding="utf-8") as src:
        spec = yaml.safe_load(src)

    if args.spec_overrides is not None:
        spec = parse_spec_overrides(spec, args.spec_overrides)

    name = spec['name']
    seed = spec['trainer']['config']['random_seed']

    # Use the results directory as a checkpoint directory for the trainer.
    results_dir = Path(os.path.expandvars(args.results_dir))
    results_dir.mkdir(exist_ok=True)

    if args.debug:
        exp_dir = results_dir.joinpath('debug')
        exp_dir.mkdir(exist_ok=True)
    else:
        exp_dir = results_dir.joinpath(name + '_' + str(seed))
        try:
            exp_dir.mkdir(exist_ok=False)
        except:
            raise FileExistsError(f'Experiment dir {exp_dir} already exists.'
                                  f' Please rename your experiment or delete the old one')

    spec["trainer"]["config"]["exp_dir"] = str(exp_dir)

    # Set training mode.
    spec["trainer"]["config"]["debug"] = args.debug

    # Set resume path, if provided.
    spec["trainer"]["config"]["resume"] = args.resume

    # wandb
    spec['trainer']['config']['use_wandb'] = args.use_wandb

    # set device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    spec['trainer']['config']['device'] = device

    return spec


def _setup_logging(spec: Mapping[str, Any]) -> None:
    """Sets up logging."""

    log_dir = Path(spec["trainer"]["config"]["exp_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "output_log_0.txt"

    debug_mode = spec["trainer"]["config"]["debug"]
    setup_logging(
        log_level=spec.get("log_level", logging.DEBUG if debug_mode else logging.INFO),
        log_file=str(log_file),
    )


def _setup_wandb(args: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    """Sets up wandb experiment tracking if enabled"""
    tag = args['wandb_tag']
    if tag == '':
        tag = None
    wandb.init(
        project=args['wandb_project'],
        entity=args['wandb_entity'],
        group=args['wandb_group'],
        name=args['wandb_run_name'],
        tags=tag,
        config=spec
    )


def main(cl_args: Optional[Sequence[str]] = None) -> None:
    """Runs the training."""
    args = parse_command_line(cl_args)

    spec = parse_spec_file(args)

    _setup_logging(spec)

    args = vars(args)
    if args['use_wandb']:
        _setup_wandb(args, spec)

    torch.multiprocessing.set_start_method("spawn")

    # Log command line, some system stats and final spec.
    logger.info(" ".join(sys.argv))
    ver_str = (
        f"PyTorch: {torch.__version__}, "
        f"CUDA: {torch.version.cuda}, "
        "Python: %s" % sys.version.replace("\n", " ")
    )
    logger.info(ver_str)
    print_spec(spec, 0)

    # Initialize the trainer object from the spec file.
    trainer = create_instance(
        spec["trainer"]["module"],
        spec["trainer"]["class"],
        **spec["trainer"]["config"],
        name=spec["name"],
    )

    # Record final spec after overrides.
    log_dir = Path(spec["trainer"]["config"]["exp_dir"])
    with open(log_dir / "experiment_spec_final.yaml", "w", encoding="utf-8") as w:
        yaml.safe_dump(spec, w)

    trainer.build(spec)
    trainer.train()

    logger.info("All done.")


if __name__ == "__main__":
    main()
