import importlib
import logging
import json
import torch
import wandb
import os
import numpy as np

from attrdict import AttrDict
from colorlog import ColoredFormatter
from typing import Mapping, Any

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", level=logging.INFO
)
logger = logging.getLogger('utils')


def config_wandb(**kwargs):
    # wandb initialization
    wandb.init(project=kwargs['wandb_project'], entity=kwargs['entity'], \
               group=kwargs['wandb_group'], name=kwargs['run_name'], \
               tags=[kwargs['tags']])
    cfg = kwargs.get('cfg', None)
    if cfg is None:
        cfg = {}
        for key, val in kwargs.items():
            cfg[key] = val
    wandb.config.update(cfg)


def save_cfg(dir, cfg):
    def to_dict(cfg):
        if isinstance(cfg, AttrDict):
            cfg = dict(cfg)
    filename = 'cfg.json'
    fp = os.path.join(dir, filename)
    with open(fp, 'w') as f:
        json.dump(cfg, f, default=to_dict, indent=4)


def create_class(module_name: str, class_name: str) -> Any:
    """Creates the specified class object from a module through importlib.

    Args:
        module_name: name of Python module to import from
        class_name: name of Python class to import
    Returns:
        A class object. Raises an exception if class/module is not found.
    """
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def create_instance(module_name: str, class_name: str, *args, **kwargs) -> Any:
    """Creates the specified instance of a class from a module through importlib.

    Args:
        module_name: name of Python module to import from
        class_name: name of Python class to import
        args: positional arguments passed to the class.
        kwargs: keyword arguments passed to the class.
    Returns:
        An instance of the class. Raises an exception if class/module is not found.
    """
    class_obj = create_class(module_name, class_name)

    return class_obj(*args, **kwargs)


def create_instance_from_spec(spec: Mapping[str, Any], *args, **kwargs) -> Any:
    """Creates the instance of a class from a provided spec.

    Args:
        spec: mapping that contains module, class name and config entries.
        args: positional arguments passed to the class.
        kwargs: keyword arguments passed to the class.
    Returns:
        An instance of the class.
    """
    return create_instance(
        spec["module"],
        spec["class"],
        *args,
        **spec["config"],
        **kwargs,
    )


def setup_logging(log_level: int = logging.INFO, log_file: str = None):
    """Sets up logging for scripts."""
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    formatter = ColoredFormatter(
        "%(log_color)s[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'white,bold',
            'INFOV': 'cyan,bold',
            'WARNING': 'yellow',
            'ERROR': 'red,bold',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
    ch.setFormatter(formatter)

    # configure streaming to logfile
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)

    # configure the console stream
    logging.root.addHandler(ch)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple):
    '''
    Extract the appropriate t-index for a batch of indices
    :param a: tensor from which we wish to extract certain timesteps
    :param t: timestep (indices) into a
    :param x_shape: shape of the batch of inputs x_t.
    '''
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def get_class_from_module(module_name, class_name):
    """Get the specified class object from a module through importlib.

    Args:
        module_name (string) : Path of the module
        class_name (string)  : Class to search for

    Returns the class if it is found or raises an exception if the module or
    class are not found.
    """
    # Initialize the manager from the given module.
    module = importlib.import_module(module_name)
    if not hasattr(module, class_name):
        raise RuntimeError("Class {} not found in {}".format(class_name, module_name))
    return getattr(module, class_name)


def get_inst_from_module(module_name, class_name, args):
    """Creates an instance of a class from specified module.

    Args:
        module_name (string) : Path of the module
        class_name (string)  : Class to search for
        args (dict)          : Constructor arguments

    Returns instance of class or raises an exception.
    """
    c = get_class_from_module(module_name, class_name)
    return c(**args)


def parse_spec_overrides(spec, other_args, allow_new_keys=False):
    """Handles spec file overrides.

    Parses optional spec file overrides and updates provided spec container.
    Spec overrides are passed as command line arguments with 'spec.' prefix .
    This must be done before any post-processing happens so the user can expect
    consistent behavior as if the spec file was changed by the user before running any code.

    Args:
        spec: spec dictionary.
        other_args: list of extra arguments returned by argparse.parse_known_args.
        allow_new_keys: allow sections and keys that don't exist in `spec`, note that new value
            literals are parsed by JSON lexer.

    Returns updated spec. Note that the spec is updated in-place.
    """

    # 1. Convert other_args to dictionary.
    # This is a very primitive parser, be gentle with it...
    spec_overrides = {}
    for a in other_args:
        items = a.split("=")
        if len(items) != 2:
            raise ValueError(
                "Invalid command line argument: {}. Expected key=value".format(a)
            )
        k, v = items
        if not k.startswith("spec."):
            raise ValueError(
                'Spec override key must start with "spec." but got: {}'.format(k)
            )
        k = k[len("spec.") :].strip()
        spec_overrides[k] = v.strip()

    # 2. Handle spec overrides, if any.
    for k, v in spec_overrides.items():
        section_keys = k.split(".")
        section = spec
        try:
            for skey in section_keys[:-1]:
                if skey not in section and allow_new_keys:
                    section[skey] = {}
                section = section[skey]
            if section_keys[-1] not in section and allow_new_keys:
                # Use JSON lexer to determine the type and value of the new key-value literal.
                try:
                    new_val = json.loads(v)
                except json.JSONDecodeError:
                    new_val = json.loads(f'"{v}"')
            else:
                cur_val = section[section_keys[-1]]
                cur_val_t = type(cur_val)
                # Return strings as-is, use JSON parser otherwise.
                # This is useful in case of complex overrides, such as lists.
                # Need to use string_types to cover str/unicode in Py2/Py3.
                new_val = v if isinstance(cur_val, str) else cur_val_t(json.loads(v))
            section[section_keys[-1]] = new_val
        except KeyError:
            logger.error("Spec override key not found: {}".format(k))
            raise
        except ValueError:
            logger.error("Spec override {} value {} has invalid type.".format(k, v))
            raise
    return spec


def print_spec(spec, indent):
    """Pretty prints spec file."""
    if indent == 0:
        logger.info("---- Options ----")
    # Print spec.
    max_len = max([len(k) for k in spec.keys()])
    fmt_string = " " * indent + "{{:<{}}}: {{}}".format(max_len)
    dicts = []
    for k, v in sorted(spec.items()):
        if type(v) is dict:
            dicts.append((k, v))
        else:
            logger.info(fmt_string.format(str(k), str(v)))
    for k, v in dicts:
        logger.info(" " * indent + k + ":")
        if v:
            print_spec(v, indent + 4)
    if indent == 0:
        logger.info("-----------------")


def grad_norm(model):
    sqsum = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sqsum += (p.grad ** 2).sum().item()
    return np.sqrt(sqsum)