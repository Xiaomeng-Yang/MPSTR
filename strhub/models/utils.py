from pathlib import PurePath
from typing import Sequence

import torch
from torch import nn

import yaml
from torch.optim.lr_scheduler import _LRScheduler, OneCycleLR


class MultiGroupOneCycleLR(_LRScheduler):
    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3, cycle_momentum=True):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.cycle_momentum = cycle_momentum

        self.step_count = 0
        self.one_cycle_schedulers = []

        for group in optimizer.param_groups:
            scheduler = OneCycleLR(
                optimizer,
                max_lr=group['lr'],
                total_steps=total_steps,
                pct_start=pct_start,
                cycle_momentum=cycle_momentum,
            )
            self.one_cycle_schedulers.append(scheduler)

        super().__init__(optimizer)

    def get_lr(self):
        self.step_count += 1
        lrs = []
        for scheduler in self.one_cycle_schedulers:
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])
        return lrs

    def step(self, epoch=None):
        self.step_count += 1
        for scheduler in self.one_cycle_schedulers:
            scheduler.step()


class InvalidModelError(RuntimeError):
    """Exception raised for any model-related error (creation, loading)"""


_WEIGHTS_URL = {
    'parseq-tiny': 'https://github.com/baudm/parseq/releases/download/v1.0.0/parseq_tiny-e7a21b54.pt',
    'parseq': 'https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt',
    'mpstr': ''
}


def _get_config(experiment: str, **kwargs):
    """Emulates hydra config resolution"""
    root = PurePath(__file__).parents[2]
    with open(root / 'configs/main.yaml', 'r') as f:
        config = yaml.load(f, yaml.Loader)['model']
    with open(root / f'configs/charset/94_full.yaml', 'r') as f:
        config.update(yaml.load(f, yaml.Loader)['model'])
    with open(root / f'configs/experiment/{experiment}.yaml', 'r') as f:
        exp = yaml.load(f, yaml.Loader)
    # Apply base model config
    model = exp['defaults'][0]['override /model']
    with open(root / f'configs/model/{model}.yaml', 'r') as f:
        config.update(yaml.load(f, yaml.Loader))
    # Apply experiment config
    if 'model' in exp:
        config.update(exp['model'])
    config.update(kwargs)
    return config


def _get_model_class(key):
    if 'pimnet' in key:
        from .pimnet.system import PIMNet as ModelClass
    elif 'mpnet' in key:
        from .mpnet.system import MPNet as ModelClass
    else:
        raise InvalidModelError("Unable to find model class for '{}'".format(key))
    return ModelClass


def create_model(experiment: str, pretrained: bool = False, **kwargs):
    try:
        config = _get_config(experiment, **kwargs)
    except FileNotFoundError:
        raise InvalidModelError("No configuration found for '{}'".format(experiment)) from None
    ModelClass = _get_model_class(experiment)
    model = ModelClass(**config)
    if pretrained:
        try:
            url = _WEIGHTS_URL[experiment]
        except KeyError:
            raise InvalidModelError("No pretrained weights found for '{}'".format(experiment)) from None
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location='cpu', check_hash=True)
        model.load_state_dict(checkpoint)
    return model


def load_from_checkpoint(checkpoint_path: str, **kwargs):
    if checkpoint_path.startswith('pretrained='):
        model_id = checkpoint_path.split('=', maxsplit=1)[1]
        model = create_model(model_id, True, **kwargs)
    else:
        ModelClass = _get_model_class(checkpoint_path)
        model = ModelClass.load_from_checkpoint(checkpoint_path, **kwargs)
    return model


def parse_model_args(args):
    kwargs = {}
    arg_types = {t.__name__: t for t in [int, float, str]}
    arg_types['bool'] = lambda v: v.lower() == 'true'  # special handling for bool
    for arg in args:
        name, value = arg.split('=', maxsplit=1)
        name, arg_type = name.split(':', maxsplit=1)
        kwargs[name] = arg_types[arg_type](value)
    return kwargs


def init_weights(module: nn.Module, name: str = '', exclude: Sequence[str] = ()):
    """Initialize the weights using the typical initialization schemes used in SOTA models."""
    if any(map(name.startswith, exclude)):
        return
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
