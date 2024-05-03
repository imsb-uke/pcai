from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from src import utils
from src.models.architecture.components.utils import (
    extract_state_dict_at_module_level,
    rename_sd_keys,
)

from .components.module_base import SequentialModuleBase

log = utils.get_pylogger(__name__)


class AdversarialNet(LightningModule):
    """Top-level network.

    Using this implementation allows handing over a single model to the "net" parameter of the
    lightning_module, instead of needing to define separate modules there. It has common modules,
    modules for the survival classification head and modules for the adversarial classification head.
    The modules are distinguished by their prefix, which should be either com__, out__ or adv__.

    Args:
        modules (Dict[str, AdversarialModuleBase]): Dictionary of modules to be used in the network.

    Example:
        To call this class from a hydra config, use the following syntax:
                _target_: src.models.architecture.adversarial_net.AdversarialNet
                defaults:
                    - _default.yaml
                    - enc_module/enc_cnn_bag.yaml@com__enc_module
                    - sa_module/sa_bag.yaml@out__sa_module
                    - agg_module/agg_bag.yaml@out__agg_module
                    - tar_module/tar_clas.yaml@out__tar_module
                    - adv_module/adv_clas.yaml@adv__adv_module
                    - agg_module/agg_bag.yaml@adv__agg_module
                    - tar_module/tar_clas.yaml@adv__tar_module
                    - _self_.yaml
    """

    def __init__(
        self,
        ckpt_path: str = None,
        allow_missing_keys: bool = False,
        allow_unexpected_keys: bool = True,
        extract_sd_level: str = "net",
        rename_sd_keys_mapping: Optional[Dict[str, str]] = None,
        **modules: Dict[str, SequentialModuleBase],
    ):
        super().__init__()
        self.clas_head = nn.Identity()
        self.adv_head = nn.Identity()

        for key in list(modules.keys()):
            if "module" not in key:
                modules.pop(key)

        com_modules = self._get_modules_by_prefix(modules, "com__")
        com_module_list = self._build_module_list(com_modules)

        self.com_embedding_len_out = com_module_list[
            list(com_module_list.keys())[-1]
        ].embedding_len_out
        self.net = nn.Sequential(self._build_module_list(com_modules))

        out_modules = self._get_modules_by_prefix(modules, "out__")
        if len(out_modules) > 0:
            self.clas_head = nn.Sequential(self._build_module_list(out_modules))

        adv_modules = self._get_modules_by_prefix(modules, "adv__")
        if len(adv_modules) > 0:
            self.adv_head = nn.Sequential(self._build_module_list(adv_modules))

        if ckpt_path is not None:
            self._load_weights_from_ckpt(
                ckpt_path,
                allow_missing_keys,
                allow_unexpected_keys,
                extract_sd_level,
                rename_sd_keys_mapping,
            )

    def _get_modules_by_prefix(self, modules, prefix):
        return {k: v for k, v in modules.items() if k.startswith(prefix)}

    def forward(self, inputs: torch.Tensor):
        x = self.net(inputs)
        return self.clas_head(x), self.adv_head(x)

    def get_meta(self):
        meta = {}
        for branch in (self.net, self.clas_head, self.adv_head):
            for name, module in branch.named_children():
                if hasattr(module, "get_meta"):
                    meta[name] = module.get_meta()
        return meta

    def _build_module_list(self, modules: Dict[str, SequentialModuleBase]) -> List:
        """Builds a sequential module list from a list of modules of type SequentialModuleBase.

        This is used to build the network from the modules defined in the config.

        If 'enc_module' is not used as first module, the input_size needs to be
        specified in the hydra config of the first module and '_partial_' needs
        to be set to False.
        """
        module_list = []

        for name, module in modules.items():
            if module is None:
                continue
            if isinstance(module, partial):
                # if module is partial, instantiate it with the output size of the previous
                # module as input size. First module must not be partial!
                if len(module_list) > 0:
                    module = module(input_size=module_list[-1][1].embedding_len_out)
                else:
                    module = module(input_size=self.com_embedding_len_out)
            log.info(f"Instantiating {name} <{module.__class__.__name__}>")
            module_list.append((name, module))

        return OrderedDict(module_list)

