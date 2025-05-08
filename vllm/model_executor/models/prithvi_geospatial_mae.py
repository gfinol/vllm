# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The vLLM team.
# Copyright 2025 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only IBM/NASA Prithvi Geospatial model."""
import copy
import dict
import importlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Optional, Set, Tuple, Union

import albumentations
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from transformers import BatchFeature
from terratorch.datamodules import Sen1Floods11NonGeoDataModule


from vllm.config import VllmConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (IsAttentionFree,
                                                   SupportsMultiModal,
                                                   SupportsV0Only)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalInputs, MultiModalKwargs)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptUpdate)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import (IntermediateTensors, PoolerOutput,
                           PoolingSequenceGroupOutput)


class PrithviGeoSpatialMAEProcessingInfo(BaseProcessingInfo):

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}


class PrithviGeoSpatialMAEInputBuilder(
        BaseDummyInputsBuilder[PrithviGeoSpatialMAEProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        # This model input is fixed and is in the form of a torch Tensor.
        # The size of pixel_values might change in the cases where we resize
        # the input but never exceeds the dimensions below.
        return {
            "pixel_values": torch.full((1, 6, 512, 512), 1.0),
            "location_coords": torch.full((1, 2), 1.0),
        }


class PrithviGeoSpatialMAEMultiModalProcessor(BaseMultiModalProcessor):

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            location_coords=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        return []

    def _preprocess(self, x: np.ndarray, img_size, location_coords, datamodule):
        # Reflect pad if not divisible by img_size
        original_h, original_w = x.shape[-2:]
        pad_h = (img_size - (original_h % img_size)) % img_size
        pad_w = (img_size - (original_w % img_size)) % img_size
        x = np.pad(
            x, ((0, 0), (0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="reflect"
        )

        # Build sliding window

        batch_size = 1
        batch = torch.tensor(x, device="cpu")
        windows = batch.unfold(3, img_size, img_size).unfold(4, img_size, img_size)
        h1, w1 = windows.shape[3:5]
        windows = rearrange(
            windows, "b c t h1 w1 h w -> (b h1 w1) c t h w", h=img_size, w=img_size
        )

        # Split into batches if number of windows > batch_size
        num_batches = windows.shape[0] // batch_size if windows.shape[0] > batch_size else 1
        windows = torch.tensor_split(windows, num_batches, dim=0)

        # location_coords = location_coords[0]

        # TODO - Add support for more than one window
        for x in windows[:1]:
            # Apply standardization
            x = datamodule.test_transform(image=x.squeeze().numpy().transpose(1, 2, 0))
            x = datamodule.aug(x)['image']

            mm_data = {
                "pixel_values": torch.empty(0) if x is None else x,
                "location_coords": torch.empty(0) if location_coords is None else location_coords,
                "img_size": img_size,
                "h1": h1,
                "w1": w1,
                "original_h": original_h,
                "original_w": original_w,
            }
            return mm_data

        return None

    def init_object_from_classpath_and_args(self, class_name: str, init_args: dict):
        class_parts = class_name.split('.')
        module_name = '.'.join(class_parts[:-1])
        class_name = class_parts[-1]

        # Import the module
        module = importlib.import_module(module_name)

        # Get the class
        cls = getattr(module, class_name)

        return cls(**init_args)

    def parse_args_list(self, args: list):
        return [
            self.init_object_from_classpath_and_args(t["class_path"], t["init_args"])
            for t in args]

    def generate_datamodule(self, classpath: str, init_args: dict):

        final_init_args = copy.deepcopy(init_args)

        final_init_args["test_transform"] = self.parse_args_list(init_args["test_transform"])
        final_init_args["val_transform"] = self.parse_args_list(init_args["val_transform"])
        final_init_args["train_transform"] = self.parse_args_list(init_args["test_transform"])

        datamodule = self.init_object_from_classpath_and_args(classpath, final_init_args)
        return datamodule

    @staticmethod
    def generate_datamodule_static():
        datamodule_config = {
            'bands': ['BLUE',
                      'GREEN',
                      'RED',
                      'NIR_NARROW',
                      'SWIR_1',
                      'SWIR_2'],
            'batch_size': 16,
            'constant_scale': 0.0001,
            'data_root': '/dccstor/geofm-finetuning/datasets/sen1floods11',
            'drop_last': True,
            'no_data_replace': 0.0,
            'no_label_replace': -1,
            'num_workers': 8,
            'test_transform': [albumentations.Resize(always_apply=False,
                                                     height=448,
                                                     interpolation=1,
                                                     p=1,
                                                     width=448),
                               albumentations.pytorch.ToTensorV2(
                                   transpose_mask=False,
                                   always_apply=True,
                                   p=1.0
                               )],
        }

        datamodule = Sen1Floods11NonGeoDataModule(data_root=datamodule_config['data_root'],
                                                  batch_size=datamodule_config["batch_size"],
                                                  num_workers=datamodule_config["num_workers"],
                                                  bands=datamodule_config["bands"],
                                                  drop_last=datamodule_config["drop_last"],
                                                  test_transform=datamodule_config["test_transform"
                                                                                   ""])

        return datamodule

    def apply(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        return_mm_hashes: bool = False,
    ) -> MultiModalInputs:

        config = {} # TODO load config from somewhere
        config = self.info
        input_data = mm_data["input_data"]
        # temporal_coords = mm_data["temporal_coords"]
        location_coords = mm_data["location_coords"]
        # datamodule = self.generate_datamodule(config["data"]["class_path"], config["data"]["init_args"])
        datamodule = self.generate_datamodule_static()
        mm_kwargs = self._preprocess(input_data, 512, location_coords, datamodule)

        # mm_kwargs = {}
        # for k, v in mm_data.items():
        #     mm_kwargs[k] = v

        return MultiModalInputs(
            type="multimodal",
            prompt=prompt,
            prompt_token_ids=[1],
            mm_kwargs=MultiModalKwargs(mm_kwargs),
            mm_hashes=None,
            mm_placeholders={},
        )


@MULTIMODAL_REGISTRY.register_processor(
    PrithviGeoSpatialMAEMultiModalProcessor,
    info=PrithviGeoSpatialMAEProcessingInfo,
    dummy_inputs=PrithviGeoSpatialMAEInputBuilder)
class PrithviGeoSpatialMAE(nn.Module, IsAttentionFree, SupportsMultiModal,
                           SupportsV0Only):
    """ Prithvi Masked Autoencoder"""

    def _instantiate_model(self, config: dict) -> Optional[nn.Module]:

        # We might be able/need to support different tasks with this same model
        if config["task_args"]["task"] == "SemanticSegmentationTask":
            from terratorch.cli_tools import SemanticSegmentationTask
            task = SemanticSegmentationTask(
                config["model_args"],
                config["task_args"]["model_factory"],
                loss=config["task_args"]["loss"],
                lr=config["task_args"]["lr"],
                ignore_index=config["task_args"]["ignore_index"],
                optimizer=config["task_args"]["optimizer"],
                optimizer_hparams=config["optimizer_params"],
                scheduler=config["task_args"]["scheduler"],
                scheduler_hparams=config["scheduler_params"],
                plot_on_val=config["task_args"]["plot_on_val"],
                freeze_decoder=config["task_args"]["freeze_decoder"],
                freeze_backbone=config["task_args"]["freeze_backbone"])

            return task.model
        else:
            return None

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        # the actual model is dynamically instantiated using terratorch
        # allowing us to perform changes to the model architecture
        # at startup time (e.g., change the model decoder class.)
        self.model = self._instantiate_model(
            vllm_config.model_config.hf_config.to_dict()["pretrained_cfg"])
        if self.model is None:
            raise ValueError(
                "Unsupported task. "
                "Only SemanticSegmentationTask is supported for now "
                "by PrithviGeospatialMAE.")

    def _parse_and_validate_multimodal_data(
            self, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        pixel_values = kwargs.pop("pixel_values", None)
        if not isinstance(pixel_values, torch.Tensor):
            raise ValueError(f"Incorrect type of pixel_values. "
                             f"Got type: {type(pixel_values)}")
        pixel_values = torch.unbind(pixel_values, dim=0)[0]

        location_coords = kwargs.pop("location_coords", None)
        if not isinstance(location_coords, torch.Tensor):
            raise ValueError(f"Incorrect type of location_coords. "
                             f"Got type: {type(location_coords)}")
        location_coords = torch.unbind(location_coords, dim=0)[0]
        if location_coords.shape == torch.Size([0]):
            location_coords = None

        return pixel_values, location_coords

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ):

        pixel_values, location_coords = (
            self._parse_and_validate_multimodal_data(**kwargs))

        img_size = kwargs.pop("img_size", 512)
        h1 = kwargs.pop("h1", 1)
        w1 = kwargs.pop("w1", 1)
        original_h = kwargs.pop("original_h", 512)
        original_w = kwargs.pop("original_w", 512)

        model_output = self.model(pixel_values,
                                  location_coords=location_coords)

        pred_imgs = []

        pred = model_output.output

        y_hat = pred.argmax(dim=1)

        y_hat = torch.nn.functional.interpolate(y_hat.unsqueeze(1).float(), size=img_size, mode="nearest")

        pred_imgs.append(y_hat)

        pred_imgs = torch.concat(pred_imgs, dim=0)

        # Build images from patches
        pred_imgs = rearrange(
            pred_imgs,
            "(b h1 w1) c h w -> b c (h1 h) (w1 w)",
            h=img_size,
            w=img_size,
            b=1,
            c=1,
            h1=h1,
            w1=w1,
        )

        # Cut padded area back to original size
        pred_imgs = pred_imgs[..., :original_h, :original_w]

        # Squeeze (batch size 1)
        pred_imgs = pred_imgs[0]

        return pred_imgs

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return PoolerOutput([PoolingSequenceGroupOutput(hidden_states)])

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        params_list = []
        model_buffers = dict(self.named_buffers())
        loaded_buffers = []
        for key, value in weights:
            if key == "state_dict":
                weights_to_parse = value
                for name, weight in weights_to_parse.items():
                    if "pos_embed" in name:
                        continue

                    if "_timm_module." in name:
                        name = name.replace("_timm_module.", "")

                    # this model requires a couple of buffers to be loaded
                    # that are not loadable with the AutoWeightsLoader
                    if name in model_buffers:
                        if "_timm_module." in name:
                            name = name.replace("_timm_module.", "")
                        buffer = model_buffers[name]
                        weight_loader = getattr(buffer, "weight_loader",
                                                default_weight_loader)
                        weight_loader(buffer, weight)
                        loaded_buffers.append(name)
                    else:
                        params_list.append((name, weight))
                break

        # Load the remaining model parameters
        loader = AutoWeightsLoader(self)
        autoloaded_weights = loader.load_weights(params_list)

        return autoloaded_weights.union(set(loaded_buffers))
