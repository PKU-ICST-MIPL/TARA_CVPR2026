# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
import torch
from PIL import Image
import io
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url


import re
import bioclip2.src.open_clip as open_clip
import torch.nn.functional as F
import torch.nn as nn
from transformers.optimization import get_scheduler

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class AlignmentProjector(nn.Module):
    def __init__(self, hidden_size, projector_dim, z_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, z_dim),
        )

    def forward(self, x):
        return self.projector(x)

class Qwen2VLGRPOTrainer_TARA(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            model_id_lower = model_id.lower()
            model_basename = model_id.rstrip("/").split("/")[-1].lower()

            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "qwen3-vl" in model_id_lower:
                model_init_kwargs.pop("use_cache", None)
                if "-a" in model_basename:
                    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(model, torch_dtype=torch.bfloat16, **model_init_kwargs)
                else:
                    model = Qwen3VLForConditionalGeneration.from_pretrained(model, torch_dtype=torch.bfloat16, **model_init_kwargs)
            elif "qwen2.5-vl" in model_id_lower:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "qwen2-vl" in model_id_lower:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "aria" in model_id_lower:
                model_init_kwargs.pop("use_cache", None)
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model
            model_id_lower = model_id.lower()
            model_basename = model_id.rstrip("/").split("/")[-1].lower()

            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "qwen3-vl" in model_id_lower:
                self.llm_hidden_dim = 2048
                model_init_kwargs.pop("use_cache", None)
                if "-a" in model_basename:
                    self.ref_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                else:
                    self.ref_model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "qwen2-vl" in model_id_lower:
                self.llm_hidden_dim = 1536
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if any(tag in model_id_lower for tag in ("qwen3-vl", "qwen2.5-vl", "qwen2-vl", "aria")):
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "qwen" in model_id_lower:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id
        else:
            pad_token_id = getattr(getattr(processing_class, "tokenizer", processing_class), "pad_token_id", None)

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            temperature=1, # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # --- 新增：加载 bioclip2，创建 projector 并绑定到 self.model 以便更新参数 ---
        # 使用 open_clip 的加载方式（遵循你的要求）
        self.bioclip_model, self.bioclip_preprocess_train, self.bioclip_preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
        self.bioclip_tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip-2')  

        # Freeze bioclip weights (只训练 projector)
        for p in self.bioclip_model.parameters():
            p.requires_grad = False
        self.bioclip_model.eval()

        # # # 推断 LLM 最后一层 hidden dim
        # print(self.model.get_input_embeddings().weight.shape)
        # llm_hidden_dim = self.model.get_input_embeddings().weight.shape[1] #2048 # 1536 #2048 

        # projector_dim = 1024

        self.text_bioclip_projector = AlignmentProjector(
            self.llm_hidden_dim,       
            1024,               
            768                  
        ).to(torch.bfloat16)  

        self.visual_bioclip_projector = AlignmentProjector(
            self.llm_hidden_dim,      
            1024,               
            1024                  
        ).to(torch.bfloat16) 

        try:
            device = getattr(self, "accelerator").device
            self.bioclip_model.to(device)
            self.text_bioclip_projector.to(device)
            self.visual_bioclip_projector.to(device)
        except Exception:
            pass

        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps)

        # -------------------------
        # create alignment_optimizer
        # -------------------------
        if getattr(self, "alignment_optimizer", None) is None:
            optimizer_cls = type(self.optimizer)
            base_defaults = self.optimizer.defaults

            base_lr = getattr(self.args, "alignment_learning_rate", base_defaults.get("lr", 1e-5))
            projector_lr = base_lr * 10  

            # 定义参数组
            param_groups = [
                {"params": self.model.parameters(), "lr": base_lr},
                {"params": self.text_bioclip_projector.parameters(), "lr": projector_lr},
                {"params": self.visual_bioclip_projector.parameters(), "lr": projector_lr},
            ]

            self.alignment_optimizer = optimizer_cls(
                param_groups,
                betas=base_defaults.get("betas", (0.9, 0.999)),
                eps=base_defaults.get("eps", 1e-8),
                weight_decay=base_defaults.get("weight_decay", 0.0),
            )

            # -------------------------
            # create alignment_lr_scheduler
            # -------------------------
            if hasattr(self, "lr_scheduler") and hasattr(self.args, "lr_scheduler_type"):
                scheduler_type = getattr(self.args, "lr_scheduler_type", "linear")
            else:
                scheduler_type = "linear"

            warmup_ratio = getattr(self.args, "warmup_ratio", 0.0)
            warmup_steps = getattr(self.args, "warmup_steps", 0)
            num_warmup_steps = (
                int(num_training_steps * warmup_ratio) if warmup_steps == 0 else warmup_steps
            )

            self.alignment_lr_scheduler = get_scheduler(
                name=scheduler_type,
                optimizer=self.alignment_optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

            # -------------------------
            # Accelerator
            # -------------------------
            try:
                self.alignment_optimizer = self.accelerator.prepare_optimizer(self.alignment_optimizer)
                self.alignment_lr_scheduler = self.accelerator.prepare(self.alignment_lr_scheduler)
            except Exception:
                pass


    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw):
        logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompts = [x["prompt"] for x in inputs]
        # prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        prompts_text = [
    maybe_apply_chat_template({k: v for k, v in example.items() if k != "label"}, self.processing_class)["prompt"]
    for example in inputs
]


        images = []
        for x in inputs:
            img_temp = Image.open(io.BytesIO(x["image"]["bytes"])).resize((384, 384), Image.Resampling.LANCZOS)
            images.append(img_temp)
        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        pixel_values = prompt_inputs["pixel_values"]
        image_grid_thw = prompt_inputs["image_grid_thw"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            gen_outputs = unwrapped_model.generate(
                **prompt_inputs,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_attentions=False,
            )

            if hasattr(gen_outputs, "sequences"):
                prompt_completion_ids = gen_outputs.sequences
            else:
                prompt_completion_ids = gen_outputs

            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

            hidden_states_all = getattr(gen_outputs, "hidden_states", None)
            if hidden_states_all is None:
                hidden_states_all = getattr(gen_outputs, "decoder_hidden_states", None)
            if hidden_states_all is None:
                raise RuntimeError("generate 没有返回 hidden_states，请确认模型的 generate 支持 output_hidden_states=True")

            # target layer of fine-grained label alignment
            target_layer = int(getattr(self.args, "target_layer", -1))
            last_layer_hs = hidden_states_all[1][target_layer]  

            idx = 0 
            last_token_embeds = last_layer_hs[:, idx, :]  

            projected_emb = self.text_bioclip_projector(last_token_embeds)  
            projected_emb = projected_emb / (projected_emb.norm(dim=1, keepdim=True) + 1e-8)

            # target layer of taxonomic visual alignment
            target_layer = int(getattr(self.args, "target_layer", len(hidden_states_all[0])//2))
            last_layer_hs = hidden_states_all[0][target_layer]

        # mask after first eos
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        pixel_values = prompt_inputs["pixel_values"].repeat(self.num_generations, 1)
        image_grid_thw = prompt_inputs["image_grid_thw"].repeat_interleave(self.num_generations, dim=0)

        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards = rewards_per_func.sum(dim=1)

        # ---------- bioclip alignment reward  ----------
        gt_texts = []
        for example in inputs:
            sol = example.get("solution", "") or ""
            m = re.search(r"<answer>\s*(.*?)\s*</answer>", sol, flags=re.S | re.I)
            if m:
                gt_texts.append(m.group(1).strip())
            else:
                gt_texts.append(sol.strip())

        tokenized = open_clip.tokenize(gt_texts).to(device)
        with torch.inference_mode():
            gt_embs = self.bioclip_model.encode_text(tokenized).detach()
        gt_embs = gt_embs / (gt_embs.norm(dim=1, keepdim=True) + 1e-8)
        gt_embs_rep = gt_embs.repeat_interleave(self.num_generations, dim=0)

        # grouped rewards -> advantages
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # -------------------- alignment joint update (LLM + projector) --------------------
        try:
            model_optimizer = self.optimizer
            if model_optimizer is None:
                raise AttributeError
        except Exception:
            raise RuntimeError("Trainer 的主 optimizer (self.optimizer) 未初始化，无法进行 alignment 的联合更新。请确保在训练开始前 Trainer 已创建 optimizer。")

        # Unfreeze the model and projector to ensure the computation graph can backpropagate back to the LLM parameters
        for p in self.model.parameters():
            p.requires_grad = True
        for p in self.text_bioclip_projector.parameters():
            p.requires_grad = True
        for p in self.visual_bioclip_projector.parameters():
            p.requires_grad = True

        # Use new alignment_optimizer
        self.alignment_optimizer.zero_grad()

        projected_emb_for_update = self.text_bioclip_projector(last_token_embeds)
        projected_emb_for_update = projected_emb_for_update / (projected_emb_for_update.norm(dim=1, keepdim=True) + 1e-8)
        text_alignment_loss_for_update = 1.0 - F.cosine_similarity(projected_emb_for_update, gt_embs_rep.detach(), dim=1).mean()

        # Obtain the embeddings of all image patches using the image encoder of BioCLIP2
        bioclip_inputs = []
        for img in images:
            img = img.resize((256, 256), Image.Resampling.LANCZOS)
            bioclip_inputs.append(self.bioclip_preprocess_train(img))
        bioclip_inputs = torch.stack(bioclip_inputs, dim=0).to(device)

        with torch.inference_mode(): 
            _, _, bioclip_patch_embs = self.bioclip_model.visual(bioclip_inputs, output_tokens=True)


        bioclip_patch_embs = F.normalize(bioclip_patch_embs, dim=-1).detach()

        # Extract all image token embeddings from the LLM (not just the last one)
        all_image_token_embs = []
        image_token_id = getattr(self.processing_class, "image_token_id", None)  # 151655
        for i in range(prompt_completion_ids.size(0)):
            pos = (prompt_completion_ids[i] == image_token_id).nonzero(as_tuple=True)[0]
            if pos.numel() == 0:
                all_image_token_embs.append(torch.zeros(1, last_layer_hs.size(2), device=device, dtype=last_layer_hs.dtype))
            else:
                img_embs_i = last_layer_hs[i, pos, :]  # (num_img_tokens, hidden_dim)
                all_image_token_embs.append(img_embs_i)
        # Align to the longest length within the batch (pad to max_len)
        max_len = max(x.size(0) for x in all_image_token_embs)
        padded_embs = []
        for x in all_image_token_embs:
            pad_len = max_len - x.size(0)
            if pad_len > 0:
                x = torch.cat([x, torch.zeros(pad_len, x.size(1), device=device, dtype=x.dtype)], dim=0)
            padded_embs.append(x)
        image_token_embs_all = torch.stack(padded_embs, dim=0)  # (B*G, max_len, hidden_dim)

        projected_image_token_embs_all = self.visual_bioclip_projector(image_token_embs_all)  # (B*G, max_len, bioclip_dim)
        projected_image_token_embs_all = F.normalize(projected_image_token_embs_all, dim=-1)

        # Compute token-wise similarity 
        bioclip_patch_embs_rep = bioclip_patch_embs.repeat_interleave(self.num_generations, dim=0) # (B*G, num_patches, dim)
        # Handle inconsistent lengths: Uniform downsampling to min_len
        len_llm = projected_image_token_embs_all.size(1)
        len_clip = bioclip_patch_embs_rep.size(1)
        min_len = min(len_llm, len_clip)

        if len_llm != len_clip:
            # Generate uniform sampling indices
            idx_llm = torch.linspace(0, len_llm - 1, steps=min_len, device=device).long()
            idx_clip = torch.linspace(0, len_clip - 1, steps=min_len, device=device).long()

            proj_b = projected_image_token_embs_all.index_select(1, idx_llm)  # (B*G, min_len, dim)
            clip_b = bioclip_patch_embs_rep.index_select(1, idx_clip)        # (B*G, min_len, dim)
        else:
            proj_b = projected_image_token_embs_all
            clip_b = bioclip_patch_embs_rep

        # Calculate per-token cosine similarity
        cos_sim = F.cosine_similarity(proj_b, clip_b, dim=-1)  # (B*G, min_len)
        visual_alignment_loss_for_update = (1.0 - cos_sim.mean(dim=-1)).mean()

        image_weight = float(getattr(self.args, "alignment_image_weight", 0.5))
        text_weight = float(getattr(self.args, "alignment_text_weight", 0.5))
        w_sum = text_weight + image_weight
        text_w = text_weight / w_sum
        img_w = image_weight / w_sum


        alignment_loss_for_update = text_w * text_alignment_loss_for_update + img_w * visual_alignment_loss_for_update

        self.accelerator.backward(alignment_loss_for_update)
        self.alignment_optimizer.step()
        self.alignment_optimizer.zero_grad()

        # Freeze the projector to ensure only the LLM is updated during subsequent RL updates
        for p in self.text_bioclip_projector.parameters():
            p.requires_grad = False
        for p in self.visual_bioclip_projector.parameters():
            p.requires_grad = False

        # Log metrics
        text_alignment_loss_value = self.accelerator.gather_for_metrics(text_alignment_loss_for_update.detach()).mean().item()
        self._metrics["text_alignment_loss"].append(text_alignment_loss_value)
        visual_alignment_loss_value = self.accelerator.gather_for_metrics(visual_alignment_loss_for_update.detach()).mean().item()
        self._metrics["visual_alignment_loss"].append(visual_alignment_loss_value)
        alignment_loss_value = self.accelerator.gather_for_metrics(alignment_loss_for_update.detach()).mean().item()
        self._metrics["total_alignment_loss"].append(alignment_loss_value)

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # Original RL update logic, where the projector is frozen
        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))




