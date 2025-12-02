#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   grpo_t5gemma_trainer.py
@Time    :   2025/10/14 15:09:49
@Author  :   Dawn Li
@Version :   1.0
@Contact :   dawnli619215645@gmail.com
@Desc    :   GRPO trainer adapted for T5/Gemma encoder-decoder architecture.
             Based on TRL's GRPOTrainer with modifications to handle encoder-decoder
             models where generation returns only completion tokens (no prompt).
"""

import copy
import re
import warnings
from collections.abc import Sequence, Sized
from contextlib import nullcontext
from typing import Any, Callable, Optional, Union

import torch
import torch.utils.data
from accelerate.utils import (
    broadcast_object_list,
    gather,
    gather_object,
)
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import Sampler
from transformers import (
    PreTrainedModel,
)
from transformers.utils import (
    is_flash_attn_2_available,
)
from trl import GRPOTrainer
from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.import_utils import is_vllm_available
from trl.models import unwrap_model_for_generation
from trl.trainer.utils import (
    entropy_from_logits,
    pad,
    selective_log_softmax,
)

if is_vllm_available():
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the dataset.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatSampler(
    ...     ["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4
    ... )
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            self.generator = torch.Generator()  # Create a local random generator
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(
                self.num_samples, generator=self.generator
            ).tolist()
        else:
            indexes = list(range(self.num_samples))

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [
            indexes[i : i + self.batch_size]
            for i in range(0, len(indexes), self.batch_size)
        ]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return (
            (self.num_samples // self.batch_size)
            * self.batch_size
            * self.mini_repeat_count
            * self.repeat_count
        )


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean(
        (tensor - torch.nanmean(tensor, keepdim=True)) ** 2
    )  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)


def split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.

    Example:
    ```python
    >>> x = torch.arange(12).reshape(6, 2)
    >>> y = torch.arange(6).reshape(6, 1)
    >>> tensor_dict = {"x": x, "y": y}
    >>> split_tensor_dict(tensor_dict, 3)
    [
        {"x": tensor([[0, 1], [2, 3]]), "y": tensor([[0], [1]])},
        {"x": tensor([[4, 5], [6, 7]]), "y": tensor([[2], [3]])},
        {"x": tensor([[ 8,  9], [10, 11]]), "y": tensor([[4], [5]])}
    ]
    ```
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    return [
        {
            key: tensor[i * chunk_size : (i + 1) * chunk_size]
            if tensor is not None
            else None
            for key, tensor in tensor_dict.items()
        }
        for i in range(num_chunks)
    ]


def shuffle_sequence_dict(
    seq_dict: dict[str, Optional[Sequence]],
) -> dict[str, Optional[Sequence]]:
    """
    Shuffles all sequence-like values in a dictionary along the first dimension in unison.

    Example:
    ```python
    >>> x = torch.arange(6).reshape(3, 2)
    >>> y = ["a", "b", "c"]
    >>> seq_dict = {"x": x, "y": y}
    >>> shuffle_sequence_dict(seq_dict)
    {'x': tensor([[2, 3],
                  [0, 1],
                  [4, 5]]),
     'y': ['b', 'a', 'c']}
    ```
    """
    # Determine batch size from the first non-None sequence
    batch_size = len(next(v for v in seq_dict.values() if v is not None))
    permutation = torch.randperm(batch_size)

    def permute(v: Optional[Sequence]) -> Optional[Sequence]:
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            return v[permutation]
        return [v[i] for i in permutation]

    return {key: permute(val) for key, val in seq_dict.items()}


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


def identity(x):
    """Do we really need docs for this?"""
    return x


def split_pixel_values_by_grid(
    batch: dict[str, torch.Tensor],
) -> dict[str, Union[torch.Tensor, list[torch.Tensor]]]:
    """
    Splits `batch["pixel_values"]` into a list of tensors based on the product of each row in
    `batch["image_grid_thw"]`, while keeping other entries unchanged.
    """
    if "image_grid_thw" not in batch or "pixel_values" not in batch:
        return batch

    lengths = batch["image_grid_thw"].prod(dim=1).tolist()  # [batch_size]
    pixel_values = batch["pixel_values"]  # [total, feature_dim]

    if sum(lengths) != pixel_values.size(0):
        raise ValueError(
            f"Mismatch: sum(lengths) = {sum(lengths)} != pixel_values.size(0) = {pixel_values.size(0)}"
        )

    split_values = list(torch.split(batch["pixel_values"], lengths, dim=0))
    return {**batch, "pixel_values": split_values}


def unsplit_pixel_values_by_grid(
    batch: dict[str, Union[torch.Tensor, list[torch.Tensor]]],
) -> dict[str, torch.Tensor]:
    """
    Opposite of `split_pixel_values_by_grid`. Merges a list of tensors in `batch["pixel_values"]`
    back into a single tensor along the first dimension.
    """
    pixel_values = batch.get("pixel_values")

    if isinstance(pixel_values, list):
        merged = torch.cat(pixel_values, dim=0)
        return {**batch, "pixel_values": merged}
    else:
        return batch


def truncate_with_protected_tokens(
    ids: torch.Tensor,
    mask: torch.Tensor,
    target_length: int,
    protected_tokens: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Truncate tensors to target length while preserving protected tokens.

    Args:
        ids (`torch.Tensor`):
            Input tensor of token IDs, shape (batch_size, sequence_length).
        mask (`torch.Tensor`):
            Input tensor of attention masks, shape (batch_size, sequence_length).
        target_length (`int`):
            Desired length of the output sequences.
        protected_tokens (`list[int]`):
            List of token IDs that should be preserved in the output.
    """
    protected_set = set(protected_tokens)

    def process_sequence(ids, mask):
        # Create boolean masks
        is_protected = torch.tensor([x.item() in protected_set for x in ids])
        is_non_protected = ~is_protected

        # Count tokens
        num_protected = is_protected.sum().item()
        num_non_protected_needed = target_length - num_protected

        if num_non_protected_needed < 0:
            raise ValueError(
                f"target_length ({target_length}) is too small for the protected tokens ({num_protected} tokens). "
                f"Please increase target length to at least {num_protected} or disable truncation."
            )

        # Select which non-protected tokens to keep (rightmost ones)
        non_protected_indices = torch.where(is_non_protected)[0]
        keep_non_protected = torch.zeros_like(is_non_protected)
        if num_non_protected_needed > 0:
            keep_indices = non_protected_indices[-num_non_protected_needed:]
            keep_non_protected[keep_indices] = True

        # Final mask: protected OR selected non-protected
        keep_mask = is_protected | keep_non_protected

        return ids[keep_mask], mask[keep_mask]

    # Process each sequence in the batch
    truncated_seq = []
    truncated_mask = []

    for i in range(ids.shape[0]):
        new_ids, new_mask = process_sequence(ids[i], mask[i])
        truncated_seq.append(new_ids)
        truncated_mask.append(new_mask)

    return torch.stack(truncated_seq), torch.stack(truncated_mask)


# ============================================================================
# Encoder-Decoder Adaptation for T5/Gemma Models
# ============================================================================
class GRPOEncoderDecoderTrainer(GRPOTrainer):
    """
    GRPO Trainer adapted for encoder-decoder models (T5, BART, T5-Gemma).

    This trainer extends the base GRPOTrainer to support encoder-decoder architectures
    where the generation process differs fundamentally from decoder-only models.

    Key Architectural Differences:
    --------------------------------
    1. **Generation Output**: Encoder-decoder models' generate() returns ONLY completion
       tokens, whereas decoder-only models return prompt+completion concatenated.

    2. **Forward Pass**: Uses separate encoder and decoder inputs instead of a single
       concatenated input_ids tensor.

    3. **Tokenization**: Prompt goes to encoder (bidirectional attention), decoder
       generates completion independently starting from decoder_start_token_id.

    Method Overrides:
    -----------------
    This trainer overrides 3 critical methods to handle encoder-decoder specifics
    while maintaining all other GRPO functionality (rewards, sampling, logging):

    - `_generate_and_score_completions`: Removes prompt slicing logic since
      generation already returns only completion tokens.

    - `_get_per_token_logps_and_entropies`: Splits concatenated inputs into
      encoder/decoder parts and passes them separately to the model.

    - `_compute_loss`: Avoids concatenating prompt+completion, instead passing
      them as separate encoder and decoder inputs.

    Compatible Models:
    ------------------
    - T5GemmaForConditionalGeneration
    - T5ForConditionalGeneration
    - BartForConditionalGeneration
    - Any HuggingFace encoder-decoder model with is_encoder_decoder=True

    Example:
    --------
    ```python
    from transformers import T5GemmaForConditionalGeneration
    from experiments.grpo_ft.grpo_t5gemma_trainer import GRPOEncoderDecoderTrainer

    model = T5GemmaForConditionalGeneration.from_pretrained("path/to/model")
    trainer = GRPOEncoderDecoderTrainer(
        model=model,
        reward_funcs=reward_function,
        args=grpo_config,
        train_dataset=dataset,
    )
    trainer.train()
    ```

    See Also:
    ---------
    - Base class: `trl.trainer.GRPOTrainer`
    - Research: DeepSeekMath paper (arXiv:2402.03300)
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the encoder-decoder GRPO trainer with model validation.

        This initializer adds validation to ensure the model is compatible with
        encoder-decoder training. All other initialization is handled by the parent
        GRPOTrainer class.

        Args:
            *args: Positional arguments passed to GRPOTrainer.__init__
            **kwargs: Keyword arguments passed to GRPOTrainer.__init__

        Raises:
            ValueError: If model is not an encoder-decoder architecture
            ValueError: If model.config.decoder_start_token_id is not set

        Note:
            The model validation happens AFTER parent initialization to ensure
            self.model is available for inspection.
        """
        # Call parent initializer first
        super().__init__(*args, **kwargs)

        # Preserve enhanced logging behavior that existed in the local GRPOTrainer
        # (not present in upstream TRL). This flag enables logging of per-component
        # reward diagnostics when reward functions expose `last_components`.
        self.log_reward_components = getattr(self.args, "log_reward_components", False)

        # Validate model is encoder-decoder
        if not getattr(self.model.config, "is_encoder_decoder", False):
            raise ValueError(
                "GRPOEncoderDecoderTrainer requires an encoder-decoder model. "
                "Got model with is_encoder_decoder=False. "
                "Use GRPOTrainer for decoder-only models (GPT, LLaMA, etc.)."
            )

        # Validate decoder_start_token_id is set
        if self.model.config.decoder_start_token_id is None:
            raise ValueError(
                "Model config must have decoder_start_token_id set. "
                "This is required for proper decoder input preparation during training."
            )

    # ========================================================================
    # Methods to be overridden in subsequent tasks
    # ========================================================================
    @profiling_decorator
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        """
        Same as TRL's _calculate_rewards, plus optional logging of reward component
        breakdowns when a custom reward function exposes `last_components`.

        This preserves the local enhancement that was previously implemented in the
        custom GRPOTrainer, so we can inherit TRL's trainer directly and only keep
        the extra behavior here.
        """
        device = self.accelerator.device
        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [
            key
            for key in inputs[0]
            if key not in ["prompt", "completion", "completion_ids"]
        ]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(
                self.reward_funcs,
                self.reward_processing_classes,
                self.reward_func_names,
            )
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(
                    reward_func, nn.Module
                ):  # Module (no PretrainedModel) for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [
                            {"messages": p + c} for p, c in zip(prompts, completions)
                        ]
                        texts = [
                            apply_chat_template(x, reward_processing_class)["text"]
                            for x in messages
                        ]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts,
                        return_tensors="pt",
                        padding=True,
                        padding_side="right",
                        add_special_tokens=False,
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[
                            :, 0
                        ]  # Shape (B*G,)
                else:
                    output_reward_func = reward_func(
                        prompts=prompts,
                        completions=completions,
                        completion_ids=completion_ids_list,
                        **reward_kwargs,
                    )
                    # Convert None values to NaN
                    output_reward_func = [
                        reward if reward is not None else torch.nan
                        for reward in output_reward_func
                    ]

                    rewards_per_func[:, i] = torch.tensor(
                        output_reward_func, dtype=torch.float32, device=device
                    )

                    # Enhanced component logging preserved from local trainer implementation
                    if (
                        self.log_reward_components
                        and hasattr(reward_func, "last_components")
                        and reward_func.last_components
                    ):
                        component_records = gather_object(reward_func.last_components)
                        if self.accelerator.is_main_process:
                            merged_components = []
                            for record in component_records:
                                if isinstance(record, dict):
                                    merged_components.append(record)
                                elif isinstance(record, (list, tuple)):
                                    merged_components.extend(
                                        item
                                        for item in record
                                        if isinstance(item, dict)
                                    )

                            metrics = {}
                            if merged_components:
                                keys = merged_components[0].keys()
                                for key in keys:
                                    values = [
                                        float(component.get(key))
                                        for component in merged_components
                                        if component.get(key) is not None
                                    ]
                                    if not values:
                                        continue
                                    mean_value = sum(values) / len(values)
                                    base_name = f"train/reward_components/{reward_func_name}/{key}"
                                    metrics[f"{base_name}_mean"] = mean_value

                                    # Convenience rates for common binary indicators
                                    if key in {
                                        "mask",
                                        "connectivity_mask",
                                        "graceful_mask",
                                    }:
                                        metrics[f"{base_name}_rate"] = mean_value
                                    if key in {"connectivity", "graceful"}:
                                        pass_rate = sum(
                                            1.0 for v in values if v >= 0.99
                                        ) / len(values)
                                        metrics[f"{base_name}_pass_rate"] = pass_rate
                                    if key == "improvement" and len(values) > 1:
                                        metrics[f"{base_name}_std"] = float(
                                            torch.tensor(values)
                                            .std(unbiased=False)
                                            .item()
                                        )

                            if metrics:
                                self.log(metrics)

                        # Clear cached components to avoid stale logging.
                        reward_func.last_components = []

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = (
                torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            )
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items()
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Generate completions and calculate rewards (encoder-decoder specific).

        This method overrides the base implementation to handle encoder-decoder
        generation which returns only completion tokens (no prompt slicing needed).

        Key Differences from Base:
        ---------------------------
        1. Regular generation: generate() returns completion_ids directly (lines 1577-1583)
        2. vLLM generation: No prompt+completion concatenation (line 1529)
        3. Paged generation: No prompt+completion concatenation (line 1563)

        Note:
        -----
        We still create prompt_completion_ids by concatenation for passing to
        _get_per_token_logps_and_entropies(). This maintains compatibility until
        that method is also overridden in Task 4 to handle encoder-decoder properly.

        Returns:
        --------
        dict with keys: prompt_ids, prompt_mask, completion_ids, completion_mask,
        advantages, rewards_per_func, old_per_token_logps, ref_per_token_logps
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]

        # We don't yet support visual reward models/function, so we keep a copy of the original text-only prompts for
        # later use in the reward computation. If images are present, we insert {"type": "image"} as required by the
        # VLM chat template.
        original_prompts = copy.deepcopy(prompts)

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color is the sky?"}]}]
        kwargs = {}
        has_images = "image" in inputs[0]
        if has_images:
            images = [example.get("image") for example in inputs]
            kwargs = {"images": [[img] for img in images]}
            for prompt in prompts:
                if isinstance(prompt, list):
                    for message in prompt:
                        if not isinstance(message, dict):
                            continue
                        content = message.get("content")
                        role = message.get("role")
                        if isinstance(content, str):
                            if role == "user":
                                message["content"] = [
                                    {"type": "image"},
                                    {"type": "text", "text": content},
                                ]
                            elif role == "system":
                                message["content"] = [{"type": "text", "text": content}]

        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]

        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            return_token_type_ids=False,
            **kwargs,
        )
        prompt_inputs = super(GRPOTrainer, self)._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        if self.max_prompt_length is not None:
            # If max_prompt_length is set, we trim the prompt to keep only the last `max_prompt_length` tokens.
            # Then we decode those tokens back into text. We manually remove leading pad tokens from the decoded text,
            # because we can't use `skip_special_tokens=True` (some special tokens are still needed for generation).
            protected = [
                self.image_token_id,
                self.vision_start_token_id,
                self.vision_end_token_id,
            ]
            protected = [token for token in protected if token is not None]
            prompt_ids, prompt_mask = truncate_with_protected_tokens(
                prompt_ids, prompt_mask, self.max_prompt_length, protected
            )

            prompts_text = self.processing_class.batch_decode(
                prompt_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            prompts_text = [
                re.sub(rf"^({re.escape(self.pad_token)})+", "", text)
                for text in prompts_text
            ]

            # The chat template inserts a single image token into the prompt text. However, when this text is later
            # tokenized, the single image token string is expanded into multiple image token IDs, depending on the
            # image size. Since we're detokenizing here, we may see repeated image tokens in the decoded text. We
            # collapse them back into a single token string to match the original template.
            if self.image_token is not None:
                prompts_text = [
                    re.sub(rf"({re.escape(self.image_token)})+", self.image_token, text)
                    for text in prompts_text
                ]

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            # First, update the vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if has_images:
                    all_images = gather_object(images)

                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]

                    if has_images:
                        ordered_set_of_images = all_images[:: self.num_generations]
                    else:
                        ordered_set_of_images = None

                    with profiling_context(self, "vLLM.generate"):
                        completion_ids = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            images=ordered_set_of_images,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                            generation_kwargs=self.args.generation_kwargs,
                        )
                else:
                    completion_ids = [None] * len(all_prompts_text)
                # Broadcast the completions from the main process to all processes, ensuring each process receives its
                # corresponding slice.
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(
                        regex=self.guided_decoding_regex
                    )
                else:
                    guided_decoding = None

                generation_kwargs = {
                    "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": self.max_completion_length,
                    "guided_decoding": guided_decoding,
                }
                if self.args.generation_kwargs is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts_text)
                    gathered_prompts = [
                        None for _ in range(self.vllm_tensor_parallel_size)
                    ]
                    torch.distributed.all_gather_object(
                        gathered_prompts, prompts_text, group=self.tp_group
                    )
                    all_prompts_text = [
                        p for sublist in gathered_prompts for p in sublist
                    ]

                    if has_images:
                        gathered_images = [
                            None for _ in range(self.vllm_tensor_parallel_size)
                        ]
                        torch.distributed.all_gather_object(
                            gathered_images, images, group=self.tp_group
                        )
                        all_images = [
                            img for sublist in gathered_images for img in sublist
                        ]
                    else:
                        all_images = None
                else:
                    all_prompts_text = prompts_text
                    all_images = images if has_images else None

                if has_images and all_images:
                    vllm_inputs = []
                    for prompt, image in zip(all_prompts_text, all_images):
                        if image is not None:
                            vllm_inputs.append(
                                {"prompt": prompt, "multi_modal_data": {"image": image}}
                            )
                        else:
                            vllm_inputs.append(prompt)
                else:
                    vllm_inputs = all_prompts_text

                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(
                        vllm_inputs, sampling_params=sampling_params, use_tqdm=False
                    )

                completion_ids = [
                    output.token_ids
                    for outputs in all_outputs
                    for output in outputs.outputs
                ]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(
                        group=self.tp_group
                    )
                    tp_slice = slice(
                        local_rank_in_group * orig_size,
                        (local_rank_in_group + 1) * orig_size,
                    )
                    completion_ids = completion_ids[tp_slice]

            # Pad the completions - NO concatenation with prompts (encoder-decoder difference)
            completion_ids = [
                torch.tensor(ids, device=device) for ids in completion_ids
            ]
            completion_ids = pad(completion_ids, padding_value=self.pad_token_id)
            # Create prompt_completion_ids for compatibility with base method calls
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        elif self.use_transformers_paged:
            # Re-process inputs for paged generation if needed
            # Note: images are already validated and preprocessed above
            paged_prompt_inputs = self.processing_class(text=prompts_text, **kwargs)
            previous_attn = self.model_wrapped.config._attn_implementation

            if is_flash_attn_2_available():
                self.model_wrapped.config._attn_implementation = "paged_attention"
            else:
                self.model_wrapped.config._attn_implementation = "sdpa_paged"
            with (
                profiling_context(self, "transformers.generate_batch"),
                unwrap_model_for_generation(
                    self.model_wrapped,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False)
                if self.is_fsdp_enabled
                else nullcontext(),
            ):
                # Cast to the appropriate dtype based on training configuration
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)
                with torch.inference_mode():
                    all_outputs = unwrapped_model.generate_batch(
                        paged_prompt_inputs.input_ids,
                        generation_config=self.generation_config,
                        progress_bar=False,
                    )
            # For encoder-decoder: generate_batch returns only completion tokens
            completion_ids = [
                output.generated_tokens for output in all_outputs.values()
            ]
            completion_ids = [
                torch.tensor(ids, device=device) for ids in completion_ids
            ]
            completion_ids = pad(
                completion_ids, padding_value=self.pad_token_id, padding_side="right"
            )
            prompt_ids = [
                torch.tensor(ids, device=device)
                for ids in paged_prompt_inputs.input_ids
            ]
            prompt_ids = pad(
                prompt_ids, padding_value=self.pad_token_id, padding_side="left"
            )
            # Create prompt_completion_ids for compatibility with base method calls
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            # Restore the original attention implementation, training mode
            self.model_wrapped.config._attn_implementation = previous_attn
        else:
            # Regular generation path - THIS IS THE KEY ENCODER-DECODER DIFFERENCE
            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False)
                if self.is_fsdp_enabled
                else nullcontext(),
            ):
                prompt_inputs["input_ids"], prompt_inputs["attention_mask"] = (
                    prompt_ids,
                    prompt_mask,
                )
                # ENCODER-DECODER CHANGE: generate() returns ONLY completion_ids
                # No prompt slicing needed (unlike decoder-only at lines 1581-1583 in base)
                completion_ids = unwrapped_model.generate(
                    **prompt_inputs,
                    generation_config=self.generation_config,
                    disable_compile=True,
                )
            # Create prompt_completion_ids for compatibility with base method calls
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Convert tensor to a list of lists of token IDs. This will be passed to the reward function, avoiding the need
        # to re-tokenize completions if the reward is computed from tokens.
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m]
            for row, mask_row in zip(completion_ids, completion_mask)
        ]

        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        completion_lengths = completion_mask.sum(1)

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = (
                completion_mask * (~truncated_completions).unsqueeze(1).int()
            )

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
        batch_size = (
            self.args.per_device_train_batch_size
            if mode == "train"
            else self.args.per_device_eval_batch_size
        )

        with torch.no_grad():
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            generate_every = (
                self.args.steps_per_generation * self.num_iterations
            )  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    pixel_values=prompt_inputs.get("pixel_values"),
                    image_grid_thw=prompt_inputs.get("image_grid_thw"),
                    pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                    image_sizes=prompt_inputs.get("image_sizes"),
                )
            else:
                old_per_token_logps = None

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        pixel_values=prompt_inputs.get("pixel_values"),
                        image_grid_thw=prompt_inputs.get("image_grid_thw"),
                        pixel_attention_mask=prompt_inputs.get("pixel_attention_mask"),
                        image_sizes=prompt_inputs.get("image_sizes"),
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = (
                            self._get_per_token_logps_and_entropies(
                                self.model,
                                prompt_completion_ids,
                                attention_mask,
                                logits_to_keep,
                                batch_size=batch_size,
                                pixel_values=prompt_inputs.get("pixel_values"),
                                image_grid_thw=prompt_inputs.get("image_grid_thw"),
                                pixel_attention_mask=prompt_inputs.get(
                                    "pixel_attention_mask"
                                ),
                                image_sizes=prompt_inputs.get("image_sizes"),
                            )
                        )
            else:
                ref_per_token_logps = None

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = (
                    prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                )
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}]
                )
        else:
            completions = completions_text

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(
            inputs, original_prompts, completions, completion_ids_list
        )

        # Apply weights to each reward function's output and sum
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(
            std_grouped_rewards, torch.zeros_like(std_grouped_rewards)
        )

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = (
            advantages.clone()
        )  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (
                self.accelerator.gather(attention_mask.sum()).sum().item()
            )
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(
            agg_completion_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_length"].append(
            agg_completion_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_length"].append(
            agg_completion_lengths.float().max().item()
        )

        # Identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(
            agg_completion_lengths
        )
        self._metrics[mode]["completions/clipped_ratio"].append(
            clipped_completions_ratio
        )
        if (
            len(term_completion_lengths) == 0
        ):  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(
            term_completion_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_terminated_length"].append(
            term_completion_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_terminated_length"].append(
            term_completion_lengths.float().max().item()
        )

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(
            is_std_zero.float().mean().item()
        )

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        # Return all relevant tensors
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "rewards_per_func": rewards_per_func,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        pixel_attention_mask=None,
        image_sizes=None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute per-token log probabilities and entropies (encoder-decoder specific).

        This method overrides the base implementation to handle encoder-decoder models
        where encoder and decoder inputs must be passed separately.

        Key Differences from Base:
        ---------------------------
        1. input_ids contains [encoder_input | decoder_input] concatenated
        2. Split and pass separately as input_ids (encoder) and decoder_input_ids (decoder)
        3. No logits slicing needed (line 1116 in base) - encoder-decoder logits already
           match decoder sequence length
        4. Get completion_ids from decoder_input_ids instead of slicing concatenated tensor

        Args:
            model: The encoder-decoder model
            input_ids: Concatenated [encoder_inputs | decoder_inputs] (will be split)
            attention_mask: Concatenated attention masks (will be split)
            logits_to_keep: Length of decoder sequence (number of completion tokens)
            batch_size: Batch size for chunking (reduces memory peak)
            compute_entropy: Whether to compute entropy values
            pixel_values: Vision model pixel values (optional)
            image_grid_thw: Vision model image grid dimensions (optional)
            pixel_attention_mask: Vision model attention mask (optional)
            image_sizes: Vision model image sizes (optional)

        Returns:
            Tuple of (log_probabilities, entropies) where:
            - log_probabilities: shape (batch, logits_to_keep)
            - entropies: shape (batch, logits_to_keep) or None if not computed
        """
        batch_size = batch_size or input_ids.size(
            0
        )  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        all_entropies = []

        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            # CRITICAL ENCODER-DECODER CHANGE: Split concatenated inputs
            # In _compute_loss, we concatenate:
            #   input_ids = torch.cat([prompt_ids, decoder_input_ids], dim=1)
            # where decoder_input_ids has BOS prepended, so its length is (logits_to_keep + 1)
            #
            # logits_to_keep = number of completion tokens (without BOS) = L
            # decoder_input_ids length = L + 1 (includes BOS)
            #
            # So we need to split based on decoder length (logits_to_keep + 1):
            prompt_length = input_ids_batch.size(1) - (logits_to_keep + 1)
            encoder_input_ids = input_ids_batch[:, :prompt_length]
            encoder_attention_mask = attention_mask_batch[:, :prompt_length]
            decoder_input_ids = input_ids_batch[:, prompt_length:]  # Last (L+1) tokens
            decoder_attention_mask = attention_mask_batch[:, prompt_length:]

            # Build model inputs for encoder-decoder architecture
            # Pass encoder and decoder inputs separately (NOT concatenated)
            model_inputs = {
                "input_ids": encoder_input_ids,  # Encoder input (prompt)
                "attention_mask": encoder_attention_mask,  # Encoder mask
                "decoder_input_ids": decoder_input_ids,  # Decoder input (completion)
                "decoder_attention_mask": decoder_attention_mask,  # Decoder mask
            }

            # Add vision inputs if present (same as base implementation)
            if image_grid_thw is not None and pixel_values is not None:
                model_inputs["image_grid_thw"] = image_grid_thw[
                    start : start + batch_size
                ]
                start_pixel_idx = image_grid_thw[:start].prod(-1).sum().item()
                end_pixel_idx = (
                    image_grid_thw[: start + batch_size].prod(-1).sum().item()
                )
                model_inputs["pixel_values"] = pixel_values[
                    start_pixel_idx:end_pixel_idx
                ]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size]
            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[
                    start : start + batch_size
                ]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start : start + batch_size]

            # Note: We don't add logits_to_keep for encoder-decoder models
            # The decoder naturally produces logits aligned with decoder_input_ids

            # Forward pass through encoder-decoder model
            logits = model(**model_inputs).logits

            # Exclude the last value: it corresponds to the next token prediction
            logits = logits[:, :-1, :]  # (B, decoder_len-1, vocab_size)

            # ENCODER-DECODER DIFFERENCE: No slicing needed!
            # For decoder-only models (base), need to slice: logits[:, -logits_to_keep:, :]
            # For encoder-decoder, logits already have shape (B, decoder_len-1, vocab_size)
            # which matches our decoder sequence length

            # Divide logits by sampling temperature
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature

            # CRITICAL FIX FOR FP16 STABILITY:
            # Compute log-probs in float32 to avoid FP16 overflows in log-softmax
            # while keeping the autograd graph (cast is differentiable).
            logits_fp32 = logits.float()

            # CRITICAL FIX: Get completion tokens by removing BOS (first position)
            # decoder_input_ids: [BOS, t1, t2, ..., tL]  shape: (B, L+1)
            # After [:, 1:]: [t1, t2, ..., tL]  shape: (B, L)
            # This matches logits shape after removing last position: (B, L)
            completion_ids_batch = decoder_input_ids[
                :, 1:
            ]  # Remove BOS, keep completion tokens

            # Compute log probabilities for the actual tokens in FP32
            logps = selective_log_softmax(logits_fp32, completion_ids_batch)
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits_fp32)
                all_entropies.append(entropies)

        # Concatenate results from all batches
        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

    def _prepare_decoder_input_ids(self, completion_ids: torch.Tensor) -> torch.Tensor:
        """
        Prepare decoder input IDs by prepending BOS token for teacher forcing.

        In encoder-decoder models, the decoder needs a starting token (BOS) to begin
        generation during training. This method prepends the decoder_start_token_id
        to the completion tokens.

        Args:
            completion_ids: Completion token IDs without BOS, shape (batch_size, seq_len)

        Returns:
            Decoder input IDs with BOS prepended, shape (batch_size, seq_len+1)

        Example:
            completion_ids = [[101, 102, 103]]  # Without BOS
            decoder_input_ids = [[0, 101, 102, 103]]  # With BOS (0 is BOS)
        """
        batch_size = completion_ids.size(0)
        bos_token_id = self.model.config.decoder_start_token_id

        # Create BOS tokens for all sequences in the batch
        bos_tokens = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=completion_ids.dtype,
            device=completion_ids.device,
        )

        # Prepend BOS to completion tokens
        decoder_input_ids = torch.cat([bos_tokens, completion_ids], dim=1)

        return decoder_input_ids

    def _compute_loss(self, model, inputs):
        """
        Compute GRPO loss (encoder-decoder specific).

        This method overrides the base implementation to handle encoder-decoder models
        where encoder and decoder inputs must be kept separate instead of concatenated.

        Key Differences from Base:
        ---------------------------
        1. Don't concatenate prompt+completion (lines 1822-1823 in base removed)
        2. Prepare decoder inputs with BOS token for teacher forcing
        3. Pass encoder and decoder inputs separately to the model
        4. Loss computation logic (advantages, KL, clipping) remains unchanged

        Args:
            model: The encoder-decoder model
            inputs: Dictionary with keys:
                - prompt_ids: Encoder input token IDs
                - prompt_mask: Encoder attention mask
                - completion_ids: Decoder output token IDs (without BOS)
                - completion_mask: Decoder attention mask
                - advantages: Computed advantages for policy gradient
                - ref_per_token_logps: Reference model log probs (optional)
                - old_per_token_logps: Old policy log probs (optional)
                - pixel_values, image_grid_thw, etc.: Vision inputs (optional)

        Returns:
            Scalar loss tensor for backpropagation
        """
        # Extract inputs
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )

        # CRITICAL ENCODER-DECODER CHANGE: Don't concatenate like base (lines 1822-1823)!
        # Base code (REMOVED):
        #   input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        #   attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        #
        # Instead, we prepare decoder inputs with BOS token for teacher forcing

        # Prepare decoder inputs by prepending BOS token
        decoder_input_ids = self._prepare_decoder_input_ids(completion_ids)

        # Extend completion_mask to include BOS position (always attended)
        bos_mask = torch.ones(
            (completion_ids.size(0), 1),
            dtype=completion_mask.dtype,
            device=completion_mask.device,
        )
        decoder_attention_mask = torch.cat([bos_mask, completion_mask], dim=1)

        # For compatibility with our overridden _get_per_token_logps_and_entropies,
        # we still create concatenated input_ids and attention_mask.
        # The override will split them correctly into encoder/decoder parts.
        input_ids = torch.cat([prompt_ids, decoder_input_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, decoder_attention_mask], dim=1)

        # Number of logits to compute (decoder sequence length minus BOS)
        # We exclude BOS from loss computation since it's not predicted
        logits_to_keep = decoder_input_ids.size(1) - 1

        # Compute per-token log probabilities and entropies
        # This calls our overridden _get_per_token_logps_and_entropies which
        # will split input_ids into encoder/decoder parts correctly
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
        )

        # Apply entropy masking if configured
        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(
                entropies, completion_mask, 1 - self.top_entropy_quantile
            )
        else:
            entropy_mask = None

        # Compute KL divergence with reference model if beta > 0
        token_mask = completion_mask
        mask_f = token_mask.float()

        if self.beta != 0.0:
            with torch.autocast(device_type="cuda", enabled=False):
                pl = per_token_logps.float()
                rl = inputs["ref_per_token_logps"].float()
                delta = (rl - pl) * mask_f
                delta = delta.clamp(min=-20.0, max=20.0)
                exp_delta = torch.exp(delta)
                per_token_kl = (exp_delta - delta - 1.0) * mask_f
                per_token_kl = torch.nan_to_num(
                    per_token_kl, nan=0.0, posinf=1e4, neginf=0.0
                )

        # Extract advantages for policy gradient
        advantages = inputs["advantages"]

        # Handle old policy log probabilities for importance sampling
        # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
        # old_per_token_logps == per_token_logps, so we can skip its computation
        # and use per_token_logps.detach() instead
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = (
            per_token_logps.detach()
            if old_per_token_logps is None
            else old_per_token_logps
        )

        # Compute log importance weights for importance sampling
        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(
                -1
            ) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. "
                "Possible values are 'token' and 'sequence'."
            )
        # From here, log_importance_weights (and all subsequent tensors) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)

        liw = log_importance_weights.float()
        liw = liw.clamp(min=-20.0, max=20.0)
        with torch.autocast(device_type="cuda", enabled=False):
            coef_1 = torch.exp(liw).float()
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        # Compute per-token loss with clipping (PPO-style)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # Apply entropy masking if enabled
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        # Add KL penalty if beta > 0
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # Aggregate per-token losses based on loss type
        if self.loss_type == "grpo":
            loss = (
                (per_token_loss * completion_mask).sum(-1)
                / completion_mask.sum(-1).clamp(min=1.0)
            ).mean()
        elif self.loss_type == "bnpo":
            loss = (
                per_token_loss * completion_mask
            ).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (
                per_token_loss.size(0) * self.max_completion_length
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log metrics for monitoring
        mode = "train" if self.model.training else "eval"

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            """Compute mean over non-masked tokens."""
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        # Log KL divergence if using reference model
        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(
                self.accelerator.gather(mean_kl).nanmean().item()
            )

        # Log entropy
        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(
            self.accelerator.gather(mean_entropy).nanmean().item()
        )

        # Compute and log clipping statistics
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (
            advantages.unsqueeze(1) > 0
        )
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(
            gathered_low_clip.nanmean().item()
        )
        self._metrics[mode]["clip_ratio/low_min"].append(
            nanmin(gathered_low_clip).item()
        )
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(
            gathered_high_clip.nanmean().item()
        )
        self._metrics[mode]["clip_ratio/high_max"].append(
            nanmax(gathered_high_clip).item()
        )
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(
            gathered_clip_ratio.nanmean().item()
        )

        return loss
