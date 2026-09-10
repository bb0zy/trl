 import copy
import importlib.resources as pkg_resources
 
from collections import defaultdict, deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import transformers
from accelerate.logging import get_logger
from accelerate.utils import gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from huggingface_hub import CommitScheduler, DatasetCard, DatasetCardData, create_repo
from packaging.version import Version
from torch import nn
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    is_trackio_available,
    is_wandb_available,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import is_peft_available, is_rich_available

from ..chat_template_utils import (
    _SUPPORTS_RESPONSE_TEMPLATE,
    add_response_schema,
    get_training_chat_template,
    is_chat_template_prefix_preserving,
    parse_response,
    supports_tool_calling,
)
from ..data_utils import is_conversational
from ..extras.profiling import profiling_context
from .grpo_trainer import GRPOTrainer
 
from ..models import unwrap_model_for_generation
 
from .utils import pad

class DOPDTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields, has_tool_images=False):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            # Sync weights if training step changed
            if self.state.global_step != self._last_loaded_step:
                with profiling_context(self, "sync_weights"):
                    self.vllm_generation.sync_weights()
                self._last_loaded_step = self.state.global_step

            # Generate using vLLM with raw token IDs
            num_generations = self.num_generations if mode == "train" else self.num_generations_eval
            _, completion_ids, logprobs, _ = self.vllm_generation.generate(
                prompts=prompt_ids,
                images=images,
                num_generations=num_generations,
                profiler=profiling_context(self, "vLLM.generate"),
            )
            # vLLM returns per-token top-k logprobs; keep only the top-1 (sampled token) logprob
            logprobs_top1 = [[lp[0] for lp in seq] for seq in logprobs]

        elif self.use_transformers_continuous_batching:
            with (
                profiling_context(self, "transformers.generate_batch"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                self._dist.summon_full_params(self.model_wrapped, recurse=False),
            ):
                # Cast to the appropriate dtype based on training configuration
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)
                if self.args.cast_lm_head_to_fp32:
                    unwrapped_model.lm_head.to(torch.float32)
                all_outputs = unwrapped_model.generate_batch(
                    prompt_ids,
                    generation_config=self.generation_config,
                    continuous_batching_config=self.continuous_batching_config,
                    progress_bar=False,
                )
                unwrapped_model.train()
            completion_ids = [output.generated_tokens for output in all_outputs.values()]
            logprobs = None

        else:
            # Regular generation path: left-pad token IDs into tensors
            prompt_tensors = [torch.tensor(ids) for ids in prompt_ids]
            padded_ids = pad(prompt_tensors, padding_value=self._tokenizer.pad_token_id, padding_side="left")
            attention_mask = pad([torch.ones_like(t) for t in prompt_tensors], padding_value=0, padding_side="left")
            generate_inputs = {"input_ids": padded_ids, "attention_mask": attention_mask}
            # For VLMs, include multimodal fields as tensors (pixel_values, image_grid_thw, etc.)
            for k, v in multimodal_fields.items():
                if isinstance(v, torch.Tensor):
                    generate_inputs[k] = v
                elif isinstance(v, list) and v and isinstance(v[0], list):
                    # Per-token field (e.g., token_type_ids): left-pad like input_ids
                    generate_inputs[k] = pad([torch.tensor(x) for x in v], padding_value=0, padding_side="left")
                else:
                    generate_inputs[k] = torch.tensor(np.array(v))

            # For VLM tool images: build token type IDs from the padded input IDs.
            if self._is_vlm and self.tools and has_tool_images:
                mm_ids = torch.zeros_like(padded_ids)
                if self._image_pad_token_id is not None:
                    mm_ids[padded_ids == self._image_pad_token_id] = 1
                if self._video_pad_token_id is not None:
                    mm_ids[padded_ids == self._video_pad_token_id] = 2

                # Use the same key the model expects: token_type_ids for models like Gemma,
                # mm_token_type_ids for models like Qwen.
                if "image_grid_thw" in generate_inputs:
                    generate_inputs["mm_token_type_ids"] = mm_ids
                else:
                    generate_inputs["token_type_ids"] = mm_ids

            generate_inputs = super()._prepare_inputs(generate_inputs)

            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                    generation_kwargs=self.generation_kwargs,  # Override model.generation_config with generation_kwargs to fix transformers#42762
                ) as unwrapped_model,
                torch.no_grad(),
                self._dist.summon_full_params(self.model_wrapped, recurse=False),
            ):
                prompt_completion_ids = unwrapped_model.generate(
                    **generate_inputs, generation_config=self.generation_config
                )
            # Compute prompt length and extract completion ids
            prompt_length = generate_inputs["input_ids"].size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # Mask everything after the first EOS token
            is_eos = completion_ids == self._tokenizer.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            completion_ids = [
                c[m].tolist() for c, m in zip(completion_ids.cpu(), completion_mask.bool().cpu(), strict=True)
            ]
            logprobs = None  # not used in this case

        return completion_ids, logprobs_top1, logprobs

    def _generate(self, prompts: list):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Copy the prompts to avoid modifying the original list
        prompts = copy.deepcopy(prompts)

        if self.rollout_func is not None:
            # Keep vLLM weights in sync for custom rollouts that rely on vLLM utilities.
            if self.use_vllm and self.state.global_step != self._last_loaded_step:
                with profiling_context(self, "sync_weights"):
                    self.vllm_generation.sync_weights()
                self._last_loaded_step = self.state.global_step

            # Pass prompts to rollout_func preserving structured messages.
            # Chat templating must happen inside rollout_func, at the backend boundary, so that
            # multimodal content (images, typed content blocks) is not lost before rollout logic runs.
            output = self.rollout_func(prompts, self)
            required_keys = {"prompt_ids", "completion_ids", "logprobs"}
            missing_keys = required_keys - output.keys()
            if missing_keys:
                missing_keys_list = sorted(missing_keys)
                raise ValueError(f"rollout_func must return keys {missing_keys_list} in its output dict.")
            extra_fields = {k: v for k, v in output.items() if k not in required_keys}
            prompt_ids, completion_ids, logprobs = output["prompt_ids"], output["completion_ids"], output["logprobs"]
            images = None
            multimodal_fields = {}
        else:
            prompt_ids, images, multimodal_fields = self._tokenize_prompts(prompts)
            completion_ids, logprobs, logprobs_topk = self._generate_single_turn(prompt_ids, images, multimodal_fields)
            extra_fields = {}
            extra_fields.pop("logprobs_topk", logprobs_topk)

        # Decode completions. It's important to use `parse_response` when possible, because it handles tool calls.
        if is_conversational({"prompt": prompts[0]}):
            if Version(transformers.__version__) >= Version("5.0.0") and (  # parse_response added in v5
                getattr(self._tokenizer, "response_template", None) is not None  # new-style
                or getattr(self._tokenizer, "response_schema", None) is not None  # old-style
            ):
                completions = [
                    [parse_response(self._tokenizer, ids, prefix=prompt_ids[i])]
                    for i, ids in enumerate(completion_ids)
                ]
            else:
                contents = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
                completions = [[{"role": "assistant", "content": content}] for content in contents]
        else:
            completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Extract tool calls from the completions and (possibly) execute them
        tool_images = []
        if self.tools:
            (
                tool_mask,
                completions,
                completion_ids,
                logprobs,
                tool_call_count,
                tool_failure_count,
                tool_images,
            ) = self._tool_call_loop(
                prompts, prompt_ids, completion_ids, completions, logprobs, images, multimodal_fields
            )
            # Merge tool response images into the images list for the forward pass
            if any(imgs for imgs in tool_images):
                if images is None:
                    images = [imgs if imgs else None for imgs in tool_images]
                else:
                    images = [(existing or []) + new for existing, new in zip(images, tool_images, strict=True)]
        else:
            # Support custom env_mask from rollout_func (e.g., for environment feedback masking)
            # Internally treated as tool_mask - marks model tokens (1) vs external tokens (0)
            tool_mask = extra_fields.pop("env_mask", None)

        # Get completion length per sequence, used for logging
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        if tool_mask is not None:  # count only model-generated tokens (tool_mask=1)
            completion_lengths = torch.tensor([sum(mask) for mask in tool_mask], device=device)
        else:
            completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        # Fail clearly if the generation backend returned no completions (avoids a cryptic min() error below).
        if agg_completion_lengths.numel() == 0:
            raise RuntimeError(
                "No completions were generated. This usually means the generation backend failed to return any "
                "results; see the generation logs above for the underlying error."
            )
        total_prompt_tokens = agg_prompt_lengths.sum()

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + agg_completion_lengths.sum()).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        eos_and_pad = [self._tokenizer.eos_token_id, self._tokenizer.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        if self.tools:
            agg_tool_call_count = self.accelerator.gather(torch.tensor(tool_call_count, device=device)).sum()
            tool_call_frequency = (agg_tool_call_count / len(agg_prompt_lengths)).item()
            self._metrics[mode]["tools/call_frequency"].append(tool_call_frequency)
            agg_tool_failure_count = self.accelerator.gather(torch.tensor(tool_failure_count, device=device)).sum()
            failure_frequency = (
                (agg_tool_failure_count / agg_tool_call_count).item() if agg_tool_call_count > 0 else 0.0
            )
            self._metrics[mode]["tools/failure_frequency"].append(failure_frequency)

        return prompt_ids, completion_ids, tool_mask, completions, logprobs, extra_fields, images, tool_images



    
