import copy
import importlib.resources as pkg_resources
from ..generation.vllm_generation import VLLMGeneration
from collections import defaultdict, deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol
from ..extras.profiling import profiling_context, profiling_decorator
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
from ..generation.vllm_client import VLLMClient
import math
import requests
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
from .utils import (
    RepeatSampler,
    create_model_from_path,
    disable_dropout_in_model,
    entropy_from_logits,
    get_callable_name,
    get_config_model_id,
    identity,
    maybe_gather_lm_head_ctx,
    nanmax,
    nanmin,
    nanstd,
    pad,
    print_prompt_completions_sample,
    repeat_iterable_dataset,
    selective_log_softmax,
    shuffle_sequence_dict,
    shutdown_event_loop_in_daemon,
    split_pixel_values_by_grid,
    split_tensor_dict,
    start_event_loop_in_daemon,
    unsplit_pixel_values_by_grid,
    use_adapter,
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
from .dopd_config import DOPDConfig
from .grpo_trainer import GRPOTrainer
from ..models import unwrap_model_for_generation

class DOPDTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        self.use_vllm = kwargs["args"].use_vllm 

        
        kwargs["args"].use_vllm = False
        super().__init__(*args, **kwargs)
        self.topk_logprobs_num = args.topk_logprobs_num
        self.teacher_ref_url = args.teacher_ref_url
        self.teacher_rl_url = args.teacher_rl_url
        

        if self.use_vllm:
            self.vllm_generation = VLLMGeneration(
                model=self.model,
                accelerator=self.accelerator,
                processing_class=self.processing_class,
                mode=kwargs["args"].vllm_mode,
                server_base_url=kwargs["args"].vllm_server_base_url,
                server_host=kwargs["args"].vllm_server_host,
                server_port=kwargs["args"].vllm_server_port,
                group_port=kwargs["args"].vllm_group_port,
                server_timeout=kwargs["args"].vllm_server_timeout,
                tensor_parallel_size=kwargs["args"].vllm_tensor_parallel_size,
                gpu_memory_utilization=kwargs["args"].vllm_gpu_memory_utilization,
                max_model_length=kwargs["args"].vllm_max_model_length,
                max_num_seqs=kwargs["args"].per_device_train_batch_size * kwargs["args"].vllm_tensor_parallel_size * kwargs["args"].steps_per_generation,
                enable_sleep_mode=kwargs["args"].vllm_enable_sleep_mode,
                model_impl=kwargs["args"].vllm_model_impl,
                trust_remote_code=kwargs["args"].trust_remote_code,
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                max_completion_length=self.max_completion_length,
                logprobs=self.topk_logprobs_num, 
                generation_kwargs=kwargs["args"].generation_kwargs,
            )
            self._last_loaded_step = -1


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
            _, completion_ids, logprobs, topk_ids = self.vllm_generation.generate(
                prompts=prompt_ids,
                images=images,
                num_generations=num_generations,
                profiler=profiling_context(self, "vLLM.generate"),
            )
            # vLLM returns per-token top-k logprobs; keep only the top-1 (sampled token) logprob
            logprobs_top1 = [[lp[0] for lp in seq] for seq in logprobs]
            logprobs = [[lp[1:] for lp in seq] for seq in logprobs]
            topk_ids = [[lp[1:] for lp in seq] for seq in topk_ids]

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

        return completion_ids, logprobs_top1, logprobs, topk_ids

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
            completion_ids, logprobs, logprobs_topk, topk_ids = self._generate_single_turn(prompt_ids, images, multimodal_fields)
            extra_fields = {}
            extra_fields["logprobs_topk"] = logprobs_topk
            extra_fields["topk_ids"] = topk_ids



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



    def get_teacher_topk_logprobs(self, 
                                    client,                          # VLLMClient 实例
                                    input_ids: torch.Tensor,         # (B, padded_total_len) — prompt + completion，已 pad
                                    prompt_lengths: list[int],       # 每个样本的 prompt 长度
                                    student_topk_ids: torch.Tensor,  # (B, padded_completion_len, K) — 学生 top-K token ID
                                    temperature: float = 1.0) -> torch.Tensor:
      
        B = input_ids.size(0)
        max_total = input_ids.size(1)
        K = student_topk_ids.size(-1)
        device = student_topk_ids.device
        sequences = [ids.tolist() for ids in input_ids]
        result = client.get_sequence_logprobs(
            sequences=sequences,
            prompt_lengths=prompt_lengths,
            top_logprobs=-1,
            temperature=temperature,
        )
        # teacher logprobs: list[list[list[float]]], 每个样本 shape (completion_len, topk)
        # teacher token_ids: list[list[list[int]]],  每个样本 shape (completion_len, topk)

        tea_logps_list = result["logprobs"]
        tea_ids_list = result["logprob_token_ids"]

        output = torch.full((B, max_total, K), -math.inf, device=device, dtype=torch.float32)

        for b in range(B):
            plen = prompt_lengths[b]
            stu_ids = student_topk_ids[b].to(torch.long)   # (padded_completion_len, K)

            tea_ids = torch.tensor(tea_ids_list[b], device=device)    # (completion_len, vocab_size)
            tea_logps = torch.tensor(tea_logps_list[b], device=device)  # (completion_len, vocab_size)

            c_len = tea_ids.size(0)                              # 该样本实际 completion 长度
            stu_slice = stu_ids[:c_len]                          # (c_len, K)

            # stu_slice: (c_len, K, 1) vs tea_ids: (c_len, 1, vocab_size) → (c_len, K, vocab_size)
            matches = (stu_slice.unsqueeze(-1) == tea_ids.unsqueeze(1))

            # tea_logps: (c_len, 1, vocab_size) → 乘 matches 取对应位置 → sum(dim=-1) → (c_len, K)
            selected = (tea_logps.unsqueeze(1) * matches).sum(dim=-1)
            selected[~matches.any(dim=-1)] = -math.inf

            output[b, plen : plen + c_len] = selected
        return output

    
    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        compute_aux_loss=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        spatial_shapes=None,
        num_tiles=None,
        image_sizes=None,
        token_type_ids=None,
        mm_token_type_ids=None,
        image_position_ids=None,
        topk_ids = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Compute log-probs, (optionally) entropies, and (optionally) the MoE load-balancing aux loss."""
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        all_entropies = []
        all_aux_losses = []
        for start in range(0, input_ids.size(0), batch_size):
            end = min(start + batch_size, input_ids.size(0))  # the last chunk can be smaller than batch_size
            input_ids_batch = input_ids[start:end]
            attention_mask_batch = attention_mask[start:end]

            # Build model inputs
            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
            if num_images is not None:
                cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                img_start, img_end = cum_imgs[start], cum_imgs[end]
            if image_grid_thw is not None and pixel_values is not None:
                rows_per_image = image_grid_thw.prod(dim=-1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
                row_start, row_end = cum_rows[start].item(), cum_rows[end].item()
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
            elif image_position_ids is not None and pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[img_start:img_end]
                model_inputs["image_position_ids"] = image_position_ids[img_start:img_end]
            elif spatial_shapes is not None and pixel_values is not None:
                # LFM2-VL tensors are tile-indexed.
                cum_tiles = torch.tensor([0] + num_tiles).cumsum(0)
                tile_start, tile_end = cum_tiles[start], cum_tiles[end]
                model_inputs["pixel_values"] = pixel_values[tile_start:tile_end]
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[tile_start:tile_end]
                model_inputs["spatial_shapes"] = spatial_shapes[tile_start:tile_end]
            elif num_tiles is not None and pixel_values is not None:
                # InternVL tensors are tile-indexed.
                cum_tiles = torch.tensor([0] + num_tiles).cumsum(0)
                tile_start, tile_end = cum_tiles[start], cum_tiles[end]
                model_inputs["pixel_values"] = pixel_values[tile_start:tile_end]
            elif pixel_values is not None and pixel_values.size(0) == sum(num_images):
                model_inputs["pixel_values"] = pixel_values[img_start:img_end]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start:end]
            if pixel_attention_mask is not None and spatial_shapes is None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[start:end]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[img_start:img_end]
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[start:end]
            if mm_token_type_ids is not None:
                model_inputs["mm_token_type_ids"] = mm_token_type_ids[start:end]

            # Only add logits_to_keep if the model supports it
            if "logits_to_keep" in self.model_kwarg_keys:
                # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

            # MoE models: request router logits so the model returns `outputs.aux_loss`. VLM wrappers honor this only
            # as a forward kwarg (not from the model config), so it must be passed here.
            if compute_aux_loss:
                model_inputs["output_router_logits"] = True

            outputs = model(**model_inputs)
            logits = outputs.logits
            # Exclude the last value: it corresponds to the next token pred
            logits = logits[:, :-1, :]  # (B, L-1, H)
            # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
            logits = logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature

            if topk_ids == None:
                completion_ids = input_ids_batch[:, -logits_to_keep:]
                logps = selective_log_softmax(logits, completion_ids)
            else:
                
                logps = selective_log_softmax(logits, topk_ids)  # compute logprobs->[batch_size,logtis_to_keep,topk]
            all_logps.append(logps) 

            if compute_entropy:
                # The entropy bonus is a differentiable loss term, so entropies must carry grad when it is
                # active. Otherwise they only feed logging and the top_entropy_quantile mask (neither
                # differentiable), so we skip grad to avoid retaining the memory-heavy full-vocab softmax.
                if self._entropy_bonus_enabled:
                    entropies = entropy_from_logits(logits)
                else:
                    with torch.no_grad():
                        entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

            if compute_aux_loss:
                all_aux_losses.append(outputs.aux_loss)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        aux_loss = torch.stack(all_aux_losses).mean() if compute_aux_loss else None
        return logps, entropies, aux_loss

    def _dopd_loss(self, per_token_logps:torch.Tensor, input_ids:torch.Tensor):

        pass

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        _, topk_ids = inputs["logprobs_topk"], inputs["topk_ids"]

        

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]

        

        batch_size = completion_ids.size(0)                   # 最长的 completion 长度
        K = len(topk_ids[0][0]) if topk_ids[0] else 0                   # top-K
        padded_topk_ids = torch.zeros(batch_size, logits_to_keep, K, dtype=torch.long, device=completion_ids.device)
        for b, ids in enumerate(topk_ids):
            t = torch.tensor(ids, dtype=torch.long)                      # (actual_completion_len, K)
            padded_topk_ids[b, :t.size(0)] = t
        topk_ids = padded_topk_ids

        # Compute the per_token_logps and the entropy at each position in the completion
        # per_token_logps -> [batch_size, logits_to_keep, topk]

        per_token_logps, entropies, aux_loss = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            compute_aux_loss=self.aux_loss_enabled,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            spatial_shapes=inputs.get("spatial_shapes"),
            num_tiles=inputs.get("num_tiles"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
            image_position_ids=inputs.get("image_position_ids"),
            topk_ids = topk_ids
        )

        with torch.no_grad():
            prompt_lengths = (prompt_ids != self._tokenizer.pad_token_id).sum(dim=-1).tolist()
            # 计算两个教师的评分
            tea_ref_topk_probs = self.get_teacher_topk_logprobs(
                client=VLLMClient(base_url=self.teacher_ref_url, connection_timeout=self.server_timeout),
                input_ids=input_ids,
                prompt_lengths=prompt_lengths,
                student_topk_ids=topk_ids,)
            tea_rl_topk_probs = self.get_teacher_topk_logprobs(
                client=VLLMClient(base_url=self.teacher_rl_url, connection_timeout=self.server_timeout),
                input_ids=input_ids,
                prompt_lengths=prompt_lengths,
                student_topk_ids=topk_ids,
            )
            r = tea_rl_topk_probs - tea_ref_topk_probs # [batch_size, input_ids.size[0], topk]
            r = r[:,-logits_to_keep:,:] # [batch_size, logits_to_keep, topk]
            stu_softmax_probs = torch.nn.functional.softmax(per_token_logps, dim=-1) # [batch_size, logits_to_keep, topk]
            r = r * stu_softmax_probs #[batch_size, logits_to_keep, topk]
        
        
        loss_1 = (r * per_token_logps * mask).sum() / mask.sum()

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            logps_top1 = per_token_logps[:, :, [0]]
            kl = F.kl_div(logps_top1.squeeze(-1), ref_per_token_logps, reduction='none', log_target=True)
            loss_2 = (kl * mask).sum() / mask.sum()
            loss = loss_1 + self.beta * loss_2
        else:
            loss_2 = torch.tensor(0.0, device=loss_1.device)
            loss = loss_1

        # ---------- metrics ----------
        mode = "train" if self.model.training else "eval"

        def global_masked_mean(x):
            if x.shape[1] == 1:
                local_sum = x.sum()
                local_count = torch.tensor(float(x.shape[0]), device=x.device)
            else:
                local_sum = (x * mask).sum()
                local_count = mask.sum().float()
            totals = self.accelerator.reduce(
                torch.stack([local_sum, local_count]), reduction="sum"
            )
            return (totals[0] / totals[1].clamp(min=1.0)).item()

        self._metrics[mode]["dopd/loss_1"].append(loss_1.detach().item())

        # teacher reward difference
        if r.dim() == 3:
            r_masked = (r.detach() * mask.unsqueeze(-1)).sum(dim=-1)
            r_mean = r_masked.sum() / mask.sum()
            self._metrics[mode]["dopd/teacher_r_mean"].append(r_mean.item())

        # student top-K entropy
        
        student_entropy = -(stu_softmax_probs * per_token_logps).sum(dim=-1)
        self._metrics[mode]["dopd/student_entropy"].append(global_masked_mean(student_entropy))

        # full-vocab entropy
        self._metrics[mode]["entropy"].append(global_masked_mean(entropies))

        # aux loss (MoE)
        if aux_loss is not None:
            self._metrics[mode]["aux_loss"].append(
                self.accelerator.gather_for_metrics(aux_loss).mean().item()
            )

        # KL
        if self.beta != 0.0:
            self._metrics[mode]["dopd/kl"].append(global_masked_mean(kl))

        self._metrics[mode]["dopd/loss"].append(loss.detach().item())

        return loss
        








        




    
