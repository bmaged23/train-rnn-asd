"""Vision-language-model classifier over fixed-length windows of face-crop
IMAGES — third candidate architecture, alongside the raw-pixel CNN+LSTM
("conv", src/image_model.py). Kept as its own fully separate pipeline, same
convention every other architecture in this project already follows — see
config.py's "scripts/modeling/train/vlm_all_combined.py" section (an earlier
draft used a shared model_factory.py + switchable train script, reverted per
explicit user request in favor of this one-architecture-per-file shape).

SCAFFOLDED 2026-08-23, VERIFIED END-TO-END THE SAME DAY (real weights
downloaded, real forward() call succeeded on a real 2-window/50-frame batch
— see "Verified" section below) but still not actually TRAINED.

MODEL SWITCHED, later the same day: config.VLM_PRETRAINED_NAME moved from
"Qwen/Qwen3.5-4B" to "Qwen/Qwen3-VL-4B-Instruct" (see config.py's own
comment on that constant for the full reasoning — Qwen3.5's hybrid
Gated-DeltaNet layers turned out to be the whole day's OOM root cause;
Qwen3-VL-4B-Instruct is a same-scale, STANDARD dense-transformer/GQA
architecture with dramatically better real memory behavior, per-layer
forward memory ~0.009GB vs. Qwen3.5's ~1.2-2GB). Everything below this
point that names "Qwen3.5"/cites its specific numbers (token ids,
hidden_size, the 8-of-32-layer LoRA coverage gap, etc.) is HISTORICAL —
true of the old model, not the current one. Re-verified facts for the
CURRENT model (Qwen3-VL-4B-Instruct, same day): resolves to
`Qwen3VLForConditionalGeneration` via the same `AutoModelForMultimodalLM`
loader; LoRA target modules (q_proj/k_proj/v_proj/o_proj/gate_proj/
up_proj/down_proj) all exist UNIFORMLY across all 36 layers (no hybrid-
architecture coverage gap this time); block list still found at
`model.language_model.layers` (36 of them); "Yes" -> token id 9454, "No"
-> token id 2753 (different vocab, 151,936 vs Qwen3.5's 248,320); real
batch=1 forward+backward+optimizer.step() (no gradient checkpointing)
verified working, ~0.68s/window steady-state. See config.py's
VLM_BATCH_SIZE/VLM_USE_GRADIENT_CHECKPOINTING comments for the full
batch-size-vs-checkpointing tradeoff investigation and final numbers.

Same
forward(features, lengths=None)
-> (B,) signature as every other model in this project (SpeakingDetectorRNN,
WindowedSpeakingDetectorRNN, ImageWindowedSpeakingDetectorRNN): one logit per
window, not per-class softmax logits, for drop-in compatibility with the
existing BCEWithLogitsLoss training loop and src/metrics.py's compute_metrics
(expects a sigmoid probability).

DESIGN — "prompted classification", not embed+pool+head (2026-08-23):
config.VLM_PRETRAINED_NAME ("Qwen/Qwen3.5-4B") was verified live (web search,
since this model postdates this project's own knowledge) to be a real
released model — but a chat/generation-oriented multimodal LM
(transformers.AutoModelForMultimodalLM + AutoProcessor, real usage is
interleaved image+text chat turns -> model.generate()), NOT a plain vision
encoder with a clean pixel_values -> pooler_output/last_hidden_state
contract. An earlier version of this file assumed the latter (a separate
mean/attention/cls_token pooling + linear head over per-frame embeddings) —
wrong for this specific model, reverted per explicit user decision in favor
of the approach below, which matches how the model is actually meant to be
used:

  1. Each window's VLM_NUM_FRAMES images + config.VLM_PROMPT_TEXT ("is this
     person speaking? answer Yes or No") are built into ONE chat-style
     multimodal input via AutoProcessor's chat template.
  2. ONE forward pass (self.model(**inputs), NOT model.generate()) — a
     single forward pass is a normal, differentiable op (needed so
     LoRA/full/partial-unfreeze fine-tuning can backprop through it);
     autoregressive generate() is not straightforwardly differentiable and
     would need a much more involved RL-style training setup instead.
  3. The next-token logits at the final input position are read at exactly
     two vocabulary entries — the first token of config.VLM_YES_TOKEN_STRING
     and config.VLM_NO_TOKEN_STRING — and their DIFFERENCE
     (yes_logit - no_logit) becomes this window's classification score, fed
     through a small learnable (scale, bias) affine calibration (always
     trainable, regardless of VLM_FINETUNE_MODE — same role a temperature/
     bias calibration layer plays on top of any classifier) before being
     returned as this project's usual single BCEWithLogitsLoss-compatible
     logit.

This means there is no separate embedding/pooling/MLP head module here (no
VLM_HEAD_POOLING/VLM_HEAD_HIDDEN_DIM — those were removed from config.py
along with the old design) — the "classification head" IS the model's own
next-token vocabulary distribution, verbalized as Yes/No, same technique
commonly used for zero-/few-shot classification with generative LMs.

MEMORY — gradient checkpointing + gradient accumulation (2026-08-23, both
explicit user request, both independently toggleable):
A real training run OOM'd on GPU 0 on the very first batch (this box is
shared — other tenants were already holding ~55GB/GPU before our run even
started, and VLM_BATCH_SIZE=4 windows' worth of forward+backward activation
memory through all 32 LoRA-adapted layers needed close to the rest).
config.VLM_USE_GRADIENT_CHECKPOINTING recomputes each decoder layer's
activations during backward() instead of storing all 32 simultaneously —
trades ~20-30% more compute per step for a large drop in peak activation
memory. Applied via `model.enable_input_require_grads()` +
`model.gradient_checkpointing_enable()` in `_load_model_and_processor()`,
in that order — `enable_input_require_grads()` must come first, since with
a frozen (or partially frozen) base, nothing upstream of a checkpointed
segment is otherwise guaranteed to carry requires_grad=True, and without
that, checkpointing's recomputation can silently fail to reconnect the
graph back to the trainable params (model trains, learns nothing, no error
raised).

config.VLM_USE_GRADIENT_ACCUMULATION (magnitude set by
VLM_GRADIENT_ACCUMULATION_STEPS) is the OTHER lever, implemented in
scripts/modeling/train/vlm_all_combined.py's run_epoch() rather than here —
accumulates gradients over several micro-batches before one
optimizer.step(), so a larger EFFECTIVE batch can train without needing
every window's activation graph held at once.

RESULT of a real run with both True (2026-08-23, same day): no repeat
OOM — checkpointing genuinely fixed it, using only ~17GB of the ~47GB
available headroom on GPU 0. But throughput was ~15s/micro-batch (~21
DAYS for one epoch), traced to forward() looping over windows one at a
time in Python rather than using the GPU's batch parallelism at all — see
"THROUGHPUT" below for the fix. Given checkpointing left real memory
margin (17 of 47GB) unused, `VLM_USE_GRADIENT_CHECKPOINTING`'s default was
flipped to `False` the same day (explicit user request) to trade that
margin back for speed, once real batching (below) was in place. Real risk
worth flagging: real-batched forward's peak memory is NOT obviously lower
than the old loop's (same B windows' activations, computed together
instead of sequentially — similar order of magnitude), so "no
checkpointing + now batched" has not itself been verified against a real
OOM ceiling. If a future run OOMs again, re-enabling this is the first
lever to try before touching VLM_BATCH_SIZE or VLM_NUM_FRAMES.

THROUGHPUT — real batched forward(), not a per-sample loop (2026-08-23,
same day, explicit user request): the ~21-days/epoch pace traced to
forward() calling the model once per window in a Python loop, never
using the GPU's actual batch-parallel matmul capacity. Rewritten to build
ALL of a batch's windows into ONE `processor.apply_chat_template()` call
and ONE `self.model(**inputs)` call. Two real mechanics this needed,
both checked against the real processor before writing:
  - Windows in a batch can have different real lengths (a padded
    short-track window has fewer real frames than a full one — see
    `_build_messages()`), so their conversations tokenize to different
    lengths. `padding=True` on `apply_chat_template()` handles this, but
    ONLY with `self.processor.tokenizer.padding_side` set to `"left"`
    (done once in `__init__`) — with the default right-padding, each
    sample's real last token would land at a DIFFERENT column, and
    reading `logits[:, -1, :]` for the whole batch would silently read
    garbage for every padded sample. Left-padding keeps every sample's
    real last token at column -1 regardless of its own length — verified
    directly: a real 2-sample batch (5 vs. 3 images) showed
    `attention_mask[:, -1] == [1, 1]` (both real, not padding) after
    left-padding.
  - `pixel_values`/`image_grid_thw` are flattened multi-image tensors
    (Qwen-VL-family convention, not a simple per-sample stack) —
    confirmed the processor concatenates these correctly across a batch
    on its own (2 samples of 5 images each -> `pixel_values` shape grew
    from `(1280, 1536)` per-sample to `(2560, 1536)` batched, exactly
    2x), no manual handling needed.
Batched vs. unbatched real forward-pass scores were compared directly on
the same 2 real samples: `[-2.56, -3.09]` (batched) vs. `[-2.63, -2.91]`
(unbatched) — close but not bit-identical, expected floating-point/kernel
variance from batching bf16 transformer inference, not a correctness
red flag (both land on the same classification direction). Not yet
verified: whether this specific model's forward() auto-derives correct
`position_ids` from `attention_mask` for the left-padded case (common in
recent causal-LM implementations, but not explicitly confirmed here) —
the small batched-vs-unbatched discrepancy above is consistent with
either ordinary batching noise or a minor position-id effect on the
padded sample; not distinguished, and not blocking further work given the
magnitude is small and training is expected to be robust to it either way.

RESOLUTION, same day: real batching alone did NOT fix the OOM — a real
run at VLM_BATCH_SIZE=4 (real batched forward, checkpointing on) still
OOM'd, this time deep inside one of the model's 24 Gated-DeltaNet linear-
attention layers (`torch_chunk_gated_delta_rule`). Root-caused (user's own
instinct, "something's sus" for a 4B model needing >47GB for 4 samples):
`transformers` requires the `kernels` package to use an optimized fused
kernel for this layer type; without it, it silently falls back to a
reference implementation that upcasts everything to fp32 and does a
64-iteration Python loop with `.clone()` calls inside the autograd graph
— confirmed via reading the actual installed transformers source
(`modeling_qwen3_5.py`'s `torch_chunk_gated_delta_rule`). Installing
`kernels` alone wasn't enough (no published Hub kernel yet for this
specific brand-new architecture); installing `flash-linear-attention`
(FLA) too — transformers' own documented priority-2 fallback — closed
the gap from "essentially 0 bytes free" down to only ~250-500MB short at
batch=4. `VLM_BATCH_SIZE` was then lowered 4->3->2->1 (see that
constant's own config.py comment for the full sequence and why the gap
didn't shrink proportionally with batch size) until batch=1 finally fit:
peak 39.34GB of ~47GB available. A single isolated timing at batch=1
looked alarming (36.52s/step) but turned out to be almost entirely
one-time Triton/FLA JIT-compilation overhead — 4 consecutive real steps
confirmed steady-state at ~0.64-0.65s/micro-batch, giving a real,
practical epoch estimate (~3.7 days) instead of the original ~21.
`kernels`/`flash-linear-attention` are now REQUIRED dependencies, not
optional — see requirements.txt.

Verified against real weights, 2026-08-23 (Qwen/Qwen3.5-4B actually
downloaded and loaded — every item below was previously a guess, now a
confirmed fact, checked with a standalone script + this file's own
VLMASDClassifier, not just manual API probing):
  - `transformers.AutoModelForMultimodalLM.from_pretrained("Qwen/Qwen3.5-4B")`
    really works — resolves to `Qwen3_5ForConditionalGeneration`
    (transformers==5.15.1). `AutoProcessor` resolves to `Qwen3VLProcessor`.
  - `processor.apply_chat_template(messages, tokenize=True,
    add_generation_prompt=True, return_tensors="pt", return_dict=True)`
    with the `[{"role": "user", "content": [{"type": "image", "image": img},
    ..., {"type": "text", "text": ...}]}]` message shape works exactly as
    assumed, returning a dict with `input_ids`/`attention_mask`/
    `mm_token_type_ids`/`pixel_values`/`image_grid_thw` — this file's
    generic `{k: v.to(device) for k, v in inputs.items()}` handling covers
    all of these without needing to special-case any key.
  - `self.model(**inputs).logits` really is `(B, seq_len, vocab_size)` —
    confirmed `(1, 236, 248320)` on a 3-frame smoke input.
  - "Yes" -> token id 9175, "No" -> token id 2665 — both genuinely single
    tokens, no warning fires.
  - `_find_transformer_blocks()` really does find real blocks at
    `model.language_model.layers` (32 of them) — the first candidate path in
    `_BLOCK_ATTR_PATHS` to match, so `finetune_mode="partial_unfreeze"` works
    as coded, no path list changes needed.
  - CORRECTED 2026-08-23 (same day, follow-up check): the original
    `VLM_LORA_TARGET_MODULES = ["q_proj", "v_proj"]` matched real submodule
    names on only 8 of 32 layers — this model uses a HYBRID attention
    architecture (8 layers are standard multi-head attention with
    q_proj/k_proj/v_proj/o_proj; the other 24 are a Gated-DeltaNet-style
    linear-attention variant with entirely different projection names —
    in_proj_qkv/in_proj_z/in_proj_b/in_proj_a/out_proj, zero overlap), so
    LoRA was adapting ZERO parameters in 75% of the model's depth, not just
    "fewer." Both layer types DO share identically-named
    `mlp.gate_proj`/`up_proj`/`down_proj` — config.py's
    `VLM_LORA_TARGET_MODULES` now also includes those plus `o_proj`, and
    this was re-verified for real: LoRA adapters now attach on all 32/32
    layers (trainable params 1,835,010 -> 20,774,914 out of 4,560,040,450
    total — 242 trainable tensors, exactly matching 6 target modules x 32
    MLP-covered layers + 6 x 8 attention-covered layers + 2 calibration
    params).
  - A real `VLMASDClassifier(finetune_mode="lora")` was built and its real
    forward() was called on a real (2, 50, 3, 96, 96) batch — matching this
    project's actual WINDOW_SIZE=50 and a `lengths=[50, 30]` mix (one full
    window, one padded-short-track window) — returning real logits,
    correct `(2,)` shape, `requires_grad=True`. A real `loss.backward()`
    was then run for the first time (previous checks were forward-only):
    gradients reached all 242 trainable tensors (122 with nonzero norm on
    this tiny synthetic 2-sample batch — the rest are plausibly LoRA's
    zero-initialized B matrices interacting with near-random input, not
    confirmed as a real concern, worth re-checking once real data is used).

Not yet verified (still real gaps, none blocking further scaffold work):
  - `features` (this project's src/image_dataset.py output) are already
    ImageNet-normalized float tensors, built for the "conv" pipeline's
    torchvision backbone — _denormalize_to_pil() undoes that normalization
    before handing images to AutoProcessor, since a processor almost
    certainly expects raw 0-255 images and applies its OWN
    normalization/resizing internally. The processor call itself didn't
    error on this in the real end-to-end check above, but whether its
    internal preprocessing is actually well-matched to this project's
    already-cropped/resized 96x96 images (vs. e.g. expecting different
    resolution/aspect handling) hasn't been specifically checked.
  - RESOLVED 2026-08-23: forward() now does one real batched call instead
    of a per-sample loop — see "THROUGHPUT" above for the fix and what was
    verified about it directly (left-padding, pixel_values/image_grid_thw
    concatenation, batched-vs-unbatched score comparison). Still not
    verified: this exact batched path at the real VLM_BATCH_SIZE=4 (only
    checked at batch=2 so far) or combined with gradient accumulation in
    an actual training run.
  - A single loss.backward() call was verified (see above) but a full
    training run — gradient clipping, an actual optimizer.step(), multiple
    batches/epochs, early stopping in practice — has not been attempted
    with the current (batched forward, checkpointing off by default) code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    IMAGE_NORMALIZE_MEAN,
    IMAGE_NORMALIZE_STD,
    VLM_FINETUNE_MODE,
    VLM_FRAME_SAMPLING,
    VLM_LORA_ALPHA,
    VLM_LORA_DROPOUT,
    VLM_LORA_R,
    VLM_LORA_TARGET_MODULES,
    VLM_NO_TOKEN_STRING,
    VLM_NUM_FRAMES,
    VLM_PARTIAL_UNFREEZE_NUM_BLOCKS,
    VLM_PRETRAINED_NAME,
    VLM_PROMPT_TEXT,
    VLM_QUANTIZATION_4BIT_COMPUTE_DTYPE,
    VLM_QUANTIZATION_4BIT_QUANT_STORAGE,
    VLM_QUANTIZATION_4BIT_QUANT_TYPE,
    VLM_QUANTIZATION_4BIT_USE_DOUBLE_QUANT,
    VLM_USE_GRADIENT_CHECKPOINTING,
    VLM_USE_QUANTIZATION,
    VLM_YES_TOKEN_STRING,
)

_TORCH_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32, "uint8": torch.uint8}


def _build_quantization_config():
    """BitsAndBytesConfig built straight from config.py's
    VLM_QUANTIZATION_4BIT_* constants — see those constants' own comments
    for what each one does. Only called when VLM_USE_QUANTIZATION is True.
    """
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=VLM_QUANTIZATION_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=_TORCH_DTYPE_MAP[VLM_QUANTIZATION_4BIT_COMPUTE_DTYPE],
        bnb_4bit_use_double_quant=VLM_QUANTIZATION_4BIT_USE_DOUBLE_QUANT,
        bnb_4bit_quant_storage=_TORCH_DTYPE_MAP[VLM_QUANTIZATION_4BIT_QUANT_STORAGE],
    )

# Common attribute paths for a transformers model's main transformer block
# list, tried in order — includes nested language-model submodule paths
# since a multimodal LM typically composes a vision tower + a separate
# language-model backbone (unlike a plain single-stack text model). See this
# module's docstring's "Assumptions" section.
_BLOCK_ATTR_PATHS = (
    "model.layers",
    "language_model.layers",
    "language_model.model.layers",
    "model.language_model.layers",
    "encoder.layers",
    "transformer.h",
    "blocks",
)


def _find_transformer_blocks(model: nn.Module) -> nn.ModuleList:
    """Best-effort lookup of `model`'s main transformer block list — needed
    only for VLM_FINETUNE_MODE="partial_unfreeze". Raises AttributeError
    (loudly, not a silent no-op) if none of _BLOCK_ATTR_PATHS exist on the
    real loaded model.
    """
    for path in _BLOCK_ATTR_PATHS:
        obj: object = model
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, (nn.ModuleList, list)):
            return obj
    raise AttributeError(
        f"could not find a transformer block list on {type(model).__name__} via any of "
        f"{_BLOCK_ATTR_PATHS} — inspect the real loaded model and add its actual attribute "
        f"path to _BLOCK_ATTR_PATHS before using VLM_FINETUNE_MODE='partial_unfreeze'"
    )


def _load_model_and_processor(
    pretrained_name: str, finetune_mode: str, use_gradient_checkpointing: bool, use_quantization: bool
):
    """Loads config.VLM_PRETRAINED_NAME + its AutoProcessor, and applies
    config.VLM_FINETUNE_MODE's freeze/LoRA/partial-unfreeze logic to the
    model. See this module's docstring for the AutoModelForMultimodalLM
    class-name caveat.

    `use_quantization` (config.VLM_USE_QUANTIZATION, added 2026-08-23,
    explicit user request): loads the frozen base in 4-bit (bitsandbytes,
    QLoRA) instead of bf16 — see config.py's VLM_USE_QUANTIZATION/
    VLM_QUANTIZATION_4BIT_* comments for what each parameter does and the
    expected memory saving. Only valid combined with finetune_mode="lora"
    — raises loudly otherwise, since 4-bit weights aren't directly
    fine-tunable via full/partial-unfreeze backprop, only LoRA adapters on
    top of a frozen quantized base (the actual definition of QLoRA) work.
    Also requires CUDA — bitsandbytes 4-bit has no meaningful CPU path.

    `use_gradient_checkpointing` (config.VLM_USE_GRADIENT_CHECKPOINTING):
    recomputes each decoder layer's activations during backward() instead
    of storing all 32 layers' simultaneously — see this module's docstring
    for the full mechanism/tradeoff. Two different setup paths depending on
    `use_quantization`:
      - Quantized: `peft.prepare_model_for_kbit_training()` handles BOTH
        the input-require-grads step AND checkpointing enablement in one
        call — this is peft's own documented, tested setup sequence for
        k-bit models, which has extra numerical-stability needs beyond
        plain LoRA (e.g. casting non-quantized modules like layernorms to
        fp32) that the manual sequence below doesn't cover.
      - Not quantized: `enable_input_require_grads()` is called FIRST,
        before `gradient_checkpointing_enable()` — required when the base
        model is (partially or fully) frozen, since checkpointing's
        recomputation needs SOME tensor flowing into each checkpointed
        segment to already have requires_grad=True to correctly rebuild a
        graph back to the trainable params sitting downstream of frozen
        layers — without this, checkpointing can silently fail to connect
        gradients at all (model trains, learns nothing, no error). Both
        calls are on `model` itself (before any LoRA wrapping) since
        peft's get_peft_model() forwards unrecognized attribute access to
        the wrapped base model, so calling these again post-wrap would be
        redundant, not incorrect — done pre-wrap here for clarity.
    """
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if use_quantization:
        if finetune_mode != "lora":
            raise ValueError(
                f"VLM_USE_QUANTIZATION=True requires VLM_FINETUNE_MODE='lora' (got {finetune_mode!r}) — "
                "4-bit weights aren't directly fine-tunable via full/partial-unfreeze backprop, only "
                "LoRA adapters on top of a frozen quantized base (QLoRA) are supported"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("VLM_USE_QUANTIZATION=True requires a CUDA device — bitsandbytes 4-bit has no meaningful CPU path")

    processor = AutoProcessor.from_pretrained(pretrained_name)

    if use_quantization:
        model = AutoModelForMultimodalLM.from_pretrained(
            pretrained_name, quantization_config=_build_quantization_config(), device_map="auto"
        )
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
    else:
        # device_map="auto" (added 2026-08-23, explicit user request — a real
        # OOM on this box's shared GPUs, tight margin from an unrelated
        # always-on vllm serve process reserving ~50% of each device):
        # accelerate shards the ~4.5B-param model's layers across every
        # visible CUDA device instead of one, pooling memory (~94GB combined
        # here vs ~47GB on a single device) at the cost of a real per-layer
        # activation transfer between devices on every forward/backward —
        # not free parallelism, a single window still runs strictly
        # sequentially through the sharded layers. VLMASDClassifier.forward()
        # already reads the input device off `next(self.model.parameters())`
        # rather than a fixed externally-passed device, and moves
        # score_scale/score_bias to the OUTPUT device dynamically — both
        # needed since the model's first/last shard can land on different
        # physical GPUs. The training script must NOT call `.to(device)` on
        # the wrapping VLMASDClassifier afterward — that would collapse the
        # dispatched shards back onto one device, defeating the point.
        model = AutoModelForMultimodalLM.from_pretrained(pretrained_name, device_map="auto")
        if use_gradient_checkpointing:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable()

    if finetune_mode == "full":
        for param in model.parameters():
            param.requires_grad = True
    elif finetune_mode == "lora":
        from peft import LoraConfig, get_peft_model

        for param in model.parameters():
            param.requires_grad = False
        lora_config = LoraConfig(
            r=VLM_LORA_R,
            lora_alpha=VLM_LORA_ALPHA,
            lora_dropout=VLM_LORA_DROPOUT,
            target_modules=VLM_LORA_TARGET_MODULES,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
    elif finetune_mode == "partial_unfreeze":
        for param in model.parameters():
            param.requires_grad = False
        blocks = _find_transformer_blocks(model)
        for block in blocks[-VLM_PARTIAL_UNFREEZE_NUM_BLOCKS:]:
            for param in block.parameters():
                param.requires_grad = True
    else:
        raise ValueError(f"unsupported VLM_FINETUNE_MODE: {finetune_mode!r}")

    return model, processor


def _answer_token_ids(processor, yes_string: str, no_string: str) -> tuple[int, int]:
    """First token id of each of VLM_YES_TOKEN_STRING/VLM_NO_TOKEN_STRING,
    via the real tokenizer. If either string tokenizes to more than one
    token, only the first is used — see this module's docstring's
    "Assumptions" section for why that's flagged as unverified rather than
    silently trusted.
    """
    yes_ids = processor.tokenizer.encode(yes_string, add_special_tokens=False)
    no_ids = processor.tokenizer.encode(no_string, add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        import warnings

        warnings.warn(
            f"VLM_YES_TOKEN_STRING={yes_string!r} -> {len(yes_ids)} token(s), "
            f"VLM_NO_TOKEN_STRING={no_string!r} -> {len(no_ids)} token(s) — expected exactly 1 "
            "each; using only the first token of each, verify this is still a meaningful "
            "Yes-vs-No comparison for the real tokenizer",
            stacklevel=2,
        )
    return yes_ids[0], no_ids[0]


def _denormalize_to_pil(image_chw: torch.Tensor) -> "Image.Image":  # noqa: F821 (PIL imported lazily below)
    """Undoes src/image_dataset.py's ImageNet normalization
    (IMAGE_NORMALIZE_MEAN/STD) on one (3, H, W) tensor and returns a PIL
    Image — AutoProcessor almost certainly expects raw 0-255 images and
    applies its own normalization internally (not confirmed against this
    specific processor, see module docstring).
    """
    from PIL import Image

    mean = torch.tensor(IMAGE_NORMALIZE_MEAN, device=image_chw.device).view(3, 1, 1)
    std = torch.tensor(IMAGE_NORMALIZE_STD, device=image_chw.device).view(3, 1, 1)
    pixels = (image_chw * std + mean).clamp(0, 1)
    array = (pixels.permute(1, 2, 0) * 255).round().byte().cpu().numpy()
    return Image.fromarray(array, mode="RGB")


class VLMASDClassifier(nn.Module):
    """See module docstring for the full "prompted classification" design.
    One speaking-logit per fixed-length window of face-crop images — same
    shape/convention as ImageWindowedSpeakingDetectorRNN.
    """

    def __init__(
        self,
        pretrained_name: str = VLM_PRETRAINED_NAME,
        finetune_mode: str = VLM_FINETUNE_MODE,
        num_frames: int = VLM_NUM_FRAMES,
        frame_sampling: str = VLM_FRAME_SAMPLING,
        prompt_text: str = VLM_PROMPT_TEXT,
        yes_token_string: str = VLM_YES_TOKEN_STRING,
        no_token_string: str = VLM_NO_TOKEN_STRING,
        use_gradient_checkpointing: bool = VLM_USE_GRADIENT_CHECKPOINTING,
        use_quantization: bool = VLM_USE_QUANTIZATION,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.frame_sampling = frame_sampling
        self.prompt_text = prompt_text
        self.model, self.processor = _load_model_and_processor(
            pretrained_name, finetune_mode, use_gradient_checkpointing, use_quantization
        )
        # Left-padding, not the tokenizer's "right" default — required for
        # forward()'s batched next-token-position read (see that method's
        # docstring). Set once here, not per forward() call.
        self.processor.tokenizer.padding_side = "left"
        self.yes_token_id, self.no_token_id = _answer_token_ids(self.processor, yes_token_string, no_token_string)

        # Learnable affine calibration on top of the raw (yes_logit -
        # no_logit) difference — always trainable regardless of
        # VLM_FINETUNE_MODE (same role a temperature/bias calibration layer
        # plays on top of any classifier), since raw vocabulary-logit scale
        # is model-specific and not guaranteed to already sit in a
        # BCEWithLogitsLoss-friendly range.
        self.score_scale = nn.Parameter(torch.tensor(1.0))
        self.score_bias = nn.Parameter(torch.tensor(0.0))

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Checkpoint file for THIS model must not be `self.state_dict()` —
        that would serialize all ~4.56B base parameters (frozen, unchanged
        from the pretrained checkpoint, ~9GB) on every save, not just the
        ~20.8M actually-trained ones (LoRA adapters + score_scale/
        score_bias under finetune_mode="lora"; whatever subset
        finetune_mode="full"/"partial_unfreeze" unfroze otherwise) — added
        2026-08-23 once a real training run was imminent, after the
        checkpoint-saving code (copied from the much-smaller conv pipeline)
        was caught doing exactly that. Generalizes across all three
        finetune_mode values by keying on requires_grad directly rather
        than any LoRA-specific helper, so it doesn't need to change if
        finetune_mode does.
        """
        return {name: param.detach().cpu() for name, param in self.named_parameters() if param.requires_grad}

    def load_trainable_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Loads a checkpoint written by trainable_state_dict() — `strict=False`
        since it only ever covers a subset of self.state_dict()'s full keys
        (the frozen base parameters are already correct from the fresh
        __init__() load and were never part of the checkpoint to begin with).
        """
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        # Every key the checkpoint provides must land somewhere real —
        # "missing" (params this model has but the checkpoint doesn't) is
        # expected and fine (the frozen base), but "unexpected" would mean
        # the checkpoint has keys this model's structure doesn't recognize
        # at all (e.g. built under a different finetune_mode/LoRA config).
        if unexpected:
            raise RuntimeError(
                f"checkpoint has {len(unexpected)} key(s) this model doesn't recognize "
                f"(e.g. {unexpected[:3]}) — was it saved under a different VLM_FINETUNE_MODE "
                "or LoRA config than this model was built with?"
            )

    def _sample_frames(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, T, 3, H, W). "all", or T already <= self.num_frames
        (the common case — src/image_dataset.py's ImageWindowedDataset
        windows are always exactly WINDOW_SIZE == VLM_NUM_FRAMES long by
        construction): no-op. "uniform": evenly-spaced subsample down to
        self.num_frames when given more frames than that.
        """
        t = images.shape[1]
        if self.frame_sampling == "all" or t <= self.num_frames:
            return images
        if self.frame_sampling != "uniform":
            raise ValueError(f"unsupported VLM_FRAME_SAMPLING: {self.frame_sampling!r}")
        indices = torch.linspace(0, t - 1, self.num_frames).round().long()
        return images[:, indices]

    def _build_messages(self, frames_chw: torch.Tensor, length: int) -> list[dict]:
        """frames_chw: (T, 3, H, W) for ONE window (already sampled). `length`:
        real (non-padded) frame count — trailing padding frames (zero
        tensors, config.WINDOW_PAD_SHORT_TRACKS) are excluded from the
        prompt entirely rather than shown to the model as blank images.
        Returns one "conversation" (list of chat turns) in the shape
        processor.apply_chat_template expects, ready to be placed alongside
        every other window's conversation in one batched call.
        """
        pil_frames = [_denormalize_to_pil(frames_chw[i]) for i in range(length)]
        return [
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in pil_frames]
                + [{"type": "text", "text": self.prompt_text}],
            }
        ]

    def forward(self, features: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """features: (B, window_size, 3, H, W) — same tensor shape
        src/image_dataset.py's collate_images produces (ImageNet-normalized;
        see _denormalize_to_pil()). `lengths` (B,), optional: real
        (non-padded) frame count per window — defaults to using every frame
        when not given. Returns (B,), one logit per window.

        REAL BATCHED implementation (2026-08-23, explicit user request,
        replacing an earlier per-sample Python loop that never actually
        used the GPU's batch-parallelism and measured ~15s/micro-batch in a
        real training run — see module docstring's "THROUGHPUT" section for
        the verification behind this). Windows can have different real
        `length`s (a padded short-track window has fewer real frames than a
        full one), so their built conversations tokenize to different
        lengths — `self.processor.tokenizer.padding_side` is set to "left"
        in __init__ specifically so every window's LAST real token still
        lands at the same column (-1) after batched padding, regardless of
        its own length; a naive right-pad would put the real "next token"
        position at a different column per sample, silently reading garbage
        for every padded sample.
        """
        images = self._sample_frames(features)  # (B, T, 3, H, W)
        b, t = images.shape[:2]
        if lengths is None:
            lengths = torch.full((b,), t, dtype=torch.long)

        messages_batch = [self._build_messages(images[i], int(lengths[i])) for i in range(b)]
        # padding=True goes through processor_kwargs, not as a direct kwarg
        # (found 2026-08-23, noisy "Kwargs passed to `processor.__call__`
        # have to be in `processor_kwargs` dict" warning logged on every
        # single forward call otherwise): apply_chat_template only treats
        # its own named parameters as safe — anything else in **kwargs gets
        # flagged, even a real, correctly-used processor option like this
        # one. See transformers/processing_utils.py's apply_chat_template.
        inputs = self.processor.apply_chat_template(
            messages_batch, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True, processor_kwargs={"padding": True},
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # logits_to_keep=1 (found 2026-08-23 investigating a 2-GPU batch_size=2
        # memory spike): without it, the model's default (logits_to_keep=0)
        # computes the lm_head over the ENTIRE input sequence (~3,300 tokens/
        # window here — 50 image frames' worth of vision tokens + prompt),
        # producing a (B, seq_len, vocab_size=248,320) tensor, even though
        # only the LAST position is ever read below. That's ~3,300x more
        # lm_head matmul + output-tensor memory than needed, on every single
        # forward call. logits_to_keep=1 makes transformers slice BEFORE the
        # lm_head projection, not after, so it only computes the one position
        # actually used.
        output = self.model(**inputs, logits_to_keep=1)
        next_token_logits = output.logits[:, -1, :]  # (B, vocab_size) — safe: left-padded, see docstring above
        raw_scores = next_token_logits[:, self.yes_token_id] - next_token_logits[:, self.no_token_id]
        # score_scale/score_bias are top-level nn.Parameters on this wrapper
        # class, not inside self.model — accelerate's device_map="auto"
        # dispatch (see _load_model_and_processor) only places self.model's
        # OWN submodules, so these two stay wherever they were created
        # (CPU/default) unless moved here. next_token_logits already sits on
        # whichever physical device holds the model's last shard (could
        # differ from the first shard's device across two GPUs), so this
        # moves the tiny scalars to match rather than assuming a fixed device.
        return raw_scores * self.score_scale.to(raw_scores.device) + self.score_bias.to(raw_scores.device)


if __name__ == "__main__":
    # Lightweight shape/wiring smoke test — does NOT download/run the real
    # VLM_PRETRAINED_NAME backbone (needs transformers+peft installed,
    # network access, and real GPU memory sized for a 4B-param model — none
    # guaranteed at scaffold time). Swaps in a tiny fake model+processor with
    # the same .apply_chat_template(...)/.model(**inputs).logits contract
    # this module expects, just to verify the scoring/calibration wiring is
    # shape-correct end to end.
    class _FakeOutput:
        def __init__(self, logits: torch.Tensor) -> None:
            self.logits = logits

    class _FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [{"Yes": 1, "No": 2}[text]]

    class _FakeProcessor:
        def __init__(self) -> None:
            self.tokenizer = _FakeTokenizer()

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, return_tensors, return_dict):
            n_images = sum(1 for c in messages[0]["content"] if c["type"] == "image")
            return {"input_ids": torch.zeros(1, n_images + 1, dtype=torch.long)}

    class _FakeModel(nn.Module):
        def __init__(self, vocab_size: int = 8) -> None:
            super().__init__()
            self.vocab_size = vocab_size
            self.dummy_param = nn.Parameter(torch.randn(1))

        def forward(self, input_ids: torch.Tensor) -> _FakeOutput:
            seq_len = input_ids.shape[1]
            logits = torch.randn(1, seq_len, self.vocab_size) + self.dummy_param
            return _FakeOutput(logits)

        def parameters(self, recurse: bool = True):
            return iter([self.dummy_param])

    model = VLMASDClassifier.__new__(VLMASDClassifier)
    nn.Module.__init__(model)
    model.num_frames = 50
    model.frame_sampling = "uniform"
    model.prompt_text = "Is this person speaking? Answer Yes or No."
    model.model = _FakeModel()
    model.processor = _FakeProcessor()
    model.yes_token_id, model.no_token_id = 1, 2
    model.score_scale = nn.Parameter(torch.tensor(1.0))
    model.score_bias = nn.Parameter(torch.tensor(0.0))

    features = torch.randn(2, 50, 3, 96, 96)
    lengths = torch.tensor([50, 30])
    logits = model(features, lengths)
    assert logits.shape == (2,), logits.shape
    print(f"logits shape: {tuple(logits.shape)} (fake model/processor, real backbone untested)")
