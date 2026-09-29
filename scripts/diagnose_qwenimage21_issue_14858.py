import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

from diffusers import QwenImage21Pipeline
from diffusers.models import attention_dispatch
from diffusers.models.transformers import transformer_qwenimage21


def tensor_stats(tensor):
    values = tensor.detach()
    finite = torch.isfinite(values)
    finite_count = int(finite.sum().item())
    non_finite_count = values.numel() - finite_count
    summary_values = values.abs() if values.is_complex() else values
    summary_values = summary_values.float()[finite]

    if finite_count:
        minimum = summary_values.min().item()
        maximum = summary_values.max().item()
        mean = summary_values.mean().item()
        std = summary_values.std(unbiased=False).item()
    else:
        minimum = maximum = mean = std = None

    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "finite": non_finite_count == 0,
        "non_finite": non_finite_count,
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "std": std,
    }


def install_probes(pipe, output):
    state = {"case": None, "mode": None, "step": -1, "layer": -1, "dispatch": -1}
    first_non_finite = {}

    def emit(stage, tensor=None, **metadata):
        row = {"case": state["case"], "mode": state["mode"], "step": state["step"], "stage": stage}
        if state["layer"] >= 0:
            row["layer"] = state["layer"]
        if state["dispatch"] >= 0:
            row["dispatch"] = state["dispatch"]
        if tensor is not None:
            row.update(tensor_stats(tensor))
        row.update(metadata)
        if tensor is not None:
            key = (state["case"], state["mode"])
            if row["non_finite"] and key not in first_non_finite:
                first_non_finite[key] = row.copy()
        output.write(json.dumps(row, allow_nan=True) + "\n")

    vae_encode = pipe.vae.encode

    def encode_probe(*args, **kwargs):
        if args:
            emit("vae_encode_input", args[0])
        result = vae_encode(*args, **kwargs)
        if hasattr(result, "latent_dist"):
            emit("vae_encode_mean", result.latent_dist.mean)
            if hasattr(result.latent_dist, "logvar"):
                emit("vae_encode_logvar", result.latent_dist.logvar)
        elif hasattr(result, "latents"):
            emit("vae_encode_latents", result.latents)
        return result

    pipe.vae.encode = encode_probe

    encode_vae_image = pipe._encode_vae_image

    def encode_vae_image_probe(image, generator):
        result = encode_vae_image(image, generator)
        emit("vae_encode_normalized_latents", result)
        return result

    pipe._encode_vae_image = encode_vae_image_probe

    pack_latents = pipe._pack_latents

    def pack_latents_probe(latents, batch_size, channels, height, width):
        result = pack_latents(latents, batch_size, channels, height, width)
        emit("latent_pack", result, input_shape=list(latents.shape), grid_hw=[height, width])
        return result

    pipe._pack_latents = pack_latents_probe

    prepare_latents = pipe.prepare_latents

    def prepare_latents_probe(*args, **kwargs):
        result = prepare_latents(*args, **kwargs)
        emit("prepared_target_latents", result[0])
        if result[1] is not None:
            emit("prepared_condition_latents", result[1])
        return result

    pipe.prepare_latents = prepare_latents_probe

    rope = pipe.transformer.pos_embed
    rope_forward = rope.forward
    rope_code = rope_forward.__func__.__code__

    def rope_probe(img_shapes, image_pad_mask, device):
        emit("rope_image_pad_mask", image_pad_mask, img_shapes=img_shapes)

        def trace_rope(frame, event, arg):
            if frame.f_code is rope_code:
                if event == "return":
                    for name in ("frame_index", "height_index", "width_index"):
                        index = frame.f_locals.get(name)
                        if isinstance(index, torch.Tensor):
                            emit(f"rope_{name}", index)
                            emit(f"rope_{name}_image_tokens", index[image_pad_mask])
                return trace_rope
            return None

        previous_trace = sys.gettrace()
        sys.settrace(trace_rope)
        try:
            result = rope_forward(img_shapes, image_pad_mask, device)
        finally:
            sys.settrace(previous_trace)
        emit("rope_embedding", result)
        return result

    rope.forward = rope_probe

    transformer_forward = pipe.transformer.forward

    def transformer_probe(*args, **kwargs):
        state["step"] += 1
        state["kv_cache_mode"] = kwargs.get("kv_cache_mode")
        shapes = kwargs.get("img_shapes")
        hidden_states = kwargs.get("hidden_states")
        img_mask = kwargs.get("img_mask")
        if hidden_states is not None:
            emit("transformer_input", hidden_states, img_shapes=shapes, img_mask_shape=list(img_mask.shape))
        result = transformer_forward(*args, **kwargs)
        output_tensor = result[0] if isinstance(result, tuple) else result.sample
        emit("transformer_output", output_tensor)
        return result

    pipe.transformer.forward = transformer_probe

    for layer_index, block in enumerate(pipe.transformer.transformer_blocks):
        block_forward = block.forward

        def make_block_probe(index, original_forward):
            def block_probe(*args, **kwargs):
                previous_layer = state["layer"]
                state["layer"] = index
                state["dispatch"] = -1
                try:
                    result = original_forward(*args, **kwargs)
                    emit("transformer_block_output", result)
                    return result
                finally:
                    state["layer"] = previous_layer

            return block_probe

        block.forward = make_block_probe(layer_index, block_forward)

    dispatch_attention = transformer_qwenimage21.dispatch_attention_fn

    def dispatch_probe(query, key, value, *args, **kwargs):
        state["dispatch"] += 1
        active_backend = attention_dispatch._AttentionBackendRegistry.get_active_backend()[0]
        mask = kwargs.get("attn_mask")
        emit(
            "attention_q",
            query,
            k_shape=list(key.shape),
            v_shape=list(value.shape),
            mask_shape=list(mask.shape) if isinstance(mask, torch.Tensor) else None,
            mask_type=type(mask).__name__ if mask is not None else None,
            backend=str(kwargs.get("backend")),
            active_backend=str(active_backend),
            kv_cache_mode=state.get("kv_cache_mode"),
        )
        if isinstance(mask, torch.Tensor):
            emit("attention_mask", mask)
        emit("attention_k", key)
        emit("attention_v", value)
        result = dispatch_attention(query, key, value, *args, **kwargs)
        emit("attention_output", result, backend=str(kwargs.get("backend")), active_backend=str(active_backend))
        return result

    transformer_qwenimage21.dispatch_attention_fn = dispatch_probe

    scheduler_step = pipe.scheduler.step

    def scheduler_probe(*args, **kwargs):
        if args:
            emit("scheduler_model_output", args[0])
        result = scheduler_step(*args, **kwargs)
        emit("scheduler_output", result[0])
        return result

    pipe.scheduler.step = scheduler_probe

    vae_decode = pipe.vae.decode

    def decode_probe(*args, **kwargs):
        if args:
            emit("vae_decode_input", args[0])
        result = vae_decode(*args, **kwargs)
        sample = result.sample if hasattr(result, "sample") else result[0]
        emit("vae_decode_output", sample)
        return result

    pipe.vae.decode = decode_probe

    return state, first_non_finite


def main():
    parser = argparse.ArgumentParser(description="Instrument the live Qwen-Image 2.1 pipeline for issue #14858.")
    parser.add_argument("--image-3-2", required=True, help="A condition image with 3:2 aspect ratio.")
    parser.add_argument("--image-4-3", required=True, help="A condition image with 4:3 aspect ratio.")
    parser.add_argument("--image-1-1", required=True, help="A square condition image.")
    parser.add_argument("--model", default="Qwen/Qwen-Image-2.1")
    parser.add_argument("--prompt", default="Change the object in the image to a bicycle.")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--steps", type=int, default=2, help="Use 40 to reproduce the full reported denoising run.")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("1024_3-2", "1024_4-3", "1024_1-1", "768_4-3"),
        default=("1024_3-2", "1024_4-3", "1024_1-1", "768_4-3"),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("cache_default", "no_cache", "cache_native"),
        default=("cache_default", "no_cache", "cache_native"),
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--log", type=Path, default=Path("qwenimage21_issue_14858.jsonl"))
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    pipe = QwenImage21Pipeline.from_pretrained(args.model, dtype=dtype).to(args.device)
    pipe.set_progress_bar_config(disable=True)

    cases = [
        ("1024_3-2", args.image_3_2, 1024),
        ("1024_4-3", args.image_4_3, 1024),
        ("1024_1-1", args.image_1_1, 1024),
        ("768_4-3", args.image_4_3, 768),
    ]
    cases = [case for case in cases if case[0] in args.cases]
    modes = [("cache_default", True, None), ("no_cache", False, None), ("cache_native", True, "native")]
    modes = [mode for mode in modes if mode[0] in args.modes]

    with args.log.open("w", encoding="utf-8") as output:
        state, first_non_finite = install_probes(pipe, output)
        for case, image_path, resolution in cases:
            for mode, use_kv_cache, backend in modes:
                state.update(case=case, mode=mode, step=-1, layer=-1, dispatch=-1, kv_cache_mode=None)
                for block in pipe.transformer.transformer_blocks:
                    block.attn.processor._attention_backend = backend

                with Image.open(image_path) as source:
                    image = source.copy()
                print(f"Running {case} / {mode} ({resolution}px, {args.steps} steps)", flush=True)
                pipe(
                    prompt=args.prompt,
                    image=image,
                    output_resolution=resolution,
                    num_inference_steps=args.steps,
                    generator=torch.Generator(device=args.device).manual_seed(args.seed),
                    output_type="pt",
                    use_kv_cache=use_kv_cache,
                )
                output.flush()

        print(f"Tensor trace written to {args.log}")
        for case, _, _ in cases:
            for mode, _, _ in modes:
                first = first_non_finite.get((case, mode))
                if first is None:
                    print(f"{case} / {mode}: no non-finite tensors")
                else:
                    print(f"{case} / {mode}: first non-finite tensor: {json.dumps(first, allow_nan=True)}")


if __name__ == "__main__":
    main()
