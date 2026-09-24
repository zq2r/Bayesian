"""Score each <prm> process step with a trained BayesianPRM checkpoint."""

import argparse
import json
import math
from pathlib import Path

import torch


def posterior_rewards(mu_heads, belief_logits, use_conservatism, beta):
    """Return reliability and conservative rewards for [steps, heads] inputs."""
    if mu_heads.ndim != 2 or belief_logits.shape != mu_heads.shape:
        raise ValueError("mu_heads and belief_logits must have the same [steps, heads] shape")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("conservatism beta must be finite and positive")

    mu_heads = mu_heads.float()
    rel_weights = torch.softmax(belief_logits.float(), dim=-1)
    mu_rel = (rel_weights * mu_heads).sum(dim=-1)
    if use_conservatism:
        post_weights = torch.softmax(
            torch.log(rel_weights.clamp_min(1e-6)) - mu_heads / beta,
            dim=-1,
        )
    else:
        post_weights = rel_weights
    mu_final = (post_weights * mu_heads).sum(dim=-1)
    return {
        "mu_heads": mu_heads,
        "rel_weights": rel_weights,
        "post_weights": post_weights,
        "mu_rel": mu_rel,
        "mu_final": mu_final,
    }


def resolve_conservatism(model, setting, beta_override):
    enabled = bool(getattr(model, "belief_use_conservatism", False))
    if setting != "auto":
        enabled = setting == "true"
    beta = float(
        beta_override if beta_override is not None
        else getattr(model, "belief_conservatism_beta", 0.1)
    )
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("conservatism beta must be finite and positive")
    return enabled, beta


def make_prompt(model, tokenizer, question, steps, num_patches):
    from internvl.conversation import get_conv_template

    if not steps or any(not isinstance(step, str) for step in steps):
        raise ValueError("each solution must contain at least one string step")
    process = "<prm>".join(steps) + "<prm>"
    content = f"Question: {question}\nProcess: {process}"
    if "<image>" not in content:
        content = "<image>\n" + content
    template = get_conv_template(model.template)
    template.append_message(template.roles[0], "")
    template.append_message(template.roles[1], content)
    image_tokens = (
        "<img>" + "<IMG_CONTEXT>" * (model.num_image_token * num_patches) + "</img>"
    )
    return template.get_prompt().replace("<image>", image_tokens, 1)


@torch.inference_mode()
def score_steps(model, tokenizer, pixels, question, steps, enabled, beta):
    prompt = make_prompt(model, tokenizer, question, steps, pixels.shape[0])
    prm_id = tokenizer.convert_tokens_to_ids("<prm>")
    image_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    if prm_id == tokenizer.unk_token_id or image_id == tokenizer.unk_token_id:
        raise ValueError("checkpoint tokenizer must contain <prm> and <IMG_CONTEXT>")

    model.img_context_token_id = image_id
    old_padding_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = "left"
        inputs = tokenizer(prompt, return_tensors="pt")
    finally:
        tokenizer.padding_side = old_padding_side

    input_ids = inputs["input_ids"].to(pixels.device)
    mask = input_ids == prm_id
    if mask.sum().item() != len(steps):
        raise ValueError(
            f"expected {len(steps)} <prm> tokens, found {mask.sum().item()}; "
            "check that steps do not contain literal <prm>"
        )
    outputs = model(
        pixel_values=pixels,
        input_ids=input_ids,
        attention_mask=inputs["attention_mask"].to(pixels.device),
        image_flags=torch.ones(pixels.shape[0], 1, dtype=torch.long, device=pixels.device),
        output_hidden_states=True,
        return_dict=True,
    )
    if not outputs.hidden_states:
        raise RuntimeError("model did not return hidden states for <prm> tokens")
    prm_hidden = outputs.hidden_states[-1][mask]
    mu_heads = torch.sigmoid(model.ensemble_prm_head(prm_hidden).float())
    mu_heads = mu_heads.transpose(0, 1).contiguous().clamp(1e-6, 1 - 1e-6)
    belief_logits = model.belief_head(prm_hidden, mu_heads)
    details = posterior_rewards(mu_heads, belief_logits, enabled, beta)
    return {key: value.cpu().tolist() for key, value in details.items()}


def load_pixels(image_path, model, dynamic, max_num, device, dtype):
    from PIL import Image
    from internvl.train.dataset import build_transform, dynamic_preprocess

    image_size = model.config.force_image_size or model.config.vision_config.image_size
    transform = build_transform(is_train=False, input_size=image_size)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        patches = (
            dynamic_preprocess(
                image,
                image_size=image_size,
                use_thumbnail=model.config.use_thumbnail,
                max_num=max_num,
            )
            if dynamic else [image]
        )
        return torch.stack([transform(patch) for patch in patches]).to(device, dtype)


def resolve_image(record, image_root):
    raw = record.get("image_path") or record.get("image")
    if not raw:
        raise ValueError("each record needs image_path or image")
    path = Path(raw)
    if not path.is_absolute():
        path = image_root / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def evaluate_record(record, image_root, model, tokenizer, device, dtype, dynamic, max_num,
                    enabled, beta):
    question = record.get("query") or record.get("query_cot") or record.get("question")
    if not isinstance(question, str) or not question:
        raise ValueError("each record needs a nonempty query, query_cot, or question")
    solutions = record.get("solutions_splits")
    if not isinstance(solutions, list) or not solutions:
        raise ValueError("each record needs nonempty solutions_splits: list[list[str]]")
    pixels = load_pixels(resolve_image(record, image_root), model, dynamic, max_num,
                         device, dtype)
    outputs = [score_steps(model, tokenizer, pixels, question, steps, enabled, beta)
               for steps in solutions]
    result = dict(record)
    result["prm_scores"] = [item["mu_final"] for item in outputs]
    result["prm_mu_rel"] = [item["mu_rel"] for item in outputs]
    result["prm_mu_heads"] = [item["mu_heads"] for item in outputs]
    result["prm_rel_weights"] = [item["rel_weights"] for item in outputs]
    result["prm_post_weights"] = [item["post_weights"] for item in outputs]
    result["belief_use_conservatism"] = enabled
    result["belief_conservatism_beta"] = beta
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--max-num", type=int, default=6)
    parser.add_argument("--belief-use-conservatism", choices=("auto", "true", "false"),
                        default="auto")
    parser.add_argument("--belief-conservatism-beta", type=float)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from internvl.model.internvl_chat.modeling_bayesian_prm import InternVLChatModel

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True,
                                               use_fast=False)
    model = InternVLChatModel.from_pretrained(
        args.checkpoint, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    if model.prm_loss_type != "bayesian_prm" or model.belief_head is None:
        raise ValueError("checkpoint must be a trained bayesian_prm checkpoint")
    enabled, beta = resolve_conservatism(
        model, args.belief_use_conservatism, args.belief_conservatism_beta
    )
    with args.annotation.open(encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise ValueError("annotation must be a JSON list")
    results = [
        evaluate_record(record, args.image_root, model, tokenizer, device, dtype,
                        args.dynamic, args.max_num, enabled, beta)
        for record in records
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
