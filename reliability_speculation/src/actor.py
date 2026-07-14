"""Actor: the agent model producing the real trajectory, exposing per-token logprobs for confidence scoring."""
from __future__ import annotations

import dataclasses
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

ACTION_SPACE = ("read_file", "edit_file", "run_tests", "submit_patch")

SYSTEM_PROMPT = (
    "You are a coding agent. Respond with exactly one JSON object describing your next "
    "action: {\"tool\": one of " + str(list(ACTION_SPACE)) + ", \"target\": <file path or empty>, "
    "\"patch\": <unified diff or empty>}. No other text. "
    "edit_file applies \"patch\" (a unified diff against \"target\") to the working tree. "
    "submit_patch ends the trajectory for grading against the working tree's accumulated edits; "
    "its own \"target\"/\"patch\" fields are ignored."
)


@dataclasses.dataclass
class Action:
    tool: str
    target: str
    patch: str
    raw_text: str


@dataclasses.dataclass
class GeneratedAction:
    action: Action
    token_logprobs: list[float]


def load_model(model_name: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a causal LM + tokenizer by HF model name, exposing per-token logprobs via generate(output_scores=True)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    model.eval()
    return model, tokenizer


def _build_prompt(tokenizer: PreTrainedTokenizerBase, state_description: str) -> str:
    """Render the state description into the model's chat template with the action-space system prompt."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": state_description},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_action(raw_text: str) -> Action:
    """Parse the model's JSON action string into a structured Action; malformed output becomes tool='invalid'."""
    tool, target, patch = "invalid", "", ""
    try:
        obj = json.loads(raw_text.strip())
        candidate_tool = obj.get("tool", "")
        if candidate_tool in ACTION_SPACE:
            tool = candidate_tool
            target = obj.get("target", "") or ""
            patch = obj.get("patch", "") or ""
    except (json.JSONDecodeError, AttributeError):
        pass
    return Action(tool=tool, target=target, patch=patch, raw_text=raw_text)


@torch.no_grad()
def generate_action(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    state_description: str,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
) -> GeneratedAction:
    """Generate one action from the model and compute per-token logprobs over the generated span."""
    prompt = _build_prompt(tokenizer, state_description)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0.0

    gen_kwargs: dict = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    output = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    generated_ids = output.sequences[0][input_len:]
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    token_logprobs = []
    for step_scores, token_id in zip(output.scores, generated_ids):
        log_probs = torch.log_softmax(step_scores[0], dim=-1)
        token_logprobs.append(log_probs[token_id].item())

    action = parse_action(raw_text)
    return GeneratedAction(action=action, token_logprobs=token_logprobs)
