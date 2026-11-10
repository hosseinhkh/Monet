"""Monet prompt construction for one control step.

Sequence for one sample (left padded in a batch), Qwen2.5-VL chat format:

  <|im_start|>system\n{system}<|im_end|>\n
  <|im_start|>user\n
    <|vision_start|><|image_pad|>*n<|vision_end|>      (agent view)
    <|vision_start|><|image_pad|>*n<|vision_end|>      (wrist view)
    Robot state: <|fim_pad|>*S\nTask: {instruction}\n...<|im_end|>\n
  <|im_start|>assistant\n<abs_vis_token>  [latent slots]  [</abs_vis_token>]

Latent slots are always the last K positions before the optional end token:
  convention "sft" : <abs_vis_token> + K x <abs_vis_token_pad>       (slots = the K pads)
  convention "vllm": <abs_vis_token> + (K-1) x <abs_vis_token_pad>   (slot 0 = <abs_vis_token>)
The ``<|fim_pad|>`` placeholders are overwritten with RobotStateProjector tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from .monet_loader import IMAGE_TOKEN_ID, LATENT_END_ID, LATENT_PAD_ID, LATENT_START_ID, PAD_TOKEN_ID


@dataclass
class MonetInputs:
    input_ids: torch.LongTensor  # [B, L]
    attention_mask: torch.LongTensor  # [B, L]
    pixel_values: Optional[torch.Tensor]  # [sum_patches, 1176]
    image_grid_thw: Optional[torch.LongTensor]  # [B * n_views, 3]
    state_mask: torch.BoolTensor  # [B, L] placeholder positions
    num_latents: int
    slot_start: int  # index of the first latent slot (== L - K - has_end)
    convention: str

    def to(self, device):
        f = lambda x: x.to(device) if isinstance(x, torch.Tensor) else x  # noqa: E731
        return MonetInputs(f(self.input_ids), f(self.attention_mask), f(self.pixel_values), f(self.image_grid_thw),
                           f(self.state_mask), self.num_latents, self.slot_start, self.convention)

    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]


def to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


class PromptBuilder:
    def __init__(self, processor, prompt_cfg, convention: str, num_latents: int):
        self.tok = processor.tokenizer
        self.image_processor = processor.image_processor
        self.cfg = prompt_cfg
        self.convention = convention
        self.K = num_latents
        self.state_token_id = self.tok.convert_tokens_to_ids(prompt_cfg.state_placeholder_token)
        if self.state_token_id is None or self.state_token_id == self.tok.unk_token_id:
            raise ValueError(f"unknown state placeholder token {prompt_cfg.state_placeholder_token}")
        merge = self.image_processor.merge_size
        patch = self.image_processor.patch_size
        side = prompt_cfg.image_size // patch
        self.tokens_per_image = (side * side) // (merge * merge)
        self._cache: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------ text
    def prompt_text(self, instruction: str, n_views: int) -> str:
        vis = "".join("<|vision_start|>" + "<|image_pad|>" * self.tokens_per_image + "<|vision_end|>"
                      for _ in range(n_views))
        state = self.cfg.state_placeholder_token * self.cfg.num_state_tokens
        body = self.cfg.template.format(state=state, instruction=instruction.strip())
        return (f"<|im_start|>system\n{self.cfg.system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{vis}{body}<|im_end|>\n<|im_start|>assistant\n<abs_vis_token>")

    def latent_suffix_ids(self) -> List[int]:
        if self.K == 0:
            n_pad = 0
        else:
            n_pad = self.K if self.convention == "sft" else self.K - 1
        ids = [LATENT_PAD_ID] * n_pad
        if self.cfg.append_latent_end_token:
            ids.append(LATENT_END_ID)
        return ids

    def token_ids(self, instruction: str, n_views: int) -> List[int]:
        key = f"{n_views}|{instruction}"
        if key not in self._cache:
            ids = self.tok(self.prompt_text(instruction, n_views), add_special_tokens=False).input_ids
            if ids[-1] != LATENT_START_ID:
                raise RuntimeError("prompt must end with <abs_vis_token>")
            n_state = sum(1 for i in ids if i == self.state_token_id)
            if n_state != self.cfg.num_state_tokens:
                raise RuntimeError(f"found {n_state} state placeholders, expected {self.cfg.num_state_tokens}")
            if sum(1 for i in ids if i == IMAGE_TOKEN_ID) != n_views * self.tokens_per_image:
                raise RuntimeError("image token count mismatch")
            self._cache[key] = ids + self.latent_suffix_ids()
        return self._cache[key]

    # ------------------------------------------------------------------ images
    def process_images(self, images: Sequence) -> Dict[str, torch.Tensor]:
        s = self.cfg.image_size
        pil = [to_pil(im).resize((s, s), Image.BICUBIC) for im in images]
        out = self.image_processor(images=pil, return_tensors="pt")
        thw = out["image_grid_thw"]
        expect = (s // self.image_processor.patch_size) ** 2
        if not bool((thw.prod(-1) == expect).all()):
            raise RuntimeError(f"unexpected image grid {thw.tolist()} for size {s}")
        return {"pixel_values": out["pixel_values"], "image_grid_thw": thw}

    # ------------------------------------------------------------------ batch
    def build(self, views: List[List], instructions: List[str]) -> MonetInputs:
        """``views[b]`` is the list of camera images (HWC uint8) for sample b."""
        B = len(instructions)
        assert len(views) == B
        n_views = len(views[0])
        seqs = [self.token_ids(ins, n_views) for ins in instructions]
        L = max(len(s) for s in seqs)
        input_ids = torch.full((B, L), PAD_TOKEN_ID, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        for b, s in enumerate(seqs):  # left padding: latent slots are aligned across the batch
            input_ids[b, L - len(s):] = torch.tensor(s)
            attn[b, L - len(s):] = 1
        flat = [im for v in views for im in v]
        img = self.process_images(flat) if flat else {"pixel_values": None, "image_grid_thw": None}
        has_end = 1 if self.cfg.append_latent_end_token else 0
        return MonetInputs(
            input_ids=input_ids, attention_mask=attn, pixel_values=img["pixel_values"],
            image_grid_thw=img["image_grid_thw"], state_mask=input_ids == self.state_token_id,
            num_latents=self.K, slot_start=L - self.K - has_end, convention=self.convention,
        )
