from typing import List, Sequence, Tuple

import torch
from transformers import (
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

import math


def _chunk_ids(
    token_ids: List[int],
    chunk_size: int,
    max_token_length: int,
) -> List[List[int]]:
    token_ids = token_ids[:max_token_length]

    if not token_ids:
        return [[]]

    return [
        token_ids[i : i + chunk_size]
        for i in range(0, len(token_ids), chunk_size)
    ]

def tokenize_long_prompt(
    text: str,
    tokenizer: CLIPTokenizer,
    max_token_length: int,
) -> torch.Tensor:
    chunk_size = tokenizer.model_max_length - 2

    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        verbose=False,
    ).input_ids

    chunks = _chunk_ids(
        token_ids,
        chunk_size,
        max_token_length,
    )

    max_chunks = max(1, math.ceil(max_token_length / chunk_size))
    while len(chunks) < max_chunks:
        chunks.append([])

    seqs = []

    for chunk in chunks:
        ids = (
            [tokenizer.bos_token_id]
            + chunk
            + [tokenizer.eos_token_id]
        )

        if len(ids) < tokenizer.model_max_length:
            ids = ids + [tokenizer.pad_token_id] * (
                tokenizer.model_max_length - len(ids)
            )
        else:
            ids = ids[: tokenizer.model_max_length]
            ids[-1] = tokenizer.eos_token_id

        seqs.append(torch.tensor(ids, dtype=torch.long))

    return torch.stack(seqs, dim=0)


def _get_pooled_output(output) -> torch.Tensor:
    """
    兼容不同 transformers / diffusers 版本
    """

    # 新版本 SDXL 常见
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds

    # 某些 CLIP 版本
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output

    # fallback
    return output.last_hidden_state[:, 0]


def encode_prompt_batch(
    prompts: Sequence[str],
    tokenizer_1: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    text_encoder_1: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection,
    clip_skip: int,
    max_token_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:

    prompt_embeds_out: List[torch.Tensor] = []
    pooled_out: List[torch.Tensor] = []

    for prompt in prompts:

        ids_1 = tokenize_long_prompt(
            prompt,
            tokenizer_1,
            max_token_length,
        ).to(device)

        ids_2 = tokenize_long_prompt(
            prompt,
            tokenizer_2,
            max_token_length,
        ).to(device)

        if len(ids_1) != len(ids_2):
            raise RuntimeError(
                f"Tokenizer chunk mismatch: "
                f"{len(ids_1)} vs {len(ids_2)}"
            )

        chunk_embeds_1 = []
        chunk_embeds_2 = []
        pooled_chunks = []

        for c1, c2 in zip(ids_1, ids_2):

            c1 = c1.unsqueeze(0)
            c2 = c2.unsqueeze(0)

            out1 = text_encoder_1(
                c1,
                output_hidden_states=True,
                return_dict=True,
            )

            out2 = text_encoder_2(
                c2,
                output_hidden_states=True,
                return_dict=True,
            )

            if clip_skip > 0:
                hs1 = out1.hidden_states[-(clip_skip + 1)]
                hs2 = out2.hidden_states[-(clip_skip + 1)]
            else:
                hs1 = out1.last_hidden_state
                hs2 = out2.last_hidden_state

            pooled = _get_pooled_output(out2)

            chunk_embeds_1.append(hs1)
            chunk_embeds_2.append(hs2)
            pooled_chunks.append(pooled)

        emb1 = torch.cat(chunk_embeds_1, dim=1)
        emb2 = torch.cat(chunk_embeds_2, dim=1)

        prompt_embeds = torch.cat(
            [emb1, emb2],
            dim=-1,
        )

        pooled_prompt_embeds = pooled_chunks[0]

        prompt_embeds_out.append(
            prompt_embeds.squeeze(0).to(dtype)
        )

        pooled_out.append(
            pooled_prompt_embeds.squeeze(0).to(dtype)
        )

    return (
        torch.stack(prompt_embeds_out, dim=0),
        torch.stack(pooled_out, dim=0),
    )
