from typing import List, Sequence, Tuple, Optional

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
        token_ids[i: i + chunk_size]
        for i in range(0, len(token_ids), chunk_size)
    ]


def _empty_chunk_ids(tokenizer: CLIPTokenizer) -> List[int]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ids = [tokenizer.bos_token_id, tokenizer.eos_token_id]
    pad_n = tokenizer.model_max_length - len(ids)
    if pad_n > 0:
        ids = ids + [pad_id] * pad_n
    else:
        ids = ids[: tokenizer.model_max_length]
        ids[-1] = tokenizer.eos_token_id
    return ids


def _pad_chunk_ids(chunk: List[int], tokenizer: CLIPTokenizer) -> List[int]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ids = [tokenizer.bos_token_id] + chunk + [tokenizer.eos_token_id]
    if len(ids) < tokenizer.model_max_length:
        ids = ids + [pad_id] * (tokenizer.model_max_length - len(ids))
    else:
        ids = ids[: tokenizer.model_max_length]
        ids[-1] = tokenizer.eos_token_id
    return ids


def tokenize_long_prompt(
        text: str,
        tokenizer: CLIPTokenizer,
        max_token_length: int,
        target_num_chunks: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    chunk_size = tokenizer.model_max_length - 2
    cap_chunks = max(1, math.ceil(max_token_length / chunk_size))

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

    if target_num_chunks is not None:
        max_chunks = max(1, int(target_num_chunks))
    else:
        max_chunks = max(1, len(chunks))
        max_chunks = min(max_chunks, cap_chunks)

    while len(chunks) < max_chunks:
        chunks.append([])

    if len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]

    seqs = [torch.tensor(_pad_chunk_ids(chunk, tokenizer), dtype=torch.long) for chunk in chunks]
    return torch.stack(seqs, dim=0), max_chunks


def _get_pooled_output(output) -> torch.Tensor:
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        return output.text_embeds

    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output

    return output.last_hidden_state[:, 0]


def _pad_chunk_stack(
        ids_list: Sequence[torch.Tensor],
        n_chunks: int,
        tokenizer: CLIPTokenizer,
) -> torch.Tensor:
    empty = torch.tensor(_empty_chunk_ids(tokenizer), dtype=torch.long)
    padded: List[torch.Tensor] = []
    for ids in ids_list:
        if ids.shape[0] >= n_chunks:
            padded.append(ids[:n_chunks])
            continue
        extra = empty.unsqueeze(0).expand(n_chunks - ids.shape[0], -1)
        padded.append(torch.cat([ids, extra], dim=0))
    return torch.stack(padded, dim=0)


def _hidden_for_clip_skip(output, clip_skip: int) -> torch.Tensor:
    # clip_skip=0 must use last_hidden_state (final layer norm), not hidden_states[-1].
    if clip_skip > 0:
        return output.hidden_states[-(clip_skip + 1)]
    return output.last_hidden_state


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
        target_num_chunks: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    ids_1_list: List[torch.Tensor] = []
    ids_2_list: List[torch.Tensor] = []

    used_num_chunks = target_num_chunks

    for prompt in prompts:
        ids_1, chunks_1 = tokenize_long_prompt(
            prompt,
            tokenizer_1,
            max_token_length,
            target_num_chunks,
        )
        ids_2, chunks_2 = tokenize_long_prompt(
            prompt,
            tokenizer_2,
            max_token_length,
            target_num_chunks,
        )
        if ids_1.shape[0] != ids_2.shape[0]:
            raise RuntimeError(
                f"Tokenizer chunk mismatch: "
                f"{ids_1.shape[0]} vs {ids_2.shape[0]}"
            )
        ids_1_list.append(ids_1)
        ids_2_list.append(ids_2)
        if used_num_chunks is None:
            used_num_chunks = max(chunks_1, chunks_2)
        else:
            used_num_chunks = max(used_num_chunks, chunks_1, chunks_2)

    if used_num_chunks is None:
        used_num_chunks = 1

    ids_1 = _pad_chunk_stack(ids_1_list, used_num_chunks, tokenizer_1).to(device)
    ids_2 = _pad_chunk_stack(ids_2_list, used_num_chunks, tokenizer_2).to(device)

    batch_size, n_chunks, seq_len = ids_1.shape
    flat_1 = ids_1.reshape(batch_size * n_chunks, seq_len)
    flat_2 = ids_2.reshape(batch_size * n_chunks, seq_len)

    out1 = text_encoder_1(
        flat_1,
        output_hidden_states=True,
        return_dict=True,
    )
    out2 = text_encoder_2(
        flat_2,
        output_hidden_states=True,
        return_dict=True,
    )

    hs1 = _hidden_for_clip_skip(out1, clip_skip)
    hs2 = _hidden_for_clip_skip(out2, clip_skip)
    pooled = _get_pooled_output(out2)

    hs1 = hs1.view(batch_size, n_chunks, seq_len, hs1.shape[-1])
    hs2 = hs2.view(batch_size, n_chunks, seq_len, hs2.shape[-1])
    prompt_embeds = torch.cat(
        [hs1.reshape(batch_size, n_chunks * seq_len, -1), hs2.reshape(batch_size, n_chunks * seq_len, -1)],
        dim=-1,
    ).to(dtype)

    pooled_prompt_embeds = pooled.view(batch_size, n_chunks, pooled.shape[-1])[:, 0, :].to(dtype)

    return prompt_embeds, pooled_prompt_embeds, used_num_chunks
