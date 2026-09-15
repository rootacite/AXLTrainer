import unittest
from types import SimpleNamespace

import torch
from torch import nn

import sys
from pathlib import Path

# `python test/test_text_processing.py` has to import the repo's own packages, exactly like
# `unittest discover -s test` does from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from text_processing import (
    encode_prompt_batch,
    tokenize_long_prompt,
)


class FakeTokenizer:
    model_max_length = 77
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, truncation=False, verbose=False):
        ids = [3 + (ord(ch) % 50) for ch in text]
        return SimpleNamespace(input_ids=ids)


class FakeTE(nn.Module):
    def __init__(self, hidden: int = 8, n_layers: int = 3, with_text_embeds: bool = False):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.with_text_embeds = with_text_embeds
        self.calls = 0
        self.last_batch = 0
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, input_ids, output_hidden_states=True, return_dict=True):
        self.calls += 1
        self.last_batch = int(input_ids.shape[0])
        batch, seq = input_ids.shape
        base = input_ids.to(dtype=torch.float32).unsqueeze(-1).expand(batch, seq, self.hidden)
        hidden_states = tuple(base + float(i) for i in range(self.n_layers + 1))
        last_hidden = hidden_states[-1] + 5.0 * self.scale
        pooled = hidden_states[-1][:, 0, :]
        return SimpleNamespace(
            hidden_states=hidden_states,
            last_hidden_state=last_hidden,
            text_embeds=pooled if self.with_text_embeds else None,
            pooler_output=None if self.with_text_embeds else pooled,
        )


class TokenizeLongPromptTest(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTokenizer()

    def test_short_prompt_is_one_chunk(self):
        ids, n_chunks = tokenize_long_prompt("short", self.tok, max_token_length=225)
        self.assertEqual(n_chunks, 1)
        self.assertEqual(tuple(ids.shape), (1, 77))

    def test_long_prompt_uses_needed_chunks_not_the_cap(self):
        text = "x" * 80
        ids, n_chunks = tokenize_long_prompt(text, self.tok, max_token_length=225)
        self.assertEqual(n_chunks, 2)
        self.assertEqual(tuple(ids.shape), (2, 77))

    def test_target_num_chunks_pads(self):
        ids, n_chunks = tokenize_long_prompt(
            "short", self.tok, max_token_length=225, target_num_chunks=3
        )
        self.assertEqual(n_chunks, 3)
        self.assertEqual(tuple(ids.shape), (3, 77))


class EncodePromptBatchTest(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTokenizer()
        self.te1 = FakeTE(with_text_embeds=False)
        self.te2 = FakeTE(with_text_embeds=True)

    def test_encoders_called_once_and_short_batch_is_one_chunk(self):
        embeds, pooled, n_chunks = encode_prompt_batch(
            prompts=["one", "two", "three"],
            tokenizer_1=self.tok,
            tokenizer_2=self.tok,
            text_encoder_1=self.te1,
            text_encoder_2=self.te2,
            clip_skip=1,
            max_token_length=225,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(n_chunks, 1)
        self.assertEqual(self.te1.calls, 1)
        self.assertEqual(self.te2.calls, 1)
        self.assertEqual(self.te1.last_batch, 3)
        self.assertEqual(tuple(embeds.shape), (3, 77, 16))
        self.assertEqual(tuple(pooled.shape), (3, 8))
        self.assertTrue(torch.isfinite(embeds).all())
        self.assertFalse(torch.allclose(embeds, torch.zeros_like(embeds)))

    def test_clip_skip_zero_uses_last_hidden_state(self):
        kwargs = dict(
            prompts=["hello world"],
            tokenizer_1=self.tok,
            tokenizer_2=self.tok,
            text_encoder_1=self.te1,
            text_encoder_2=self.te2,
            max_token_length=225,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        skip0, pooled0, _ = encode_prompt_batch(clip_skip=0, **kwargs)
        self.te1.calls = 0
        self.te2.calls = 0
        skip1, pooled1, _ = encode_prompt_batch(clip_skip=1, **kwargs)
        self.assertTrue(torch.isfinite(skip0).all())
        self.assertTrue(torch.isfinite(skip1).all())
        self.assertFalse(torch.allclose(skip0, skip1))
        self.assertTrue(torch.isfinite(pooled0).all())
        self.assertTrue(torch.isfinite(pooled1).all())

    def test_mixed_lengths_pad_to_batch_max(self):
        embeds, _, n_chunks = encode_prompt_batch(
            prompts=["short", "y" * 80],
            tokenizer_1=self.tok,
            tokenizer_2=self.tok,
            text_encoder_1=self.te1,
            text_encoder_2=self.te2,
            clip_skip=1,
            max_token_length=225,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(n_chunks, 2)
        self.assertEqual(self.te1.last_batch, 4)
        self.assertEqual(tuple(embeds.shape), (2, 154, 16))

    def test_target_num_chunks_aligns_negative(self):
        _, _, n_pos = encode_prompt_batch(
            prompts=["z" * 80],
            tokenizer_1=self.tok,
            tokenizer_2=self.tok,
            text_encoder_1=self.te1,
            text_encoder_2=self.te2,
            clip_skip=1,
            max_token_length=225,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.te1.calls = 0
        embeds_neg, _, n_neg = encode_prompt_batch(
            prompts=["no"],
            tokenizer_1=self.tok,
            tokenizer_2=self.tok,
            text_encoder_1=self.te1,
            text_encoder_2=self.te2,
            clip_skip=1,
            max_token_length=225,
            device=torch.device("cpu"),
            dtype=torch.float32,
            target_num_chunks=n_pos,
        )
        self.assertEqual(n_pos, n_neg)
        self.assertEqual(embeds_neg.shape[1], n_pos * 77)


if __name__ == "__main__":
    unittest.main()
