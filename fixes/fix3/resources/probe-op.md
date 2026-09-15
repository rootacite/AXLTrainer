# What `probe_op.py` found: the faulting operation is a text-encoder LoRA A-weight gradient

Two runs of the committed configuration (`network_dim = 48`, `train_batch_size = 2`, seed
`1145141919`) with `probe_op.py` as the trainer's entry point and `HIP_LAUNCH_BLOCKING=1` in the child
environment. Both aborted, and both ended on the same operation.

## The last line before the fault

`probe_op.py` prints every `aten` GEMM / convolution / attention call to the child's stdout, which is the
same stream MIOpen writes its kernel selection to, so the two interleave in `train.out`. With launches
serialized, the op that launches the faulting kernel is the last one printed:

```
OP 8344 backward aten.mm.default [[48, 308], [308, 1280]]
VGPU=0x5645c6184410 SWq=0x7f329c6ba000, HWq=0x7f2fb6e00000, id=1
Dispatch Header =0xd02 (type=2, barrier=1, acquire=2, release=1), setup=3
grid=[2560, 1, 1], workgroup=[128, 1, 1]
private_seg_size=0, group_seg_size=1638
kernel_obj=0x7f2cd2682380, kernarg_address=0x0x7f2fb4a00000
completion_signal=0x0, correlation_id=0
rptr=62022, wptr=62024
:0:rocdevice.cpp            :3678: 3194366185 us:  Memory Fault Error [host: acitehost, GPU index: 0,
    faulting addr: 0x7f2d68800000,
    kernel: Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x128x16_MI16x16x1_SN_…_ISA1201_…_WS32_WG64_2_1]
Memory access fault by GPU node-1 (Agent handle: 0x5645b71d8d60) on address 0x7f2d68800000.
    Reason: Page not present or supervisor privilege.
```

`[[48, 308], [308, 1280]]` is `aten.mm(A, B)` with `A` of shape `[48, 308]` and `B` of shape
`[308, 1280]`, i.e. `M = 48`, `K = 308`, `N = 1280`.

## Where those numbers come from

The loop around it is one transformer layer's backward, four GEMMs per layer, repeating for every layer
of the text encoder (`OP` lines are numbered by the probe; `pass` is the autograd state at dispatch):

```
8331 backward aten.mm.default [[308, 1280], [1280, 48]]    lora_A input grad      [B*seq, r] = [B*seq, in] @ [in, r]
8332 backward aten.mm.default [[48, 308], [308, 1280]]     grad_A                 [r, B*seq] @ [B*seq, in]   <-- the fault
8333 backward aten.mm.default [[308, 48], [48, 1280]]      lora_B input grad
8334 backward aten.mm.default [[308, 1280], [1280, 1280]]  base layer input grad  (a 1280 -> 1280 projection)
```

| dim | value | source |
| --- | --- | --- |
| `M` | 48 | `network_dim` (the rank; `lora_A.weight` is `[r, in_features]` and its gradient has the same shape) |
| `K` | 308 | `train_batch_size` x encoder sequence length = 2 x 154 |
| `N` | 1280 | `in_features` = `text_encoder_2`'s hidden size |

`N = 1280` is the discriminator: `text_encoder_2` is the only 1280-wide module in the graph. The UNet's
cross-attention projections have `in_features = 2048` (their `grad_A` would be `[48, 308] @ [308, 2048]`)
and its spatial layers have `K = batch x latent pixels` (7680 for this bucket, not 308). So the faulting
kernel is launched by **a LoRA A-weight gradient in one of `text_encoder_2`'s attention projections**
(`q_proj`, `k_proj`, `v_proj`, `out_proj` — all `1280 -> 1280`).

This is a refinement of `fixes/fix2.txt`, which described the overrun as living in the *`K = network_dim`*
GEMMs (the `lora_A`/`lora_B` *forward*-side products) and in `d(encoder_hidden_states)`. Here the
offending GEMM is the *weight-gradient* side, where the rank is `M` and the sequence length is `K`.

## Why this explains the whole scan

- **`network_dim` is `M`.** `M = 48` is a partial `MT64` tile (`48 = 0x64 + 48`); `M = 64` — the one rank
  that never aborted in any scan — is exactly one full tile. The rank is the strongest knob in this bug
  because it selects the kernel *of this GEMM*.
- **`batch x chunks` is `K`.** `K = 308 = 19x16 + 4` is a partial `K` tile too; `K = 154` (one CLIP
  chunk) is the other partial case and does *not* abort, which is why `max_token_length = 75` survives
  while `150` dies on the same bucket, images, batch and rank.
- **The bucket is nowhere in this GEMM.** The text encoder sees `batch x sequence` tokens and never sees
  the image. The bucket can therefore only matter through the allocator: the UNet's activations for
  bucket `B` are sized by `B`, and those allocations decide whether the TE LoRA buffers end up against a
  hole. That is exactly the pattern the scans show — the same `(rank, M)` cell is fatal in one bucket and
  fine in another, and the committed seed (bucket `1280x768`, `M = 308`) is fatal in 9 runs out of 9.

## The caveat: without `HIP_LAUNCH_BLOCKING` the same fault is reported elsewhere

The plain runs (no blocking) end their logs on MIOpen's candidate-selection dump for a 3x3 convolution
whose implicit-GEMM problem is `M = 30720`, `N = 384`, `K = 2880` — `M` is this bucket's level-0 feature
map times the batch (`2 x 160 x 96 = 30720`) and `K = 320 x 3 x 3`, i.e. a 320-channel 3x3 conv, which is
the UNet's ResNet block, and `N = 384` is `C_out`. In the blocked run no conv selection is in flight at
the abort at all.

Both are the same Tensile solution (`MT64x128x16`) and the same read-past-the-buffer mechanism, so the
difference is *where the fault is attributed*, not which fault it is: with a queue of pending kernels,
the KFD fault surfaces at the next point the process synchronizes — which was MIOpen's next kernel
lookup — while with blocking it surfaces on the launch that caused it. Anyone re-running this should use
`HIP_LAUNCH_BLOCKING=1` to get an attributable operation, and should not treat the plain log's conv dump
as the culprit.
