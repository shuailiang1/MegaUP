# Mega-UP

Mega-UP provides two CuTe DSL kernels that fuse an attention
projection with the data movement of a Ulysses all-to-all:

1. **Peer-write GEMM** (sequence-parallel to head-parallel). It computes
   `x @ W.T (+ b)` and its epilogue stores every head shard of the output
   directly into the symmetric-memory slot of the GPU that owns those heads.
   It replaces `linear + permute + all_to_all`, the exchange in front of
   attention.
2. **Head-shard pull GEMM** (head-parallel to sequence-parallel). Dedicated
   warps pull the head shards of each row from the peers' symmetric memory and
   assemble the dense input while the MMA warps already consume completed
   tiles. It replaces `all_to_all + permute + linear`, the exchange after
   attention.

Neither kernel launches a separate collective or permutation, and both produce
the same bits as the PyTorch reference (`torch.nn.functional.linear` +
`all_to_all_single`) unless cuBLAS itself splits K for the reference at very
small M; see [Correctness](#correctness).

The repository contains only the two kernels, their symmetric-memory runtime,
host APIs for the BSHD, SBHD and packed THD layouts, tests and benchmarks. It
does not contain attention kernels, model code or framework integration.

| Layout | `*peer_write_gemm_*`: input to output | `*headshard_pull_gemm_*`: input to output | Layout copies |
| --- | --- | --- | --- |
| BSHD | `[B, S, H, D]` to `[B, U*S, N/(U*D), D]` | `[B, U*S, H/U, D]` to `[B, S, N/D, D]` | contiguous for B = 1, strided for B > 1 |
| SBHD | `[S, B, H, D]` to `[U*S, B, N/(U*D), D]` | `[U*S, B, H/U, D]` to `[S, B, N/D, D]` | contiguous |
| THD (packed) | `[T, H, D]` to `[U*T, N/(U*D), D]` | `[U*T, H/U, D]` to `[T, N/D, D]` | contiguous, or one row gather with a `THDLayout` |

The weight `W` is `[N, K]` (the `torch.nn.functional.linear` layout) with
`K = H*D`, and it does not have to be square: grouped-query K and V
projections, fused QKV projections and output projections whose input width
differs from the model width are all supported. `S` and `T` are the rank-local
sequence length and token count; every rank of the group uses the same shape.

## Supported configurations

The ranks exchange the output of a peer-write projection and the input of a
pull projection, so the exchanged width is `N` for a peer-write and `K` for a
pull. That width has to split into equal head shards; the other width stays
on each rank and only has to fit the GEMM tiles:

| Constraint | Why |
| --- | --- |
| `U` in {2, 4, 8}, ranks of one node with peer access | symmetric memory over NVLink |
| exchanged width a multiple of 256 | the peer-write GEMM tiles N with 256-wide tiles |
| head shard `width / U` a multiple of 128 and of `D` | one warp-wide 64-bit pull copy moves 128 columns; each rank receives whole heads |
| pull output width `N` a multiple of 256 and of `D` | the pull GEMM tiles N with 256-wide tiles |
| peer-write input width `K` a multiple of 8 | TMA rows must be 16-byte aligned |
| BF16 activations, weight and bias, FP32 accumulation | |

`mega_up.supports_shape(width, U)` evaluates the rule for an exchanged width
and `mega_up.unsupported_shape_reason(width, U)` explains a rejection; the
`prepare_*` functions check every constraint. The rule contains no list of
validated sizes: with `head_dim = 128` or `256` every exchanged head count
divisible by `U` is supported, and with `head_dim = 64` every count divisible
by `2U`. Examples:

| Model, projection | `W` (`[N, K]`) | exchanged heads x dim | U = 2 | U = 4 | U = 8 |
| --- | ---: | :---: | :---: | :---: | :---: |
| Wan2.1 1.3B, Q/K/V/O | 1536 x 1536 | 12 x 128 | yes | yes | no (12 heads) |
| LTX-Video, Q/K/V/O | 2048 x 2048 | 32 x 64 | yes | yes | yes |
| Flux.1, HunyuanVideo, Q/K/V/O | 3072 x 3072 | 24 x 128 | yes | yes | yes |
| Llama-3 8B, Q and O | 4096 x 4096 | 32 x 128 | yes | yes | yes |
| Llama-3 8B, K or V | 1024 x 4096 | 8 x 128 | yes | yes | yes |
| Llama-3 8B, fused QKV | 6144 x 4096 | 48 x 128 | yes | yes | yes |
| Wan2.1 / Wan2.2 14B, Cosmos-Predict2.5 14B, Q/K/V/O | 5120 x 5120 | 40 x 128 | yes | yes | yes |
| Step-Video-T2V, Q/K/V/O | 6144 x 6144 | 48 x 128 | yes | yes | yes |
| Llama-3 70B, Q and O | 8192 x 8192 | 64 x 128 | yes | yes | yes |
| Llama-3 70B, K or V | 1024 x 8192 | 8 x 128 | yes | yes | yes |

A fused QKV projection is one peer-write. The kernel splits the output
columns into `U` contiguous blocks, and rank `p` receives columns
`[p*N/U, (p+1)*N/U)`, so the rows of the weight (and the bias) must be grouped
by rank:

- Rows packed per head as `[H, 3, D]` (each head's query, key and value rows
  adjacent, the layout of Megatron-style `query_key_value` layers) are
  already grouped: rank `p` receives heads `[p*H/U, (p+1)*H/U)` as
  `[..., 3*H/U, D]`, which views as `[..., H/U, 3, D]`.
- Rows packed as `[3, H, D]` (all query rows, then all key rows, then all
  value rows, as in `torch.nn.MultiheadAttention.in_proj_weight`) need one row
  permutation of the weight and the bias beforehand.
- With grouped-query attention the block of rank `p` holds its `H_q/U` query
  heads and its `H_kv/U` key and value heads, in the order the caller chose;
  the received `[..., (H_q + 2*H_kv)/U, D]` tensor splits accordingly.

Both kernels accept an optional bias, which most diffusion-transformer
attention projections carry. It is added to the FP32 accumulator before the
single rounding to BF16, exactly like `F.linear(x, W, b)`.

Head shards do not have to align with the 256-wide GEMM tiles: at
N = 5120 / U = 8 each rank owns 640 columns, and the peer-write epilogue
routes every 64-column subtile on its own. An odd number of 256-wide N tiles
(for example N = 2304) is also supported.

## Requirements

- A GPU supported by the kernels; `initialize_runtime` checks the device and
  rejects unsupported ones.
- All ranks of a Ulysses group on one node and mutually peer-accessible.
- Python 3.10 or newer.
- PyTorch 2.7 or newer with NCCL and symmetric-memory support. The runtime
  uses private PyTorch symmetric-memory APIs; revalidate it when upgrading
  PyTorch.
- NVIDIA CUTLASS DSL `>=4.4.2,<4.9` with `apache-tvm-ffi`, and CUDA Python
  `>=12.9,<14`. Tested combinations: CUTLASS DSL 4.4.2 with apache-tvm-ffi
  0.1.8, 0.1.9 and 0.1.15, and CUTLASS DSL 4.8.1 (`nvidia-cutlass-dsl[cu13]`
  for CUDA 13) with apache-tvm-ffi 0.1.15. DSL 4.8.1 needs apache-tvm-ffi
  0.1.10 or newer and has been validated for U = 2 and 4 (see
  [Correctness](#correctness)); the DSL releases in between were not run.
- Contiguous BF16 tensors; a bias must be 16-byte aligned.

The first peer-write compilation temporarily installs, then restores, a
CUTLASS DSL argument-converter compatibility shim. Do not compile an
unrelated CuTe DSL kernel concurrently in another Python thread during that
first call.

## Install

```bash
pip install -e .
```

The kernels compile on first use, once per static configuration (weight
shape, U, with or without bias). The token count is a runtime value: changing
the sequence length never recompiles.

## Usage

Launch one process per GPU and pass the Ulysses process group explicitly.

```python
import os

import torch
import torch.distributed as dist

from mega_up import (
    initialize_runtime,
    prepare_headshard_pull_gemm_bshd,
    prepare_peer_write_gemm_bshd,
)

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

U = dist.get_world_size()
B, S, D = 1, 32768, 128                   # S: local sequence length
H, H_kv = 32, 8                           # query and key/value heads
hidden = H * D                            # model width
runtime = initialize_runtime(dist.group.WORLD, hidden=hidden, max_rows=B * S, device=device)

torch.manual_seed(0)                      # identical parameters on every rank
w_v = torch.randn(H_kv * D, hidden, dtype=torch.bfloat16, device=device) / hidden**0.5
b_v = torch.randn(H_kv * D, dtype=torch.bfloat16, device=device)
w_o = torch.randn(hidden, H * D, dtype=torch.bfloat16, device=device) / hidden**0.5
b_o = torch.randn(hidden, dtype=torch.bfloat16, device=device)

# linear + all-to-all: [B, S, H, D] -> [B, U*S, H_kv/U, D]
v_proj = prepare_peer_write_gemm_bshd(
    w_v, runtime, batch=B, sequence=S, heads=H, head_dim=D, bias=b_v
)
# all-to-all + linear: [B, U*S, H/U, D] -> [B, S, hidden/D, D]
out_proj = prepare_headshard_pull_gemm_bshd(
    w_o, runtime, batch=B, sequence=S, heads=H, head_dim=D, bias=b_o
)

x = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)  # [B, S, hidden] as heads
v = v_proj(x)                                                      # [B, U*S, H_kv/U, D]
attn = torch.randn(B, U * S, H // U, D, dtype=torch.bfloat16, device=device)  # attention output
y = out_proj(attn)                                                 # [B, S, hidden/D, D]
```

The peer-write input only has to factor `K` as `heads * head_dim`, and
`head_dim` is the head size of the exchanged output. `hidden` sizes the arena
for the widest exchanged width of the projections that share the runtime (the
peer-write `N` or the pull `K`). Projections of different exchanged widths
can also use a runtime each: a process group holds one runtime per `hidden`;
see [Runtime](#runtime).

`prepare_*` validates the weight, bias and shape on all ranks, compiles the
kernel if this configuration has not been compiled yet, and binds it to the
weight, the bias and the arena slots. A call of the returned executable
checks its input, launches the kernel and the layout copy, and returns. The
peer-write executable returns a new tensor (or fills `out=`); the pull
executable owns its materialized-input and output buffers, so its result is
overwritten by the next call (clone it to keep it). `peer_write_gemm_*` and
`headshard_pull_gemm_*` are one-shot wrappers that prepare on every call.

The APIs are forward-only building blocks and do not register autograd
formulas; the backward of each projection is the other kernel (the data
gradient of a peer-write projection with weight `W` (`[N, K]`) is a pull GEMM
with weight `W.T` (`[K, N]`), and vice versa).

Every call is collective: all ranks of the group must issue the same calls in
the same order. Before any device-side barrier, the APIs exchange the call's
metadata and every rank's local validation error, so an invalid argument on one
rank raises on all ranks instead of leaving the others waiting in a barrier.
The exchange is a host-side `all_gather_object` and synchronizes the host with
the device; pass `validate_group=False` to the calls of a validated
steady-state loop. A runtime may be used from different CUDA streams one
operation at a time.

### Packed THD

Packed (varlen) training concatenates samples along one token axis. With
`x` of shape `[T, H, D]` (the same `T` on every rank), the THD APIs follow the
usual Ulysses contract: rank `r` holds rows `[r*T, (r+1)*T)` of the global
packed stream, the head-parallel result lists the `U` slabs in rank order
(the global packed order, so global `cu_seqlens` apply unchanged), and the
materialization is one contiguous copy.

```python
from mega_up import prepare_headshard_pull_gemm_thd, prepare_peer_write_gemm_thd

v_proj = prepare_peer_write_gemm_thd(w_v, runtime, heads=H, head_dim=D, tokens=T, bias=b_v)
out_proj = prepare_headshard_pull_gemm_thd(w_o, runtime, heads=H, head_dim=D, tokens=T, bias=b_o)
v = v_proj(x)                                                           # [U*T, H_kv/U, D]
y = out_proj(attn)                                                      # [T, hidden/D, D]
```

The token count is a runtime argument of the kernels, so preparing for a new
`T` binds the compiled kernels again but does not recompile them.

For pipelines that instead split every sample evenly across the ranks (so all
ranks share the rank-local `cu_seqlens`), build a `THDLayout` once per step
with `prepare_thd_layout(cu_seqlens, runtime)` and pass `layout=` to both
calls. Every sample of the head-parallel tensor is then contiguous with its
`U` chunks in rank order and boundaries `layout.cu_seqlens_out`
(`U * cu_seqlens`). The regroup is one row gather (`index_select`) driven by two
precomputed permutations and moves the same bytes as the contiguous copy. The
layout's fingerprint takes part in the collective validation, so a rank whose
`cu_seqlens` differ is rejected on every rank.

### Sequence-first SBHD

Frameworks that keep activations sequence-first (`[S, B, H, D]`, Transformer
Engine's `sbhd` format) use the SBHD APIs. Rank `r` holds sequence chunk `r`,
and the head-parallel tensor lists the chunks in rank order along the sequence
dimension. Because rows are sequence-major, the peer-write receive slots
already have the head-parallel layout and the head-parallel input is already
destination-major, so both copies are contiguous for every `B`.

```python
from mega_up import prepare_headshard_pull_gemm_sbhd, prepare_peer_write_gemm_sbhd

v_proj = prepare_peer_write_gemm_sbhd(
    w_v, runtime, sequence=S, batch=B, heads=H, head_dim=D, bias=b_v
)
out_proj = prepare_headshard_pull_gemm_sbhd(
    w_o, runtime, sequence=S, batch=B, heads=H, head_dim=D, bias=b_o
)
v = v_proj(x_sbhd)                                                       # [U*S, B, H_kv/U, D]
y = out_proj(attn_sbhd)                                                  # [S, B, hidden/D, D]
```

### BSHD with B > 1

The kernels see a matrix with `M = B*S` rows. For `B > 1`, the head-parallel
layout `[B, U*S, H/U, D]` interleaves batches and sources, so the
materialization (peer-write) and the staging (pull) are strided copies instead
of contiguous ones. At hidden 5120, U = 8 and 65,536 tokens per rank, `B = 4`
took 22-25% more API time than `B = 1` with the same token count. SBHD and
packed THD avoid it.

## Design

### Peer-write GEMM

The kernel is a persistent GEMM (256x256 tiles on 2-CTA clusters, dynamic tile
scheduling) derived from QuACK's GEMM. Output column block `p`
(`N/U` columns) belongs to rank `p`. The epilogue converts each 128x64
accumulator subtile to BF16 and issues one TMA store into the destination
rank's slot of the symmetric arena:

- Every row of a subtile reaches the peer as one NVLink write, so the subtile
  width sets the write size. With all eight GPUs exchanging, 128-byte rows
  (64-column subtiles) sustained about 1.6x the peer-write bandwidth of
  64-byte rows (32 columns); wider subtiles gained nothing more. At hidden
  2048 / U = 8 the wider subtile took the peer-write kernel
  from 1.6-1.8x to 1.1-1.2x the dense GEMM, and at 3072 / U = 8 from
  1.16-1.23x to 1.01-1.04x.
- When the 256-wide N tile divides the head shard, the whole tile goes to one
  peer. Otherwise (a shard narrower than the tile, or a shard such as 640
  columns that straddles two tiles) each subtile picks its destination from
  its global column, so the GEMM keeps its full-size tiles.
- N tiles map to destination peers, so if every rank visited the N tiles in
  the same order, all ranks would write into the same one or two peers at any
  moment. The tile scheduler therefore starts each rank at its own column
  block (`peer_n_rotation`). Without the rotation the peer-write kernel runs
  at 1.14x the dense GEMM at hidden 5120 / U = 8 (1.28x with 32-column
  subtiles) and within 3% of it at 4096 / U = 8 and 5120 / U = 4; with it,
  at 1.00-1.01x.
- The optional bias is read per subtile, partitioned exactly like the
  accumulator fragment, and added in FP32 before the conversion.

### Head-shard pull GEMM

Each CTA of the persistent kernel runs four epilogue warps, one MMA warp, one
TMA warp and two pull warps. Pull warps take tickets of 4, 8 or 16 rows from a
global atomic queue, copy the rows' head shards from every peer into the local
dense input, and publish one release-ordered ready count per M tile; the TMA
warp waits for an M tile's count before loading its A tiles. Two CTAs of a 1x2
cluster share each M tile and multicast A.

- The copy uses the widest per-lane vector that tiles the shard: 256-, 128- or
  64-bit lanes (shards that are multiples of 512, 256 or 128 columns).
- A ticket loads its rows of a peer shard into registers before storing any
  of them, so the ticket size sets the remote loads in flight per lane. Each
  load waits for a full NVLink round trip: with 8-row tickets, shards of 256
  columns (512 bytes per row and peer) kept too few loads in flight, and the
  hidden 2048 / U = 8 pull GEMM ran at 1.9-2.0x the dense GEMM; 16-row
  tickets brought it to 1.4-1.5x. Wider shards gained nothing from more rows.
- The row group is bounded to fit the pull warps' registers: 16-row tickets
  up to 256-column shards, 8-row tickets up to 1280, 4-row tickets beyond, and
  row groups wider than 4 x 3072 elements are copied in column chunks.
  Without the bound, 4096-column shards compiled with a stack frame and the
  16384/U=4 pull GEMM ran at 1.33x the dense GEMM instead of 1.17x.
- For 4096-element rows (K = 4096) a bulk-copy ring (one warp, three
  full-row SMEM stages, `cp.async.bulk`) replaces the vector copy; it
  measured faster only for that row length.
- With an odd number of 256-wide N tiles, the second CTA of the last cluster
  computes an out-of-bounds tile: TMA zero-fills its loads and drops its
  stores.
- The C staging buffer is acquired before every RF-to-SMEM write, including
  the first subtile of a tile, so the store pipeline's write-after-read
  dependency holds across tile boundaries regardless of warp timing (see
  [Epilogue regression test](#epilogue-regression-test)).

### Runtime

`initialize_runtime` collectively allocates one BF16 symmetric arena of
`max_rows * hidden` elements per rank and validates that every rank requested
the same configuration. An operation on `rows` rows of exchanged width `w`
fits when `rows * w <= max_rows * hidden`, so one runtime serves projections
of different widths. A process group can also hold several runtimes, one per
`hidden`, each with its own arena (for example `hidden = 3F` for a fused QKV
projection and `hidden = F` for the output projection); a repeated call with
the `hidden` of an existing runtime returns that runtime and has to repeat
its `max_rows`. Both kernels use densely packed slots (slot `p` at
`p * rows * w/U`): peer writes land source-major, pull inputs are staged
destination-major. Every operation follows the same protocol:

1. peer writes or peer reads complete, then a symmetric barrier;
2. the result is copied into ordinary CUDA memory (no returned tensor aliases
   the arena);
3. a second barrier before the arena is reused.

A non-reentrant lease with completion events keeps the arena from being reused
while earlier work on another stream is still running.

A prepared call checks its input against the prepared shape, takes the lease
and enqueues the kernel, the barriers and the layout copy. On the benchmark
node at U = 8 that costs about 0.11 ms of host time for the peer-write and
0.21 ms for the pull, independent of the token count. The one-shot
peer-write, which also validates and binds the weight and the arena slots,
takes about 0.47 ms.

## Correctness

| Test | Command | Covers |
| --- | --- | --- |
| Layout and shape rules (CPU) | `pytest tests/test_layout.py tests/test_shapes.py` | BSHD/SBHD/THD permutations, the shape rule, tile rotation, pull lane/ticket/chunk selection |
| Distributed correctness | `torchrun --standalone --nproc-per-node=U tests/run_distributed_correctness.py --up U --heads H [--peer-out-heads E] [--pull-in-heads P] [--pull-out-heads Q] [--runtime-per-width] [--bias]` | both kernels through the one-shot and the prepared APIs (a prepared executable is called on two different inputs), square or rectangular weights (peer-write `[E*D, H*D]`, pull `[Q*D, P*D]`), one shared runtime or one runtime per exchanged width, BSHD, SBHD and THD (contiguous and split-sample), bias, collective rejection of rank-local errors |
| The same, as pytest | `pytest -m gpu tests/test_distributed.py` | a matrix of square and rectangular weights, U and bias; cases with too few GPUs are skipped |
| Epilogue under SM contention (1 GPU) | `python tests/run_pull_gemm_contention.py --hidden 5120 [--out-features N] --up 8 --contention [--bias]` | pull-GEMM output bit-stable across calls while other kernels share the SMs |

The distributed runner compares against explicit PyTorch references
(`F.linear` + `all_to_all_single`, and slice-by-slice regroups for THD that do
not share the library's index tensors) and reports, per check, whether the
result is bitwise equal.

Results on 8-GPU nodes of one type (PyTorch 2.13, CUDA 13.2), with CUTLASS DSL
4.4.2 unless stated otherwise:

- hidden 1280, 2048, 2304, 2560, 3072, 3328, 4096, 4864, 5120, 6144 and 8192
  with every supported U, with and without bias (48 runs), each with BSHD
  cases `(B, S)` = (1, 64), (2, 17), (1, 4096), (3, 1365), SBHD cases
  `(S, B)` = (64, 2), (17, 3), (1365, 3) and THD cases with unequal, odd and
  length-1 samples, contiguous and split-sample: the pulled input is always
  bitwise equal, and both GEMM outputs, through the one-shot and the prepared
  APIs, are bitwise equal to the reference except for the 34-row cases at
  hidden 4864-6144 without bias, which differ by one or two BF16 ulp because
  the cuBLAS reference depends on M there (see "Small M" below).
- With bias, every one of these cases is bitwise equal.
- The same cases, with and without bias, at other head layouts: 128-dim heads
  x 8 (U = 8, 128-column shards), x 12 (U = 2 and 4) and x 56 (U = 8);
  256-dim heads x 8 (U = 8), x 12 (U = 4), x 20 (U = 2) and x 24 (U = 8);
  64-dim heads x 16 (U = 8) and x 32 (U = 4 and 8). Every check is bitwise
  equal except, again, the 34-row cases at hidden 5120 and 6144 without bias.
- Rectangular weights (`W` as `[N, K]`), with the BSHD, SBHD and THD cases
  above: peer-write 6144x4096 (fused grouped-query QKV, U = 8, bias),
  8192x3072 with pull 3072x4096 (U = 4), 1024x2048 with pull 2048x1024 (U = 2,
  bias), 5120x2880 with pull 3072x4096 (64-dim heads, U = 8, bias), 2560x1040
  with pull 1280x2560 (80-dim heads, K not a multiple of 64, U = 2),
  15360x5120 (fused QKV, U = 8), 8192x2048 with pull 2048x8192 (U = 4, bias),
  6144x2048 for both kernels (U = 2 and 4) and 2048x6144 for both kernels
  (U = 4 and 8). Every check is bitwise equal except small-M outputs of the
  2048x8192 pull, the 5120x5120 pull without bias and both 2048x6144 kernels
  (see "Small M").
- One runtime per exchanged width on the same process group
  (`--runtime-per-width`): 6144x4096 with pull 4096x4096 (U = 8, bias),
  8192x3072 with pull 3072x4096 (U = 4, bias), 6144x2048 (U = 2 and 4) and
  2048x6144 (U = 4 and 8). Every check is bitwise equal except the small-M
  outputs of the 2048x6144 kernels.
- Small M: the fused kernels accumulate K in one pass for every M, so a row
  of their output does not depend on the number of rows in the call. The
  cuBLAS reference does for large K: its result for a small call differs by
  one or two BF16 ulp from the same rows of its own 4096-row result (seen for
  K from 4864 to 8192 and calls of 17 to 96 rows; the range depends on the
  shape). These are exactly the reference outputs the fused kernels differ
  from, and both kernels reproduce the rows of the 4096-row cuBLAS result bit
  for bit.
- CUTLASS DSL 4.8.1 (four GPUs, so U = 2 and 4 only): hidden 1280, 2304 and
  8192 at U = 2 and 4096 at U = 4, and the U = 2 and U = 4 rectangular
  configurations above, with a shared runtime and with one runtime per width
  (12 runs). The outcome is the same as with 4.4.2, including the small-M
  reference outputs.
- Contention: 200 calls each at hidden 2304/U=2, 4096/U=8, 5120/U=8 and
  8192/U=2, with and without bias, and for the rectangular pulls 3072x4096
  (U = 4, bias), 2048x8192 (U = 4), 4096x1024 (U = 8, bias) and 1280x4096
  (U = 8), while elementwise or mixed elementwise and GEMM kernels run on a
  side stream: zero changed outputs. With CUTLASS DSL 4.8.1: hidden 2048/U=8,
  2304/U=2 and 5120/U=8 and the rectangular pulls 3072x4096 (U = 4),
  2048x8192 (U = 4), 4096x1024 (U = 8), 1280x4096 (U = 8) and 6144x2048
  (U = 4): zero changed outputs.

### Epilogue regression test

An earlier revision of the pull GEMM waited for the previous TMA store to
finish reading the C staging buffer only between subtiles of one tile, not
across a tile boundary. When SMEM-free kernels share the SM, the epilogue
warps lose issue slots, the MMA runs ahead, and the next tile's first subtile
could overwrite the buffer while the previous tile's last store was still
reading it, silently corrupting the last 16 columns of a tile.
`tests/run_pull_gemm_contention.py` reproduces that regime on one GPU (it
emulates a UP group with local buffers) and checks every call bitwise against
the first; with the old revision it reported 4/200 corrupted calls under
elementwise contention and 72/200 under mixed contention at hidden 2048.
`--kernel-file` loads another kernel revision for A/B runs.

## Performance

Setup: one 8-GPU node with NVLink and no other workload, CUTLASS DSL 4.4.2
(no performance numbers were measured with 4.8.1),
`benchmarks/bench_bshd.py` with the packed THD layout and 65,536 tokens per
rank, 10 warmups and 40 measured iterations. Every configuration ran twice,
back to back: with the SM clocks of the used GPUs locked to one frequency
(`nvidia-smi -lgc`) and at the default clocks; the clocks were sampled during
every run. Every sample is the maximum over the ranks of one
CUDA-event interval; the paths are timed in rotating order and the tables show
medians. Before timing, the benchmark checks the fused outputs against the
PyTorch pipeline; every configuration below was bitwise equal.

- **kernel / GEMM**: the fused kernel alone (input already staged, no
  barriers, no output copy) relative to a dense `torch.mm` with the same M, N
  and K. This is the cost of moving the data inside the GEMM.
- **API vs PyTorch**: speedup of a prepared executable (staging or
  materialization copy and symmetric barriers included, `validate_group=False`)
  over the PyTorch pipeline it replaces: `linear` + destination-major permute +
  NCCL `all_to_all_single` + output copy for the peer-write GEMM, and the
  reverse for the pull GEMM.

Each cell shows the run with the SM clock locked, then the run at the default
clocks. In both settings most runs at hidden 4096 and above reached the board
power limit and ran below the locked or the maximum clock; because the paths
alternate within a run, a ratio still compares them at the same clock.

| hidden | U | peer-write kernel / GEMM | pull kernel / GEMM | peer-write API vs PyTorch | pull API vs PyTorch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 8 | 1.09 / 1.20 | 1.40 / 1.48 | 2.51 / 2.19 | 2.09 / 1.83 |
| 2048 | 4 | 1.01 / 1.04 | 1.34 / 1.37 | 2.70 / 2.46 | 2.19 / 1.96 |
| 2048 | 2 | 1.00 / 1.02 | 1.34 / 1.35 | 2.70 / 2.48 | 2.19 / 2.02 |
| 2304 | 2 | 1.03 / 1.06 | 1.42 / 1.39 | 2.56 / 2.36 | 2.01 / 1.89 |
| 2560 | 4 | 1.01 / 1.05 | 1.26 / 1.28 | 2.53 / 2.34 | 2.11 / 2.00 |
| 2560 | 2 | 1.02 / 1.04 | 1.30 / 1.29 | 2.52 / 2.38 | 2.08 / 1.95 |
| 3072 | 8 | 1.01 / 1.04 | 1.23 / 1.25 | 2.35 / 2.28 | 1.98 / 1.93 |
| 3072 | 4 | 1.01 / 1.03 | 1.23 / 1.24 | 2.38 / 2.33 | 2.01 / 1.94 |
| 3072 | 2 | 1.01 / 1.03 | 1.23 / 1.23 | 2.37 / 2.31 | 2.03 / 1.96 |
| 4096 | 8 | 1.00 / 1.01 | 1.16 / 1.16 | 2.12 / 2.13 | 1.86 / 1.88 |
| 4096 | 4 | 1.00 / 1.00 | 1.16 / 1.16 | 2.14 / 2.13 | 1.89 / 1.88 |
| 4096 | 2 | 1.01 / 1.00 | 1.16 / 1.17 | 2.13 / 2.13 | 1.88 / 1.88 |
| 4864 | 2 | 1.01 / 1.00 | 1.25 / 1.25 | 1.84 / 1.85 | 1.51 / 1.51 |
| 5120 | 8 | 0.99 / 1.00 | 1.19 / 1.19 | 1.93 / 1.93 | 1.64 / 1.60 |
| 5120 | 4 | 1.00 / 1.00 | 1.20 / 1.21 | 1.94 / 1.93 | 1.62 / 1.63 |
| 5120 | 2 | 1.00 / 1.00 | 1.21 / 1.20 | 1.83 / 1.81 | 1.52 / 1.55 |
| 6144 | 8 | 0.95 / 0.95 | 1.15 / 1.14 | 1.86 / 1.84 | 1.56 / 1.55 |
| 6144 | 4 | 0.95 / 0.95 | 1.12 / 1.12 | 1.87 / 1.84 | 1.60 / 1.59 |
| 6144 | 2 | 0.94 / 0.95 | 1.15 / 1.14 | 1.72 / 1.73 | 1.48 / 1.48 |
| 8192 | 8 | 1.00 / 1.00 | 1.15 / 1.16 | 1.64 / 1.63 | 1.42 / 1.42 |
| 8192 | 4 | 1.01 / 1.00 | 1.11 / 1.13 | 1.62 / 1.63 | 1.47 / 1.45 |
| 8192 | 2 | 1.01 / 1.01 | 1.12 / 1.13 | 1.54 / 1.54 | 1.38 / 1.37 |

The peer-write kernel costs what the GEMM alone costs (0.94-1.06x; below 1.0
at hidden 6144, where its GEMM is faster than the cuBLAS kernel `torch.mm`
selects) everywhere except at hidden 2048 / U = 8 (1.09-1.20x). The pull
kernel takes 1.11-1.30x the GEMM from hidden 2560 on and 1.34-1.48x at hidden
2048 and 2304. The exchange is hardest to hide at small hidden and large U:
the bytes a rank exchanges grow with `hidden` while the GEMM work grows with
`hidden^2`, and a larger U sends a larger share, `(U-1)/U`, of the output to
peers. At hidden 2048 / U = 8 hiding it completely would take more peer-write
bandwidth than the kernel sustains (see [Peer-write GEMM](#peer-write-gemm)).

Token count, hidden 5120 / U = 8:

| tokens / rank | peer-write kernel / GEMM | pull kernel / GEMM | peer-write API vs PyTorch | pull API vs PyTorch |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 0.99 / 1.04 | 1.38 / 1.50 | 1.61 / 1.58 | 1.18 / 1.09 |
| 16384 | 1.01 / 1.03 | 1.34 / 1.35 | 1.79 / 1.73 | 1.40 / 1.30 |
| 32768 | 1.00 / 1.01 | 1.24 / 1.25 | 1.89 / 1.89 | 1.56 / 1.53 |
| 65536 | 0.99 / 1.00 | 1.19 / 1.19 | 1.93 / 1.93 | 1.64 / 1.60 |
| 131072 | 1.00 / 1.01 | 1.17 / 1.18 | 1.96 / 1.96 | 1.68 / 1.69 |

The pull kernel's overhead grows as M shrinks, and the API speedups fall
because each call's barriers and host-side work do not shrink with M (the
benchmark synchronizes after every sample, so the host work before a call's
first launch is part of the sample, for the PyTorch pipeline as well).

Bias, 65,536 tokens per rank (the PyTorch pipeline adds the bias with
`addmm`):

| hidden | U | peer-write kernel time, bias / no bias | pull kernel time, bias / no bias | peer-write API vs PyTorch | pull API vs PyTorch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3072 | 8 | 1.00 / 1.01 | 1.00 / 1.00 | 2.35 / 2.26 | 2.00 / 1.92 |
| 5120 | 8 | 1.00 / 1.00 | 1.00 / 1.01 | 1.95 / 1.93 | 1.62 / 1.63 |
| 5120 | 4 | 1.00 / 1.00 | 1.00 / 0.99 | 1.95 / 1.92 | 1.62 / 1.61 |
| 5120 | 2 | 1.01 / 1.00 | 1.00 / 0.99 | 1.83 / 1.82 | 1.53 / 1.54 |

The fused bias costs nothing measurable inside either kernel.

### Rectangular weights

The exchanged bytes per GEMM FLOP scale with `1/K` for a peer-write and with
`1/N` for a pull, so a rectangular projection costs about what a square one
with the same `K` (peer-write) or the same `N` (pull) costs. Measured on
another node of the same type at the default clocks, with the square shapes
in the same session for reference (65,536 tokens per rank, packed THD):

| Peer-write `W` | pull `W` | U | peer-write kernel / GEMM | pull kernel / GEMM | peer-write API vs PyTorch | pull API vs PyTorch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6144x4096 | 4096x4096 | 8 | 0.92 | 1.14 | 2.15 | 1.82 |
| 15360x5120 | 5120x5120 | 8 | 1.00 | 1.18 | 1.96 | 1.58 |
| 12288x3072 | 3072x4096 | 4 | 1.01 | 1.20 | 2.32 | 1.97 |
| 5120x2880 | 3072x4096 | 8 | 1.02 | 1.25 | 2.27 | 1.81 |
| 1024x4096 | 4096x1024 | 8 | 0.99 | 1.32 | 1.62 | 1.23 |
| 8192x2048 | 2048x8192 | 4 | 1.21 | 1.26 | 2.43 | 2.35 |
| 8192x2048 | 2048x8192 | 8 | 1.23 | 1.30 | 2.22 | 2.14 |
| 5120x5120 | 5120x5120 | 8 | 0.99 | 1.18 | 1.90 | 1.60 |
| 4096x4096 | 4096x4096 | 8 | 1.01 | 1.14 | 2.05 | 1.86 |
| 3072x3072 | 3072x3072 | 4 | 1.02 | 1.23 | 2.23 | 1.93 |
| 2048x2048 | 2048x2048 | 8 | 1.15 | 1.43 | 2.07 | 1.71 |
| 2048x2048 | 2048x2048 | 4 | 1.01 | 1.34 | 2.31 | 1.82 |

From `K = 2880` on, the peer-write runs at the GEMM's speed for fused QKV and
narrow K/V projections alike (the 6144x4096 kernel is faster than the cuBLAS
kernel `torch.mm` selects for that shape). At `K = 2048` it is limited by the
peer-write bandwidth, as the square 2048 / U = 8 case is: the 8192x2048
projection writes four times the bytes of the square 2048 projection, while
its GEMM takes only 3.7 times as long. The pulls with `N` from 2048 to 5120
take 1.14-1.30x the GEMM, the 2048x8192 pull less than the square 2048 pull.
A pull with a narrow input (`K = 1024` at U = 8, 128-column shards, for
example the data gradient of a grouped-query K/V projection) adds 0.12 ms,
about what the square 2048 pulls add, to a GEMM of only 0.36 ms; 32-row pull
tickets did not shorten it.

### End to end

In an end-to-end study with the diffusers implementations of Wan2.2-A14B and
Cosmos-Predict2.5-14B (hidden 5120 = 40 heads x 128, U = 8, 16 equal-length
480p or 8 720p videos packed per step, 65K to 85K rows per GPU, SM clocks
pinned), an earlier revision of these kernels replaced the Q, K and V
projections (peer-write) and the output projection (pull) together with their
Ulysses exchange. One denoising step became 5.9-7.8% faster on Cosmos and
5.2-6.9% faster on Wan2.2, and one training step (forward and backward with
activation checkpointing) 4.2-6.1% and 3.7-5.6% faster. Removing the exchange
altogether would gain 5-10%, so the kernels realized 83-88% (Cosmos) and
69-73% (Wan2.2) of that bound. Cosmos's outputs were bit-identical to the
conventional path. Most of Wan2.2's remaining gap was work outside the
kernels: its projection biases and its QK RMSNorm over all heads were applied
to the head shards separately (about 1% of the step). The bias is now fused;
the cross-head normalization still needs a per-token statistic from every
rank. At the default clocks the steps gained 4.2-6.0% and 3.1-3.7%
(denoising) and 2.5-4.5% and 2.5-3.3% (training): under the power limit the
GPUs throttled by different amounts, and every exchange waited for the
slowest GPU.

## Limitations

- Widths: the exchanged width follows the shape rule above, and the pull
  output width must be a multiple of 256 (a 2880-wide output projection is not
  supported; a 2880-wide peer-write input is).
- A pull with a narrow input pays a roughly fixed cost on a short GEMM: at
  `K = 512` (4096x512, U = 4) the kernel took 1.56x the GEMM and the API was no
  faster than the PyTorch pipeline.
- BF16 only; no FP8.
- One node: U in {2, 4, 8} over NVLink.
- Forward-only host APIs; no autograd integration.
- A normalization across all heads between the projection and attention (for
  example Wan's QK RMSNorm over the full hidden dimension) needs a per-token
  statistic from every rank and is not fused.

## Source layout

```text
mega_up/
  kernels/
    peer_write_gemm.py       peer-write GEMM (QuACK-derived, Apache-2.0)
    headshard_pull_gemm.py   head-shard pull GEMM
  ops.py                     BSHD / SBHD / THD host APIs
  runtime.py                 symmetric arena, barriers, shape rule
benchmarks/
  bench_bshd.py
tests/
  test_layout.py
  test_shapes.py
  test_distributed.py
  run_distributed_correctness.py
  run_pull_gemm_contention.py
```

## License

Project-authored code is MIT-licensed. `peer_write_gemm.py` retains its
Apache-2.0 QuACK lineage and notices. See `LICENSE`, `NOTICE`, and
`THIRD_PARTY_LICENSES.md`.
