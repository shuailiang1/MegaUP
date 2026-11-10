# Third-party licenses

Mega-UP contains modified code with the following open-source lineage.
Files identified as Apache-2.0 below remain under Apache-2.0; the other
project-authored files are MIT-licensed.

## QuACK 0.4.0

- Project: https://github.com/Dao-AILab/quack
- Release: https://pypi.org/project/quack-kernels/0.4.0/
- License: Apache-2.0
- Copyright: 2025-2026, QuACK team
- Local file: `mega_up/kernels/peer_write_gemm.py`

The local file retains the GEMM structure and selected helpers from
QuACK 0.4.0. It removes unused dtype and epilogue paths, carries compatibility
code for CUTLASS DSL 4.4.2, and adds dynamic tile scheduling and peer-routed
stores.
The Apache-2.0 text is in `licenses/Apache-2.0.txt`.

## NVIDIA CUTLASS CuTe DSL examples

- Project: https://github.com/NVIDIA/cutlass
- License: BSD-3-Clause
- Copyright: 2025-2026 NVIDIA CORPORATION & AFFILIATES

QuACK's GEMM is based on the persistent dense GEMM example of CUTLASS
CuTe DSL. The corresponding license text is in
`licenses/NVIDIA-CUTLASS-BSD-3-Clause.txt`.

## Head-shard pull kernel provenance

`mega_up/kernels/headshard_pull_gemm.py` is project-authored code
released under the repository's MIT license. It uses NVIDIA CUTLASS
CuTe DSL APIs as a runtime dependency.

## Runtime dependencies not bundled here

Mega-UP imports, but does not vendor:

- PyTorch (`torch`)
- NVIDIA CUTLASS DSL (`nvidia-cutlass-dsl`)
- Apache TVM FFI (`apache-tvm-ffi`), used by the CUTLASS DSL export of the
  peer-write kernel
- CUDA Python (`cuda-python`)

Those packages remain subject to their own distribution terms.
