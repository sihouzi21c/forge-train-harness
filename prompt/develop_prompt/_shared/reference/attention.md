# Attention Operator Optimization Reference (used by Scout when building PROMPT.md)

> This file is referenced by the Scout Agent when generating PROMPT.md for the attention operator.
> Integrate the relevant content into the attention PROMPT.md.
>
> Copying open-source implementations is not allowed; you must write the code yourself.

## Stage 1 → Stage 2 baseline state

When entering Stage 2, the attention site has already achieved bitwise alignment in Stage 1 against the active ref stack on the deterministic path (under the torch backend: `ref/reference/train_pure_mup_mtp.py` → `model_pure_mup_mtp.py:TransformerLayer.forward`; under the megatron backend: the equivalent Megatron-LM + TE fused-attention path). Specific characteristics:


| Dimension     | Value / implementation                                                                                                  |
| ------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `is_causal`   | True                                                                                                                    |
| Model shape   | NUM_HEADS=16, NUM_KV_HEADS=2 (GQA, group=8), HEAD_DIM per active [model] axis (0.5B=64, MiniCPM5-1B=128 — read FORGE_HEAD_DIM, do NOT assume hidden//heads), MAX_SEQ_LEN=4096 |
| dtype         | bf16 input / bf16 output / fp32 accumulation                                                                            |
| MTP branch    | Inside Eagle 1, the same `TransformerLayer.forward` is taken, calling the same attention kernel again (identical shape; only the hidden input from `eagle_fc` differs) |


**Stage 2 optimization goal**: while keeping the above interface and mathematical behavior unchanged, replace the attention implementation with a **high-MFU FlashAttention on H100** (CuTeDSL / CUTLASS / hand-written CUDA), and pass the `op-long` operator-level hard gate (DP=2 200 steps vs ref-script trajectory mean rel diff ≤ 1%); the local `test_op.py` uses an FP64 ground truth + bf16 ref wrapper for accuracy validation.

MFU target: Forward at least 45%, Backward at least 40% (causal attention computation method). If the MFU you compute is significantly higher, you should reflect and investigate yourself.
Using open-source projects is not allowed; you must write the operator yourself.

---

## FlashAttention core idea

The core of FlashAttention is **tiling + online softmax + not materializing the S/P matrix**:

1. **Block tiling**: split Q into blocks by BLOCK_M, and K/V into blocks by BLOCK_N. Each thread block
  handles the interaction between one Q block and all K/V blocks.
2. **Online softmax**: incrementally update the softmax max and sum while iterating over the K blocks,
  avoiding computing the full S matrix before applying softmax.
3. **Don't materialize S/P**: the result of Q*K^T is multiplied with V directly in SRAM, without being written back to HBM.
  Memory complexity drops from O(N²) to O(N).
4. **Recomputation**: during backward, the S/P matrices are not stored but recomputed.
  Trading compute for memory is a net gain in memory-bound scenarios.

## Key points for attention optimization on H100

### Tensor Core utilization

- Q*K^T and Score*V are both matrix multiplications, **they must go through Tensor Core**
- In CuTe DSL / CUDA C++: use SM90 WGMMA (SS mode) to handle the Q*K^T and Score*V matmuls
- Typical tile: BLOCK_M=64, BLOCK_N=128, BLOCK_HEADDIM=64/128

### Producer / Consumer model and Warp Specialization

The compute ceiling for attention forward on H100 comes from **thorough overlap between TMA memory access and WGMMA compute**.
A single-warp serial model that performs "fetch K/V → compute QK → softmax → compute PV" wastes a large amount of compute on SM90.
The recommended paradigm on SM90 is **warp specialization**: split warps within a CTA by function into
**producer** (DMA / TMA) and **consumer** (WGMMA + softmax) groups, synchronized via mbarrier
producer-consumer, decoupling issue and compute at the hardware level.

**Basic split of duties** (typical 2 WG = 256 threads configuration):


| Role           | warps    | Responsibility                                                                | setmaxnreg               |
| -------------- | -------- | ----------------------------------------------------------------------------- | ------------------------ |
| Producer (WG0) | warp 0-3 | Issue TMA loads for K / V, commit mbarrier per stage                          | dealloc (~24 regs/thread) |
| Consumer (WG1) | warp 4-7 | `consumer_wait` for TMA completion → WGMMA QK → online softmax → WGMMA PV → TMA store O | alloc (~200 regs/thread) |


### GQA (Grouped Query Attention) adaptation

When the model uses GQA (number of Q heads > number of KV heads):

- Multiple Q heads share the same group of K/V heads
- The kernel's grid dimension and head indexing must be mapped correctly
- The GQA structure can be exploited: K/V memory access across multiple Q heads can be reused

**GQA native implementation**: do not perform `repeat_interleave` inside the kernel; instead, index the KV head directly via compile-time `FastDivmod`:

```python
head_idx_kv = head_idx // qhead_per_kvhead_divmod
mK_cur = mK[None, None, head_idx_kv, batch_idx]
mV_cur = mV[None, None, head_idx_kv, batch_idx]
```

The grid is launched along the Q head dimension (`head_idx ∈ [0, H_q)`); multiple Q heads share the same group of K/V tiles.

---

## fwd prompt supplement

### Main tuning dimensions


| Parameter           | Allowed range | Description                       |
| ------------------- | ------------- | --------------------------------- |
| `n_block_size`      | 64, 128, 256  | KV tile width                     |
| `num_stages`        | 2, 3, 4       | K/V pipeline depth                |
| `num_threads`       | 256, 384      | 2 WG or 3 WG                      |
| `num_producer_regs` | 24–40         | Producer register budget          |
| `num_mma_regs`      | 160–240       | Consumer register budget          |
| QK mode             | SS            | WGMMA operand mode                |
| PV mode             | SS, RS        | RS = P from registers             |
| Softmax             | online        | Fused with QK loop (log2e scaling) |
| `body_unroll`       | 1, 2          | Loop unroll factor (usually 1 for causal) |


**Occupancy trade-off**: registers and shared memory jointly limit CTAs/SM. Increasing `num_stages` or growing the tile
will increase smem usage, possibly degrading from 2 CTAs/SM to 1 CTAs/SM; increasing `num_mma_regs` increases
register usage and may likewise reduce occupancy. Balancing is required under the H100 228KB smem / 65536 regs/SM
capacity constraint.

### Kernel data flow

```
Grid: (m_block, head_idx_q, batch)    // head_idx_q ∈ [0, H_q)
Each CTA owns one Q tile [m_block × D], iterates over K/V tiles

    Q[m_block] ──TMA──→ smem_Q (one-shot load)
    head_idx_kv = head_idx_q // qhead_per_kvhead   // GQA mapping

    for n = n_block_max-1 .. 0:     // reverse for causal
        K[n, head_idx_kv] ──TMA pipeline──→ smem_K[stage]
        V[n, head_idx_kv] ──TMA pipeline──→ smem_V[stage]

        GEMM_QK:  S = Q @ K^T          [m_block × n_block]   (SS WGMMA)
        causal_mask(S)                   // only on diagonal tile
        online_softmax(S, running_max, running_sum)
        P = softmax(S)                   // FP32 → FP16 conversion
        GEMM_PV:  O += P @ V           [m_block × D]         (SS WGMMA)
        rescale_O(O, row_scale)

    finalize_softmax(O)
    O ──stmatrix──→ smem_O ──TMA store──→ gmem
    LSE ──store──→ gmem
```

### Warp Specialization (2 WG, 256 threads)

```
┌──────────────────────────────────────┐
│ WG0 (warps 0-3): Producer           │  setmaxnreg(24)
│   warp0: TMA issue (K, V pipeline)  │
│   warp1-3: idle / scheduler         │
├──────────────────────────────────────┤
│ WG1 (warps 4-7): Consumer           │  setmaxnreg(200)
│   WGMMA: QK^T, PV                   │
│   online softmax (in registers)     │
│   TMA store O                       │
└──────────────────────────────────────┘
```

For D=192/256, 3 WG (1 producer + 2 consumer) is feasible, with each consumer WG handling a split of PV.

### Pipeline design

- **K pipeline**: `PipelineTmaAsync<num_stages>` (currently 3 stages)
- **V pipeline**: independent pipeline, also num_stages
- Producer issues `tma_load` with barrier; consumer calls `consumer_wait` / `consumer_release`
- Q is loaded once, using an independent `ClusterTransactionBarrier`

### Online Softmax (log2e)

Use `log2(e)` scaling, entirely in registers:

```
m_new = max(m_old, rowmax(S * scale_log2))
P = exp2(S * scale_log2 - m_new)
l_new = exp2(m_old - m_new) * l_old + rowsum(P)
O_new = exp2(m_old - m_new) * O_old + P @ V
```

After the loop: `O = O / l_final`, `LSE = m_final / log2(e) + log(l_final)`.

```python
row_max_cur = warp_reduce(fmax_reduce(acc_S_row), fmax, width=4)
acc_S_row_exp = exp2f(acc_S_row * scale_log2 - row_max_cur_scaled)
row_scale = exp2f((row_max_prev - row_max_cur) * scale_log2)
row_sum = fadd_reduce(acc_S_row_exp, init_val=row_sum * row_scale)
```

### Body Loop scheduling

```
for n_tile in range(n_block_max - n_block_min):
    # 1. Issue QK GEMM (wg_wait=-1, no wait)
    consumer_wait(K[stage])
    WGMMA QK: acc_S = Q @ K^T

    # 2. Issue PV GEMM (wg_wait=-1, using the previous tOrP)
    consumer_wait(V[stage])
    WGMMA PV: acc_O += tOrP @ V^T

    # 3. Wait QK done, do softmax
    wg_wait(1)              // wait for QK to finish
    consumer_release(K)
    row_scale = online_softmax(acc_S)

    # 4. Wait PV done, prepare next P
    wg_wait(0)              // wait for PV to finish
    consumer_release(V)
    tOrP = cvt_f16(acc_S)   // FP32 → FP16

    # 5. Rescale O
    rescale_O(acc_O, row_scale)
```

**Key**: softmax (FMA pipe) and PV GEMM (Tensor pipe) execute in parallel between steps 3 and 4.

### Performance ceiling analysis

At small head dim (e.g. D=64), QK and PV must be serialized (PV depends on softmax(QK)), and the FMA
serialization overhead of softmax is significant, so Tensor Core utilization will be pinned at a relatively low level —
this is the structural ceiling for FlashAttention forward at small D. Increasing D (128/192/256) better amortizes
softmax overhead, and Tensor Core utilization usually rises.

To gauge whether the ceiling has been reached, look at NCU's `tensor_active` + `fma_active`: their sum approaching
100% means the compute side is saturated, and the next step should be to reduce softmax serialization or move to
an attention variant with larger D.

### Architectural-level optimizations to explore


| Direction                  | Complexity | Description                                                  |
| -------------------------- | --- | ------------------------------------------------------------ |
| 3 WG (1 prod + 2 consumer) | High | More warps to hide latency; with small D each consumer WG only computes half an m tile |
| Increase m_block           | Medium | Larger M tile, more efficient GEMM, but register/smem usage rises |
| RS mode for PV             | High | P is fed directly from registers as the WGMMA A-operand, saving smem bandwidth |
| Softmax warp_reduce latency optimization | Low | Push row_sum's warp_reduce as late as possible to the finalize stage |
| Merge prologue into body   | Medium | Eliminate the overhead of a standalone prologue QK |


### CuTe DSL SM90 mode reference

**Warp Specialization**:

```python
warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
if warp_idx < 4:
    cute.arch.warpgroup_reg_dealloc(24)   # Producer: release registers
else:
    cute.arch.warpgroup_reg_alloc(200)    # Consumer: acquire registers
```

**Note**: `setmaxnreg` must be called immediately after the branch; no other instructions are allowed in between.

**TMA Pipeline**:

```python
# Producer
pipeline_k.producer_acquire(state)
cute.copy(tma_atom_K, src, dst, tma_bar_ptr=pipeline_k.producer_get_barrier(state))
state.advance()

# Consumer
pipeline_k.consumer_wait(state, pipeline_k.consumer_try_wait(state))
pipeline_k.consumer_release(state)
state.advance()
```

**wg_wait semantics**:


| Value         | Behavior                              | Use case                                       |
| ------------- | ------------------------------------- | ---------------------------------------------- |
| `wg_wait(-1)` | No wait, continue immediately         | Pipeline next GEMM                             |
| `wg_wait(0)`  | Wait for all outstanding MMA groups to finish | Before reading the accumulator         |
| `wg_wait(1)`  | Wait until ≤1 outstanding             | Previous GEMM done while the current one is still running |


**FP32→FP16 Conversion**:

```python
tOrP_acc = cute.make_tensor(acc_S.iterator, convert_layout_acc_frgA(acc_S.layout))
cvt_f16(tOrP_acc, tOrP)
```

### RS Mode for PV GEMM (advanced optimization reference)

When `SdP_swapAB=true`, the accumulator register layout of the QK GEMM exactly matches the A-operand layout of the PV MMA. P can be fed directly from registers as the WGMMA A-operand, saving the P→smem→WGMMA round-trip.

```python
tiled_mma_pv = make_trivial_tiled_mma(
    dtype, dtype, OperandMajorMode.K, OperandMajorMode.MN, Float32,
    tiler_mn=(m_block, head_dim),
)
rP = cute.make_fragment_like(acc_S, dtype)
rP.store(acc_S.load().to(dtype))
tdVrP = cute.make_tensor(rP.data(), tiled_mma_pv.layout_A_TV())
sm90_utils.gemm(tiled_mma_pv, acc_O, tdVrP, tOrVt, zero_init=False, wg_wait=-1)
```

**Prerequisite**: requires SdP_swapAB=true, which changes the operand order of the QK GEMM to K@Q^T.

### Shared Memory Layout reference

```
smem_Q:  [m_block, D]                    // one-shot load
smem_K:  [n_block, D, num_stages]        // pipeline, replicated by num_stages
smem_V:  [n_block, D, num_stages]        // pipeline, replicated by num_stages
smem_O:  [m_block, D]                    // epilogue stage, can share space with V
```

Total usage ≈ `bytes(Q) + num_stages × (bytes(K) + bytes(V))`. This must stay within the H100 228KB / SM
budget and leave headroom for the target occupancy (with 2 CTAs/SM, the per-CTA cap is ~114KB).
