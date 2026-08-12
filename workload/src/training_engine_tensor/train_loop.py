"""Self-developed training engine — public CLI entry contract.

This module is the **single source of truth** for the ours-side training
entry point.  Every harness gate — M1 single-step tensor-capture gates
(``forward-align`` / ``backward-align``), M2 / M3 / M4 bitwise-trajectory
runs, M5 resume gate, M6 long-train, and the stage-2 op-long runner —
invokes :func:`run_training_loop` as a black-box subroutine and either
parses its ``[<loss_tag>]`` stdout lines (M2+) or loads the tensor
dump it writes when ``capture_output_file`` is set (M1).  No gate script may
build its own dataloader, model, optimizer, or training loop — that
orchestration belongs entirely to this module.

Architectural contract
----------------------

From milestone M1 onward the self-developed training engine owns the
full training stack, including the dataloader.  The harness layer
therefore treats :func:`run_training_loop` as the **only** ours-side
entry, mirroring how the L0 ref script (``ref/reference/${ref_script}``)
is the only reference-side entry — regardless of which ref stack is
selected (Megatron sibling, pure-torch sibling, …).  Splitting these
two SSOTs anywhere else (in ``evals/_common.py``, in a gate script,
in a test fixture) is a boundary violation that the framework guard
rejects.

The agent-loop implementation lives in this same package (``backward``,
``forward``, ``optimizer``, ``parameters``, ``nccl``, ``kernels``,
``triton_kernels``, ``dataloader``, etc.).  This file ties them
together behind a stable signature so the harness scripts never have
to touch the internals.  Replacing the body with the real loop is a
strict local refactor: signatures, stdout grammar, and exit codes
must remain identical to keep the dispatcher parsers stable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainLoopConfig:
    """Immutable configuration for a single ``run_training_loop`` invocation.

    All fields are required unless documented otherwise.  The values
    flow from ``config/eval.toml`` through the dispatcher into env
    variables, and from there into the gate scripts that build this
    config; see ``[evals.<name>.ref_env]`` for the per-gate authoritative
    shape (NUM_STEPS_OVERRIDE / MICRO_BATCH_SIZE_OVERRIDE / …).

    Resume semantics
    ~~~~~~~~~~~~~~~~

    The resume gate runs :func:`run_training_loop` three times:

    * ``Phase R`` — uninterrupted reference run; *save_path* /
      *resume_from* are both ``None`` and *start_step* is zero.
    * ``Phase A`` — initial leg; *save_path* is set so the engine
      writes a full checkpoint AFTER the final step in this invocation
      (i.e. at absolute step ``start_step + num_steps``).  ``num_steps``
      is the number of steps to train before saving.
    * ``Phase B`` — continuation leg; *resume_from* is set and
      ``start_step`` is the step counter the save was taken at, so
      the emitted ``[<loss_tag>]`` lines carry the original global
      step number.

    Implementations MUST honour ``start_step``: the per-step lines
    must read ``step=<start_step + i>`` for ``i ∈ [0, num_steps)``,
    not zero-based; the gate dispatcher diffs trajectories on the
    absolute step number.  When *resume_from* is set the dataloader
    must be advanced to ``start_step`` (Megatron's
    ``consumed_train_samples`` handles this when the saved state
    includes ``num_samples``).

    M1 capture semantics
    ~~~~~~~~~~~~~~~~~~~~

    The M1 forward-align / backward-align gates run :func:`run_training_loop`
    with ``capture_output_file`` set to an **absolute file path** chosen
    by the dispatcher.  The candidate engine never picks the basename
    itself — the dispatcher computes
    ``<artifact_dir>/<[defaults].candidate_capture_basename>`` from
    ``config/eval.toml`` and passes that complete path in.  This
    keeps the on-disk filename a single dispatcher-side SSOT and lets
    the suite-runtime move the dump under any artifact root without
    touching engine code.

    In capture mode the engine MUST:

    * run exactly one forward + backward pass (``num_steps`` is pinned
      to 1 by the dispatcher; grad-accum / global batch shape are pinned
      to ``WORLD_SIZE * MICRO_BATCH_SIZE``);
    * collect per-module forward activations and per-parameter gradients
      into a single ``dict[str, torch.Tensor]`` keyed with
      ``fwd.<module_fqn>`` and ``grad.<param_fqn>`` respectively, where
      both FQN namespaces match what the reference side's
      ``named_modules()`` / ``named_parameters()`` publish — same wire
      format as :func:`evals.harness_hook.install` (the standard hook
      the agent-generated M1 bridge loads on the reference side);
    * write the dict to ``capture_output_file`` AFTER backward completes
      but BEFORE the optimizer step (the optimizer step is otherwise a
      no-op for capture mode);
    * optionally write a sibling ``<capture_output_file>.graph.json``
      mirroring the reference's per-module execution order
      (``[{order, fqn, class, input_shapes, input_dtypes, output_shape,
      output_dtype}, …]``) so the dispatcher can diff structural
      drift when tensor keys diverge;
    * return normally without emitting per-step ``[<loss_tag>]`` lines
      (the dispatcher only consumes the dump file in capture mode).

    The dispatcher's diff loop intersects on keys present on both
    sides and PASSes iff every common key is bitwise identical, so the
    engine controls the comparison surface by choosing how exhaustively
    it mirrors the reference side's FQN namespace.  No rename table on
    either side — the baseline's natural FQN is the single source of truth.
    """

    num_steps: int
    micro_batch_size: int
    seq_length: int
    grad_accum_steps: int
    seed: int
    world_size: int

    checkpoint_root: str
    # data_path is required for both backends. The harness path-shim
    # (build_suite_env) auto-derives it from [data].conf_path / DATA_CONF
    # when only the conf is set, so the candidate process always sees a
    # populated DATA_PATH regardless of which ref stack is selected.
    data_path: str

    start_step: int = 0
    save_path: str | None = None
    resume_from: str | None = None
    capture_output_file: str | None = None

    # ── Hash-record capture (M1–M5)
    #
    # ``hash_capture_level`` is the per-suite TOML knob threaded from
    # ``[evals.<suite>].hash_capture_level`` through the dispatcher's
    # CLI-arg passthrough (--hash-capture-level <N>). See
    # ``evals/harness_hook/__init__.py`` for the level semantics:
    #   0 = no hash capture (today's stdout-only behaviour);
    #   1 = loss family + grad family;
    #   2 = level 1 + per-module fwd / bwd activations.
    #
    # ``hash_output`` is the absolute path the engine writes the JSON
    # hash dump to when ``hash_capture_level > 0`` (--hash-output <path>
    # from the dispatcher; SSOT for the on-disk wire format).
    #
    # ``persistent`` switches the harness_hook lifecycle:
    #   False = M1 single-step dump-and-exit (today's M1.fwd / M1.bwd);
    #   True  = M2 / M3 / M4 / M5 multi-step persistent capture (the
    #           engine must drive ``CaptureSession.begin_step`` /
    #           ``capture`` / ``capture_grads`` / ``dump`` at the hook
    #           points; see harness_hook docstring + recipes/README.md).
    #
    # M5 resume-gate convention: ``hash_output`` is a base path; the
    # engine writes two dumps as ``<hash_output>.ref.json`` (Phase R
    # un-resumed trajectory) and ``<hash_output>.res.json`` (Phase
    # A+B resumed trajectory) so the dispatcher can ours-vs-ours diff.
    hash_capture_level: int = 0
    hash_output: str | None = None
    persistent: bool = False

    # ── Backend selector and backend-specific paths
    #
    # ``backend`` is the SSOT for which L0 ref stack the gate is paired
    # with (mirrors ``[ref].backend`` in config/ref.toml). Allowed
    # values are listed in ``config_runtime.BACKENDS``:
    #
    #   * ``"megatron"`` — Megatron-LM ref + ours-side Megatron-native
    #     pipeline (BlendedMegatronDatasetBuilder + GPTDataset);
    #     ``megatron_root`` MUST be non-empty.
    #   * ``"torch"``    — pure-PyTorch ref (train_pure_mup_mtp.py +
    #     muP + MTP) + ours-side framework-native pipeline (no
    #     Megatron); ``megatron_root`` MUST be empty.
    #
    # ``run_training_loop`` branches on ``backend`` to pick the
    # dataloader / process-group bootstrap path.
    backend: str = "megatron"
    megatron_root: str = ""

    # ── WSD-SFT 3-phase schedule (resume-milestone wsd-sft gate + the
    #    single-job production line)
    #
    # These make the per-phase learning-rate schedule and the SFT
    # weights-only handoff first-class on the config, so the 3-phase
    # driver (evals/scripts/train_ours_al.sh → train_ours_phase.py) can
    # drive stable → decay → sft as three ``run_training_loop``
    # invocations that differ ONLY in this block plus ``resume_from`` /
    # ``save_path`` / ``start_step`` / ``data_path`` — the same
    # ``dataclasses.replace`` idiom the resume gate already uses.
    # Defaults reproduce today's behaviour (no separate WSD anneal
    # window), so existing construction sites that never set them are
    # unchanged.
    #
    #   * ``lr`` / ``min_lr`` — peak and floor of the phase's schedule.
    #   * ``lr_warmup_iters`` — linear warmup length; lr ramps ``min_lr`` →
    #     ``lr`` over absolute steps ``[0, lr_warmup_iters)`` (warmup ramps
    #     from ``min_lr``, NOT 0 — matching the ref's ``compute_lr``).
    #   * ``lr_decay_iters`` — the absolute step at/after which lr is pinned
    #     to ``min_lr``; also the right anchor of the WSD anneal window,
    #     which spans absolute
    #     ``[lr_decay_iters - lr_wsd_decay_iters, lr_decay_iters]``.
    #     For a phase that only "holds the peak" (stable), set this far past
    #     the phase's last step so the floor is never reached.
    #   * ``lr_wsd_decay_iters`` — length of the trailing WSD anneal window
    #     (``lr`` → ``min_lr``, exponential curve); ``0`` means "hold the
    #     peak" (stable phase).
    #
    # The schedule is evaluated at the ABSOLUTE global step — the same
    # integer emitted in the ``[LOSS] step=`` line — NOT a per-phase local
    # index. So a decay leg that ``resume_from``s the stable run at global
    # step S sees warmup as already past and picks the anneal up at its
    # absolute position. Concretely lr / min_lr / lr_warmup_iters /
    # lr_decay_iters / lr_wsd_decay_iters are the per-phase schedule knobs
    # the 3-phase driver exports (LR / MIN_LR / LR_WARMUP_ITERS /
    # LR_DECAY_ITERS / LR_WSD_DECAY_ITERS). Defaults below are inert
    # placeholders (the 3-phase entry always sets all five explicitly);
    # the legacy single-leg gates never consume them.
    #
    # ``init_weights_only`` is the SFT handoff and is DISTINCT from a plain
    # ``resume_from``:
    #   * ``resume_from`` alone  → full resume (params + optimizer moments +
    #     step counter + dataloader position); this is the decay phase
    #     continuing the stable run.
    #   * ``resume_from`` + ``init_weights_only=True`` → load ONLY the model
    #     weights from that directory, then start a fresh run: optimizer
    #     state re-initialised, step counter reset to ``start_step`` (0),
    #     dataloader rebuilt from the phase's own ``data_path`` (the SFT
    #     corpus, whose stream carries the loss_mask). This is the sft
    #     phase taking the decay weights as its init point.
    lr: float = 1e-2
    min_lr: float = 0.0
    lr_warmup_iters: int = 0
    lr_decay_iters: int = 0
    lr_wsd_decay_iters: int = 0
    init_weights_only: bool = False
    # ``save_interval`` — periodic versioned-checkpoint cadence in ABSOLUTE
    # steps. >0 writes a full resumable state to ``<save_path>/step_<abs>/``
    # every interval (production crash-resume; the bash driver's
    # latest_ckpt_step() rediscovers the newest step_<abs>/ to restart a
    # crashed phase) AND one final ckpt (also under step_<phase_end>/) at
    # the phase end. 0 (gate/bitwise) = save only at run end, directly
    # under ``save_path`` (today's behaviour).
    save_interval: int = 0
    # ``phase_name`` — WSD-SFT phase label (stable/decay/sft) for the
    # machine ``[PHASE] name=…`` banner (see "Stdout grammar" below). Set
    # explicitly by the phase driver so a crash-restarted phase does not
    # mis-infer its identity from resume_from. None (single-phase /
    # bitwise) → no banner emitted.
    phase_name: str | None = None

    @property
    def global_batch_size(self) -> int:
        return self.grad_accum_steps * self.micro_batch_size * self.world_size


def run_training_loop(config: TrainLoopConfig, *, loss_tag: str = "LOSS") -> None:
    """Run ``num_steps`` of training and emit one line per global step.

    The agent-loop implementation must satisfy every clause below; the
    harness dispatcher and gate scripts depend on each of them.

    Behaviour
    ~~~~~~~~~

    1. **Process group**: the caller has already launched the process
       under ``torchrun`` (or the equivalent in
       ``evals/scripts/launch_dp.py``).  ``train_loop`` is responsible
       for initialising whatever process-group / NCCL / framework-
       specific state it needs (``initialize_megatron`` + ``mpu`` when
       ``config.backend == "megatron"``; bare ``torch.distributed``
       when ``config.backend == "torch"``) and for tearing it down
       before returning.
    2. **Dataloader**: build a dataloader that reads the same data as
       the L0 ref script. ``config.backend`` selects the pipeline:

       * ``"megatron"`` — build a Megatron ``GPTDataset`` over
         ``config.data_path`` (either an HF gsm8k binary prefix
         produced by ``ref/reference/prepare_gsm8k_data.sh``, or a
         modelbest_sdk weighted-shard string the harness path-shim
         already resolved from ``[data].conf_path`` — see the matching
         ref script's ``--data-path`` contract).
         ``config.megatron_root`` MUST be non-empty so that
         ``initialize_megatron`` + ``mpu`` find the source tree.
       * ``"torch"`` — build a framework-native dataloader that
         consumes the same weighted-shard list the pure-torch ref's
         ``train_pure_mup_mtp.py`` reads (see how
         ``run_16gpu_1000step_pure_mup_mtp.sh`` materializes
         ``$DATA_CONF`` into ``--data-path-file``). ``config.data_path``
         is the same shard string the harness path-shim already
         resolved from ``[data].conf_path``. ``config.megatron_root``
         is empty.

       Either way, both sides feed the same physical data, which is
       what keeps trajectories aligned.  When ``config.resume_from``
       is set the dataloader must be rebuilt with the same seed so
       the resumed leg sees the same micro-batch sequence as the
       reference leg.

       NOTE: bare ``torch.utils.data`` is on
       ``framework_guard.TORCH_BANNED_SUBMODULES`` — the engine must
       rely on a framework-native pipeline (Megatron's
       ``BlendedMegatronDatasetBuilder`` + ``pretrain_gpt.get_batch``,
       or modelbest_sdk's ``ModelbestDataloader``) and not wrap a
       vanilla ``DataLoader`` itself.
    3. **Weights**: load the canonical FP32 checkpoint from
       ``config.checkpoint_root``.  When ``config.resume_from`` is set,
       load the full state (params + optimizer moments + step counter
       + num_samples) from that directory instead and seek the
       dataloader forward to ``config.start_step``.  When
       ``config.init_weights_only`` is ALSO set, load only the model
       weights from ``resume_from`` (fresh optimizer, step counter =
       ``start_step``, dataloader built fresh from ``config.data_path``)
       — the SFT weights-only handoff.  ``resume_from`` may be either a
       phase save root holding a ``*training_state.pt`` or a versioned
       ``step_<abs>/`` directory; glob ``*training_state.pt`` inside it.
       Data-state note: when ``resume_from`` is set but ``data_path``
       differs from the corpus recorded in the checkpoint, the engine
       must NOT replay the saved dataloader cursor — it restores RNG +
       optimizer + step and builds the new corpus's stream fresh (the
       ref's ``--no-load-data-state`` semantics, inferred from the
       corpus switch rather than a dedicated flag).
    4. **Training**: from absolute step ``config.start_step`` to
       ``config.start_step + config.num_steps - 1`` (inclusive) run
       grad-accumulated forward + backward + optimizer + LR schedule.
       The grad accumulation count is ``config.grad_accum_steps``.
       The LR schedule is the per-phase WSD curve defined by
       ``lr / min_lr / lr_warmup_iters / lr_decay_iters /
       lr_wsd_decay_iters`` (see the field block on
       :class:`TrainLoopConfig`), evaluated at the absolute step.
       Emit the ``[PHASE]`` banner (see stdout grammar) once before the
       first step so the gate can verify the resolved schedule.
    5. **Save hook**: when ``config.save_path`` is set and
       ``config.save_interval == 0``, persist the full state (params +
       optimizer moments + step counter + num_samples + dataloader
       position) to that directory AFTER the final step in this
       invocation completes.  When ``config.save_interval > 0``,
       ADDITIONALLY write a full resumable state to
       ``<save_path>/step_<abs>/training_state.pt`` every
       ``save_interval`` absolute steps, and write the final state
       versioned under ``step_<start_step + num_steps>/`` as well —
       this is what the 3-phase driver's crash-restart branch globs.
       The contract for the corresponding resumed leg is to read this
       directory, seek the dataloader to the saved position, and
       continue.
    6. **Capture hook**: when ``config.capture_output_file`` is set,
       switch to single-step tensor capture (see "M1 capture semantics"
       on :class:`TrainLoopConfig`).  The engine writes the
       fwd-activation / param-grad dict to that exact file path and
       returns; no ``[<loss_tag>]`` lines are emitted.
       ``capture_output_file`` and ``save_path`` are mutually exclusive
       at the dispatcher level.
    7. **Cleanup (ordered teardown — exit 0 is a gate requirement)**:
       the gate PASSes iff the process exits with ``returncode == 0`` AND
       the artifact (loss trajectory / tensor dump) is valid. A crash
       during interpreter shutdown — the NCCL / kernel-workspace /
       dataloader-thread destructor-order race that surfaces as SIGABRT
       (``terminate called without an active exception``,
       ``returncode == -6``) — is a FAIL even when every gate-visible
       byte was already flushed. Do NOT bypass the finalizer with
       ``os._exit``; tear the process down cleanly in this order before
       returning, so nothing is left to crash in the interpreter's
       destructor pass:

         a. **Drain** in-flight GPU work: ``torch.cuda.synchronize()`` on
            the active device so no kernel is still executing against a
            context that is about to be released.
         b. **Stop background threads** that touch CUDA: join / close the
            dataloader's prefetch + pinned-memory workers (and any
            ``modelbest_sdk`` / framework iterator threads). A daemon
            thread mid ``.pin_memory()`` / H2D copy when the context dies
            is the most common SIGABRT source — it must be quiesced here,
            not abandoned to the finalizer.
         c. **Release NCCL, multi-rank-coordinated**: if the process
            group is initialised, ``dist.barrier()`` so every rank
            reaches teardown together, then
            ``dist.destroy_process_group()``. The barrier prevents one
            rank from destroying its communicator while a peer is still
            mid-collective.
         d. **Release the rest**: free any kernel / attention workspace
            the engine allocated, then ``torch.cuda.empty_cache()``.
         e. **Return normally.** The runner does not call ``os._exit``;
            control returns through ``run_training_loop`` to the gate
            script, which exits 0 only if this teardown left the
            interpreter with nothing to crash on.

       A teardown that *hangs* (artifact valid but the process never
       exits) is also a FAIL — it consumes the gate timeout, which is a
       real cost in hours-long loops. Bound every join / barrier so a
       stuck rank surfaces as a timeout rather than an indefinite hang.

    Stdout grammar
    ~~~~~~~~~~~~~~

    Phase banner (rank-0 only, emitted ONCE at the start of each
    ``run_training_loop`` invocation, BEFORE the first per-step line;
    emitted iff ``config.phase_name`` is set)::

        [PHASE] name=<str> start_step=<N> lr=<float .9e> min_lr=<float .9e>
                warmup=<N> decay=<N> wsd_decay=<N> init_weights_only=<0|1>
                data_path=<str>

    This line is the SSOT the wsd-sft gate dispatcher parses to assert
    **switch correctness** across the stable → decay → sft handoffs:
    that the peak LR steps to the phase's value, that ``start_step``
    resets to 0 on the ``init_weights_only`` SFT leg (vs. carrying over
    on the decay full-resume leg), and that ``data_path`` swaps to the
    phase's corpus. ``name`` is the phase label the runner passes; the
    remaining fields echo the resolved ``TrainLoopConfig`` so the gate
    can check emitted == the frozen phase table without a second engine.
    Gates that do not exercise the phase schedule (single-leg bitwise /
    resume / long) may omit it — the parser only requires it for wsd-sft.

    Per-step (one line per global step, rank-0 only)::

        [<loss_tag>] step=<N> global_loss=<float .9e> grad_norm=<float .9e>
                     time_s=<float6> mfu_e2e_standard=<float6>

    The gate-decisive floats (``global_loss`` / ``grad_norm``) MUST be
    emitted with the ``.9e`` format — fp32 needs ≥9 significant decimal
    digits to round-trip uniquely, and anything narrower silently masks
    sub-print fp32 ULP drift from the per-step ``max_abs_diff == 0``
    gate. Canonical constant: ``harness.wire_format.LOSS_FLOAT_FORMAT``.

    Trailing marker::

        ALL DONE

    The harness ``evals/dispatcher`` parser keys off the bracketed
    ``loss_tag`` (default ``LOSS``; the resume gate uses ``LOSS_REF``
    for the uninterrupted phase and ``LOSS_RES`` for the save+load
    leg so a single subprocess can emit both trajectories).  Lines
    are picked up live via ``run_streaming_subprocess``; no
    end-of-run dump file is required.

    In M1 capture mode (``config.capture_output_file`` set) no ``[LOSS]``
    lines are emitted — the dispatcher consumes only the tensor dump
    produced at that exact path.

    Failure semantics
    ~~~~~~~~~~~~~~~~~

    Errors propagate as exceptions; the gate wrapper turns them into
    a non-zero exit so the dispatcher records a failed run.  Do not
    swallow ``KeyboardInterrupt`` or torch-distributed timeouts —
    they must visibly kill the worker so launch_dp.py can clean up.

    Implementation note
    ~~~~~~~~~~~~~~~~~~~

    This module is intentionally a stub at harness check-in time; the
    agent loop fills in the body via the rest of
    ``training_engine_tensor/`` (forward, backward, optimizer, nccl,
    kernels, …).  The signature, dataclass shape, and stdout grammar
    are the long-lived contract — agents must not change them without
    a synchronous update to every gate script and the dispatcher
    parsers.
    """
    raise NotImplementedError(
        "training_engine_tensor.train_loop.run_training_loop is the SSOT "
        "entry point for the self-developed training engine.  The agent-loop "
        "implementation belongs in this module; harness gate scripts are "
        "thin wrappers and must not provide their own training body."
    )
