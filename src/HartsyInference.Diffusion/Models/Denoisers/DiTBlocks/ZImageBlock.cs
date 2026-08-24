using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Z-Image transformer block (Lumina2/NextDiT). Used for both <c>noise_refiner</c> blocks and the 30 main <c>layers</c> — they're structurally identical and only differ in which tokens they're called on. Uses AdaLN with 4 outputs (scale_msa, gate_msa, scale_mlp, gate_mlp — scale + gate, no shifts) and a fused QKV projection. The SwarmUI single-file checkpoint stores QKV as one big <c>[3*hidden, hidden]</c> tensor and the output projection as <c>attention.out</c>; QK-norm is <c>q_norm</c>/<c>k_norm</c> (not <c>norm_q</c>/<c>norm_k</c>). All attention linears have NO bias.</summary>
public sealed unsafe class ZImageBlock
{
    /// <summary>F16-mode damping for the two sandwich-normed projections. Z-Image's attention out-projection
    /// and SwiGLU produce raw magnitudes past F16's 65504 (traced live: attnProjected INF in the FIRST refiner
    /// block; the historical layer-0 ffnOut INF is the silu(w1·x)·(w3·x) product, ~±160k) — but BOTH feed
    /// straight into an RmsNorm, and RMSNorm(c·x) ≡ RMSNorm(x). Scaling <c>attention.out</c> and
    /// <c>feed_forward.w3</c> (folded into the GEMM alpha via Fp8ScaleFactor — zero extra kernels, any weight
    /// dtype) divides every intermediate on those paths with a bit-exact post-norm result: F16 floating point
    /// loses NO relative precision to a power-of-two exponent shift. 1/64 because late-step magnitudes grow
    /// ~4× past step 1's (raw ffnOut traced to ~1.05M at step 6 — 1/16 left exactly ONE element at INF).</summary>
    private const float F16SandwichDamp = 1.0f / 64.0f;

    /// <summary>HARTSY_ZIMAGE_F16TRACE=1: logs min/max/nan of every block intermediate for the first few block
    /// forwards — locates the first F16 overflow site. Each probe D2H-drains the tensor (very slow); debug only.</summary>
    private static readonly bool F16TraceEnabled =
        Environment.GetEnvironmentVariable("HARTSY_ZIMAGE_F16TRACE") == "1";
    private static int _traceCallsLeft = F16TraceEnabled ? 300 : 0;   // covers all blocks of an 8-step gen

    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _headDim;
    private readonly int _ffnDim;
    private readonly float _eps;
    private readonly bool _useF16SandwichDamp;
    private readonly bool _allowF16Attention;

    private readonly QkNorm _normQ;
    private readonly QkNorm _normK;

    // AdaLN: Linear(adaLNEmbedDim → 4 * hidden). t_emb is fed raw — Z-Image's t_embedder applies SiLU
    // internally between its two Linears, but the final output is the second Linear's output (not silu'd).
    // Z-Image's adaLN_modulation is `Sequential(Linear)` — index 0 — so no SiLU here.
    private Tensor? _adaLNWeight;
    private Tensor? _adaLNBias;

    // Fused QKV is split into separate Q/K/V weights at load (GPU-residency rewrite): 3 Linears write
    // directly into [B, S, H, D] so the head-split needs no host memcopy. The scalar fp8 weight_scale is
    // per-tensor, so the three splits share it (see SplitQkv).
    private Tensor? _toQWeight, _toKWeight, _toVWeight;

    // Output projection: Linear(hidden → hidden), no bias.
    private Tensor? _attnOutWeight;

    // RMSNorm scales
    private Tensor? _attnNorm1Weight, _attnNorm2Weight;
    private Tensor? _ffnNorm1Weight, _ffnNorm2Weight;

    // SwiGLU FFN: w1 (gate), w3 (linear), w2 (output). No biases.
    private Tensor? _w1Weight;
    private Tensor? _w2Weight;
    private Tensor? _w3Weight;

    public ZImageBlock(int hiddenSize, int numHeads, int ffnDim, float eps = 1e-5f,
        bool? useF16SandwichDamp = null, bool allowF16Attention = true)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _headDim = hiddenSize / numHeads;
        _ffnDim = ffnDim;
        _eps = eps;
        _useF16SandwichDamp = useF16SandwichDamp ?? DitDtype.Act == DType.F16;
        _allowF16Attention = allowF16Attention;

        _normQ = new QkNorm(_headDim, eps);
        _normK = new QkNorm(_headDim, eps);
    }

    /// <summary>Loads weights using Z-Image's Lumina-style naming. <paramref name="prefix"/> is e.g. "noise_refiner.0" or "layers.5".</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _adaLNWeight = weights[$"{prefix}.adaLN_modulation.0.weight"];
        weights.TryGetValue($"{prefix}.adaLN_modulation.0.bias", out _adaLNBias);

        (_toQWeight, _toKWeight, _toVWeight) = SplitQkv(weights[$"{prefix}.attention.qkv.weight"], _hiddenSize);
        _attnOutWeight = weights[$"{prefix}.attention.out.weight"];

        // F16 mode: damp the two RmsNorm-sandwiched projections so their raw outputs fit F16 range (see
        // F16SandwichDamp). Applied once per weight (LoadWeights runs once per transformer instance); the F32
        // path is untouched when the flag is off so the baseline stays bit-identical.
        if (_useF16SandwichDamp)
        {
            _attnOutWeight.Fp8ScaleFactor *= F16SandwichDamp;
        }

        _normQ.LoadWeights(weights[$"{prefix}.attention.q_norm.weight"]);
        _normK.LoadWeights(weights[$"{prefix}.attention.k_norm.weight"]);

        // CudaBackend.RmsNorm reads weight as float* directly, so RMSNorm scales MUST be F32.
        // BF16-stored norms (e.g., from a BF16 or nvfp8-mixed checkpoint) would otherwise be
        // bit-reinterpreted as garbage F32. Cheap one-time cast (each tensor is just [hidden]).
        _attnNorm1Weight = TensorCasts.LoadF32(weights, $"{prefix}.attention_norm1.weight");
        _attnNorm2Weight = TensorCasts.LoadF32(weights, $"{prefix}.attention_norm2.weight");
        _ffnNorm1Weight = TensorCasts.LoadF32(weights, $"{prefix}.ffn_norm1.weight");
        _ffnNorm2Weight = TensorCasts.LoadF32(weights, $"{prefix}.ffn_norm2.weight");

        _w1Weight = weights[$"{prefix}.feed_forward.w1.weight"];
        _w2Weight = weights[$"{prefix}.feed_forward.w2.weight"];
        _w3Weight = weights[$"{prefix}.feed_forward.w3.weight"];
        if (_useF16SandwichDamp)
        {
            // Damps silu(w1·x)·(w3·x) AND the w2 output linearly; ffn_norm2 cancels the factor exactly.
            _w3Weight.Fp8ScaleFactor *= F16SandwichDamp;
        }
    }

    /// <summary>Enumerates all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_adaLNWeight is not null) yield return _adaLNWeight;
        if (_adaLNBias is not null) yield return _adaLNBias;
        if (_toQWeight is not null) yield return _toQWeight;
        if (_toKWeight is not null) yield return _toKWeight;
        if (_toVWeight is not null) yield return _toVWeight;
        if (_attnOutWeight is not null) yield return _attnOutWeight;
        foreach (Tensor w in _normQ.EnumerateWeights()) yield return w;
        foreach (Tensor w in _normK.EnumerateWeights()) yield return w;
        if (_attnNorm1Weight is not null) yield return _attnNorm1Weight;
        if (_attnNorm2Weight is not null) yield return _attnNorm2Weight;
        if (_ffnNorm1Weight is not null) yield return _ffnNorm1Weight;
        if (_ffnNorm2Weight is not null) yield return _ffnNorm2Weight;
        if (_w1Weight is not null) yield return _w1Weight;
        if (_w2Weight is not null) yield return _w2Weight;
        if (_w3Weight is not null) yield return _w3Weight;
    }

    /// <summary>Runs the AdaLN-modulated self-attention → SwiGLU FFN sandwich; rotates Q/K via <paramref name="rope"/> when non-null.</summary>
    /// <param name="x">Token sequence [B, seqLen, hidden].</param>
    /// <param name="tEmb">Timestep embedding [B, adaLNEmbedDim] — already through t_embedder.mlp (Linear → SiLU → Linear), output is NOT SiLU'd.</param>
    public Tensor Forward(IBackend backend, Tensor x, Tensor tEmb, ZImageRope? rope, Tensor? attnBias = null)
    {
        int batch = (int)x.Shape[0];
        int seqLen = (int)x.Shape[1];
        // Activation dtype follows the INPUT (the Krea2Block pattern): the transformer casts the token stream
        // to F16 once before the block loop on the HARTSY_DIT_F16 path, and every block activation follows.
        // The AdaLN modulation vectors stay F32 (tiny per-channel params — the F16 norm/affine/gate kernels
        // take an F16 activation + F32 params); the classic F32 path is byte-identical to the baseline.
        DType act = x.DType;
        TensorShape shape = new TensorShape(batch, seqLen, _hiddenSize);
        TensorShape headsShape = new TensorShape(batch, seqLen, _numHeads, _headDim);
        TensorShape mhShape = new TensorShape(batch, _numHeads, seqLen, _headDim);

        // GPU-residency rewrite (mirrors the verified ErnieImageBlock / FluxSingleStreamBlock ports): the scale-only
        // AdaLN affines, the two gated residuals, the QK-norm and the head-split/merge reshapes all run as IBackend
        // ops so the activation chain stays device-resident — no per-op DataPointer D2H sync barriers around the
        // attention/FFN GEMMs. ApplyScale → DiTUtils.Modulate (shift null = pure broadcast multiply x·(1+scale));
        // ApplyGatedResidual → GatedResidualLastDim; the ReshapeToMultiHead/FromMultiHead host loops → Permute0213
        // (Q/K/V declared directly as [B, S, H, D], byte-identical to [B, S, hidden]); the fused-QKV host split →
        // three Linears against the load-time-split Q/K/V weights. The ONLY ops left on the CPU are the tiny AdaLN
        // chunk ([1, 15360]) and ZImageRope.Forward — both contained host excursions the current block already runs.

        // ── AdaLN: Linear(t_emb) → device 4-way split (ModulationSplit4: (1+scale), tanh(gate)) ──
        // mod[0]/mod[2] come out ALREADY (1+scale)-folded, so the affines below apply them directly
        // (no Modulate/AddScalar). The old host split read projected.DataPointer — a full D2H sync
        // barrier per block (34×/step).
        Tensor[] mod = ComputeAdaLN(backend, tEmb, batch);

        bool trace = _traceCallsLeft > 0;
        if (trace)
        {
            _traceCallsLeft--;
            Trace("x-in", x);
            Trace("mod0-scaleMsa", mod[0]);
            Trace("mod1-gateMsa", mod[1]);
        }

        // ── Attention sub-block ──
        Tensor norm1 = new Tensor(shape, act);
        backend.RmsNorm(norm1, x, _attnNorm1Weight!, _eps);
        Tensor modulated = new Tensor(shape, act);
        backend.AffineBroadcastLastDim(modulated, norm1, mod[0], null); // x·(1+scale_msa), no shift
        norm1.Dispose();
        if (trace) Trace("modulated-msa", modulated);

        // Separate Q/K/V projections declared directly as [B, S, H, D] so RmsNorm normalizes over headDim and
        // Permute0213 runs with no explicit reshape.
        Tensor qHeads = new Tensor(headsShape, act);
        Tensor kHeads = new Tensor(headsShape, act);
        Tensor v = new Tensor(headsShape, act);
        backend.Linear(qHeads, modulated, _toQWeight!, null);
        backend.Linear(kHeads, modulated, _toKWeight!, null);
        backend.Linear(v, modulated, _toVWeight!, null);
        modulated.Dispose();
        if (trace) { Trace("qHeads", qHeads); Trace("kHeads", kHeads); Trace("vHeads", v); }

        Tensor qNormed = new Tensor(headsShape, act);
        Tensor kNormed = new Tensor(headsShape, act);
        backend.RmsNorm(qNormed, qHeads, _normQ.Weight, _normQ.Eps);
        backend.RmsNorm(kNormed, kHeads, _normK.Weight, _normK.Eps);
        qHeads.Dispose();
        kHeads.Dispose();
        if (trace) { Trace("qNormed", qNormed); Trace("kNormed", kNormed); }

        // RoPE on the PRE-permute [B, S, H, D] layout, device-resident for B=1 (bit-identical: RoPE indexes only
        // (position, dim)). The old post-permute host rope.Forward D2H-round-tripped the full ~63 MB Q and K
        // every block every step — the dominant residual Z-Image cost. B>1 keeps the host path post-permute.
        bool gpuRope = rope is not null && batch == 1;
        if (gpuRope)
        {
            rope!.ApplyGpu(backend, qNormed, kNormed, _numHeads);
        }

        // [B, S, H, D] → [B, H, S, D] for SDPA.
        Tensor qMh = new Tensor(mhShape, act);
        Tensor kMh = new Tensor(mhShape, act);
        Tensor vMh = new Tensor(mhShape, act);
        backend.Permute0213(qMh, qNormed, seqLen, _numHeads, _headDim);
        backend.Permute0213(kMh, kNormed, seqLen, _numHeads, _headDim);
        backend.Permute0213(vMh, v, seqLen, _numHeads, _headDim);
        qNormed.Dispose();
        kNormed.Dispose();
        v.Dispose();

        if (rope is not null && !gpuRope)
        {
            rope.Forward(qMh, kMh, batch, _numHeads, seqLen);
        }

        // Q/K RMS normalization bounds the scores, but that alone does NOT bound V. Base has produced V values
        // above F16's 65504 ceiling in late main blocks, so its variant policy disables the internal F16 SDPA
        // narrowing even though this tensor stream is F32. Turbo retains the validated fused F16 path.
        float scale = 1.0f / MathF.Sqrt(_headDim);
        Tensor attnOut = new Tensor(mhShape, act);
        backend.ScaledDotProductAttention(attnOut, qMh, kMh, vMh, attnBias, scale, allowF16: _allowF16Attention);
        qMh.Dispose();
        kMh.Dispose();
        vMh.Dispose();
        if (trace) Trace("attnOut", attnOut);

        // [B, H, S, D] → [B, S, hidden]
        Tensor attnFlat = new Tensor(shape, act);
        backend.Permute0213(attnFlat, attnOut, _numHeads, seqLen, _headDim);
        attnOut.Dispose();

        Tensor projected = new Tensor(shape, act);
        backend.Linear(projected, attnFlat, _attnOutWeight!, null);
        attnFlat.Dispose();
        if (trace) Trace("attnProjected", projected);

        Tensor postAttnNorm = new Tensor(shape, act);
        backend.RmsNorm(postAttnNorm, projected, _attnNorm2Weight!, _eps);
        projected.Dispose();

        Tensor afterAttn = new Tensor(shape, act);
        backend.GatedResidualLastDim(afterAttn, x, postAttnNorm, mod[1]);   // x + gate_msa·attn_out
        postAttnNorm.Dispose();
        if (trace) Trace("afterAttn", afterAttn);

        // ── FFN sub-block ──
        Tensor normF1 = new Tensor(shape, act);
        backend.RmsNorm(normF1, afterAttn, _ffnNorm1Weight!, _eps);
        Tensor modulatedF = new Tensor(shape, act);
        backend.AffineBroadcastLastDim(modulatedF, normF1, mod[2], null); // x·(1+scale_mlp), no shift
        normF1.Dispose();

        // The SwiGLU runs at the block dtype: F16 is range-safe because w3 is damped at load (F16SandwichDamp)
        // — the silu(w1·x)·(w3·x) product and the w2 output are both /16, and ffn_norm2 cancels it exactly.
        Tensor ffnOut = ForwardSwiGlu(backend, modulatedF, batch, seqLen, trace);
        modulatedF.Dispose();
        if (trace) Trace("ffnOut", ffnOut);

        Tensor postFfnNorm = new Tensor(shape, act);
        backend.RmsNorm(postFfnNorm, ffnOut, _ffnNorm2Weight!, _eps);
        ffnOut.Dispose();

        Tensor result = new Tensor(shape, act);
        backend.GatedResidualLastDim(result, afterAttn, postFfnNorm, mod[3]);  // afterAttn + gate_mlp·ffn_out
        afterAttn.Dispose();
        postFfnNorm.Dispose();

        for (int i = 0; i < mod.Length; i++) mod[i].Dispose();
        if (trace) Trace("blockOut", result);

        return result;
    }

    /// <summary>Splits the fused <c>attention.qkv.weight</c> <c>[3*hidden, hidden]</c> into separate contiguous Q/K/V
    /// weights <c>[hidden, hidden]</c> at load. Rows [0,H)=Q, [H,2H)=K, [2H,3H)=V (matches the old feature-dim split).
    /// The per-tensor scalar <see cref="Tensor.Fp8ScaleFactor"/> is shared by all three splits (mirrors
    /// <c>CheckpointConvertUtils.SplitInProjWeight</c>). Dtype-agnostic byte copy — works for fp8/F16/F32.</summary>
    internal static (Tensor q, Tensor k, Tensor v) SplitQkv(Tensor qkv, int h)
    {
        if (qkv.Shape[0] != 3L * h || qkv.Shape[1] != h)
            throw new ArgumentException($"Expected fused QKV weight [{3 * h}, {h}], got [{qkv.Shape[0]}, {qkv.Shape[1]}].");

        long chunkBytes = (long)h * h * qkv.DType.SizeInBytes;
        TensorShape splitShape = new TensorShape(h, h);

        Tensor q = new Tensor(splitShape, qkv.DType) { Fp8ScaleFactor = qkv.Fp8ScaleFactor };
        Tensor k = new Tensor(splitShape, qkv.DType) { Fp8ScaleFactor = qkv.Fp8ScaleFactor };
        Tensor v = new Tensor(splitShape, qkv.DType) { Fp8ScaleFactor = qkv.Fp8ScaleFactor };

        byte* src = (byte*)qkv.DataPointer;
        Buffer.MemoryCopy(src, (void*)q.DataPointer, chunkBytes, chunkBytes);
        Buffer.MemoryCopy(src + chunkBytes, (void*)k.DataPointer, chunkBytes, chunkBytes);
        Buffer.MemoryCopy(src + 2 * chunkBytes, (void*)v.DataPointer, chunkBytes, chunkBytes);
        return (q, k, v);
    }

    /// <summary>AdaLN: Linear(t_emb) → device 4-way split via <see cref="IBackend.ModulationSplit4"/>, which emits
    /// <c>(1+scale_msa, tanh(gate_msa), 1+scale_mlp, tanh(gate_mlp))</c> — the diffusers Z-Image convention
    /// (tanh'd gates, transformer_z_image.py:233) with the <c>(1+scale)</c> pre-folded so the block applies the
    /// scales directly via AffineBroadcast. Fully device-resident: the old host split read
    /// <c>projected.DataPointer</c>, a D2H sync barrier per block (34×/step).</summary>
    private Tensor[] ComputeAdaLN(IBackend backend, Tensor tEmb, int batch)
    {
        int outDim = 4 * _hiddenSize;
        Tensor projected = new Tensor(new TensorShape(batch, outDim), DType.F32);
        backend.Linear(projected, tEmb, _adaLNWeight!, _adaLNBias);

        Tensor[] results = new Tensor[4];
        TensorShape paramShape = new TensorShape(batch, _hiddenSize);
        for (int p = 0; p < 4; p++) results[p] = new Tensor(paramShape, DType.F32);
        backend.ModulationSplit4(results[0], results[1], results[2], results[3], projected);
        projected.Dispose();
        return results;
    }

    /// <summary>Debug probe (HARTSY_ZIMAGE_F16TRACE): min/max/nan of a tensor, F16 or F32. D2H-drains.</summary>
    private static void Trace(string name, Tensor t)
    {
        float min = float.MaxValue, max = float.MinValue;
        long nan = 0, inf = 0;
        long n = t.Shape.ElementCount;
        if (t.DType == DType.F16)
        {
            Half* p = (Half*)t.DataPointer;
            for (long i = 0; i < n; i++)
            {
                float v = (float)p[i];
                if (float.IsNaN(v)) { nan++; continue; }
                if (float.IsInfinity(v)) { inf++; continue; }
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }
        else
        {
            float* p = (float*)t.DataPointer;
            for (long i = 0; i < n; i++)
            {
                float v = p[i];
                if (float.IsNaN(v)) { nan++; continue; }
                if (float.IsInfinity(v)) { inf++; continue; }
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }
        HartsyInference.Core.Logging.Logs.Info(
            $"[zimage-f16trace] {name} [{t.DType.Name}] min={min:F3} max={max:F3} nan={nan} inf={inf} n={n}");
    }

    /// <summary>SwiGLU FFN with no biases: output = w2(silu(w1(x)) * w3(x)).</summary>
    private Tensor ForwardSwiGlu(IBackend backend, Tensor input, int batch, int seqLen, bool trace = false)
    {
        TensorShape ffShape = new TensorShape(batch, seqLen, _ffnDim);

        Tensor gate = new Tensor(ffShape, input.DType);
        backend.Linear(gate, input, _w1Weight!, null);
        Tensor gateActivated = new Tensor(ffShape, input.DType);
        backend.Silu(gateActivated, gate);
        gate.Dispose();

        Tensor linear = new Tensor(ffShape, input.DType);
        backend.Linear(linear, input, _w3Weight!, null);

        Tensor gated = new Tensor(ffShape, input.DType);
        backend.Mul(gated, gateActivated, linear);
        if (trace) { Trace("ffn-siluG", gateActivated); Trace("ffn-u", linear); Trace("ffn-gated", gated); }
        gateActivated.Dispose();
        linear.Dispose();

        TensorShape outShape = new TensorShape(batch, seqLen, _hiddenSize);
        Tensor output = new Tensor(outShape, input.DType);
        backend.Linear(output, gated, _w2Weight!, null);
        gated.Dispose();

        return output;
    }

}
