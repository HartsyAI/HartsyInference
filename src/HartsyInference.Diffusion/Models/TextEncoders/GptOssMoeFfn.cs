using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.Nvfp4;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>GPT-OSS Mixture-of-Experts feed-forward block — the per-layer FFN substitute used by
/// <see cref="LlamaStyleEncoder"/> when <see cref="LlamaStyleEncoderConfig.NumLocalExperts"/> is non-zero.
/// One router (linear with bias) + a bank of <c>numExperts</c> experts, each storing a fused
/// gate+up projection (interleaved every other column — <b>not</b> first/second half) plus a separate
/// down projection. Routes each token to the top-k experts (k=<see cref="LlamaStyleEncoderConfig.NumExpertsPerToken"/>),
/// runs the GPT-OSS clamped gated MLP per expert, and recombines outputs weighted by the softmax over
/// the top-k logits only.
///
/// <para><b>Tensor layout (upstream verbatim, from <c>transformers/models/gpt_oss/modeling_gpt_oss.py</c>):</b></para>
/// <list type="bullet">
/// <item><c>router.weight</c>     — <c>[numExperts, hidden]</c></item>
/// <item><c>router.bias</c>       — <c>[numExperts]</c></item>
/// <item><c>experts.gate_up_proj</c>       — <c>[numExperts, hidden, 2*intermediate]</c> (gate=cols 0,2,4,... / up=cols 1,3,5,...)</item>
/// <item><c>experts.gate_up_proj_bias</c>  — <c>[numExperts, 2*intermediate]</c></item>
/// <item><c>experts.down_proj</c>          — <c>[numExperts, intermediate, hidden]</c></item>
/// <item><c>experts.down_proj_bias</c>     — <c>[numExperts, hidden]</c></item>
/// </list>
///
/// <para><b>Forward (per token):</b></para>
/// <code>
/// logits = hidden @ router.weight^T + router.bias            # [numExperts]
/// topk_logits, topk_idx = topk(logits, k)                    # [k], [k]
/// scores = softmax(topk_logits)                              # [k] — softmax over k only, NOT all experts
/// out = 0
/// for j in range(k):
///   e = topk_idx[j]
///   gate_up = hidden @ gate_up_proj[e] + gate_up_proj_bias[e]   # [2*intermediate]
///   gate = gate_up[0::2].clamp(max=L)                            # [intermediate]
///   up   = gate_up[1::2].clamp(-L, L)
///   glu  = gate * sigmoid(α * gate)
///   gated = (up + 1) * glu
///   out += scores[j] * (gated @ down_proj[e] + down_proj_bias[e])
/// </code>
/// Reference: <c>GptOssTopKRouter</c> + <c>GptOssExperts</c> in <c>modeling_gpt_oss.py</c>. The custom
/// gated activation matches OpenAI's open-weights GLU variant — see <see cref="LlamaStyleEncoderConfig.ClampedSwiGluLimit"/>
/// and <see cref="LlamaStyleEncoderConfig.ClampedSwiGluAlpha"/>.
///
/// <para><b>Expert weight storage — two mutually exclusive modes:</b> (a) <b>dense F32</b> banks under
/// the bare <c>experts.gate_up_proj</c> / <c>experts.down_proj</c> keys (the diffusers MXFP4 path
/// dequantizes at load); (b) <b>NVFP4-packed</b> banks under <c>experts.gate_up_proj.weight</c> (U8) +
/// <c>.weight_scale</c> + <c>.weight_scale_2</c> (the ComfyUI <c>gpt_oss_20b_nvfp4</c> file). Packed
/// banks stay mmap-backed and are dequantized ONE EXPERT AT A TIME transiently during
/// <see cref="Forward"/> (~100 MB peak instead of ~76 GB for the full 20B bank at F32 — the latter
/// OOM-kills 64 GB hosts). The packed forward groups tokens by routed expert so each expert bank is
/// dequantized exactly once per layer forward.</para></summary>
public sealed unsafe class GptOssMoeFfn
{
    private readonly int _hidden;
    private readonly int _intermediate;
    private readonly int _numExperts;
    private readonly int _topK;
    private readonly float _clampLimit;
    private readonly float _alpha;

    private Tensor? _routerWeight;
    private Tensor? _routerBias;
    private Tensor? _expertsGateUp;       // dense F32 [numExperts, hidden, 2*intermediate]
    private Tensor? _expertsGateUpBias;   // [numExperts, 2*intermediate]
    private Tensor? _expertsDown;         // dense F32 [numExperts, intermediate, hidden]
    private Tensor? _expertsDownBias;     // [numExperts, hidden]

    // NVFP4-packed alternative to _expertsGateUp/_expertsDown (see class doc): U8 nibbles [E, out, in/2]
    // + swizzled F8E4M3 block scales + F32 per-expert globals, dequantized per expert at forward time.
    private Tensor? _gateUpPacked;
    private Tensor? _gateUpBlockScale;
    private Tensor? _gateUpGlobalScale;
    private Tensor? _downPacked;
    private Tensor? _downBlockScale;
    private Tensor? _downGlobalScale;

    /// <summary>Creates a GPT-OSS MoE FFN.</summary>
    /// <param name="hiddenSize">Input/output channel count (2880 for GPT-OSS).</param>
    /// <param name="intermediateSize">Per-expert FFN inner dim (2880 for GPT-OSS).</param>
    /// <param name="numExperts">Total experts in the bank (32 for GPT-OSS).</param>
    /// <param name="topK">Number of experts each token routes to (4 for GPT-OSS).</param>
    /// <param name="clampLimit">Clamp ceiling for the gated activation (7.0 for GPT-OSS).</param>
    /// <param name="alpha">SiLU-approximation α coefficient (1.702 for GPT-OSS).</param>
    public GptOssMoeFfn(int hiddenSize, int intermediateSize, int numExperts, int topK,
        float clampLimit, float alpha)
    {
        if (topK <= 0 || topK > numExperts)
            throw new ArgumentOutOfRangeException(nameof(topK), $"topK must be in (0, {numExperts}]; got {topK}.");
        _hidden = hiddenSize;
        _intermediate = intermediateSize;
        _numExperts = numExperts;
        _topK = topK;
        _clampLimit = clampLimit;
        _alpha = alpha;
    }

    /// <summary>Loads MoE weights under <paramref name="prefix"/> (typical: <c>model.layers.{i}.mlp</c>).
    /// Expert banks are accepted either dense F32 (bare <c>experts.gate_up_proj</c> key) or NVFP4-packed
    /// (<c>experts.gate_up_proj.weight</c> U8 + scale companions) — see the class doc.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _routerWeight = weights[$"{prefix}.router.weight"];
        _routerBias = weights[$"{prefix}.router.bias"];
        _expertsGateUpBias = weights[$"{prefix}.experts.gate_up_proj_bias"];
        _expertsDownBias = weights[$"{prefix}.experts.down_proj_bias"];

        ValidateShape(_routerWeight, "router.weight", _numExperts, _hidden);
        ValidateShape(_routerBias, "router.bias", _numExperts);
        ValidateShape(_expertsGateUpBias, "experts.gate_up_proj_bias", _numExperts, 2 * _intermediate);
        ValidateShape(_expertsDownBias, "experts.down_proj_bias", _numExperts, _hidden);

        if (weights.TryGetValue($"{prefix}.experts.gate_up_proj", out Tensor? gateUpDense))
        {
            _expertsGateUp = gateUpDense;
            _expertsDown = weights[$"{prefix}.experts.down_proj"];
            ValidateShape(_expertsGateUp, "experts.gate_up_proj", _numExperts, _hidden, 2 * _intermediate);
            ValidateShape(_expertsDown, "experts.down_proj", _numExperts, _intermediate, _hidden);
            return;
        }

        if (!weights.ContainsKey($"{prefix}.experts.gate_up_proj.weight"))
            throw new InvalidOperationException(
                $"MoE experts missing under '{prefix}.experts' — expected dense F32 'gate_up_proj'/'down_proj' " +
                "or NVFP4-packed 'gate_up_proj.weight' (+ '.weight_scale'/'.weight_scale_2') keys.");

        _gateUpPacked = weights[$"{prefix}.experts.gate_up_proj.weight"];
        _gateUpBlockScale = weights[$"{prefix}.experts.gate_up_proj.weight_scale"];
        _gateUpGlobalScale = weights[$"{prefix}.experts.gate_up_proj.weight_scale_2"];
        _downPacked = weights[$"{prefix}.experts.down_proj.weight"];
        _downBlockScale = weights[$"{prefix}.experts.down_proj.weight_scale"];
        _downGlobalScale = weights[$"{prefix}.experts.down_proj.weight_scale_2"];

        // On-disk packed layout is [E, out, in/2]: gate_up out = 2*intermediate, in = hidden;
        // down out = hidden, in = intermediate. Two FP4 elements per byte.
        ValidateShape(_gateUpPacked, "experts.gate_up_proj.weight", _numExperts, 2 * _intermediate, _hidden / 2);
        ValidateShape(_downPacked, "experts.down_proj.weight", _numExperts, _hidden, _intermediate / 2);
        ValidateShape(_gateUpGlobalScale, "experts.gate_up_proj.weight_scale_2", _numExperts);
        ValidateShape(_downGlobalScale, "experts.down_proj.weight_scale_2", _numExperts);
    }

    /// <summary>Enumerates every weight tensor for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_routerWeight is not null) yield return _routerWeight;
        if (_routerBias is not null) yield return _routerBias;
        if (_expertsGateUp is not null) yield return _expertsGateUp;
        if (_expertsGateUpBias is not null) yield return _expertsGateUpBias;
        if (_expertsDown is not null) yield return _expertsDown;
        if (_expertsDownBias is not null) yield return _expertsDownBias;
        if (_gateUpPacked is not null) yield return _gateUpPacked;
        if (_gateUpBlockScale is not null) yield return _gateUpBlockScale;
        if (_gateUpGlobalScale is not null) yield return _gateUpGlobalScale;
        if (_downPacked is not null) yield return _downPacked;
        if (_downBlockScale is not null) yield return _downBlockScale;
        if (_downGlobalScale is not null) yield return _downGlobalScale;
    }

    /// <summary>Forward pass. Input/output shape: <c>[B, S, hidden]</c>, F32.
    /// <para>Implementation note: this is the reference CPU path. Dense mode loops over (batch×seq)
    /// tokens, routing each to top-k experts and running the GPT-OSS-style clamped GLU per (token,
    /// expert) pair. Packed (NVFP4) mode inverts the loop — tokens are grouped by routed expert so each
    /// expert bank is dequantized to a transient F32 slice exactly once per forward (memory-bounded:
    /// slice scratch only, never the full bank) and evaluated as batched GEMMs. Both modes compute the
    /// same math (fp accumulation order differs). Asymptotically <c>O(B·S·k·hidden·intermediate)</c> —
    /// fine for the Lens encoder's single pre-denoise pass over ≤512-token prompts.</para></summary>
    public Tensor Forward(IBackend backend, Tensor input)
    {
        int batch = (int)input.Shape[0];
        int seqLen = (int)input.Shape[1];
        int tokens = batch * seqLen;
        Tensor output = new Tensor(input.Shape, DType.F32);

        // Step 1: router logits — Linear projection of input to [B, S, numExperts]
        TensorShape logitsShape = new TensorShape(batch, seqLen, _numExperts);
        Tensor logits = new Tensor(logitsShape, DType.F32);
        backend.Linear(logits, input, _routerWeight!, _routerBias);

        // Step 2: per-token top-k selection + softmax over top-k only.
        int[] slotExpert = new int[tokens * _topK];
        float[] slotScore = new float[tokens * _topK];
        ComputeRouting(logits, tokens, slotExpert, slotScore);
        logits.Dispose();

        // Zero output up-front — we accumulate into it.
        float* outPtr = (float*)output.DataPointer;
        long outElements = output.Shape.ElementCount;
        for (long i = 0; i < outElements; i++) outPtr[i] = 0f;

        // Step 3: MoE evaluation.
        if (_expertsGateUp is not null)
            ForwardDense(input, output, tokens, slotExpert, slotScore);
        else
            ForwardPacked(backend, input, output, tokens, slotExpert, slotScore);
        return output;
    }

    /// <summary>Fills the flat per-slot routing tables: slot <c>t·topK + j</c> holds token <c>t</c>'s
    /// j-th expert index and its softmax-over-top-k score.</summary>
    private void ComputeRouting(Tensor logits, int tokens, int[] slotExpert, float[] slotScore)
    {
        float* logitsPtr = (float*)logits.DataPointer;
        Span<int> topKIdx = stackalloc int[_topK];
        Span<float> topKLogits = stackalloc float[_topK];
        Span<float> topKScores = stackalloc float[_topK];

        for (int t = 0; t < tokens; t++)
        {
            TopKRouter(logitsPtr + (long)t * _numExperts, topKIdx, topKLogits);
            SoftmaxOverTopK(topKLogits, topKScores);
            for (int j = 0; j < _topK; j++)
            {
                slotExpert[t * _topK + j] = topKIdx[j];
                slotScore[t * _topK + j] = topKScores[j];
            }
        }
    }

    /// <summary>Token-major evaluation over the dense F32 expert banks.</summary>
    private void ForwardDense(Tensor input, Tensor output, int tokens, int[] slotExpert, float[] slotScore)
    {
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        float* gateUpBank = (float*)_expertsGateUp!.DataPointer;
        float* gateUpBiasBank = (float*)_expertsGateUpBias!.DataPointer;
        float* downBank = (float*)_expertsDown!.DataPointer;
        float* downBiasBank = (float*)_expertsDownBias!.DataPointer;
        long gateUpPerExpert = (long)_hidden * (2 * _intermediate);
        long downPerExpert = (long)_intermediate * _hidden;

        // Scratch buffers reused per token.
        float[] gateUpScratch = new float[2 * _intermediate];
        float[] gatedScratch = new float[_intermediate];

        for (int t = 0; t < tokens; t++)
        {
            float* tokIn = inPtr + (long)t * _hidden;
            float* tokOut = outPtr + (long)t * _hidden;
            for (int j = 0; j < _topK; j++)
            {
                int slot = t * _topK + j;
                int e = slotExpert[slot];
                AccumulateExpertKernel(tokIn, tokOut,
                    gateUpBank + e * gateUpPerExpert, gateUpBiasBank + (long)e * (2 * _intermediate),
                    downBank + e * downPerExpert, downBiasBank + (long)e * _hidden,
                    slotScore[slot], gateUpScratch, gatedScratch);
            }
        }
    }

    /// <summary>Expert-major evaluation over the NVFP4-packed banks. Tokens are bucketed by routed
    /// expert (CSR layout); each expert with a non-empty bucket is dequantized ONCE into two transient
    /// F32 slices in the on-disk <c>[out, in]</c> orientation, its routed tokens are gathered into one
    /// matrix, and the expert runs as two <c>backend.Linear</c> GEMMs (gather → GEMM → clamped-GLU →
    /// GEMM → scatter into per-slot contribution rows, folded serially afterwards). Memory-bounded:
    /// peak transient is the live expert slices (~100 MB each for GPT-OSS-20B) + the contributions
    /// buffer, regardless of bank size.
    /// <para><b>Concurrency contract:</b> on the CPU backend experts run in parallel with per-worker
    /// reusable slices (the CPU ops are stateless; each (token, slot) contribution row is owned by
    /// exactly one expert). On GPU backends experts run STRICTLY SEQUENTIALLY with a FRESH slice tensor
    /// per expert: the CUDA backend's stream, reference-keyed weight/activation caches, lazy D2H sync
    /// callbacks, and upload auto-promotion are none of them safe against concurrent calls or against
    /// reusing a Tensor object whose bytes changed — the parallel+reuse shape corrupted the GPU cache
    /// bookkeeping and the native heap in production (<c>malloc(): unsorted double linked list
    /// corrupted</c>). Each expert ends with <c>Sync</c> + <c>FreeWeights</c> before its slices are
    /// disposed so no async op or cache entry outlives the host memory.</para></summary>
    private void ForwardPacked(IBackend backend, Tensor input, Tensor output, int tokens,
        int[] slotExpert, float[] slotScore)
    {
        int totalSlots = tokens * _topK;
        int[] counts = new int[_numExperts];
        for (int s = 0; s < totalSlots; s++) counts[slotExpert[s]]++;
        int[] starts = new int[_numExperts + 1];
        for (int e = 0; e < _numExperts; e++) starts[e + 1] = starts[e] + counts[e];
        int[] slotsByExpert = new int[totalSlots];
        int[] cursor = new int[_numExperts];
        Array.Copy(starts, cursor, _numExperts);
        for (int s = 0; s < totalSlots; s++) slotsByExpert[cursor[slotExpert[s]]++] = s;

        List<int> activeExperts = new(_numExperts);
        for (int e = 0; e < _numExperts; e++)
            if (starts[e + 1] > starts[e]) activeExperts.Add(e);

        int twoI = 2 * _intermediate;
        Tensor contributions = new Tensor(new TensorShape(totalSlots, _hidden), DType.F32);
        try
        {
            float* inPtr = (float*)input.DataPointer;
            float* contribPtr = (float*)contributions.DataPointer;

            if (backend.Device.IsCpu)
            {
                // Parallelism bounded by slice memory (each worker holds ~100 MB of dequant slices) and
                // by the CPU GEMM being single-threaded per call.
                ParallelOptions po = new ParallelOptions
                {
                    MaxDegreeOfParallelism = Math.Clamp(Environment.ProcessorCount / 2, 1, 8),
                };
                Parallel.ForEach(activeExperts, po,
                    () => (GateUp: new Tensor(new TensorShape(twoI, _hidden), DType.F32),
                           Down: new Tensor(new TensorShape(_hidden, _intermediate), DType.F32)),
                    (e, _, slices) =>
                    {
                        RunPackedExpert(backend, e, starts, slotsByExpert, slotScore,
                            inPtr, contribPtr, slices.GateUp, slices.Down);
                        return slices;
                    },
                    slices =>
                    {
                        slices.GateUp.Dispose();
                        slices.Down.Dispose();
                    });
            }
            else
            {
                // GPU backends: sequential experts, fresh slices — see the concurrency contract above.
                foreach (int e in activeExperts)
                {
                    Tensor gateUpSlice = new Tensor(new TensorShape(twoI, _hidden), DType.F32);
                    Tensor downSlice = new Tensor(new TensorShape(_hidden, _intermediate), DType.F32);
                    try
                    {
                        RunPackedExpert(backend, e, starts, slotsByExpert, slotScore,
                            inPtr, contribPtr, gateUpSlice, downSlice);
                    }
                    finally
                    {
                        // Drain the stream, then drop any cached/auto-promoted device copies keyed by
                        // these tensors BEFORE freeing their host memory.
                        backend.Sync();
                        backend.FreeWeights([gateUpSlice, downSlice]);
                        gateUpSlice.Dispose();
                        downSlice.Dispose();
                    }
                }
            }

            // Serial reduction: fold every slot's contribution into its token's output row.
            float* outPtr = (float*)output.DataPointer;
            for (int slot = 0; slot < totalSlots; slot++)
            {
                float* src = contribPtr + (long)slot * _hidden;
                float* dst = outPtr + (long)(slot / _topK) * _hidden;
                for (int h = 0; h < _hidden; h++) dst[h] += src[h];
            }
        }
        finally
        {
            contributions.Dispose();
        }
    }

    /// <summary>Evaluates one packed expert: dequant into <paramref name="gateUpSlice"/> /
    /// <paramref name="downSlice"/>, gather routed tokens, two <c>backend.Linear</c> GEMMs with the
    /// clamped-GLU between them, scatter into the per-slot contribution rows. All host reads of GEMM
    /// outputs go through <c>DataPointer</c> (which lazily syncs GPU-cached activations), so on GPU
    /// backends the stream is drained before the transient tensors are disposed.</summary>
    private void RunPackedExpert(IBackend backend, int e, int[] starts, int[] slotsByExpert,
        float[] slotScore, float* inPtr, float* contribPtr, Tensor gateUpSlice, Tensor downSlice)
    {
        int begin = starts[e];
        int n = starts[e + 1] - begin;
        int twoI = 2 * _intermediate;
        int hidden = _hidden;
        int intermediate = _intermediate;
        float clampLimit = _clampLimit;
        float alpha = _alpha;

        Nvfp4Codec.DequantExpertSlice(_gateUpPacked!, _gateUpBlockScale!, _gateUpGlobalScale!, e, gateUpSlice);
        Nvfp4Codec.DequantExpertSlice(_downPacked!, _downBlockScale!, _downGlobalScale!, e, downSlice);
        float* gateUpB = (float*)_expertsGateUpBias!.DataPointer + (long)e * twoI;
        float* downB = (float*)_expertsDownBias!.DataPointer + (long)e * hidden;

        // Gather this expert's routed token rows into one [n, hidden] matrix.
        Tensor x = new Tensor(new TensorShape(n, hidden), DType.F32);
        Tensor gateUpOut = new Tensor(new TensorShape(n, twoI), DType.F32);
        Tensor gated = new Tensor(new TensorShape(n, intermediate), DType.F32);
        Tensor downOut = new Tensor(new TensorShape(n, hidden), DType.F32);
        try
        {
            float* xPtr = (float*)x.DataPointer;
            for (int i = 0; i < n; i++)
            {
                int t = slotsByExpert[begin + i] / _topK;
                Buffer.MemoryCopy(inPtr + (long)t * hidden, xPtr + (long)i * hidden,
                    hidden * sizeof(float), hidden * sizeof(float));
            }

            backend.Linear(gateUpOut, x, gateUpSlice, null);

            // GPT-OSS clamped gated activation, bias folded in here (the GEMM ran bias-free):
            // gate = (gu[2k] + b[2k]).clamp(max=L); up = (gu[2k+1] + b[2k+1]).clamp(-L, L);
            // gated = (up + 1) · gate · sigmoid(α·gate).
            float* guPtr = (float*)gateUpOut.DataPointer;
            float* gatedPtr = (float*)gated.DataPointer;
            for (int i = 0; i < n; i++)
            {
                float* guRow = guPtr + (long)i * twoI;
                float* gatedRow = gatedPtr + (long)i * intermediate;
                for (int k = 0; k < intermediate; k++)
                {
                    float g = guRow[2 * k] + gateUpB[2 * k];
                    float u = guRow[2 * k + 1] + gateUpB[2 * k + 1];
                    if (g > clampLimit) g = clampLimit;
                    if (u > clampLimit) u = clampLimit;
                    else if (u < -clampLimit) u = -clampLimit;
                    float sig = 1.0f / (1.0f + MathF.Exp(-alpha * g));
                    gatedRow[k] = (u + 1.0f) * g * sig;
                }
            }

            backend.Linear(downOut, gated, downSlice, null);

            // Scatter: contribution row for slot s = score · (down_out + down_bias). Reading
            // downOut.DataPointer also drains any pending GPU work on this expert's tensors.
            float* downPtr = (float*)downOut.DataPointer;
            for (int i = 0; i < n; i++)
            {
                int slot = slotsByExpert[begin + i];
                float score = slotScore[slot];
                float* src = downPtr + (long)i * hidden;
                float* dstRow = contribPtr + (long)slot * hidden;
                for (int h = 0; h < hidden; h++)
                    dstRow[h] = score * (src[h] + downB[h]);
            }
        }
        finally
        {
            x.Dispose();
            gateUpOut.Dispose();
            gated.Dispose();
            downOut.Dispose();
        }
    }

    /// <summary>Selects the top-<see cref="_topK"/> expert indices for a token by raw logit value.
    /// Uses a simple partial sort — O(numExperts · topK), good enough for k=4 over 32 experts.</summary>
    private void TopKRouter(float* logits, Span<int> outIdx, Span<float> outLogits)
    {
        // Initialize with first topK candidates.
        for (int i = 0; i < _topK; i++)
        {
            outIdx[i] = i;
            outLogits[i] = logits[i];
        }
        // Sort the initial topK by logit descending.
        for (int i = 1; i < _topK; i++)
        {
            int curIdx = outIdx[i];
            float curLogit = outLogits[i];
            int j = i - 1;
            while (j >= 0 && outLogits[j] < curLogit)
            {
                outIdx[j + 1] = outIdx[j];
                outLogits[j + 1] = outLogits[j];
                j--;
            }
            outIdx[j + 1] = curIdx;
            outLogits[j + 1] = curLogit;
        }
        // Scan remaining experts; replace the smallest when we find something larger.
        for (int e = _topK; e < _numExperts; e++)
        {
            float v = logits[e];
            if (v <= outLogits[_topK - 1]) continue;
            int j = _topK - 1;
            while (j > 0 && outLogits[j - 1] < v)
            {
                outIdx[j] = outIdx[j - 1];
                outLogits[j] = outLogits[j - 1];
                j--;
            }
            outIdx[j] = e;
            outLogits[j] = v;
        }
    }

    /// <summary>Softmax over the top-k logits (NOT over all experts). Numerically stable via max-subtract.</summary>
    private void SoftmaxOverTopK(ReadOnlySpan<float> topKLogits, Span<float> outScores)
    {
        float max = topKLogits[0];
        for (int i = 1; i < _topK; i++) if (topKLogits[i] > max) max = topKLogits[i];
        float sum = 0f;
        for (int i = 0; i < _topK; i++)
        {
            float v = MathF.Exp(topKLogits[i] - max);
            outScores[i] = v;
            sum += v;
        }
        float inv = 1.0f / sum;
        for (int i = 0; i < _topK; i++) outScores[i] *= inv;
    }

    /// <summary>Runs one expert on one token and accumulates <c>weight · expert(input)</c> into <paramref name="tokOut"/>.
    /// gate/up are interleaved in <c>gate_up_proj</c>'s last dim (cols 0,2,4,... are gate; 1,3,5,... are up).
    /// Weight pointers address ONE expert's matrices in the runtime layout (<c>gateUpW</c>:
    /// <c>[hidden, 2·intermediate]</c>, <c>downW</c>: <c>[intermediate, hidden]</c>) — the dense path
    /// passes bank-offset pointers, the packed path passes the transient dequant slices.</summary>
    private void AccumulateExpertKernel(float* tokIn, float* tokOut,
        float* gateUpW, float* gateUpB, float* downW, float* downB, float weight,
        float[] gateUpScratch, float[] gatedScratch)
    {
        // Step A: gate_up = tokIn @ gate_up_proj[expert] + bias
        //   gate_up_proj[expert] has shape [hidden, 2*intermediate]
        //   tokIn has shape [hidden]
        //   result has shape [2*intermediate]
        // Bias already shaped [2*intermediate].
        int twoI = 2 * _intermediate;
        for (int j = 0; j < twoI; j++) gateUpScratch[j] = gateUpB[j];
        for (int h = 0; h < _hidden; h++)
        {
            float a = tokIn[h];
            if (a == 0f) continue;
            float* row = gateUpW + (long)h * twoI;
            for (int j = 0; j < twoI; j++)
                gateUpScratch[j] += a * row[j];
        }

        // Step B: GPT-OSS clamped gated activation
        //   gate = gate_up[0::2].clamp(max=L)
        //   up   = gate_up[1::2].clamp(-L, L)
        //   glu  = gate * sigmoid(α * gate)
        //   gated = (up + 1) * glu
        float L = _clampLimit;
        float alpha = _alpha;
        for (int k = 0; k < _intermediate; k++)
        {
            float g = gateUpScratch[2 * k];
            float u = gateUpScratch[2 * k + 1];
            if (g > L) g = L;
            if (u > L) u = L;
            else if (u < -L) u = -L;
            float sig = 1.0f / (1.0f + MathF.Exp(-alpha * g));
            float glu = g * sig;
            gatedScratch[k] = (u + 1.0f) * glu;
        }

        // Step C: down_proj @ gated + bias, scaled by `weight`, accumulated into tokOut.
        //   down_proj[expert] has shape [intermediate, hidden]
        //   tokOut accumulates [hidden]
        for (int h = 0; h < _hidden; h++) tokOut[h] += weight * downB[h];
        for (int k = 0; k < _intermediate; k++)
        {
            float a = gatedScratch[k];
            if (a == 0f) continue;
            float wa = weight * a;
            float* row = downW + (long)k * _hidden;
            for (int h = 0; h < _hidden; h++)
                tokOut[h] += wa * row[h];
        }
    }

    private static void ValidateShape(Tensor t, string name, params int[] expected)
    {
        if (t.Shape.Rank != expected.Length)
            throw new InvalidOperationException($"GPT-OSS MoE weight '{name}' rank {t.Shape.Rank} != expected {expected.Length}.");
        for (int i = 0; i < expected.Length; i++)
            if (t.Shape[i] != expected[i])
                throw new InvalidOperationException($"GPT-OSS MoE weight '{name}' dim {i} = {t.Shape[i]} != expected {expected[i]}.");
    }
}
