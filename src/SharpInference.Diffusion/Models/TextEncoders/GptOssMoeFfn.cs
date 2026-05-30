using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.TextEncoders;

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
/// and <see cref="LlamaStyleEncoderConfig.ClampedSwiGluAlpha"/>.</summary>
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
    private Tensor? _expertsGateUp;       // [numExperts, hidden, 2*intermediate]
    private Tensor? _expertsGateUpBias;   // [numExperts, 2*intermediate]
    private Tensor? _expertsDown;         // [numExperts, intermediate, hidden]
    private Tensor? _expertsDownBias;     // [numExperts, hidden]

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

    /// <summary>Loads MoE weights under <paramref name="prefix"/> (typical: <c>model.layers.{i}.mlp</c>).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _routerWeight = weights[$"{prefix}.router.weight"];
        _routerBias = weights[$"{prefix}.router.bias"];
        _expertsGateUp = weights[$"{prefix}.experts.gate_up_proj"];
        _expertsGateUpBias = weights[$"{prefix}.experts.gate_up_proj_bias"];
        _expertsDown = weights[$"{prefix}.experts.down_proj"];
        _expertsDownBias = weights[$"{prefix}.experts.down_proj_bias"];

        ValidateShape(_routerWeight, "router.weight", _numExperts, _hidden);
        ValidateShape(_routerBias, "router.bias", _numExperts);
        ValidateShape(_expertsGateUp, "experts.gate_up_proj", _numExperts, _hidden, 2 * _intermediate);
        ValidateShape(_expertsGateUpBias, "experts.gate_up_proj_bias", _numExperts, 2 * _intermediate);
        ValidateShape(_expertsDown, "experts.down_proj", _numExperts, _intermediate, _hidden);
        ValidateShape(_expertsDownBias, "experts.down_proj_bias", _numExperts, _hidden);
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
    }

    /// <summary>Forward pass. Input/output shape: <c>[B, S, hidden]</c>, F32.
    /// <para>Implementation note: this is the reference CPU path — loop over (batch×seq) tokens, route
    /// each to top-k experts, run the GPT-OSS-style clamped GLU per (token, expert) pair, accumulate.
    /// Asymptotically <c>O(B·S·k·hidden·intermediate)</c>. A real-production CUDA path would batch tokens
    /// per expert (gather → expert GEMM → scatter), but for the Lens encoder running on short prompts
    /// (≤512 tokens, 24 layers, k=4) this CPU loop completes in ~1-2 seconds on commodity hardware —
    /// fast enough for the single pre-denoise forward pass.</para></summary>
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

        // Step 2: per-token top-k selection + softmax over top-k only
        // Step 3: per-token MoE evaluation
        float* logitsPtr = (float*)logits.DataPointer;
        float* inPtr = (float*)input.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        // Zero output up-front — we accumulate into it.
        long outElements = output.Shape.ElementCount;
        for (long i = 0; i < outElements; i++) outPtr[i] = 0f;

        // Scratch buffers reused per token.
        float[] gateUpScratch = new float[2 * _intermediate];
        float[] gatedScratch = new float[_intermediate];
        Span<int> topKIdx = stackalloc int[_topK];
        Span<float> topKLogits = stackalloc float[_topK];
        Span<float> topKScores = stackalloc float[_topK];

        for (int t = 0; t < tokens; t++)
        {
            float* tokLogits = logitsPtr + (long)t * _numExperts;
            float* tokIn = inPtr + (long)t * _hidden;
            float* tokOut = outPtr + (long)t * _hidden;

            TopKRouter(tokLogits, topKIdx, topKLogits);
            SoftmaxOverTopK(topKLogits, topKScores);

            for (int j = 0; j < _topK; j++)
            {
                int expertIdx = topKIdx[j];
                float weight = topKScores[j];
                AccumulateExpert(tokIn, tokOut, expertIdx, weight, gateUpScratch, gatedScratch);
            }
        }

        logits.Dispose();
        return output;
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
    /// gate/up are interleaved in <c>gate_up_proj</c>'s last dim (cols 0,2,4,... are gate; 1,3,5,... are up).</summary>
    private void AccumulateExpert(float* tokIn, float* tokOut, int expertIdx, float weight,
        float[] gateUpScratch, float[] gatedScratch)
    {
        float* gateUpW = (float*)_expertsGateUp!.DataPointer + (long)expertIdx * _hidden * (2 * _intermediate);
        float* gateUpB = (float*)_expertsGateUpBias!.DataPointer + (long)expertIdx * (2 * _intermediate);
        float* downW = (float*)_expertsDown!.DataPointer + (long)expertIdx * _intermediate * _hidden;
        float* downB = (float*)_expertsDownBias!.DataPointer + (long)expertIdx * _hidden;

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
