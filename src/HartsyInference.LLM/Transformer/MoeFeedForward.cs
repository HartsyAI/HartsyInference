using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Mixture-of-Experts feed-forward block: a router picks the top-k experts per token, each selected
/// expert (a SwiGLU FFN) runs only on the tokens routed to it (gather → expert GEMM → weighted scatter-add),
/// plus an optional always-on shared expert applied to every token. Replaces the dense SwiGLU on MoE layers.
/// Covers Qwen2-MoE / Qwen3-MoE / Mixtral (softmax routing) and DeepSeek (sigmoid routing); experts run through
/// the same quant-aware <see cref="GenericTransformer.Project"/> path as the dense FFN.
///
/// <para><b>Routing residency:</b> the router logits are read to the host to pick top-k experts and build the
/// per-expert token groups (one D2H per MoE layer). The expert GEMMs and the gather/scatter combine stay
/// device-resident. A fully on-device top-k + dispatch is a follow-up; correctness is unaffected.</para></summary>
public sealed class MoeFeedForward
{
    private readonly MoeConfig _moe;
    private readonly int _hidden;
    private readonly bool _lowVram;

    private Tensor _routerW = null!;                 // [E, hidden]
    private Tensor[] _gateW = null!, _upW = null!, _downW = null!;   // per expert
    private Tensor? _shGateW, _shUpW, _shDownW, _shGateScoreW;       // shared expert (optional)

    public MoeFeedForward(MoeConfig moe, int hiddenSize, bool lowVram)
    {
        _moe = moe;
        _hidden = hiddenSize;
        _lowVram = lowVram;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _routerW = w[$"{prefix}.mlp.gate.weight"];
        int e = _moe.NumExperts;
        _gateW = new Tensor[e];
        _upW = new Tensor[e];
        _downW = new Tensor[e];
        for (int i = 0; i < e; i++)
        {
            string ep = $"{prefix}.mlp.experts.{i}";
            _gateW[i] = w[$"{ep}.gate_proj.weight"];
            _upW[i] = w[$"{ep}.up_proj.weight"];
            _downW[i] = w[$"{ep}.down_proj.weight"];
        }
        if (_moe.SharedExpertIntermediateSize > 0)
        {
            _shGateW = w[$"{prefix}.mlp.shared_expert.gate_proj.weight"];
            _shUpW = w[$"{prefix}.mlp.shared_expert.up_proj.weight"];
            _shDownW = w[$"{prefix}.mlp.shared_expert.down_proj.weight"];
            // The shared-expert sigmoid gate ([1, hidden]) is optional (Qwen2-MoE has it; some variants don't).
            w.TryGetValue($"{prefix}.mlp.shared_expert_gate.weight", out Tensor? sg);
            _shGateScoreW = sg;
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        yield return _routerW;
        for (int i = 0; i < _gateW.Length; i++) { yield return _gateW[i]; yield return _upW[i]; yield return _downW[i]; }
        if (_shGateW is not null) { yield return _shGateW; yield return _shUpW!; yield return _shDownW!; }
        if (_shGateScoreW is not null) yield return _shGateScoreW;
    }

    /// <summary>MoE FFN: <paramref name="x"/> <c>[1, N, hidden]</c> → <c>[1, N, hidden]</c>.</summary>
    public unsafe Tensor Forward(IBackend backend, Tensor x, int n)
    {
        int e = _moe.NumExperts;
        int topK = _moe.NumExpertsPerTok;
        int inter = _moe.MoeIntermediateSize;

        // 1. Router logits, read to host for top-k selection.
        Tensor routerLogits = new(new TensorShape(1, n, e), DType.F32);
        GenericTransformer.Project(backend, routerLogits, x, _routerW, null, lowVram: false);
        float[] logits = new float[(long)n * e];
        fixed (float* dst = logits)
        {
            float* src = (float*)routerLogits.DataPointer;   // D2H sync on CUDA
            Buffer.MemoryCopy(src, dst, logits.Length * 4L, logits.Length * 4L);
        }
        routerLogits.Dispose();

        // 2. Per-token top-k routing → per-expert token groups + weights.
        List<int>[] expertTokens = new List<int>[e];
        List<float>[] expertWeights = new List<float>[e];
        for (int i = 0; i < e; i++) { expertTokens[i] = []; expertWeights[i] = []; }
        Route(logits, n, e, topK, expertTokens, expertWeights);

        // 3. Output accumulator, seeded with the shared expert (if any).
        Tensor output = new(new TensorShape(1, n, _hidden), DType.F32);
        backend.Fill(output, 0f);
        if (_shGateW is not null)
        {
            Tensor shared = SwiGlu(backend, x, n, _shGateW, _shUpW!, _shDownW!, _moe.SharedExpertIntermediateSize);
            int[] identity = new int[n];
            for (int i = 0; i < n; i++) identity[i] = i;
            float[] gateVals;
            if (_shGateScoreW is not null)
            {
                // sigmoid(x @ shared_expert_gate) scales the shared expert per token. The gate is a per-token
                // scalar already going to the host (as a scatter scale), so apply the sigmoid there — no need
                // for a device Sigmoid op.
                Tensor g = new(new TensorShape(1, n, 1), DType.F32);
                GenericTransformer.Project(backend, g, x, _shGateScoreW, null, lowVram: false);
                gateVals = HostCopy(g, n);
                g.Dispose();
                for (int i = 0; i < n; i++) gateVals[i] = 1f / (1f + MathF.Exp(-gateVals[i]));
            }
            else
            {
                gateVals = new float[n];
                Array.Fill(gateVals, 1f);
            }
            // Identity scatter folds the per-token gate in and seeds the accumulator in one device op.
            backend.ScatterAddWeightedRows(output, shared, identity, gateVals);
            shared.Dispose();
        }

        // 4. Routed experts: gather → expert SwiGLU → weighted scatter-add.
        for (int ex = 0; ex < e; ex++)
        {
            int ne = expertTokens[ex].Count;
            if (ne == 0) continue;
            int[] idx = [.. expertTokens[ex]];
            float[] wts = [.. expertWeights[ex]];

            Tensor gathered = new(new TensorShape(1, ne, _hidden), DType.F32);
            backend.GatherRows(gathered, x, idx);
            Tensor expOut = SwiGlu(backend, gathered, ne, _gateW[ex], _upW[ex], _downW[ex], inter);
            gathered.Dispose();
            backend.ScatterAddWeightedRows(output, expOut, idx, wts);
            expOut.Dispose();
        }
        return output;
    }

    /// <summary>SwiGLU FFN over <paramref name="rows"/> tokens: down(silu(gate(x)) * up(x)).</summary>
    private Tensor SwiGlu(IBackend backend, Tensor x, int rows, Tensor gateW, Tensor upW, Tensor downW, int inter)
    {
        TensorShape ff = new(1, rows, inter);
        Tensor gate = new(ff, DType.F32);
        GenericTransformer.Project(backend, gate, x, gateW, null, _lowVram);
        Tensor act = new(ff, DType.F32);
        backend.Silu(act, gate);
        gate.Dispose();
        Tensor up = new(ff, DType.F32);
        GenericTransformer.Project(backend, up, x, upW, null, _lowVram);
        Tensor comb = new(ff, DType.F32);
        backend.Mul(comb, act, up);
        act.Dispose(); up.Dispose();
        Tensor outp = new(new TensorShape(1, rows, _hidden), DType.F32);
        GenericTransformer.Project(backend, outp, comb, downW, null, _lowVram);
        comb.Dispose();
        return outp;
    }

    /// <summary>Per-token top-k expert selection + routing weights (softmax or sigmoid, optional renorm).</summary>
    private void Route(float[] logits, int n, int e, int topK, List<int>[] expertTokens, List<float>[] expertWeights)
    {
        float[] score = new float[e];
        int[] pick = new int[topK];
        for (int t = 0; t < n; t++)
        {
            long baseOff = (long)t * e;
            if (_moe.Scoring == MoeScoring.Softmax)
            {
                float max = float.NegativeInfinity;
                for (int i = 0; i < e; i++) max = MathF.Max(max, logits[baseOff + i]);
                float sum = 0f;
                for (int i = 0; i < e; i++) { float v = MathF.Exp(logits[baseOff + i] - max); score[i] = v; sum += v; }
                for (int i = 0; i < e; i++) score[i] /= sum;
            }
            else
            {
                for (int i = 0; i < e; i++) score[i] = 1f / (1f + MathF.Exp(-logits[baseOff + i]));
            }

            // Top-k by score (k is small; a partial selection scan is fine).
            float wsum = 0f;
            for (int kk = 0; kk < topK; kk++)
            {
                int best = -1;
                float bestVal = float.NegativeInfinity;
                for (int i = 0; i < e; i++)
                {
                    bool already = false;
                    for (int j = 0; j < kk; j++) if (pick[j] == i) { already = true; break; }
                    if (already) continue;
                    if (score[i] > bestVal) { bestVal = score[i]; best = i; }
                }
                pick[kk] = best;
                wsum += bestVal;
            }
            for (int kk = 0; kk < topK; kk++)
            {
                int ex = pick[kk];
                float wt = score[ex];
                if (_moe.NormTopKProb) wt /= wsum;
                expertTokens[ex].Add(t);
                expertWeights[ex].Add(wt);
            }
        }
    }

    private static unsafe float[] HostCopy(Tensor t, int count)
    {
        float[] r = new float[count];
        fixed (float* dst = r)
        {
            float* src = (float*)t.DataPointer;   // D2H sync on CUDA
            Buffer.MemoryCopy(src, dst, count * 4L, count * 4L);
        }
        return r;
    }
}
