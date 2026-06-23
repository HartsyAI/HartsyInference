using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.LLM.Transformer;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Validates the CUDA MoE path — the gather / weighted-scatter-add expert-dispatch kernels and the
/// resident expert GEMMs — by checking <see cref="MoeFeedForward"/> on CUDA matches the same block on the CPU
/// backend (which the LLM suite proves against a pure-math reference).</summary>
[Collection("CudaSerial")]
public sealed unsafe class CudaMoeTests
{
    private readonly ITestOutputHelper _output;
    public CudaMoeTests(ITestOutputHelper output) => _output = output;

    private static uint _rng = 0x1234ABCDu;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.5f; }
    private static Tensor F2(int a, int b) { Tensor t = new(new TensorShape(a, b), DType.F32); float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor X(int n, int h) { Tensor t = new(new TensorShape(1, n, h), DType.F32); float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }

    [Fact]
    public void CudaMoe_MatchesCpuMoe()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");

        const int hidden = 32, inter = 48, shInter = 40, e = 6, topK = 2, n = 8;
        MoeConfig moe = new()
        {
            NumExperts = e, NumExpertsPerTok = topK, MoeIntermediateSize = inter,
            NormTopKProb = true, Scoring = MoeScoring.Softmax, SharedExpertIntermediateSize = shInter,
        };
        const string p = "model.layers.0";
        Dictionary<string, Tensor> w = new()
        {
            [$"{p}.mlp.gate.weight"] = F2(e, hidden),
            [$"{p}.mlp.shared_expert.gate_proj.weight"] = F2(shInter, hidden),
            [$"{p}.mlp.shared_expert.up_proj.weight"] = F2(shInter, hidden),
            [$"{p}.mlp.shared_expert.down_proj.weight"] = F2(hidden, shInter),
            [$"{p}.mlp.shared_expert_gate.weight"] = F2(1, hidden),
        };
        for (int i = 0; i < e; i++)
        {
            w[$"{p}.mlp.experts.{i}.gate_proj.weight"] = F2(inter, hidden);
            w[$"{p}.mlp.experts.{i}.up_proj.weight"] = F2(inter, hidden);
            w[$"{p}.mlp.experts.{i}.down_proj.weight"] = F2(hidden, inter);
        }

        MoeFeedForward moeFf = new(moe, hidden, lowVram: false);
        moeFf.LoadWeights(w, p);

        using Tensor x = X(n, hidden);
        float[] cpuOut;
        using (CpuBackend cpu = new())
        using (Tensor oc = moeFf.Forward(cpu, x, n))
        {
            cpuOut = new float[oc.ElementCount];
            float* pc = (float*)oc.DataPointer;
            for (long i = 0; i < cpuOut.Length; i++) cpuOut[i] = pc[i];
        }

        using CudaBackend cuda = new(0, ptxDir);
        using Tensor og = moeFf.Forward(cuda, x, n);
        cuda.Sync();
        float* pg = (float*)og.DataPointer;
        float max = 0f;
        for (long i = 0; i < cpuOut.Length; i++) max = MathF.Max(max, MathF.Abs(cpuOut[i] - pg[i]));
        _output.WriteLine($"max |cpu - cuda| = {max:E3}");
        Assert.True(max <= 2e-3f, $"CUDA MoE diverges from CPU MoE by {max:E3}");

        foreach (Tensor t in w.Values) t.Dispose();
    }
}
