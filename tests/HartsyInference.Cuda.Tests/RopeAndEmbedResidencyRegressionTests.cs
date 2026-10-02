using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Cuda.Tests;

/// <summary>Regression coverage for the per-decode-step residency bug fixed alongside the 3060 TTFT work: the
/// RoPE cos/sin table and the layer-0 embedding input are built fresh on the host once per forward call and
/// read by EVERY layer, but <see cref="HartsyInference.Gpu.GpuResidencyCache{TBuffer}.CopyToDevice"/> only
/// persists a tensor that became some op's OUTPUT — a plain input that misses is uploaded, used, and freed in
/// that call's own <c>finally</c>, every single read. Before the fix this cost ~4 PCIe re-uploads per layer per
/// forward call (prefill and decode alike) for a tensor that never changed within the call. The fix
/// (<see cref="GenericTransformer"/>'s <c>MakeRopeTableResident</c> helper and the analogous embedding-residency
/// code in <c>ForwardEmbeds</c>/<c>ForwardBatchDecode</c>) uploads each of the three tensors (cos, sin,
/// embedding) exactly ONCE per forward call via a cheap <c>Scale(_, _, 1f)</c> identity op, so all 36-layer (or,
/// here, N-layer) reads hit instead of re-uploading. That per-call upload is unavoidable — the content is new
/// every call — so the true floor is a SMALL, LAYER-COUNT-INDEPENDENT constant, not zero. What this test
/// guards is exactly that independence: a 2-layer and an 8-layer model of otherwise-identical shape must cost
/// the SAME small number of misses per decode step. A regression that makes misses scale with layer count
/// again (the original bug) fails this immediately; a regression that makes the per-call upload itself
/// disappear would be a correctness bug elsewhere, not something this test is positioned to catch.</summary>
[Trait("Category", "GpuIntegration")]
public sealed unsafe class RopeAndEmbedResidencyRegressionTests
{
    /// <summary>The 3060 with every card visible; a runner pinning one card (CUDA_VISIBLE_DEVICES) leaves only 0.</summary>
    private const int Ordinal = 1;

    // Comfortably above the true per-call floor (3 on the real Qwen3-4B path: cos, sin, embedding) but far
    // below what ANY reintroduction of the per-layer bug would produce even for the smaller layer count below
    // (2 layers × ~4 touches/layer = 8, before even counting the per-call floor).
    private const long MaxMissesPerDecodeStep = 12;

    private static uint _rng = 0x9E3779B9u;
    private static float Rand()
    {
        _rng ^= _rng << 13;
        _rng ^= _rng >> 17;
        _rng ^= _rng << 5;
        return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f;
    }
    private static Tensor Fill(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = Rand();
        return t;
    }
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static Tensor Ones(int n)
    {
        Tensor t = new(new TensorShape(n), DType.F32);
        new Span<float>((float*)t.DataPointer, n).Fill(1f);
        return t;
    }

    private static TransformerConfig TinyConfig(int numLayers) => new()
    {
        HiddenSize = 16, NumLayers = numLayers, NumHeads = 4, NumKvHeads = 2, HeadDim = 4,
        IntermediateSize = 32, VocabSize = 24, MaxPositionEmbeddings = 256, AttentionBias = true, QkNorm = false,
        RopeTheta = 10000f, RotaryDim = 4, RopeScaling = RopeScaling.None,
    };

    private static Dictionary<string, Tensor> TinyWeights(TransformerConfig c)
    {
        int h = c.HiddenSize, qDim = c.QDim, kvDim = c.KvDim;
        Dictionary<string, Tensor> w = new()
        {
            ["model.embed_tokens.weight"] = F2(c.VocabSize, h),
            ["model.norm.weight"] = Ones(h),
        };
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"model.layers.{i}";
            w[$"{p}.input_layernorm.weight"] = Ones(h);
            w[$"{p}.post_attention_layernorm.weight"] = Ones(h);
            w[$"{p}.self_attn.q_proj.weight"] = F2(qDim, h);
            w[$"{p}.self_attn.k_proj.weight"] = F2(kvDim, h);
            w[$"{p}.self_attn.v_proj.weight"] = F2(kvDim, h);
            w[$"{p}.self_attn.o_proj.weight"] = F2(h, qDim);
            w[$"{p}.self_attn.q_proj.bias"] = F1(qDim);
            w[$"{p}.self_attn.k_proj.bias"] = F1(kvDim);
            w[$"{p}.self_attn.v_proj.bias"] = F1(kvDim);
            w[$"{p}.mlp.gate_proj.weight"] = F2(c.IntermediateSize, h);
            w[$"{p}.mlp.up_proj.weight"] = F2(c.IntermediateSize, h);
            w[$"{p}.mlp.down_proj.weight"] = F2(h, c.IntermediateSize);
        }
        return w;
    }

    [Theory]
    [InlineData(2)]
    [InlineData(8)]
    public void MissesPerDecodeStep_DoesNotScaleWithLayerCount(int numLayers)
    {
        Assert.True(CudaContext.IsAvailable(), $"CUDA unavailable: {CudaContext.LastUnavailableReason}");
        string? ptxDir = BackendGate.KernelDir("Ptx", "HartsyInference.Cuda");
        Assert.False(ptxDir is null, "no compiled PTX directory beside the tests or in the repo");

        _rng = 0xA5A5u;
        TransformerConfig cfg = TinyConfig(numLayers);
        Dictionary<string, Tensor> weights = TinyWeights(cfg);
        try
        {
            using CudaBackend backend = new(Math.Min(Ordinal, CudaContext.GetDeviceCount() - 1), ptxDir);
            using GenericTransformer model = new(cfg);
            model.LoadWeights(weights, "model");
            backend.PreloadWeights(model.EnumerateWeights());

            int capacity = 16;
            using FixedKvCache cache = new(cfg.NumLayers, 1, cfg.NumKvHeads, cfg.HeadDim, capacity);
            using Tensor prefillHidden = model.Forward(backend, [1, 2, 3], 0, cache);

            long before1 = backend.TransferState.Misses;
            using Tensor step1 = model.Forward(backend, [4], cache.CurrentLength, cache);
            long afterStep1 = backend.TransferState.Misses;

            long before2 = afterStep1;
            using Tensor step2 = model.Forward(backend, [5], cache.CurrentLength, cache);
            long afterStep2 = backend.TransferState.Misses;

            long delta1 = afterStep1 - before1;
            long delta2 = afterStep2 - before2;

            Assert.True(delta1 <= MaxMissesPerDecodeStep,
                $"{numLayers}-layer decode step 1 cost {delta1} cache misses (bound {MaxMissesPerDecodeStep}); "
                + "a per-layer re-upload (the original bug) would scale with layer count, not stay bounded.");
            Assert.True(delta2 <= MaxMissesPerDecodeStep,
                $"{numLayers}-layer decode step 2 cost {delta2} cache misses (bound {MaxMissesPerDecodeStep}).");
            Assert.Equal(delta1, delta2);   // steady-state cost must not grow step over step
        }
        finally
        {
            foreach (Tensor t in weights.Values) t.Dispose();
        }
    }
}
