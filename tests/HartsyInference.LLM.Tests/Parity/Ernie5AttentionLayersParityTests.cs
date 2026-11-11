using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Backends.Latent;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.LLM.Transformer.Compression;
using HartsyInference.LLM.Transformer.Dsa;
using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>The ERNIE-5.5 attention layers on the device against the CPU: HCA (grouped <c>wo_a</c> through
/// <see cref="IBackend.BlockDiagonalLinear"/>, F32 and block-FP8) and DSA (device top-k and <see cref="IBackend.MergeWindowTopK"/>), each
/// with its single-query MQA attention lowered onto the device flash entry. A prefill chunk then decode steps on synthetic weights, at
/// small widths and at ERNIE's head widths. Each test is reported as skipped (never passed) when its device or kernels are absent.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class Ernie5AttentionLayersParityTests
{
    [CudaFact]
    public void Cuda_HcaAttention_F32WoA_MatchesCpu() => HcaParity(Cuda, fp8: false, headDim: 32);

    [VulkanFact]
    public void Vulkan_HcaAttention_F32WoA_MatchesCpu() => HcaParity(Vulkan, fp8: false, headDim: 32);

    [CudaFact]
    public void Cuda_HcaAttention_Fp8WoA_ErnieHeadDim_MatchesCpu() => HcaParity(Cuda, fp8: true, headDim: 512);

    [VulkanFact]
    public void Vulkan_HcaAttention_Fp8WoA_ErnieHeadDim_MatchesCpu() => HcaParity(Vulkan, fp8: true, headDim: 512);

    [CudaFact]
    public void Cuda_DsaAttention_MatchesCpu() => DsaParity(Cuda, LatentEncoding.F32, kvLoraRank: 32, ropeDim: 8);

    [VulkanFact]
    public void Vulkan_DsaAttention_MatchesCpu() => DsaParity(Vulkan, LatentEncoding.F32, kvLoraRank: 32, ropeDim: 8);

    [CudaFact]
    public void Cuda_DsaAttention_ErnieLatent_MatchesCpu() => DsaParity(Cuda, LatentEncoding.F32, kvLoraRank: 512, ropeDim: 64);

    [VulkanFact]
    public void Vulkan_DsaAttention_ErnieLatent_MatchesCpu() => DsaParity(Vulkan, LatentEncoding.F32, kvLoraRank: 512, ropeDim: 64);

    [CudaFact]
    public void Cuda_DsaAttention_Fp8Indexer_StaysCloseToCpu() => DsaParity(Cuda, LatentEncoding.Fp8E4M3Ue8m0x128, kvLoraRank: 32, ropeDim: 8);

    [VulkanFact]
    public void Vulkan_DsaAttention_Fp8Indexer_StaysCloseToCpu() =>
        DsaParity(Vulkan, LatentEncoding.Fp8E4M3Ue8m0x128, kvLoraRank: 32, ropeDim: 8);

    private static IBackend Cuda() => new CudaBackend(0, CudaPtxDirectory.Resolve());

    private static IBackend Vulkan() => new VulkanBackend();

    private static void HcaParity(Func<IBackend> open, bool fp8, int headDim)
    {
        using SyntheticHca layer = new(fp8, headDim);
        float[] want = layer.Run(new CpuBackend());
        using IBackend device = open();
        float[] got = layer.Run(device);
        AssertClose(want, got, 1e-3f, $"HCA {(fp8 ? "fp8" : "f32")} wo_a, head dim {headDim}");
    }

    // F32 indexer encodings keep the device's picks equal to the CPU's, so every output must match closely. With the FP8x128 round
    // trips a device rounding can move a code and so a near-tied pick; most outputs must still match.
    private static void DsaParity(Func<IBackend> open, LatentEncoding indexerEncoding, int kvLoraRank, int ropeDim)
    {
        using SyntheticDsa layer = new(indexerEncoding, kvLoraRank, ropeDim);
        float[] want = layer.Run(new CpuBackend());
        using IBackend device = open();
        float[] got = layer.Run(device);
        if (indexerEncoding == LatentEncoding.F32)
        {
            AssertClose(want, got, 1e-3f, $"DSA latent {kvLoraRank}+{ropeDim}");
            return;
        }
        int close = 0;
        for (int i = 0; i < want.Length; i++)
        {
            Assert.True(float.IsFinite(got[i]), $"DSA element {i} is {got[i]}");
            if (MathF.Abs(want[i] - got[i]) <= 1e-2f * MathF.Max(1f, MathF.Abs(want[i]))) close++;
        }
        Assert.True(close >= want.Length * 9 / 10, $"DSA with FP8 indexer: {close} of {want.Length} outputs within 1e-2");
    }

    private static void AssertClose(float[] want, float[] got, float tolerance, string label)
    {
        Assert.Equal(want.Length, got.Length);
        for (int i = 0; i < want.Length; i++)
            Assert.True(MathF.Abs(want[i] - got[i]) <= tolerance * MathF.Max(1f, MathF.Abs(want[i])),
                $"{label} element {i}: cpu {want[i]} device {got[i]}");
    }

    private static float[] Values(Random rng, int count, float amplitude)
    {
        float[] v = new float[count];
        for (int i = 0; i < count; i++) v[i] = (float)((rng.NextDouble() * 2 - 1) * amplitude);
        return v;
    }

    private static float[] Gains(Random rng, int count) => Values(rng, count, 0.2f).Select(v => 1f + v).ToArray();

    // Runs a prefill chunk then decode steps and returns every output row; the weights live as long as the fixture.
    private abstract class SyntheticLayer : IDisposable
    {
        protected const int Dim = 64, Prefill = 37, Decode = 3;
        private readonly List<Tensor> _owned = new();
        protected readonly Random Rng = new(41);
        protected readonly float[] X;

        protected SyntheticLayer() => X = Values(Rng, (Prefill + Decode) * Dim, 1f);

        protected Tensor Own(float[] values, params long[] shape)
        {
            Tensor t = new(new TensorShape(shape), DType.F32);
            values.CopyTo(t.AsSpan<float>());
            _owned.Add(t);
            return t;
        }

        protected Tensor OwnFp8(int rows, int cols, out Tensor scale)
        {
            Tensor w = new(new TensorShape(rows, cols), DType.F8E4M3);
            Span<byte> codes = w.AsSpan<byte>();
            for (int i = 0; i < codes.Length; i++)
            {
                byte c = LatentCodec.EncodeE4M3((float)(Rng.NextDouble() * 2 - 1));
                codes[i] = (c & 0x7F) == 0x7F ? (byte)0 : c;
            }
            _owned.Add(w);
            int gridRows = (rows + 127) / 128, gridCols = (cols + 127) / 128;
            scale = Own(Values(Rng, gridRows * gridCols, 0.1f).Select(v => 0.05f + MathF.Abs(v)).ToArray(), gridRows, gridCols);
            return w;
        }

        public float[] Run(IBackend backend)
        {
            float[] y = new float[X.Length];
            Forward(backend, y);
            return y;
        }

        protected abstract void Forward(IBackend backend, float[] y);

        public void Dispose()
        {
            foreach (Tensor t in _owned) t.Dispose();
        }
    }

    private sealed class SyntheticHca : SyntheticLayer
    {
        private readonly HcaAttentionSettings _settings;
        private readonly HcaAttentionWeights _weights;
        private readonly DeepSeekV41RopeTable _rope;

        public SyntheticHca(bool fp8, int headDim)
        {
            _settings = new HcaAttentionSettings
            {
                Dim = Dim, Heads = 8, HeadDim = headDim, RopeDim = 8, QLoraRank = 24, OGroups = 8, OLoraRank = 16, Window = 6,
                CompressRatio = 8, NormEps = 1e-6f, QueryHeadNorm = true,
            };
            HcaAttentionSettings s = _settings;
            int h = s.Heads, hd = s.HeadDim, grouped = s.OGroups * s.OLoraRank;
            Tensor? woaScale = null;
            Tensor woa;
            if (fp8)
            {
                woa = OwnFp8(grouped, s.GroupDim, out Tensor scale);
                woaScale = scale;
            }
            else
            {
                woa = Own(Values(Rng, grouped * s.GroupDim, 0.2f), grouped, s.GroupDim);
            }
            _weights = new HcaAttentionWeights
            {
                WqA = Own(Values(Rng, s.QLoraRank * Dim, 0.2f), s.QLoraRank, Dim),
                QNorm = Own(Gains(Rng, s.QLoraRank), s.QLoraRank),
                WqB = Own(Values(Rng, h * hd * s.QLoraRank, 0.2f), h * hd, s.QLoraRank),
                Wkv = Own(Values(Rng, hd * Dim, 0.2f), hd, Dim),
                KvNorm = Own(Gains(Rng, hd), hd),
                CompressorWkv = Own(Values(Rng, hd * Dim, 0.2f), hd, Dim),
                CompressorWgate = Own(Values(Rng, hd * Dim, 0.2f), hd, Dim),
                CompressorApe = Values(Rng, s.CompressRatio * hd, 0.1f),
                CompressorNorm = Gains(Rng, hd),
                Sink = Own(Values(Rng, h, 1f), h),
                WoA = woa,
                WoAScaleInv = woaScale,
                WoB = Own(Values(Rng, Dim * grouped, 0.2f), Dim, grouped),
                OGate = Own(Values(Rng, grouped * s.QLoraRank, 0.2f), grouped, s.QLoraRank),
                HeadMix = Own(Values(Rng, h * h, 0.3f), h, h),
            };
            _rope = DeepSeekV41RopeTable.Build(s.RopeDim, Prefill + Decode, 10000, null);
        }

        protected override void Forward(IBackend backend, float[] y)
        {
            using HcaAttention layer = new(backend, _settings, _weights, _rope);
            HcaAttentionState state = new(Prefill + Decode, _settings);
            layer.Forward(X.AsSpan(0, Prefill * Dim), Prefill, 0, state, y.AsSpan(0, Prefill * Dim));
            for (int t = Prefill; t < Prefill + Decode; t++)
                layer.Forward(X.AsSpan(t * Dim, Dim), 1, t, state, y.AsSpan(t * Dim, Dim));
        }
    }

    private sealed class SyntheticDsa : SyntheticLayer
    {
        private readonly DsaAttentionSettings _settings;
        private readonly DsaAttentionWeights _weights;
        private readonly DeepSeekV41RopeTable _rope, _indexerRope;

        public SyntheticDsa(LatentEncoding indexerEncoding, int kvLoraRank, int ropeDim)
        {
            DsaIndexerSettings indexer = new()
            {
                Heads = 2, HeadDim = 128, RopeDim = 16, TopK = 6, QueryEncoding = indexerEncoding, KeyEncoding = indexerEncoding,
            };
            _settings = new DsaAttentionSettings
            {
                Dim = Dim, Heads = 4, QLoraRank = 24, KvLoraRank = kvLoraRank, QkNopeDim = 16, QkRopeDim = ropeDim, VDim = 16, Window = 5,
                NormEps = 1e-6f, Indexer = indexer,
            };
            DsaAttentionSettings s = _settings;
            int h = s.Heads, ih = indexer.Heads, ihd = indexer.HeadDim;
            _weights = new DsaAttentionWeights
            {
                QA = Own(Values(Rng, s.QLoraRank * Dim, 0.2f), s.QLoraRank, Dim),
                QANorm = Own(Gains(Rng, s.QLoraRank), s.QLoraRank),
                QB = Own(Values(Rng, h * s.QkHeadDim * s.QLoraRank, 0.2f), h * s.QkHeadDim, s.QLoraRank),
                KvA = Own(Values(Rng, s.LatentDim * Dim, 0.2f), s.LatentDim, Dim),
                KvANorm = Own(Gains(Rng, s.KvLoraRank), s.KvLoraRank),
                KvB = Own(Values(Rng, h * (s.QkNopeDim + s.VDim) * s.KvLoraRank, 0.2f), h * (s.QkNopeDim + s.VDim), s.KvLoraRank),
                WoB = Own(Values(Rng, Dim * h * s.VDim, 0.2f), Dim, h * s.VDim),
                OGate = Own(Values(Rng, h * s.VDim * s.QLoraRank, 0.2f), h * s.VDim, s.QLoraRank),
                Sink = Own(Values(Rng, h, 1f), h),
                HeadMix = Own(Values(Rng, h * h, 0.3f), h, h),
                Indexer = new DsaIndexerWeights
                {
                    WqB = Own(Values(Rng, ih * ihd * s.QLoraRank, 0.3f), ih * ihd, s.QLoraRank),
                    Wk = Own(Values(Rng, ihd * Dim, 0.3f), ihd, Dim),
                    KeyNormWeight = Own(Gains(Rng, ihd), ihd),
                    KeyNormBias = Own(Values(Rng, ihd, 0.1f), ihd),
                    WeightsProj = Own(Values(Rng, ih * Dim, 0.3f), ih, Dim),
                },
            };
            _rope = DeepSeekV41RopeTable.Build(s.QkRopeDim, Prefill + Decode, 10000, null);
            _indexerRope = DeepSeekV41RopeTable.Build(indexer.RopeDim, Prefill + Decode, 10000, null);
        }

        protected override void Forward(IBackend backend, float[] y)
        {
            using DsaAttention layer = new(backend, _settings, _weights, _rope, _indexerRope);
            using DsaAttentionState state = new(Prefill + Decode, _settings);
            layer.Forward(X.AsSpan(0, Prefill * Dim), Prefill, 0, state, y.AsSpan(0, Prefill * Dim));
            for (int t = Prefill; t < Prefill + Decode; t++)
                layer.Forward(X.AsSpan(t * Dim, Dim), 1, t, state, y.AsSpan(t * Dim, Dim));
        }
    }
}
