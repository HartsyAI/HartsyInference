using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>SeedVR2's windowed joint attention: no RowScatterAdd call names a row twice (the CUDA kernel adds without
/// atomics, so a repeated row races there while the sequential host loop hides it), and every text row comes back as
/// its mean over the windows.</summary>
[Collection("CudaSerial")]
public sealed class SeedVr2WindowedAttentionTests
{
    private const int Heads = 2, HeadDim = 128, Dim = Heads * HeadDim, MlpDim = 64, TxtLen = 5;

    private readonly ITestOutputHelper _output;

    private static AttentionLayer Layer(int heads, int headDim) => new(new AttentionLayerConfig { Heads = heads, KvHeads = heads, HeadDim = headDim });

    public SeedVr2WindowedAttentionTests(ITestOutputHelper output) => _output = output;

    private static SeedVr2Config Config => new()
    {
        VidDim = Dim, TxtInDim = 32, EmbDim = 6 * Dim, Heads = Heads, HeadDim = HeadDim,
        MlpDim = MlpDim, NumLayers = 4, MmLayers = 2, InChannels = 33, OutChannels = 16,
    };

    /// <summary>Blocks 0 and 1 (regular, then shifted partition) on grids where a shape group holds several windows,
    /// which is when the text rows used to repeat inside one scatter.</summary>
    [Theory]
    [InlineData(8, 4, 4)]
    [InlineData(3, 16, 28)]
    public void Block_ScatterNamesEachRowOncePerCall(int gridT, int gridH, int gridW)
    {
        using UniqueScatterBackend backend = new();
        using SeedVr2DevicePlan plan = new(backend, Config, gridT, gridH, gridW, TxtLen);
        Assert.All(plan.Partitions, p => Assert.Contains(p.Groups, g => g.WindowCount >= 2));

        Dictionary<string, Tensor> weights = BlockWeights(blockCount: 2, seed: 11);
        float[] emb = Values(6 * Dim, seed: 12, amplitude: 0.1f);
        using Tensor ones = Filled(Dim, 1f);
        Tensor vid = RandomMatrix(gridT * gridH * gridW, Dim, seed: 13);
        Tensor txt = RandomMatrix(TxtLen, Dim, seed: 14);
        for (int index = 0; index < 2; index++)
        {
            SeedVr2DitBlock block = new(Config, index);
            block.LoadWeights(weights, index);
            block.PrepareMod(backend, emb);
            (vid, txt) = block.Forward(backend, vid, txt, plan, ones);
            block.ReleaseOwned(backend);
        }
        vid.Dispose();
        txt.Dispose();
        foreach (Tensor w in weights.Values)
            w.Dispose();

        Assert.True(backend.ScatterCalls > 0, "the blocks issued no RowScatterAdd");
        Assert.True(backend.RepeatedRows.Count == 0, string.Join("; ", backend.RepeatedRows));
    }

    /// <summary>Video rows against softmax attention inside their one window and text rows against the mean over
    /// every window, from a double-precision reference that partitions the grid with <see cref="SeedVr2Windowing"/>.</summary>
    [Theory]
    [InlineData(8, 4, 4)]
    [InlineData(3, 16, 28)]
    public void AttendWindows_MatchesPerWindowReference(int gridT, int gridH, int gridW)
    {
        int lv = gridT * gridH * gridW;
        (int nT, int nH, int nW) = Config.WindowCounts;
        using UniqueScatterBackend backend = new();
        using SeedVr2DevicePlan plan = new(backend, Config, gridT, gridH, gridW, TxtLen);
        using Tensor q = RandomMatrix(lv + TxtLen, Dim, seed: 21);
        using Tensor k = RandomMatrix(lv + TxtLen, Dim, seed: 22);
        using Tensor v = RandomMatrix(lv + TxtLen, Dim, seed: 23);
        for (int parity = 0; parity < 2; parity++)
        {
            SeedVr2WindowSlice[] windows = parity == 0
                ? SeedVr2Windowing.MakeWindows(gridT, gridH, gridW, nT, nH, nW)
                : SeedVr2Windowing.MakeShiftedWindows(gridT, gridH, gridW, nT, nH, nW);
            double[] expected = WindowAttention(q, k, v, windows, gridH, gridW, lv);
            (Tensor vid, Tensor txt) = SeedVr2DitBlock.AttendWindows(
                backend, Layer(Heads, HeadDim), q, k, v, plan.Partitions[parity], lv, TxtLen);
            double vidError = MaxAbsDiff(expected, 0, vid), txtError = MaxAbsDiff(expected, lv * Dim, txt);
            vid.Dispose();
            txt.Dispose();
            _output.WriteLine($"parity {parity}: {windows.Length} windows, max |Δ| video {vidError:e2}, text {txtError:e2}");
            Assert.True(vidError < 1e-5 && txtError < 1e-5, $"parity {parity}: video {vidError:e2}, text {txtError:e2}");
        }
        Assert.True(backend.RepeatedRows.Count == 0, string.Join("; ", backend.RepeatedRows));
    }

    /// <summary>At the 3B's 640×360 geometry the windowed attention is bit-identical across two CUDA runs and matches
    /// the CPU: the scatter that summed the text rows raced and lost updates from run to run.</summary>
    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void Cuda_AttendWindows_IsDeterministicAndMatchesCpu()
    {
        if (!BackendGate.TryOpen("cuda", _output.WriteLine, out IBackend? opened))
            return;
        using IBackend cuda = opened!;
        using IBackend cpu = new CpuBackend();
        const int gridT = 3, gridH = 22, gridW = 40, txtLen = 58, heads = 20;
        SeedVr2Config config = Config with { Heads = heads, VidDim = heads * HeadDim, EmbDim = 6 * heads * HeadDim };
        int rows = gridT * gridH * gridW + txtLen;
        using Tensor q = RandomMatrix(rows, config.VidDim, seed: 31);
        using Tensor k = RandomMatrix(rows, config.VidDim, seed: 32);
        using Tensor v = RandomMatrix(rows, config.VidDim, seed: 33);

        float[] first = Attend(cuda, config, q, k, v, gridT, gridH, gridW, txtLen);
        float[] second = Attend(cuda, config, q, k, v, gridT, gridH, gridW, txtLen);
        float[] reference = Attend(cpu, config, q, k, v, gridT, gridH, gridW, txtLen);

        Assert.Equal(first.Length, reference.Length);
        int mismatch = FirstBitMismatch(first, second);
        double maxError = 0;
        for (int i = 0; i < first.Length; i++)
            maxError = Math.Max(maxError, Math.Abs(first[i] - reference[i]));
        _output.WriteLine($"CUDA vs CPU max |Δ| {maxError:e2}; first run-to-run bit mismatch at {mismatch}");
        if (mismatch >= 0)
            Assert.Fail($"two CUDA runs differ first at element {mismatch}: {first[mismatch]:R} vs {second[mismatch]:R}");
        Assert.True(maxError < 1e-3, $"CUDA vs CPU max |Δ| {maxError:e2}");
    }

    /// <summary>Both partitions' video then text rows from <see cref="SeedVr2DitBlock.AttendWindows"/>, on the host.</summary>
    private static float[] Attend(IBackend backend, SeedVr2Config config, Tensor q, Tensor k, Tensor v,
        int gridT, int gridH, int gridW, int txtLen)
    {
        int lv = gridT * gridH * gridW;
        List<float> values = [];
        SeedVr2DevicePlan plan = new(backend, config, gridT, gridH, gridW, txtLen);
        try
        {
            foreach (SeedVr2DevicePlan.Partition part in plan.Partitions)
            {
                (Tensor vid, Tensor txt) = SeedVr2DitBlock.AttendWindows(
                    backend, Layer(config.Heads, config.HeadDim), q, k, v, part, lv, txtLen);
                backend.Sync();
                values.AddRange(vid.AsReadOnlySpan<float>().ToArray());
                values.AddRange(txt.AsReadOnlySpan<float>().ToArray());
                vid.Dispose();
                txt.Dispose();
            }
        }
        finally
        {
            plan.Release(backend);
        }
        return [.. values];
    }

    /// <summary>Softmax attention per head inside each window over its video rows plus every text row, in double;
    /// a text row takes 1/windows of each window's output.</summary>
    private static double[] WindowAttention(Tensor q, Tensor k, Tensor v, SeedVr2WindowSlice[] windows,
        int gridH, int gridW, int lv)
    {
        ReadOnlySpan<float> qs = q.AsReadOnlySpan<float>(), ks = k.AsReadOnlySpan<float>(), vs = v.AsReadOnlySpan<float>();
        double[] output = new double[(lv + TxtLen) * Dim];
        double scale = 1.0 / Math.Sqrt(HeadDim);
        foreach (SeedVr2WindowSlice window in windows)
        {
            List<int> members = [];
            for (int t = window.T0; t < window.T1; t++)
            for (int h = window.H0; h < window.H1; h++)
            for (int w = window.W0; w < window.W1; w++)
                members.Add((t * gridH + h) * gridW + w);
            for (int p = 0; p < TxtLen; p++)
                members.Add(lv + p);
            double[] probs = new double[members.Count];
            foreach (int row in members)
            {
                double share = row >= lv ? 1.0 / windows.Length : 1.0;
                for (int head = 0; head < Heads; head++)
                {
                    int off = head * HeadDim;
                    double max = double.NegativeInfinity, sum = 0;
                    for (int j = 0; j < members.Count; j++)
                    {
                        double dot = 0;
                        for (int d = 0; d < HeadDim; d++)
                            dot += (double)qs[row * Dim + off + d] * ks[members[j] * Dim + off + d];
                        probs[j] = dot * scale;
                        max = Math.Max(max, probs[j]);
                    }
                    for (int j = 0; j < members.Count; j++)
                    {
                        probs[j] = Math.Exp(probs[j] - max);
                        sum += probs[j];
                    }
                    for (int d = 0; d < HeadDim; d++)
                    {
                        double acc = 0;
                        for (int j = 0; j < members.Count; j++)
                            acc += probs[j] * vs[members[j] * Dim + off + d];
                        output[row * Dim + off + d] += share * acc / sum;
                    }
                }
            }
        }
        return output;
    }

    private static double MaxAbsDiff(double[] expected, int offset, Tensor actual)
    {
        ReadOnlySpan<float> a = actual.AsReadOnlySpan<float>();
        double max = 0;
        for (int i = 0; i < a.Length; i++)
            max = Math.Max(max, Math.Abs(expected[offset + i] - a[i]));
        return max;
    }

    private static int FirstBitMismatch(float[] a, float[] b)
    {
        for (int i = 0; i < a.Length; i++)
        {
            if (BitConverter.SingleToUInt32Bits(a[i]) != BitConverter.SingleToUInt32Bits(b[i]))
                return i;
        }
        return -1;
    }

    private static Dictionary<string, Tensor> BlockWeights(int blockCount, int seed)
    {
        Dictionary<string, Tensor> weights = [];
        int s = seed;
        for (int index = 0; index < blockCount; index++)
        {
            foreach (string b in new[] { "vid", "txt" })
            {
                string p = $"blocks.{index}";
                weights[$"{p}.attn.proj_qkv.{b}.weight"] = RandomMatrix(3 * Dim, Dim, ++s, 0.06f);
                weights[$"{p}.attn.proj_out.{b}.weight"] = RandomMatrix(Dim, Dim, ++s, 0.06f);
                weights[$"{p}.attn.proj_out.{b}.bias"] = RandomVector(Dim, ++s, 0.02f);
                weights[$"{p}.attn.norm_q.{b}.weight"] = Shifted(RandomVector(HeadDim, ++s, 0.1f), 1f);
                weights[$"{p}.attn.norm_k.{b}.weight"] = Shifted(RandomVector(HeadDim, ++s, 0.1f), 1f);
                weights[$"{p}.mlp.{b}.proj_in_gate.weight"] = RandomMatrix(MlpDim, Dim, ++s, 0.06f);
                weights[$"{p}.mlp.{b}.proj_in.weight"] = RandomMatrix(MlpDim, Dim, ++s, 0.06f);
                weights[$"{p}.mlp.{b}.proj_out.weight"] = RandomMatrix(Dim, MlpDim, ++s, 0.12f);
                foreach (string ada in new[] { "attn_shift", "attn_scale", "attn_gate", "mlp_shift", "mlp_scale", "mlp_gate" })
                    weights[$"{p}.ada.{b}.{ada}"] = RandomVector(Dim, ++s, 0.1f);
            }
        }
        return weights;
    }

    private static Tensor RandomMatrix(long rows, int cols, int seed, float amplitude = 1f)
    {
        Tensor t = new Tensor(new TensorShape(rows, cols), DType.F32);
        Values(t.AsSpan<float>(), seed, amplitude);
        return t;
    }

    private static Tensor RandomVector(int count, int seed, float amplitude)
    {
        Tensor t = new Tensor(new TensorShape(count), DType.F32);
        Values(t.AsSpan<float>(), seed, amplitude);
        return t;
    }

    private static Tensor Filled(int count, float value)
    {
        Tensor t = new Tensor(new TensorShape(count), DType.F32);
        t.AsSpan<float>().Fill(value);
        return t;
    }

    private static Tensor Shifted(Tensor t, float offset)
    {
        Span<float> s = t.AsSpan<float>();
        for (int i = 0; i < s.Length; i++)
            s[i] += offset;
        return t;
    }

    private static float[] Values(int count, int seed, float amplitude)
    {
        float[] values = new float[count];
        Values(values, seed, amplitude);
        return values;
    }

    private static void Values(Span<float> values, int seed, float amplitude)
    {
        Random rng = new(seed);
        for (int i = 0; i < values.Length; i++)
            values[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * amplitude;
    }

    /// <summary>CpuBackend behind a recorder of every RowScatterAdd call that names a destination row twice.</summary>
    private sealed class UniqueScatterBackend : IBackend
    {
        private readonly IBackend _cpu = new CpuBackend();

        public int ScatterCalls { get; private set; }

        public List<string> RepeatedRows { get; } = [];

        public void RowScatterAdd(Tensor output, Tensor input, Tensor indices, int m, int channels)
        {
            ScatterCalls++;
            HashSet<int> seen = [];
            foreach (int row in indices.AsSpan<int>()[..m])
            {
                if (!seen.Add(row))
                {
                    RepeatedRows.Add($"call {ScatterCalls} repeats row {row}");
                    break;
                }
            }
            _cpu.RowScatterAdd(output, input, indices, m, channels);
        }

        public void AffineBroadcastLastDim(Tensor output, Tensor input, Tensor scale, Tensor? shift) =>
            _cpu.AffineBroadcastLastDim(output, input, scale, shift);
        public void GatedResidualLastDim(Tensor output, Tensor residual, Tensor value, Tensor gate) =>
            _cpu.GatedResidualLastDim(output, residual, value, gate);
        public void QkvSplitNorm(Tensor q, Tensor k, Tensor v, Tensor qkv, Tensor qWeight, Tensor kWeight, float eps) =>
            _cpu.QkvSplitNorm(q, k, v, qkv, qWeight, kWeight, eps);
        public void WanRopeInterleaved(Tensor x, Tensor cos, Tensor sin, int seqLen, int heads, int headDim) =>
            _cpu.WanRopeInterleaved(x, cos, sin, seqLen, heads, headDim);
        public void RowGather(Tensor output, Tensor input, Tensor indices, int m, int channels) =>
            _cpu.RowGather(output, input, indices, m, channels);
        public void SliceRows(Tensor output, Tensor input, int rowOffset) => _cpu.SliceRows(output, input, rowOffset);
        public void MoeCombine(Tensor output, Tensor expertOut, Tensor pairSlot, Tensor topkWeight, int k, bool accumulate) =>
            _cpu.MoeCombine(output, expertOut, pairSlot, topkWeight, k, accumulate);
        public void PreloadWeights(IEnumerable<Tensor> weights) => _cpu.PreloadWeights(weights);
        public void FreeWeights(IEnumerable<Tensor> weights) => _cpu.FreeWeights(weights);
        public void Sync() => _cpu.Sync();

        public DeviceKind Device => _cpu.Device;
        public BackendCapabilities Capabilities => _cpu.Capabilities;
        public void Dispose() => _cpu.Dispose();

        public void MatMul(Tensor output, Tensor a, Tensor b) => _cpu.MatMul(output, a, b);
        public void BatchedMatMul(Tensor output, Tensor a, Tensor b) => _cpu.BatchedMatMul(output, a, b);
        public void Linear(Tensor output, Tensor input, Tensor weight, Tensor? bias) => _cpu.Linear(output, input, weight, bias);
        public void Conv2D(Tensor output, Tensor input, Tensor weight, Tensor? bias, int strideH, int strideW, int padH, int padW) =>
            _cpu.Conv2D(output, input, weight, bias, strideH, strideW, padH, padW);
        public void Conv1d(Tensor output, Tensor input, Tensor weight, Tensor? bias, int stride, int padLeft, int padRight, int dilation, int groups) =>
            _cpu.Conv1d(output, input, weight, bias, stride, padLeft, padRight, dilation, groups);
        public void ConvTranspose1d(Tensor output, Tensor input, Tensor weight, Tensor? bias, int stride, int padding, int dilation, int groups, int outputPadding) =>
            _cpu.ConvTranspose1d(output, input, weight, bias, stride, padding, dilation, groups, outputPadding);
        public void GroupNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps) =>
            _cpu.GroupNorm(output, input, weight, bias, groups, eps);
        public void LayerNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, float eps) => _cpu.LayerNorm(output, input, weight, bias, eps);
        public void RmsNorm(Tensor output, Tensor input, Tensor weight, float eps) => _cpu.RmsNorm(output, input, weight, eps);
        public void AdaInstanceNorm1d(Tensor output, Tensor input, Tensor gamma, Tensor beta, float eps) =>
            _cpu.AdaInstanceNorm1d(output, input, gamma, beta, eps);
        public void ScaledDotProductAttention(Tensor output, Tensor query, Tensor key, Tensor value, Tensor? mask, float scale, bool allowF16 = false) =>
            _cpu.ScaledDotProductAttention(output, query, key, value, mask, scale, allowF16);
        public void Gelu(Tensor output, Tensor input) => _cpu.Gelu(output, input);
        public void Silu(Tensor output, Tensor input) => _cpu.Silu(output, input);
        public void Sigmoid(Tensor output, Tensor input) => _cpu.Sigmoid(output, input);
        public void Tanh(Tensor output, Tensor input) => _cpu.Tanh(output, input);
        public void Elu(Tensor output, Tensor input, float alpha) => _cpu.Elu(output, input, alpha);
        public void LeakyRelu(Tensor output, Tensor input, float slope) => _cpu.LeakyRelu(output, input, slope);
        public void Snake(Tensor output, Tensor input, Tensor alpha, Tensor? beta) => _cpu.Snake(output, input, alpha, beta);
        public void Add(Tensor output, Tensor a, Tensor b) => _cpu.Add(output, a, b);
        public void Mul(Tensor output, Tensor a, Tensor b) => _cpu.Mul(output, a, b);
        public void Scale(Tensor output, Tensor input, float scalar) => _cpu.Scale(output, input, scalar);
        public void Clamp(Tensor output, Tensor input, float min, float max) => _cpu.Clamp(output, input, min, max);
        public void Transpose2D(Tensor output, Tensor input, int d1, int d2) => _cpu.Transpose2D(output, input, d1, d2);
        public void Permute0213(Tensor output, Tensor input, int s, int h, int d) => _cpu.Permute0213(output, input, s, h, d);
        public void GeGlu(Tensor output, Tensor input) => _cpu.GeGlu(output, input);
        public void BroadcastAdd(Tensor hidden, Tensor bias, int channels, int spatial) => _cpu.BroadcastAdd(hidden, bias, channels, spatial);
        public void Concat(Tensor output, ReadOnlySpan<Tensor> inputs, int dim) => _cpu.Concat(output, inputs, dim);
        public void Split(ReadOnlySpan<Tensor> outputs, Tensor input, int dim) => _cpu.Split(outputs, input, dim);
        public void UpsampleNearest2D(Tensor output, Tensor input, int scaleH, int scaleW) => _cpu.UpsampleNearest2D(output, input, scaleH, scaleW);
        public void UpsampleBilinear2D(Tensor output, Tensor input, int scaleH, int scaleW) => _cpu.UpsampleBilinear2D(output, input, scaleH, scaleW);
        public void CopyTo(Tensor destination, Tensor source) => _cpu.CopyTo(destination, source);
        public void Fill(Tensor tensor, float value) => _cpu.Fill(tensor, value);
        public void Fft(Tensor output, Tensor input) => _cpu.Fft(output, input);
        public void Stft(Tensor output, Tensor input, int fftSize, int hopLength, Tensor window) => _cpu.Stft(output, input, fftSize, hopLength, window);
        public void MelFilterbank(Tensor output, Tensor input, Tensor filters) => _cpu.MelFilterbank(output, input, filters);
    }
}
