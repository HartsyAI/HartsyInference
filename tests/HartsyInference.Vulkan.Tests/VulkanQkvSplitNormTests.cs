using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>Parity gate for the fused QKV split + per-head QK-RMSNorm on Vulkan.
///
/// <para>This was the one op on the Flux/DiT path whose host default is a TRUE fallback rather than composition:
/// it reads <c>DataPointer</c> on six tensors, so every call cost a device-to-host sync, a scalar loop over every
/// token and head, and an upload of the results. Flux runs 19 double plus 38 single blocks per forward, each
/// calling it once per step.</para></summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanQkvSplitNormTests
{
    private readonly ITestOutputHelper _output;

    public VulkanQkvSplitNormTests(ITestOutputHelper output) => _output = output;

    private static Tensor Filled(TensorShape shape, int seed, double centre = 0.0)
    {
        Tensor t = new(shape, DType.F32);
        Random rng = new(seed);
        Span<float> span = t.AsSpan<float>();
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = (float)(rng.NextDouble() * 2.0 - 1.0 + centre);
        }
        return t;
    }

    /// <summary>Head dims that straddle every plausible subgroup width, because the cross-subgroup fold is where a
    /// per-head reduction silently normalizes by the wrong denominator.</summary>
    [Theory]
    [InlineData(6, 24, 64)]    // Flux double-stream: 24 heads x 64
    [InlineData(4, 8, 128)]    // headDim twice a 64-wide subgroup
    [InlineData(3, 5, 40)]     // headDim below a subgroup and not a power of two
    [InlineData(1, 1, 256)]    // a single wide head, several subgroups deep
    public void MatchesCpuReference(int tokens, int heads, int headDim)
    {
        if (!VulkanAvailable(out string? reason))
        {
            _output.WriteLine($"SKIPPED: {reason}");
            return;
        }
        int w = heads * headDim;
        using VulkanBackend gpu = new(0, SpirvDir());
        IBackend cpu = new CpuBackend();

        // Off-centre input: a norm that dropped the scale or used the wrong denominator still looks plausible on
        // well-conditioned data.
        using Tensor qkv = Filled(new TensorShape(tokens, 3 * w), seed: 5, centre: 0.4);
        using Tensor qWeight = Filled(new TensorShape(headDim), seed: 6, centre: 1.0);
        using Tensor kWeight = Filled(new TensorShape(headDim), seed: 7, centre: 1.0);

        using Tensor gq = new(new TensorShape(tokens, w), DType.F32);
        using Tensor gk = new(new TensorShape(tokens, w), DType.F32);
        using Tensor gv = new(new TensorShape(tokens, w), DType.F32);
        using Tensor cq = new(new TensorShape(tokens, w), DType.F32);
        using Tensor ck = new(new TensorShape(tokens, w), DType.F32);
        using Tensor cv = new(new TensorShape(tokens, w), DType.F32);

        gpu.QkvSplitNorm(gq, gk, gv, qkv, qWeight, kWeight, 1e-6f);
        cpu.QkvSplitNorm(cq, ck, cv, qkv, qWeight, kWeight, 1e-6f);

        double worstQ = Worst(gq, cq), worstK = Worst(gk, ck), worstV = Worst(gv, cv);
        _output.WriteLine($"[{tokens}t x {heads}h x {headDim}d] q={worstQ:E3} k={worstK:E3} v={worstV:E3}");
        Assert.True(worstQ < 1e-4, $"q diverges: {worstQ:E3}");
        Assert.True(worstK < 1e-4, $"k diverges: {worstK:E3}");
        Assert.True(worstV == 0.0, $"v is a straight copy and must be exact, got {worstV:E3}");
    }

    /// <summary>LayerNorm + per-batch modulation, the other norm every DiT block runs.</summary>
    /// <remarks>The batch index is <c>row / seqLen</c>, so the shapes below deliberately include more than one
    /// batch entry: a kernel that ignored the divide would read batch 0's modulation for every row and still look
    /// correct on a single-entry batch.</remarks>
    [Theory]
    [InlineData(1, 16, 320)]
    [InlineData(3, 7, 128)]
    [InlineData(2, 1, 64)]
    public void LayerNormModulate_MatchesCpuReference(int batch, int seqLen, int dim)
    {
        if (!VulkanAvailable(out string? reason))
        {
            _output.WriteLine($"SKIPPED: {reason}");
            return;
        }
        using VulkanBackend gpu = new(0, SpirvDir());
        IBackend cpu = new CpuBackend();
        TensorShape shape = new(batch, seqLen, dim);
        using Tensor input = Filled(shape, seed: 31, centre: 0.7);
        using Tensor scale = Filled(new TensorShape(batch, dim), seed: 32);
        using Tensor shift = Filled(new TensorShape(batch, dim), seed: 33);
        using Tensor got = new(shape, DType.F32);
        using Tensor want = new(shape, DType.F32);

        gpu.LayerNormModulate(got, input, scale, shift, 1e-6f);
        cpu.LayerNormModulate(want, input, scale, shift, 1e-6f);

        double worst = Worst(got, want);
        _output.WriteLine($"[{batch}b x {seqLen}s x {dim}d] max abs err = {worst:E3}");
        Assert.True(worst < 1e-4, $"LayerNormModulate diverges: {worst:E3}");
    }

    /// <summary>In-place rotary on one tensor, full and partial.</summary>
    /// <remarks>The partial cases are the ones worth having: a kernel that ignored rotaryDim and rotated the whole
    /// head would pass every full-rotary shape and corrupt exactly the models that use partial rotary.</remarks>
    [Theory]
    [InlineData(1, 6, 4, 64, 0)]     // full rotary, rotaryDim defaulted
    [InlineData(2, 3, 2, 128, 128)]  // full rotary, stated explicitly
    [InlineData(1, 5, 3, 128, 64)]   // partial: half the head rotates, half must be untouched
    [InlineData(1, 2, 1, 96, 32)]    // partial with an odd-ish head width
    public void ApplyRopeSingle_MatchesCpuReference(int batch, int seqLen, int heads, int headDim, int rotaryDim)
    {
        if (!VulkanAvailable(out string? reason))
        {
            _output.WriteLine($"SKIPPED: {reason}");
            return;
        }
        using VulkanBackend gpu = new(0, SpirvDir());
        IBackend cpu = new CpuBackend();
        TensorShape shape = new(batch, seqLen, heads, headDim);
        using Tensor gpuX = Filled(shape, seed: 41, centre: 0.3);
        using Tensor cpuX = Filled(shape, seed: 41, centre: 0.3);
        using Tensor cos = Filled(new TensorShape(batch, seqLen, headDim), seed: 42);
        using Tensor sin = Filled(new TensorShape(batch, seqLen, headDim), seed: 43);

        gpu.ApplyRopeSingle(gpuX, cos, sin, rotaryDim);
        cpu.ApplyRopeSingle(cpuX, cos, sin, rotaryDim);

        double worst = Worst(gpuX, cpuX);
        _output.WriteLine($"[{batch}b x {seqLen}s x {heads}h x {headDim}d rot={rotaryDim}] max abs err = {worst:E3}");
        Assert.True(worst < 1e-5, $"ApplyRopeSingle diverges: {worst:E3}");
    }

    /// <summary>AdaLN modulation split. The scales get 1+x and the gates get tanh(x), and swapping them produces
    /// plausible output that is wrong everywhere — so all four outputs are checked, not just the first.</summary>
    [Theory]
    [InlineData(1, 320)]
    [InlineData(4, 128)]
    public void ModulationSplit4_MatchesCpuReference(int batch, int dim)
    {
        if (!VulkanAvailable(out string? reason))
        {
            _output.WriteLine($"SKIPPED: {reason}");
            return;
        }
        using VulkanBackend gpu = new(0, SpirvDir());
        IBackend cpu = new CpuBackend();
        using Tensor proj = Filled(new TensorShape(batch, 4 * dim), seed: 51, centre: 0.2);
        TensorShape outShape = new(batch, dim);
        Tensor[] got = [new(outShape, DType.F32), new(outShape, DType.F32), new(outShape, DType.F32), new(outShape, DType.F32)];
        Tensor[] want = [new(outShape, DType.F32), new(outShape, DType.F32), new(outShape, DType.F32), new(outShape, DType.F32)];

        gpu.ModulationSplit4(got[0], got[1], got[2], got[3], proj);
        cpu.ModulationSplit4(want[0], want[1], want[2], want[3], proj);

        string[] names = ["scaleMsa", "gateMsa", "scaleMlp", "gateMlp"];
        for (int i = 0; i < 4; i++)
        {
            double worst = Worst(got[i], want[i]);
            _output.WriteLine($"[{batch}b x {dim}d] {names[i]} err = {worst:E3}");
            Assert.True(worst < 1e-5, $"{names[i]} diverges: {worst:E3}");
        }
        foreach (Tensor tensor in got.Concat(want))
        {
            tensor.Dispose();
        }
    }

    /// <summary>The two row-indexed ops, whose parameters are gathered through a per-row index.</summary>
    /// <remarks>The index is deliberately non-identity and repeats rows: an implementation that ignored it and read
    /// row r of the table would pass an identity index and be wrong for every real call, since the whole point is
    /// that many rows share few modulation vectors.</remarks>
    [Theory]
    [InlineData(6, 128, true)]
    [InlineData(6, 128, false)]
    [InlineData(3, 64, true)]
    public void AffineBroadcastRowIndexed_MatchesCpuReference(int rows, int dim, bool withShift)
    {
        if (!VulkanAvailable(out string? reason))
        {
            _output.WriteLine($"SKIPPED: {reason}");
            return;
        }
        using VulkanBackend gpu = new(0, SpirvDir());
        IBackend cpu = new CpuBackend();
        const int tableRows = 2;
        using Tensor input = Filled(new TensorShape(rows, dim), seed: 61, centre: 0.5);
        using Tensor scaleTable = Filled(new TensorShape(tableRows, dim), seed: 62);
        using Tensor shiftTable = Filled(new TensorShape(tableRows, dim), seed: 63);
        using Tensor rowIndex = new(new TensorShape(rows), DType.I32);
        Span<int> idx = rowIndex.AsSpan<int>();
        for (int i = 0; i < rows; i++)
        {
            idx[i] = (i * 7) % tableRows;   // non-identity, repeating
        }
        using Tensor got = new(new TensorShape(rows, dim), DType.F32);
        using Tensor want = new(new TensorShape(rows, dim), DType.F32);

        gpu.AffineBroadcastRowIndexed(got, input, scaleTable, withShift ? shiftTable : null, rowIndex);
        cpu.AffineBroadcastRowIndexed(want, input, scaleTable, withShift ? shiftTable : null, rowIndex);

        double worst = Worst(got, want);
        _output.WriteLine($"[{rows}r x {dim}d shift={withShift}] err = {worst:E3}");
        Assert.True(worst < 1e-5, $"AffineBroadcastRowIndexed diverges: {worst:E3}");
    }

    [Theory]
    [InlineData(6, 128)]
    [InlineData(5, 64)]
    public void GatedResidualRowIndexed_MatchesCpuReference(int rows, int dim)
    {
        if (!VulkanAvailable(out string? reason))
        {
            _output.WriteLine($"SKIPPED: {reason}");
            return;
        }
        using VulkanBackend gpu = new(0, SpirvDir());
        IBackend cpu = new CpuBackend();
        const int tableRows = 3;
        using Tensor residual = Filled(new TensorShape(rows, dim), seed: 71, centre: 0.2);
        using Tensor value = Filled(new TensorShape(rows, dim), seed: 72, centre: -0.3);
        using Tensor gateTable = Filled(new TensorShape(tableRows, dim), seed: 73);
        using Tensor rowIndex = new(new TensorShape(rows), DType.I32);
        Span<int> idx = rowIndex.AsSpan<int>();
        for (int i = 0; i < rows; i++)
        {
            idx[i] = (i * 5) % tableRows;
        }
        using Tensor got = new(new TensorShape(rows, dim), DType.F32);
        using Tensor want = new(new TensorShape(rows, dim), DType.F32);

        gpu.GatedResidualRowIndexed(got, residual, value, gateTable, rowIndex);
        cpu.GatedResidualRowIndexed(want, residual, value, gateTable, rowIndex);

        double worst = Worst(got, want);
        _output.WriteLine($"[{rows}r x {dim}d] err = {worst:E3}");
        Assert.True(worst < 1e-5, $"GatedResidualRowIndexed diverges: {worst:E3}");
    }

    /// <summary>Token sequence back to an image plane.</summary>
    /// <remarks>Both packings are covered because both are in use and neither is a default: an implementation that
    /// hardcoded one produces a plausible image with the patch interior transposed, which is the kind of wrong that
    /// survives a smoke test. Non-square packed grids catch an h/w swap for the same reason.</remarks>
    [Theory]
    [InlineData(1, 4, 2, 2, 2, true)]
    [InlineData(1, 4, 2, 2, 2, false)]
    [InlineData(2, 3, 3, 5, 2, true)]     // non-square packed grid
    [InlineData(1, 8, 4, 4, 1, false)]    // patch 1: the degenerate case that must still route correctly
    public void UnpatchifyTokens_MatchesCpuReference(int batch, int channels, int hPacked, int wPacked, int patch, bool innerChannelFastest)
    {
        if (!VulkanAvailable(out string? reason))
        {
            _output.WriteLine($"SKIPPED: {reason}");
            return;
        }
        using VulkanBackend gpu = new(0, SpirvDir());
        IBackend cpu = new CpuBackend();
        int seq = hPacked * wPacked;
        int patchVolume = channels * patch * patch;
        int h = hPacked * patch, w = wPacked * patch;

        using Tensor tokens = Filled(new TensorShape(batch, seq, patchVolume), seed: 81);
        using Tensor got = new(new TensorShape(batch, channels, h, w), DType.F32);
        using Tensor want = new(new TensorShape(batch, channels, h, w), DType.F32);

        gpu.UnpatchifyTokens(got, tokens, channels, hPacked, wPacked, patch, innerChannelFastest);
        cpu.UnpatchifyTokens(want, tokens, channels, hPacked, wPacked, patch, innerChannelFastest);

        double worst = Worst(got, want);
        _output.WriteLine($"[{batch}b {channels}c {hPacked}x{wPacked} p{patch} inner={innerChannelFastest}] err = {worst:E3}");
        // A shuffle moves values without arithmetic, so anything but exact means the indexing disagrees.
        Assert.True(worst == 0.0, $"UnpatchifyTokens is a pure shuffle and must match exactly; got {worst:E3}");
    }

    /// <summary>The activation set: exact-erf GELU, erf itself, Mish, PReLU and erf-gated GEGLU.</summary>
    /// <remarks>Inputs span the tails on purpose. The two GELUs agree near zero and diverge by about 1e-3 further
    /// out, so a kernel wired to the tanh approximation when the exact one was asked for passes any test that only
    /// samples the middle. Mish's softplus is the other tail hazard: the naive form overflows exp for large x.</remarks>
    [Theory]
    [InlineData("GeluErf")]
    [InlineData("Mish")]
    public void Activations_MatchCpuReference(string op)
    {
        if (!VulkanAvailable(out string? reason))
        {
            _output.WriteLine($"SKIPPED: {reason}");
            return;
        }
        using VulkanBackend gpu = new(0, SpirvDir());
        IBackend cpu = new CpuBackend();
        TensorShape shape = new(4, 257);           // not a multiple of the workgroup
        using Tensor input = new(shape, DType.F32);
        Span<float> span = input.AsSpan<float>();
        Random rng = new(91);
        for (int i = 0; i < span.Length; i++)
        {
            // Wide: +/-12 reaches where the GELU variants disagree and where a naive softplus overflows.
            span[i] = (float)(rng.NextDouble() * 24.0 - 12.0);
        }
        using Tensor got = new(shape, DType.F32);
        using Tensor want = new(shape, DType.F32);

        switch (op)
        {
            case "GeluErf": gpu.GeluErf(got, input); cpu.GeluErf(want, input); break;
            default: gpu.Mish(got, input); cpu.Mish(want, input); break;
        }

        double worst = Worst(got, want);
        _output.WriteLine($"{op}: max abs err = {worst:E3}");
        // The erf approximation carries ~1.5e-7 of its own, and GELU multiplies it by x.
        Assert.True(worst < 2e-5, $"{op} diverges: {worst:E3}");
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Prelu_MatchesCpuReference(bool perChannel)
    {
        if (!VulkanAvailable(out string? reason))
        {
            _output.WriteLine($"SKIPPED: {reason}");
            return;
        }
        using VulkanBackend gpu = new(0, SpirvDir());
        IBackend cpu = new CpuBackend();
        const int batch = 2, channels = 5, timeDim = 33;
        using Tensor input = Filled(new TensorShape(batch, channels, timeDim), seed: 92);
        using Tensor alpha = Filled(new TensorShape(perChannel ? channels : 1), seed: 93);
        using Tensor got = new(new TensorShape(batch, channels, timeDim), DType.F32);
        using Tensor want = new(new TensorShape(batch, channels, timeDim), DType.F32);

        gpu.Prelu(got, input, alpha);
        cpu.Prelu(want, input, alpha);

        double worst = Worst(got, want);
        _output.WriteLine($"Prelu perChannel={perChannel}: max abs err = {worst:E3}");
        Assert.True(worst == 0.0, $"Prelu is a select and a multiply; it must match exactly, got {worst:E3}");
    }

    [Theory]
    [InlineData(4, 96)]
    [InlineData(3, 65)]
    public void GegluErf_MatchesCpuReference(int rows, int inner)
    {
        if (!VulkanAvailable(out string? reason))
        {
            _output.WriteLine($"SKIPPED: {reason}");
            return;
        }
        using VulkanBackend gpu = new(0, SpirvDir());
        IBackend cpu = new CpuBackend();
        using Tensor proj = Filled(new TensorShape(rows, 2 * inner), seed: 94, centre: 1.5);
        using Tensor got = new(new TensorShape(rows, inner), DType.F32);
        using Tensor want = new(new TensorShape(rows, inner), DType.F32);

        gpu.GegluErf(got, proj, rows, inner);
        cpu.GegluErf(want, proj, rows, inner);

        double worst = Worst(got, want);
        _output.WriteLine($"GegluErf [{rows}r x {inner}]: max abs err = {worst:E3}");
        Assert.True(worst < 2e-5, $"GegluErf diverges: {worst:E3}");
    }

    [Theory]
    [InlineData(1025, 0.25f, -1.5f)]
    [InlineData(64, 1.0f, 0.0f)]
    public void AffineMix_MatchesCpuReference(int count, float xScale, float yScale)
    {
        if (!VulkanAvailable(out string? reason))
        {
            _output.WriteLine($"SKIPPED: {reason}");
            return;
        }
        using VulkanBackend gpu = new(0, SpirvDir());
        IBackend cpu = new CpuBackend();
        using Tensor x = Filled(new TensorShape(count), seed: 101);
        using Tensor y = Filled(new TensorShape(count), seed: 102);
        using Tensor got = new(new TensorShape(count), DType.F32);
        using Tensor want = new(new TensorShape(count), DType.F32);

        gpu.AffineMix(got, x, y, xScale, yScale);
        cpu.AffineMix(want, x, y, xScale, yScale);

        double worst = Worst(got, want);
        _output.WriteLine($"AffineMix [{count}] {xScale}/{yScale}: err = {worst:E3}");
        Assert.True(worst < 1e-6, $"AffineMix diverges: {worst:E3}");
    }

    /// <summary>Bias fill, with and without a bias — the absent case must zero rather than leave the buffer as it
    /// found it, since an unbiased convolution still needs its output cleared before the taps accumulate.</summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void FillBias_MatchesCpuReference(bool withBias)
    {
        if (!VulkanAvailable(out string? reason))
        {
            _output.WriteLine($"SKIPPED: {reason}");
            return;
        }
        using VulkanBackend gpu = new(0, SpirvDir());
        IBackend cpu = new CpuBackend();
        const int batch = 1, cOut = 4, tOut = 3, h = 5, w = 7;
        using Tensor bias = Filled(new TensorShape(cOut), seed: 103);
        using Tensor got = new(new TensorShape([batch, cOut, tOut, h, w]), DType.F32);
        using Tensor want = new(new TensorShape([batch, cOut, tOut, h, w]), DType.F32);

        gpu.FillBias(got, withBias ? bias : null);
        cpu.FillBias(want, withBias ? bias : null);

        double worst = Worst(got, want);
        _output.WriteLine($"FillBias withBias={withBias}: err = {worst:E3}");
        Assert.True(worst == 0.0, $"FillBias writes exact values; got {worst:E3}");
    }

    /// <summary>Depth-to-space. A ratio above 2 is included because the channel packing is
    /// <c>(c*r + p1)*r + p2</c> — read as <c>c + (p1*r + p2)*cOut</c> it produces a plausible image with the
    /// sub-pixel grid scrambled, and at r=2 with few channels the two orders can coincide.</summary>
    [Theory]
    [InlineData(2, 2)]
    [InlineData(3, 2)]
    [InlineData(1, 4)]
    public void PixelShuffle2d_MatchesCpuReference(int ratio, int cOut)
    {
        if (!VulkanAvailable(out string? reason))
        {
            _output.WriteLine($"SKIPPED: {reason}");
            return;
        }
        using VulkanBackend gpu = new(0, SpirvDir());
        IBackend cpu = new CpuBackend();
        const int batch = 2, depth = 3, inH = 4, inW = 5;
        int cIn = cOut * ratio * ratio;
        using Tensor input = Filled(new TensorShape([batch, cIn, depth, inH, inW]), seed: 104);
        TensorShape outShape = new([batch, cOut, depth, inH * ratio, inW * ratio]);
        using Tensor got = new(outShape, DType.F32);
        using Tensor want = new(outShape, DType.F32);

        gpu.PixelShuffle2d(got, input, ratio);
        cpu.PixelShuffle2d(want, input, ratio);

        double worst = Worst(got, want);
        _output.WriteLine($"PixelShuffle2d r={ratio} cOut={cOut}: err = {worst:E3}");
        Assert.True(worst == 0.0, $"PixelShuffle2d is a pure shuffle and must match exactly; got {worst:E3}");
    }

    private static double Worst(Tensor a, Tensor b)
    {
        ReadOnlySpan<float> x = a.AsReadOnlySpan<float>();
        ReadOnlySpan<float> y = b.AsReadOnlySpan<float>();
        double worst = 0;
        for (int i = 0; i < x.Length; i++)
        {
            worst = Math.Max(worst, Math.Abs(x[i] - y[i]));
        }
        return worst;
    }

    private static string SpirvDir()
    {
        string local = Path.Combine(AppContext.BaseDirectory, "Spirv");
        return Directory.Exists(local) ? local : Path.Combine(RepoRoot.Path, "src", "HartsyInference.Vulkan", "Spirv");
    }

    private static bool VulkanAvailable(out string? reason)
    {
        try
        {
            using VulkanBackend probe = new(0, SpirvDir());
            reason = null;
            return true;
        }
        catch (Exception ex)
        {
            reason = ex.Message;
            return false;
        }
    }
}
