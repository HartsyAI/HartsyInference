using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.LLM.Tests.Parity;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Ssm;

/// <summary>The CUDA and Vulkan Mamba-1 selective scan against <see cref="SelectiveScanReference"/>: output and carried state, with the
/// state kept on the device between two calls. Each test reports itself as skipped when its device is absent.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class SelectiveScanGpuTests
{
    private const float Tol = 1e-4f;

    private readonly ITestOutputHelper _output;
    public SelectiveScanGpuTests(ITestOutputHelper output) => _output = output;

    // rows, frames, channels, states, gated, stateless
    [CudaTheory]
    [InlineData(1, 1, 4, 16, true, false)]    // one frame, the voice isolator's 16-state shape
    [InlineData(2, 7, 5, 16, true, false)]    // batched rows, gate on
    [InlineData(1, 5, 3, 5, false, true)]     // no gate, no carried state: starts from zero
    [InlineData(1, 4, 2, 256, true, false)]   // maximum state count: eight states per lane
    public void Cuda_MatchesReference(int rows, int frames, int channels, int states, bool gated, bool stateless)
    {
        using CudaBackend backend = new(0, PtxDir());
        Compare(backend, rows, frames, channels, states, gated, stateless);
        // Without the module the backend answers from the host reference and this test would pass without touching the device.
        CudaKernels? kernels = (CudaKernels?)typeof(CudaBackend).GetField("_kernels", BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(backend);
        Assert.True(kernels?.HasSelectiveScanKernel == true, "ssm_selective_scan_f32.ptx did not load; the device path did not run.");
    }

    [VulkanTheory]
    [InlineData(1, 1, 4, 16, true, false)]
    [InlineData(2, 7, 5, 16, true, false)]
    [InlineData(1, 5, 3, 5, false, true)]
    [InlineData(1, 4, 2, 256, true, false)]
    public void Vulkan_MatchesReference(int rows, int frames, int channels, int states, bool gated, bool stateless)
    {
        using VulkanBackend backend = new();
        Compare(backend, rows, frames, channels, states, gated, stateless);
    }

    /// <summary>Runs the device scan over the whole sequence in one call, then again over its second half from the device-resident
    /// state of the first half, and compares each call's output and the final state with the reference run in one piece.</summary>
    private static void Compare(IBackend device, int rows, int frames, int channels, int states, bool gated, bool stateless)
    {
        Random rng = new(rows * 7919 + frames * 131 + states);
        float[] uV = Floats(rng, rows * frames * channels, -1f, 1f);
        float[] deltaV = Floats(rng, rows * frames * channels, 0.05f, 1f);
        float[] aV = Floats(rng, channels * states, -2f, -0.1f);
        float[] bV = Floats(rng, rows * frames * states, -1f, 1f);
        float[] cV = Floats(rng, rows * frames * states, -1f, 1f);
        float[] dV = Floats(rng, channels, -0.5f, 0.5f);
        float[] zV = Floats(rng, rows * frames * channels, -2f, 2f);
        float[] state0 = stateless ? new float[rows * channels * states] : Floats(rng, rows * channels * states, -0.5f, 0.5f);
        int half = frames / 2;

        // Reference: one call over the whole sequence.
        float[] wantOut;
        float[] wantState;
        using (Tensor expectedOut = new(new TensorShape(rows, frames, channels), DType.F32))
        using (Tensor expectedState = Make(state0, rows, channels, states))
        using (Tensor u = Make(uV, rows, frames, channels))
        using (Tensor delta = Make(deltaV, rows, frames, channels))
        using (Tensor a = Make(aV, channels, states))
        using (Tensor b = Make(bV, rows, frames, states))
        using (Tensor c = Make(cV, rows, frames, states))
        using (Tensor d = Make(dV, channels))
        using (Tensor z = gated ? Make(zV, rows, frames, channels) : null!)
        {
            SelectiveScanReference.Apply(expectedOut, u, delta, a, b, c, d, z, stateless ? null : expectedState, rows, frames, channels, states);
            wantOut = expectedOut.AsReadOnlySpan<float>().ToArray();
            wantState = expectedState.AsReadOnlySpan<float>().ToArray();
        }

        // Device: two calls over the halves when there are two or more frames and a state is carried, the second continuing from the
        // state the first left on the device; one call otherwise (a stateless call keeps nothing, so it cannot be split).
        bool split = frames >= 2 && !stateless;
        (int Start, int Count)[] segments = split ? new[] { (0, half), (half, frames - half) } : new[] { (0, frames) };
        using Tensor actualState = Make(state0, rows, channels, states);
        using Tensor a0 = Make(aV, channels, states);
        using Tensor d0 = Make(dV, channels);
        foreach ((int start, int count) in segments)
        {
            using Tensor segOut = new(new TensorShape(rows, count, channels), DType.F32);
            using Tensor u = Make(Slice(uV, rows, frames, channels, start, count), rows, count, channels);
            using Tensor delta = Make(Slice(deltaV, rows, frames, channels, start, count), rows, count, channels);
            using Tensor b = Make(Slice(bV, rows, frames, states, start, count), rows, count, states);
            using Tensor c = Make(Slice(cV, rows, frames, states, start, count), rows, count, states);
            using Tensor z = gated ? Make(Slice(zV, rows, frames, channels, start, count), rows, count, channels) : null!;
            device.SelectiveScan(segOut, u, delta, a0, b, c, d0, z, stateless ? null : actualState, rows, count, channels, states);
            device.OffloadActivation(segOut);
            AssertClose(Slice(wantOut, rows, frames, channels, start, count), segOut.AsReadOnlySpan<float>().ToArray(),
                $"output[{start}..{start + count})");
        }
        device.OffloadActivation(actualState);
        if (!stateless)
            AssertClose(wantState, actualState.AsReadOnlySpan<float>().ToArray(), "state");
    }

    /// <summary>Frames [start, start + count) of every row of a <c>[rows, frames, width]</c> buffer, with width inferred from the length.</summary>
    private static float[] Slice(float[] source, int rows, int frames, int width, int start, int count)
    {
        float[] result = new float[rows * count * width];
        for (int r = 0; r < rows; r++)
            Array.Copy(source, ((long)r * frames + start) * width, result, (long)r * count * width, (long)count * width);
        return result;
    }

    private static void AssertClose(float[] want, float[] got, string what)
    {
        Assert.Equal(want.Length, got.Length);
        for (int i = 0; i < want.Length; i++)
            Assert.True(MathF.Abs(want[i] - got[i]) <= Tol, $"{what}[{i}] want {want[i]} got {got[i]}");
    }

    private static Tensor Make(float[] values, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    private static float[] Floats(Random rng, int n, float lo, float hi)
    {
        float[] values = new float[n];
        for (int i = 0; i < n; i++) values[i] = lo + (float)rng.NextDouble() * (hi - lo);
        return values;
    }

    private static string PtxDir()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate)) return candidate;
        }
        return Path.Combine(AppContext.BaseDirectory, "Ptx");
    }
}
