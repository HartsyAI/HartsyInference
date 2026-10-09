using HartsyInference.Core.Moe;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>
/// The expert FFN kernels (<see cref="CudaExpertKernels"/>) against <see cref="ExpertProgramReference"/> on a real device, for
/// every <see cref="ExpertProgram"/> variant. No backend is built, so these run on sm_75 cards too.
/// </summary>
[Collection("CudaSerial")]
public sealed unsafe class CudaExpertKernelTests
{
    private const int Hidden = 64;
    private const int Intermediate = 32;
    private const int Rows = 6;

    private readonly ITestOutputHelper _output;

    public CudaExpertKernelTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }

    private static float[] Fill(Random rng, int count, float scale)
    {
        float[] values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * scale;
        return values;
    }

    private static ulong Upload(float[] host)
    {
        ulong device = CudaMemory.Allocate((nuint)(host.Length * sizeof(float)));
        fixed (float* p = host) CudaMemory.CopyHostToDevice(device, p, (nuint)(host.Length * sizeof(float)));
        return device;
    }

    /// <summary>Runs <paramref name="program"/> on the device and on the reference, and returns the measured errors.</summary>
    private (double MaxAbs, double ScaleRelative, int Clamped) Measure(CudaExpertKernels kernels, CudaContext context, ExpertProgram program,
        int seed)
    {
        Random rng = new(seed);
        float[] gate = Fill(rng, Intermediate * Hidden, 0.25f);
        float[] up = Fill(rng, Intermediate * Hidden, 0.25f);
        float[] down = Fill(rng, Hidden * Intermediate, 0.25f);
        float[] x = Fill(rng, Rows * Hidden, 1.0f);
        F32ExpertWeights weights = new(Hidden, Intermediate, gate, up, down);

        float[] reference = new float[Rows * Hidden];
        ExpertProgramReference.Apply(program, weights, x, Rows, reference);

        int clamped = 0;
        if (program.IsClamped)
        {
            for (int r = 0; r < Rows; r++)
            {
                // Pre-activations are gate·x and up·x; count how many the clamp actually moves.
                for (int i = 0; i < Intermediate; i++)
                {
                    float g = 0f, u = 0f;
                    for (int j = 0; j < Hidden; j++)
                    {
                        g += gate[i * Hidden + j] * x[r * Hidden + j];
                        u += up[i * Hidden + j] * x[r * Hidden + j];
                    }
                    (float cg, float cu) = program.Clamp(g, u);
                    if (cg != g || cu != u) clamped++;
                }
            }
        }

        ulong w1 = Upload(gate), w3 = Upload(up), w2 = Upload(down), xd = Upload(x);
        ulong gd = CudaMemory.Allocate((nuint)(Rows * Intermediate * sizeof(float)));
        ulong ud = CudaMemory.Allocate((nuint)(Rows * Intermediate * sizeof(float)));
        ulong hd = CudaMemory.Allocate((nuint)(Rows * Intermediate * sizeof(float)));
        ulong yd = CudaMemory.Allocate((nuint)(Rows * Hidden * sizeof(float)));
        float[] output = new float[Rows * Hidden];
        try
        {
            kernels.RunExpert(new CudaExpertKernels.Buffers(w1, w3, w2, xd, gd, ud, hd, yd), Rows, Hidden, Intermediate, program, 0);
            context.Synchronize();
            fixed (float* p = output) CudaMemory.CopyDeviceToHost(p, yd, (nuint)(output.Length * sizeof(float)));
        }
        finally
        {
            foreach (ulong buffer in new[] { w1, w3, w2, xd, gd, ud, hd, yd }) CudaMemory.Free(buffer);
        }

        double maxAbs = 0, maxRef = 0;
        for (int i = 0; i < output.Length; i++)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(output[i] - reference[i]));
            maxRef = Math.Max(maxRef, Math.Abs(reference[i]));
        }
        return (maxAbs, maxRef > 0 ? maxAbs / maxRef : maxAbs, clamped);
    }

    private void Check(ExpertProgram program, string name, int seed)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaContext context = new(0);
        using CudaExpertKernels kernels = new(PtxDir());
        (double maxAbs, double scaleRelative, int clamped) = Measure(kernels, context, program, seed);
        _output.WriteLine($"{name}: device={context.DeviceName} sm={context.Sm} maxAbs={maxAbs:E3} "
            + $"maxAbs/maxRef={scaleRelative:E3} clampedPreActivations={clamped}");
        Assert.True(maxAbs <= ToleranceAbs, $"{name}: max abs error {maxAbs:E3} exceeds {ToleranceAbs:E1}.");
    }

    /// <summary>Absolute tolerance on outputs whose magnitude is about 1. Set from the measured error (see the PR), not guessed.</summary>
    private const double ToleranceAbs = 1e-5;

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void Swiglu_MatchesReference()
    {
        Check(ExpertProgram.Swiglu, nameof(Swiglu_MatchesReference), seed: 1);
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void GeGlu_MatchesReference()
    {
        Check(ExpertProgram.GeGlu, nameof(GeGlu_MatchesReference), seed: 2);
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void SwigluClamped_MatchesReference_AndTheClampIsExercised()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaContext context = new(0);
        using CudaExpertKernels kernels = new(PtxDir());
        (double maxAbs, double scaleRelative, int clamped) = Measure(kernels, context, ExpertProgram.SwigluClamped(0.5f), seed: 3);
        _output.WriteLine($"SwigluClamped(0.5): device={context.DeviceName} sm={context.Sm} maxAbs={maxAbs:E3} "
            + $"maxAbs/maxRef={scaleRelative:E3} clampedPreActivations={clamped}");
        Assert.True(clamped > 0, "The clamp did not move any pre-activation, so the clamped path was not tested.");
        Assert.True(maxAbs <= ToleranceAbs, $"SwigluClamped: max abs error {maxAbs:E3} exceeds {ToleranceAbs:E1}.");
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void Relu_MatchesReference()
    {
        Check(new ExpertProgram(ExpertActivation.Relu, float.PositiveInfinity, float.NegativeInfinity, float.PositiveInfinity),
            nameof(Relu_MatchesReference), seed: 4);
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void ReluSquared_MatchesReference()
    {
        Check(new ExpertProgram(ExpertActivation.ReluSquared, float.PositiveInfinity, float.NegativeInfinity, float.PositiveInfinity),
            nameof(ReluSquared_MatchesReference), seed: 5);
    }
}
