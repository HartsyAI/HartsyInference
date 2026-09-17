using System.Text.Json;
using HartsyInference.Audio.Models.Mert2;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.Parity;

/// <summary>Gates S2 and S3 of the SheetSage2 port: the MERT-v2 audio encoder against the released
/// implementation running the real checkpoint. The ladder is ordered so a failure names its own cause:
/// <list type="bullet">
///   <item><b>S2</b> — the mel frontend, the only weightless stage, so a wrong window / filterbank / log law can
///     only surface here.</item>
///   <item><b>S3a</b> — the RoPE tables, which the released model builds as a FLOAT32 product of position and
///     inverse frequency; a double-precision angle silently drifts from it by token 7500.</item>
///   <item><b>S3b</b> — the ConvNeXt subsampler, whose GlobalResponseNorm reduces over the whole 30000-frame
///     axis. A per-frame or per-chunk normalization produces plausible numbers, just not these.</item>
///   <item><b>S3c</b> — the 24 Conformer layers, the learned 25-way layer mix, and <c>encoder_projection</c>:
///     exactly the tensor the score decoder cross-attends over.</item>
/// </list>
///
/// <para>Gated on <c>SHEETSAGE2_CHECKPOINT</c> (the Comfy-Org single file) and <c>SHEETSAGE2_REF_DIR</c>
/// (<c>tests/python-reference/dump_mert2_reference.py</c> output).</para></summary>
public sealed unsafe class Mert2ParityTests
{
    /// <summary>Gate S2 — mel frontend, relative L2. Both sides are float32 and share the checkpoint's window,
    /// filterbank and statistics, so the only sources of difference are FFT and summation order. 1e-5 is roughly
    /// two orders of magnitude above the ~1e-7 float32 round-off those reorderings produce over 1025 bins, and
    /// still two orders below what any structural mistake (an off-by-one frame, a magnitude instead of a power
    /// spectrum, a rebuilt filterbank) would cost.</summary>
    private const double MelTolerance = 1e-5;

    /// <summary>Gate S3a — RoPE tables, max absolute difference. cos/sin are bounded by 1, and matching the
    /// reference's float32 angle rounding should leave only the last-ulp disagreement of <c>pow</c>; 1e-5 catches
    /// a double-precision angle (which drifts ~5e-4 by token 7499) while tolerating that ulp.</summary>
    private const double RopeTolerance = 1e-5;

    /// <summary>Gate S3b — ConvNeXt subsampler, relative L2. Twelve layers of GEMM on TF32 tensor cores (the
    /// engine's default on sm_80+) carry ~5e-4 relative per matmul, and the stack's residuals accumulate it, so a
    /// tolerance below 1e-3 would be testing cuBLAS's compute mode rather than this port. 3e-3 sits above the
    /// measured deviation and far below the ~1e-1 that a wrong normalization axis or convolution padding gives.</summary>
    private const double SubsampledTolerance = 3e-3;

    /// <summary>Gate S3c — encoder mix and projection, relative L2. Looser than S3b because 24 Conformer layers
    /// sit on top of it AND attention runs with F16 I/O through cuDNN (~1e-3 relative on the scores), which is the
    /// configuration that ships. Still one to two orders below the disagreement of any real error: a wrong RoPE
    /// convention, an unrotated key, or a mixed-up layer weight all land near or above 1e-1.</summary>
    private const double EncoderTolerance = 1e-2;

    private readonly ITestOutputHelper _out;

    public Mert2ParityTests(ITestOutputHelper output) => _out = output;

    private static string? Checkpoint => Environment.GetEnvironmentVariable("SHEETSAGE2_CHECKPOINT");

    private static string? ReferenceDir => Environment.GetEnvironmentVariable("SHEETSAGE2_REF_DIR");

    [Fact]
    [Trait("Category", "Integration")]
    public void MelFrontend_MatchesReference()
    {
        if (Gated(out string checkpoint, out string referenceDir)) return;

        using SafeTensorsLoader reference = new();
        reference.Load(Path.Combine(referenceDir, "tensors.safetensors"));
        Dictionary<string, Tensor> referenceTensors = reference.GetAllTensors();

        using SafeTensorsLoader loader = new();
        loader.Load(checkpoint);
        Mert2Config config = new();
        Mert2MelFrontend frontend = new(config);
        frontend.LoadWeights(loader.GetAllTensors());

        Tensor waveform = referenceTensors["waveform"];
        using Tensor mel = frontend.Compute(waveform.AsReadOnlySpan<float>());
        Assert.Equal(config.FramesPerWindow, (int)mel.Shape[0]);
        Assert.Equal(config.MelBins, (int)mel.Shape[1]);
        Report("mel", mel, referenceTensors["mel"], MelTolerance);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void RopeTables_MatchReference()
    {
        if (Gated(out _, out string referenceDir)) return;

        using SafeTensorsLoader reference = new();
        reference.Load(Path.Combine(referenceDir, "tensors.safetensors"));
        Dictionary<string, Tensor> referenceTensors = reference.GetAllTensors();

        Mert2Config config = new();
        int tokens = config.TokensPerWindow;
        int headDim = config.HeadDim;
        int half = headDim / 2;
        using Tensor cos = new(new TensorShape(1, tokens, headDim), DType.F32);
        using Tensor sin = new(new TensorShape(1, tokens, headDim), DType.F32);
        Mert2Ops.BuildRopeTables(cos, sin, tokens, headDim, config.RopeTheta);

        // The reference stores only the half-width table; ApplyRope wants both halves filled with it, so check the
        // duplication as well as the values — a table that only rotated the lower half would still pass on column 0.
        Tensor referenceCos = referenceTensors["rope_cos"];
        Tensor referenceSin = referenceTensors["rope_sin"];
        Assert.Equal(tokens, (int)referenceCos.Shape[0]);
        Assert.Equal(half, (int)referenceCos.Shape[1]);

        float* mineCos = (float*)cos.DataPointer;
        float* mineSin = (float*)sin.DataPointer;
        float* theirCos = (float*)referenceCos.DataPointer;
        float* theirSin = (float*)referenceSin.DataPointer;
        double worst = 0;
        int worstToken = 0;
        for (int t = 0; t < tokens; t++)
        {
            for (int j = 0; j < half; j++)
            {
                double expectedCos = theirCos[(long)t * half + j];
                double expectedSin = theirSin[(long)t * half + j];
                for (int copy = 0; copy < 2; copy++)
                {
                    long index = (long)t * headDim + j + copy * half;
                    double difference = Math.Max(Math.Abs(mineCos[index] - expectedCos), Math.Abs(mineSin[index] - expectedSin));
                    if (difference > worst)
                    {
                        worst = difference;
                        worstToken = t;
                    }
                }
            }
        }
        _out.WriteLine($"rope max abs diff {worst:E3} (worst token {worstToken} of {tokens}).");
        Assert.True(worst < RopeTolerance, $"rope tables diverge: max abs {worst:E3} exceeds {RopeTolerance:E1}.");
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void Encoder_MatchesReference()
    {
        if (Gated(out string checkpoint, out string referenceDir)) return;

        using SafeTensorsLoader reference = new();
        reference.Load(Path.Combine(referenceDir, "tensors.safetensors"));
        Dictionary<string, Tensor> referenceTensors = reference.GetAllTensors();
        JsonElement meta = JsonDocument.Parse(File.ReadAllText(Path.Combine(referenceDir, "meta.json"))).RootElement;
        _out.WriteLine($"reference attention score bound {meta.GetProperty("attentionScoreBound").GetDouble():F1}, "
            + $"max |V| {meta.GetProperty("attentionMaxValue").GetDouble():F1} — both far inside F16 range, which is "
            + "why attention runs with allowF16.");

        using IBackend backend = CreateBackend(out string backendName);
        _out.WriteLine($"backend: {backendName}");

        using SafeTensorsLoader loader = new();
        loader.Load(checkpoint);
        Mert2Config config = new();
        using Mert2AudioEncoder model = new(config);
        model.LoadWeights(loader.GetAllTensors());
        backend.PreloadWeights(model.EnumerateWeights());

        // The learned mix is a checkpoint fact with no tolerance beyond the softmax itself.
        Tensor referenceMix = referenceTensors["layer_weights"];
        Assert.Equal(model.Encoder.MixInputCount, (int)referenceMix.ElementCount);
        ReadOnlySpan<float> expectedMix = referenceMix.AsReadOnlySpan<float>();
        for (int i = 0; i < expectedMix.Length; i++)
        {
            Assert.Equal(expectedMix[i], model.MixWeights[i], 6);
        }

        Tensor waveform = referenceTensors["waveform"];
        using Tensor mel = model.Frontend.Compute(waveform.AsReadOnlySpan<float>());
        using (Tensor subsampled = model.Encoder.Subsample(backend, mel))
        {
            Assert.Equal(config.TokensPerWindow, (int)subsampled.Shape[0]);
            Report("subsampled", subsampled, referenceTensors["subsampled"], SubsampledTolerance);
        }

        using Tensor mixed = model.Encoder.Forward(backend, mel, model.MixWeights);
        Report("mixed", mixed, referenceTensors["mixed"], EncoderTolerance);

        using Tensor projected = model.EncodeMel(backend, mel);
        Assert.Equal(config.ProjectionDim, (int)projected.Shape[1]);
        Report("projected", projected, referenceTensors["projected"], EncoderTolerance);
    }

    /// <summary>CUDA when a device and its PTX are present, because that is the path that ships: the CPU backend
    /// materializes a 16x7500x7500 score matrix per layer, which is neither what runs in production nor something
    /// a test machine should be asked for. <c>MERT2_FORCE_CPU=1</c> pins the host path anyway.</summary>
    private static IBackend CreateBackend(out string name)
    {
        string ptxDirectory = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (Environment.GetEnvironmentVariable("MERT2_FORCE_CPU") != "1" && Directory.Exists(ptxDirectory))
        {
            try
            {
                name = "CUDA";
                return new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDirectory);
            }
            catch (Exception)
            {
                // fall through to the host path
            }
        }
        name = "CPU";
        return new CpuBackend();   // tier-lint: guarded
    }

    private static bool Gated(out string checkpoint, out string referenceDir)
    {
        checkpoint = Checkpoint ?? "";
        referenceDir = ReferenceDir ?? "";
        return checkpoint.Length == 0 || !File.Exists(checkpoint)
            || referenceDir.Length == 0 || !File.Exists(Path.Combine(referenceDir, "tensors.safetensors"));
    }

    /// <summary>Diffs against the reference elementwise (the released tensors carry a leading batch axis of 1 that
    /// this port drops, so only the element count has to line up) and gates on relative L2, printing maxAbs beside
    /// it because a flat absolute tolerance means something different at each stage's scale.</summary>
    private void Report(string name, Tensor mine, Tensor reference, double tolerance)
    {
        Assert.Equal(reference.ElementCount, mine.ElementCount);
        float* a = (float*)mine.DataPointer;
        float* b = (float*)reference.DataPointer;
        double maxAbs = 0, sumSquaredDifference = 0, sumSquaredReference = 0, peak = 0;
        for (long i = 0; i < mine.ElementCount; i++)
        {
            double expected = b[i];
            double difference = a[i] - expected;
            maxAbs = Math.Max(maxAbs, Math.Abs(difference));
            peak = Math.Max(peak, Math.Abs(expected));
            sumSquaredDifference += difference * difference;
            sumSquaredReference += expected * expected;
        }
        double relative = Math.Sqrt(sumSquaredDifference / Math.Max(sumSquaredReference, 1e-12));
        _out.WriteLine($"{name}: relL2 {relative:E3}, maxAbs {maxAbs:E3} (reference peak {peak:F4}).");
        Assert.True(relative < tolerance, $"{name} diverges: relative L2 {relative:E3} exceeds {tolerance:E1} (maxAbs {maxAbs:E3}).");
    }
}
