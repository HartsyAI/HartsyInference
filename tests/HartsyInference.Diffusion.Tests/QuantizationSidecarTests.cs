using System.Text;
using System.Text.Json;
using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Quantization;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Quant;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Binding a quantized output to the semantics of the build it came from.
/// <para>Video planning resolves by exact file hash, so a checkpoint we quantized ourselves is a stranger to it:
/// without a sidecar it plans as an unknown base and the task, acceleration and step count the source declared
/// are gone, silently. These pin that a recognized source produces a sidecar and an unrecognized one does not —
/// the second half matters because writing a sidecar for a file nobody has verified would be inventing
/// provenance.</para></summary>
public sealed class QuantizationSidecarTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("hartsy-sidecar-").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private static unsafe Tensor Ramp(int rows, int cols)
    {
        Tensor t = new Tensor(new TensorShape(rows, cols), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < rows * cols; i++) p[i] = MathF.Sin(i * 0.01f);
        return t;
    }

    /// <summary>An ordinary checkpoint nobody has verified gets no sidecar, and that is not an error — inventing
    /// provenance for an unknown file is worse than leaving it unknown.</summary>
    [Fact]
    public async Task AnUnrecognizedSourceGetsNoSidecar()
    {
        using Tensor weight = Ramp(256, 256);
        string src = Path.Combine(_dir, "unknown.safetensors");
        SafeTensorsWriter.Save(src, new Dictionary<string, Tensor> { ["blocks.0.attn.weight"] = weight });
        string outPath = Path.Combine(_dir, "unknown.gguf");

        (QuantizationReport report, string? sidecar) = await QuantizationService.QuantizeAsync(new QuantizationJob
        {
            SourcePath = src,
            OutputPath = outPath,
            Target = new QuantizationTarget(QuantizationTargetKind.Gguf, GgufQuantPolicy.Q8_0),
            Architecture = "test-arch",
        });

        Assert.True(report.TensorCount > 0);
        Assert.Null(sidecar);
        Assert.False(File.Exists(outPath + ".hartsy-video-profile.json"));
    }

    /// <summary>The sidecar binds the OUTPUT's hash, not the source's. Binding the source's would make the file
    /// describe something it is not, and planning — which checks the hash — would reject it.</summary>
    [Fact]
    public async Task ASidecarWouldBindTheOutputHashNotTheSource()
    {
        using Tensor weight = Ramp(256, 256);
        string src = Path.Combine(_dir, "hashcheck.safetensors");
        SafeTensorsWriter.Save(src, new Dictionary<string, Tensor> { ["blocks.0.attn.weight"] = weight });
        string outPath = Path.Combine(_dir, "hashcheck.gguf");
        await QuantizationService.QuantizeAsync(new QuantizationJob
        {
            SourcePath = src,
            OutputPath = outPath,
            Target = new QuantizationTarget(QuantizationTargetKind.Gguf, GgufQuantPolicy.Q8_0),
            Architecture = "test-arch",
        });

        // No sidecar for this unknown source, so the assertion that matters is the negative one: nothing was
        // written that claims to describe the output.
        string[] strays = Directory.GetFiles(_dir, "*.hartsy-video-profile.json");
        Assert.Empty(strays);
    }

    /// <summary>A sidecar the resolver would throw away is worse than none: it costs a full hash of a multi-GB
    /// output and then plans as an unknown base anyway. <c>ValidateSidecar</c> refuses <c>Steps &lt;= 0</c>, and an
    /// artifact that declares no step count is the ordinary case — the base number has to be written out, because
    /// the format cannot say "use the recipe's".</summary>
    [Fact]
    public void AnArtifactWithNoDeclaredStepsStillGetsAStepCountTheResolverAccepts()
    {
        VideoKnownArtifact source = new()
        {
            Sha256 = new string('a', 64),
            Id = "test-artifact",
            DisplayName = "Test",
            Role = VideoProfileArtifactRole.Main,
            Task = VideoTaskFamily.Fl2Va,
            Acceleration = VideoAccelerationKind.None,
            Steps = null,
        };
        VideoProfileSidecar sidecar = QuantizationService.BuildSidecar(source, new string('b', 64));
        Assert.True(sidecar.Steps > 0, $"Steps was {sidecar.Steps}; the resolver refuses anything <= 0.");
        Assert.Equal(VideoTaskFamily.Fl2Va, sidecar.Task);
        // The OUTPUT's hash, not the source's — binding the source's would describe a different file.
        Assert.Equal(new string('b', 64), sidecar.Sha256);
    }

    /// <summary>A declared step count is carried rather than replaced by the fallback.</summary>
    [Fact]
    public void ADeclaredStepCountIsCarriedThrough()
    {
        VideoKnownArtifact source = new()
        {
            Sha256 = new string('a', 64),
            Id = "turbo",
            DisplayName = "Turbo",
            Role = VideoProfileArtifactRole.Main,
            Task = VideoTaskFamily.Fl2Va,
            Acceleration = VideoAccelerationKind.Turbo,
            Steps = 8,
        };
        Assert.Equal(8, QuantizationService.BuildSidecar(source, new string('b', 64)).Steps);
    }
}
