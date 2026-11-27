using System.Text.Json;
using HartsyInference.Diffusion.Models.Duo;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Duo;

/// <summary>The experimental run folder of the CLI route: a complete folder reads back the values written to it, and a folder that does not match the
/// sample is refused before a run starts. Synthetic values only; no Intent model runs.</summary>
public sealed class DuoRunFilesTests : IDisposable
{
    private const int TextWidth = 5, VideoWidth = 96, AudioWidth = 32;
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "duo-run-" + Guid.NewGuid().ToString("N"));

    public DuoRunFilesTests() => Directory.CreateDirectory(_directory);

    public void Dispose()
    {
        if (Directory.Exists(_directory))
        {
            Directory.Delete(_directory, recursive: true);
        }
    }

    [Fact]
    public void ACompleteFolderReadsBackTheValuesWritten()
    {
        DuoSampleInput sample = Sample(blocks: 2);
        float[] shared = Write("shared.f32", Values(1, 3 * TextWidth));
        float[] reference = Write("reference.f32", Values(2, 4 * VideoWidth));
        float[][] body = Enumerable.Range(0, 2).Select(b => Write($"body{b}.f32", Values(10 + b, 2 * TextWidth))).ToArray();
        float[][] video = Enumerable.Range(0, 2).Select(b => Write($"video{b}.f32", Values(20 + b, 20 * VideoWidth))).ToArray();
        float[][] audio = Enumerable.Range(0, 2).Select(b => Write($"audio{b}.f32", Values(30 + b, 6 * AudioWidth))).ToArray();
        Manifest(blocks: 2, shared: "shared.f32", reference: "reference.f32", steps: 4, shift: 1.5);

        DuoRun run = DuoRunFiles.Read(_directory, sample, TextWidth, VideoWidth, AudioWidth);

        Assert.Equal(shared, run.SharedText);
        Assert.Equal(reference, run.ReferenceVideo);
        Assert.Equal(body, run.BodyText);
        Assert.Equal(video, run.NoiseVideo);
        Assert.Equal(audio, run.NoiseAudio);
        Assert.Equal(4, run.Steps);
        Assert.Equal(1.5, run.Shift);
    }

    [Fact]
    public void AFileOfTheWrongLengthIsRefused()
    {
        DuoSampleInput sample = Sample(blocks: 1);
        Write("shared.f32", Values(1, 3 * TextWidth));
        Write("reference.f32", Values(2, 4 * VideoWidth));
        Write("body0.f32", Values(3, 2 * TextWidth + 1));
        Write("video0.f32", Values(4, 20 * VideoWidth));
        Write("audio0.f32", Values(5, 6 * AudioWidth));
        Manifest(blocks: 1, shared: "shared.f32", reference: "reference.f32", steps: 4, shift: 1.0);

        Assert.Throws<InvalidDataException>(() => DuoRunFiles.Read(_directory, sample, TextWidth, VideoWidth, AudioWidth));
    }

    [Fact]
    public void AManifestWithTheWrongBlockCountIsRefused()
    {
        DuoSampleInput sample = Sample(blocks: 2);
        Write("shared.f32", Values(1, 3 * TextWidth));
        Write("reference.f32", Values(2, 4 * VideoWidth));
        Manifest(blocks: 1, shared: "shared.f32", reference: "reference.f32", steps: 4, shift: 1.0);

        Assert.Throws<InvalidDataException>(() => DuoRunFiles.Read(_directory, sample, TextWidth, VideoWidth, AudioWidth));
    }

    private float[] Write(string name, float[] values)
    {
        byte[] bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(Path.Combine(_directory, name), bytes);
        return values;
    }

    private void Manifest(int blocks, string shared, string reference, int steps, double shift)
    {
        object manifest = new
        {
            steps,
            shift,
            sharedText = shared,
            referenceVideo = reference,
            blocks = Enumerable.Range(0, blocks).Select(b => new
            {
                bodyText = $"body{b}.f32",
                noiseVideo = $"video{b}.f32",
                noiseAudio = $"audio{b}.f32",
            }).ToArray(),
        };
        File.WriteAllText(Path.Combine(_directory, "manifest.json"), JsonSerializer.Serialize(manifest));
    }

    private static DuoSampleInput Sample(int blocks) => new()
    {
        SampleId = 5,
        SharedDescriptionRows = 2,
        SharedCueRows = 1,
        Reference = new DuoGrid(1, 4, 4),
        Blocks = Enumerable.Range(0, blocks).Select(_ => new DuoBlockInput
        {
            StartMs = 0,
            BodyTextRows = 2,
            Video = new DuoGrid(5, 4, 4),
            AudioSteps = 3,
            FrameStartsMs = [0, 100, 200, 300, 400],
        }).ToArray(),
        CleanBlocks = 0,
        NoisyBlocks = [0],
    };

    private static float[] Values(int seed, int count)
    {
        float[] data = new float[count];
        for (int i = 0; i < count; i++)
        {
            data[i] = (float)Math.Sin((seed * 1.7) + (i * 0.37));
        }
        return data;
    }
}
