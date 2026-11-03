using HartsyInference.Cli.Dispatch;
using HartsyInference.Cli.Infra;
using HartsyInference.Engine;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Cli.Tests;

/// <summary>A layered image result on the CLI: the composite is the main PNG and each layer lands next to it as
/// <c>&lt;stem&gt;.layer-NN.png</c>, bottom first, RGBA with the layer's own alpha. The encoder is deterministic, so
/// each file is compared with the RGBA encoding of the layer it should hold.</summary>
public sealed class LayeredArtifactTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "hartsy-layered-" + Path.GetRandomFileName());

    public void Dispose()
    {
        if (Directory.Exists(_dir))
            Directory.Delete(_dir, recursive: true);
    }

    [Fact]
    public void Layers_AreWrittenNextToTheCompositeInOrder()
    {
        ImageLayer bottom = new ImageLayer { Index = 0, Width = 1, Height = 1, Rgb = [10, 20, 30], Alpha = [255], Name = "background" };
        ImageLayer top = new ImageLayer { Index = 1, Width = 1, Height = 1, Rgb = [200, 100, 50], Alpha = [77] };
        ImageResult result = new ImageResult { Rgb = [9, 9, 9], Alpha = [255], Width = 1, Height = 1, Layers = [bottom, top] };
        GeneratedArtifact artifact = new GeneratedArtifact
        {
            Kind = ArtifactKind.Image,
            FileBytes = PngEncoder.Encode(result.Rgb, result.Alpha, 1, 1),
            Extension = "png",
            Companions = GenerationDispatch.LayerCompanions(result),
        };

        string main = ArtifactWriter.Write(artifact, _dir, "layered test", force: false)!;

        string stem = Path.Combine(_dir, Path.GetFileNameWithoutExtension(main));
        Assert.Equal(PngEncoder.Encode(bottom.Rgb, bottom.Alpha, 1, 1), File.ReadAllBytes($"{stem}.layer-00.png"));
        Assert.Equal(PngEncoder.Encode(top.Rgb, top.Alpha, 1, 1), File.ReadAllBytes($"{stem}.layer-01.png"));
        Assert.Equal(artifact.FileBytes, File.ReadAllBytes(main));
        Assert.False(File.Exists($"{stem}.layer-02.png"));
    }

    [Fact]
    public void SingleImage_HasNoCompanions()
    {
        Assert.Empty(GenerationDispatch.LayerCompanions(new ImageResult { Rgb = [0, 0, 0], Width = 1, Height = 1 }));
    }
}
