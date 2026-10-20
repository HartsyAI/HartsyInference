using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The tiny MOSS-Image checkpoint and reference outputs of <c>tests/python-reference/moss_image_numpy_reference.py</c>
/// (hidden 64, 2 heads, 3 joint and 2 text blocks, 16 latent channels, 48-wide text features), loaded through
/// <see cref="MossImageCheckpointConverter"/> under a ComfyUI prefix into a ready transformer. <see cref="Tensors"/>
/// holds the raw file, inputs and expected outputs included.</summary>
public sealed class MossImageFixture : IDisposable
{
    /// <summary>Packed latent rows of the fixture's inputs.</summary>
    public const int Height = 3;

    /// <summary>Packed latent columns of the fixture's inputs.</summary>
    public const int Width = 4;

    private static readonly string Path_ = Path.Combine(AppContext.BaseDirectory, "Fixtures", "MossImage", "moss_image_tiny.safetensors");
    private readonly SafeTensorsLoader _loader = new();

    public MossImageFixture()
    {
        _loader.Load(Path_);
        Tensors = _loader.GetAllTensors();
        Dictionary<string, Tensor> raw = new(StringComparer.Ordinal);
        foreach ((string key, Tensor value) in Tensors)
        {
            if (key.StartsWith("weights.", StringComparison.Ordinal))
                raw["model.diffusion_model." + key["weights.".Length..]] = value;
        }
        Weights = MossImageCheckpointConverter.ConvertTransformer(raw);
        Transformer = new MossImageTransformer(MossImageConfig.FromWeights(Weights));
        Transformer.LoadWeights(Weights);
    }

    /// <summary>Every tensor in the fixture file.</summary>
    public IReadOnlyDictionary<string, Tensor> Tensors { get; }

    /// <summary>The converted transformer weights.</summary>
    public IReadOnlyDictionary<string, Tensor> Weights { get; }

    /// <summary>The loaded transformer.</summary>
    public MossImageTransformer Transformer { get; }

    /// <summary>The edit cases' two reference latents as borrowed tokens (grids 2×2 and 1×3).</summary>
    public ReferenceLatentTokens[] References() =>
    [
        new ReferenceLatentTokens(Tensors["inputs.ref0"], 2, 2),
        new ReferenceLatentTokens(Tensors["inputs.ref1"], 1, 3),
    ];

    /// <inheritdoc/>
    public void Dispose()
    {
        Transformer.Dispose();
        _loader.Dispose();
    }
}
