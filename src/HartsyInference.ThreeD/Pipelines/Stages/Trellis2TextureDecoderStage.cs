using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.TrellisV2;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>
/// TRELLIS.2 texture decoding: the guided <see cref="ShapeVaeDecoder"/> over the texture latent, then upstream's <c>* 0.5 + 0.5</c> that
/// maps the head from [-1, 1] to [0, 1]. A checkpoint path is read for the call and released afterwards; a decoder passed in
/// stays resident and is owned by the stage.
/// </summary>
public sealed class Trellis2TextureDecoderStage : ITrellis2TextureDecoderStage, IDisposable
{
    private readonly string? _path;
    private readonly ShapeVaeDecoder? _resident;
    private bool _disposed;

    /// <param name="path">The <c>tex_dec_next_dc_f16c32_fp16.safetensors</c> file, read per decode.</param>
    public Trellis2TextureDecoderStage(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        _path = path;
    }

    /// <param name="decoder">A texture decoder (<see cref="ShapeVaeDecoderConfig.Texture"/>); this stage owns it.</param>
    public Trellis2TextureDecoderStage(ShapeVaeDecoder decoder) => _resident = decoder ?? throw new ArgumentNullException(nameof(decoder));

    /// <inheritdoc/>
    public SparseTensor Decode(IBackend backend, SparseTensor textureLatent, ShapeVaeSubdivisions guides)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(textureLatent);
        ObjectDisposedException.ThrowIf(_disposed, this);
        SparseTensor head;
        if (_resident is not null)
        {
            head = _resident.DecodeGuided(backend, textureLatent, guides);
        }
        else
        {
            using SafeTensorsLoader loader = new();
            loader.Load(_path!);
            using ShapeVaeDecoder decoder = new(ShapeVaeDecoderConfig.Texture, loader.GetAllTensors());
            head = decoder.DecodeGuided(backend, textureLatent, guides);
        }

        Span<float> values = head.Feats.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
            values[i] = values[i] * 0.5f + 0.5f;
        return head;
    }

    /// <summary>Releases the resident decoder, if any.</summary>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _resident?.Dispose();
    }
}
