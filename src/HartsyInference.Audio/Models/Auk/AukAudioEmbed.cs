using HartsyInference.Audio.Models.F5Tts;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Auk;

/// <summary>Shared latent embedding: <c>Linear(latent, dim)</c> then <c>x + ConvPos(x)</c>, applied separately to the reference and the noisy latent.</summary>
public sealed class AukAudioEmbed
{
    private readonly AukConfig _config;
    private readonly F5ConvPosEmbed _convPos;
    private Tensor? _linW, _linB;

    public AukAudioEmbed(AukConfig config)
    {
        _config = config;
        _convPos = new F5ConvPosEmbed(new F5TtsConfig
        {
            Dim = config.Dim,
            ConvPosKernel = config.ConvPosKernel,
            ConvPosGroups = config.ConvPosGroups,
        });
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        int dim = _config.Dim;
        long convIn = dim / _config.ConvPosGroups;
        string conv = $"{prefix}.conv_pos_embed.conv1d";
        foreach (string i in new[] { "0", "2" })
        {
            AukOps.Require(weights, $"{conv}.{i}.weight", dim, convIn, _config.ConvPosKernel);
            AukOps.Require(weights, $"{conv}.{i}.bias", dim);
        }
        _linW = AukOps.Take(weights, $"{prefix}.linear.weight", dim, _config.LatentDim);
        _linB = AukOps.Take(weights, $"{prefix}.linear.bias", dim);
        _convPos.LoadWeights(weights, $"{prefix}.conv_pos_embed");
    }

    /// <summary>Input <c>[1, n, latent]</c>, output <c>[1, n, dim]</c>; the caller owns the result.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, int n)
    {
        Tensor lin = WhisperOps.ProjectLinear(backend, latent, _linW!, _linB, 1, n, _config.LatentDim, _config.Dim);
        Tensor pos = _convPos.Forward(backend, lin, n, _config.Dim);
        Tensor result = new(lin.Shape, DType.F32);
        backend.Add(result, lin, pos);
        lin.Dispose(); pos.Dispose();
        return result;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_linW is not null) yield return _linW;
        if (_linB is not null) yield return _linB;
        foreach (Tensor t in _convPos.EnumerateWeights()) yield return t;
    }
}
