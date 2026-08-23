using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Music;

/// <summary>Sana DCAE residual block (diffusers <c>autoencoder_dc.ResBlock</c>): 3×3 conv → SiLU → 3×3 conv (no bias)
/// → channelwise RMSNorm → + residual. Channel count is preserved.</summary>
public sealed unsafe class ResBlock2d
{
    private Tensor? _conv1W, _conv1B, _conv2W, _normW, _normB;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _conv1W = w[$"{p}.conv1.weight"]; w.TryGetValue($"{p}.conv1.bias", out _conv1B);
        _conv2W = w[$"{p}.conv2.weight"];
        _normW = TensorCasts.LoadF32(w, $"{p}.norm.weight"); w.TryGetValue($"{p}.norm.bias", out _normB);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _conv1W, _conv1B, _conv2W, _normW, _normB })
            if (t is not null) yield return t;
    }

    public Tensor Forward(IBackend backend, Tensor x)
    {
        int c = (int)x.Shape[1];
        Tensor h1 = MusicDcaeDecoder.Conv3x3(backend, x, _conv1W!, _conv1B, c);
        backend.Silu(h1, h1);
        Tensor h2 = MusicDcaeDecoder.Conv3x3(backend, h1, _conv2W!, null, c);
        h1.Dispose();
        MusicDcaeDecoder.RmsNormChannels(h2, _normW!, _normB);

        float* hp = (float*)h2.DataPointer;
        float* xp = (float*)x.DataPointer;
        long n = x.Shape.ElementCount;
        for (long i = 0; i < n; i++) hp[i] += xp[i];
        return h2;
    }
}
