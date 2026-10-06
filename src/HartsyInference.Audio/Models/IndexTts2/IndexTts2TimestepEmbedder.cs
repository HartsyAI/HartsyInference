using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.IndexTts2;

/// <summary>IndexTTS-2's S2Mel DiT's <c>TimestepEmbedder</c>: sinusoidal timestep embedding → 2-layer SiLU MLP.
/// Two independent instances exist in the real checkpoint (<c>t_embedder</c> for the DiT's own hidden size,
/// <c>t_embedder2</c> for the WaveNet final stage's) — same class, different weights. The sinusoid's inverse
/// frequencies are loaded directly from the checkpoint's own <c>freqs</c> buffer rather than recomputed, to
/// avoid any transcription risk in the formula (confirmed present in the real <c>s2mel.pth</c>:
/// <c>t_embedder.freqs [128]</c>). Real source (<c>diffusion_transformer.py:TimestepEmbedder</c>) — note the
/// **cos-then-sin** concatenation order and the division by <c>half</c> (not <c>half-1</c>), both different
/// from this codebase's existing F5-TTS <c>F5TimestepEmbedding</c> (sin-then-cos, <c>half-1</c>) — a different
/// model's sinusoidal convention, not a bug to reconcile:
/// <code>
/// half = frequency_embedding_size // 2
/// freqs = exp(-log(10000) * arange(0, half) / half)      # buffer: self.freqs
/// args = 1000 * t[:, None] * freqs[None]
/// embedding = cat([cos(args), sin(args)], dim=-1)
/// t_emb = self.mlp(embedding)                             # Linear → SiLU → Linear
/// </code></summary>
internal sealed unsafe class IndexTts2TimestepEmbedder
{
    private readonly int _hidden, _freqEmbedSize;
    private Tensor? _freqs;   // [freqEmbedSize / 2]
    private Tensor? _mlp0W, _mlp0B, _mlp2W, _mlp2B;

    public IndexTts2TimestepEmbedder(int hidden, int freqEmbedSize = 256)
    {
        _hidden = hidden;
        _freqEmbedSize = freqEmbedSize;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _freqs = WhisperOps.EnsureF32(w[$"{prefix}.freqs"]);
        _mlp0W = WhisperOps.EnsureF32(w[$"{prefix}.mlp.0.weight"]);
        _mlp0B = WhisperOps.EnsureF32(w[$"{prefix}.mlp.0.bias"]);
        _mlp2W = WhisperOps.EnsureF32(w[$"{prefix}.mlp.2.weight"]);
        _mlp2B = WhisperOps.EnsureF32(w[$"{prefix}.mlp.2.bias"]);
    }

    /// <summary>Returns the <c>[1, hidden]</c> timestep embedding for scalar flow time <paramref name="t"/>.</summary>
    public Tensor Forward(IBackend backend, float t)
    {
        int half = _freqEmbedSize / 2;
        Tensor emb = new(new TensorShape(1, _freqEmbedSize), DType.F32);
        float* ep = (float*)emb.DataPointer;
        float* fp = (float*)_freqs!.DataPointer;
        const float scale = 1000f;
        for (int i = 0; i < half; i++)
        {
            float angle = scale * t * fp[i];
            ep[i] = MathF.Cos(angle);
            ep[half + i] = MathF.Sin(angle);
        }

        Tensor h1 = new(new TensorShape(1, _hidden), DType.F32);
        backend.Linear(h1, emb, _mlp0W!, _mlp0B);
        emb.Dispose();

        Tensor activated = new(h1.Shape, DType.F32);
        backend.Silu(activated, h1);
        h1.Dispose();

        Tensor h2 = new(new TensorShape(1, _hidden), DType.F32);
        backend.Linear(h2, activated, _mlp2W!, _mlp2B);
        activated.Dispose();
        return h2;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_freqs, _mlp0W, _mlp0B, _mlp2W, _mlp2B];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
