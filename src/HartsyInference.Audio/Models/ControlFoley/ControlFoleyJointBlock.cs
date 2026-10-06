using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>Port of <c>JointBlock</c>: the latent, audio, CLIP and text streams keep their own projections and
/// modulation but attend jointly over the concatenation <c>[latent, audio, clip, text]</c>. In a pre-only block the
/// three condition streams are not updated (nothing consumes them afterwards).</summary>
internal sealed unsafe class ControlFoleyJointBlock
{
    private readonly ControlFoleyBlock _latent, _clip, _text, _audio;

    private ControlFoleyJointBlock(ControlFoleyBlock latent, ControlFoleyBlock clip, ControlFoleyBlock text, ControlFoleyBlock audio)
    {
        _latent = latent;
        _clip = clip;
        _text = text;
        _audio = audio;
    }

    internal static ControlFoleyJointBlock Load(IReadOnlyDictionary<string, Tensor> w, string prefix, int heads, bool preOnly)
        => new(ControlFoleyBlock.Load(w, $"{prefix}.latent_block", heads, false),
            ControlFoleyBlock.Load(w, $"{prefix}.clip_block", heads, preOnly),
            ControlFoleyBlock.Load(w, $"{prefix}.text_block", heads, preOnly),
            ControlFoleyBlock.Load(w, $"{prefix}.audio_block", heads, preOnly));

    internal IEnumerable<Tensor> Weights() => _latent.Weights().Concat(_clip.Weights()).Concat(_text.Weights()).Concat(_audio.Weights());

    /// <summary>Updates the four streams in place.</summary>
    internal void Forward(IBackend backend, Tensor latent, Tensor clip, Tensor audio, Tensor text, Tensor globalC, Tensor extendedC,
        ControlFoleyRope latentRope, ControlFoleyRope clipRope)
    {
        using ControlFoleyBlock.PreAttention x = _latent.Pre(backend, latent, extendedC, latentRope);
        using ControlFoleyBlock.PreAttention c = _clip.Pre(backend, clip, globalC, clipRope);
        using ControlFoleyBlock.PreAttention t = _text.Pre(backend, text, globalC, null);
        using ControlFoleyBlock.PreAttention a = _audio.Pre(backend, audio, globalC, null);

        int latentLen = (int)latent.Shape[1], audioLen = (int)audio.Shape[1], clipLen = (int)clip.Shape[1], textLen = (int)text.Shape[1];
        using Tensor q = Concat(x.Q, a.Q, c.Q, t.Q);
        using Tensor k = Concat(x.K, a.K, c.K, t.K);
        using Tensor v = Concat(x.V, a.V, c.V, t.V);
        using Tensor attn = ControlFoleyBlock.Attend(backend, q, k, v);

        using (Tensor part = ControlFoleyBlock.TokenSlice(attn, 0, latentLen))
        {
            _latent.Post(backend, latent, part, x.Modulation);
        }

        if (_clip.PreOnly)
        {
            return;
        }

        using (Tensor part = ControlFoleyBlock.TokenSlice(attn, latentLen, audioLen))
        {
            _audio.Post(backend, audio, part, a.Modulation);
        }

        using (Tensor part = ControlFoleyBlock.TokenSlice(attn, latentLen + audioLen, clipLen))
        {
            _clip.Post(backend, clip, part, c.Modulation);
        }

        using (Tensor part = ControlFoleyBlock.TokenSlice(attn, latentLen + audioLen + clipLen, textLen))
        {
            _text.Post(backend, text, part, t.Modulation);
        }
    }

    private static Tensor Concat(Tensor first, Tensor second, Tensor third, Tensor fourth)
    {
        Tensor[] parts = [first, second, third, fourth];
        int b = (int)first.Shape[0], h = (int)first.Shape[1], hd = (int)first.Shape[3];
        int total = parts.Sum(p => (int)p.Shape[2]);
        Tensor o = new(new TensorShape(b, h, total, hd), DType.F32);
        float* op = (float*)o.DataPointer;
        for (int bh = 0; bh < b * h; bh++)
        {
            long at = (long)bh * total * hd;
            foreach (Tensor p in parts)
            {
                long bytes = p.Shape[2] * hd * 4L;
                Buffer.MemoryCopy((float*)p.DataPointer + (long)bh * p.Shape[2] * hd, op + at, bytes, bytes);
                at += p.Shape[2] * hd;
            }
        }

        return o;
    }
}
