using System.Collections.Generic;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Models.Music;

/// <summary>MiniMax-H3 audio VAE decoder (<c>audio_vae/model.safetensors</c>): <c>dec_in_proj</c> (kernel-1 conv,
/// 32 → 2048) followed by a BigVGAN vocoder — <c>conv_pre</c> (k7) → 7 × [<c>ConvTranspose1d</c> upsample (rates
/// 5,5,2,2,2,2,2; channels halve each stage) → mean of 3 anti-aliased SnakeBeta AMP resblocks (kernels 3/7/11,
/// dilations 1/3/5)] → <c>activation_post</c> → <c>conv_post</c> (k7, no bias) → clamp to [-1, 1]. ∏rates = 800, so
/// one 40 Hz latent frame becomes 800 samples at 32 kHz.</summary>
///
/// <remarks><para>The vocoder body is <see cref="LtxBigVganGenerator"/> parameterized to H3's geometry — the two
/// architectures are the same BigVGAN-v2 AMP generator and differ only in checkpoint conventions, which
/// <see cref="LoadWeights"/> normalizes: PyTorch <c>weight_norm</c> (<c>weight_g</c>/<c>weight_v</c>) is fused at
/// load, each upsample's <c>ModuleList</c> wrapper level (<c>ups.i.0</c>) is flattened, the interleaved
/// <c>activations.2j</c>/<c>activations.2j+1</c> pairs are split into <c>acts1.j</c>/<c>acts2.j</c>, and
/// <c>activation_post</c> is renamed <c>act_post</c>.</para>
/// <para>The decoder is mono; stereo comes from the latent's dedicated channel axis. Latents are
/// <c>[1, LatentChannels, S, T]</c> and each of the S channels is decoded independently, matching the reference
/// <c>permute → reshape(b·s, c, t)</c>. Output is <c>[1, S, T · 800]</c> in [-1, 1].</para></remarks>
public sealed unsafe class MiniMaxH3AudioVaeDecoder : IDisposable
{
    private readonly MiniMaxH3AudioVaeConfig _config;
    private readonly LtxBigVganGenerator _generator;
    private readonly List<Tensor> _owned = [];
    private Tensor? _decInW;
    private Tensor? _decInB;

    public MiniMaxH3AudioVaeDecoder(MiniMaxH3AudioVaeConfig? config = null)
    {
        _config = config ?? new MiniMaxH3AudioVaeConfig();
        if (_config.LatentsMean.Length != _config.LatentChannels || _config.LatentsStd.Length != _config.LatentChannels)
            throw new ArgumentException(
                $"LatentsMean/LatentsStd must have {_config.LatentChannels} entries, got " +
                $"{_config.LatentsMean.Length}/{_config.LatentsStd.Length}.", nameof(config));
        _generator = new LtxBigVganGenerator(
            inChannels: _config.DecoderInputChannels,
            hiddenChannels: _config.DecoderDim,
            outChannels: 1,
            upsampleFactors: _config.UpsampleRates,
            upsampleKernels: _config.UpsampleKernels,
            resnetKernels: _config.ResblockKernels,
            resnetDilations: _config.ResblockDilations,
            applyTanh: false);
    }

    /// <summary>Waveform sample rate of <see cref="Decode"/>'s output.</summary>
    public int SampleRate => _config.SampleRate;

    /// <summary>Waveform samples produced per latent frame (800 for the shipped configuration).</summary>
    public int SamplesPerLatentFrame => _generator.TotalUpsample;

    /// <summary>Fuses weight-norm, folds the latent mean/std into <c>dec_in_proj</c>, and maps the H3 key names onto
    /// the ones <see cref="LtxBigVganGenerator"/> expects.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        FoldLatentNorm(w);
        _generator.LoadWeights(Remap(w), prefix: "decoder");
    }

    /// <summary>Every loaded weight tensor, for <see cref="IBackend.PreloadWeights"/>/<see cref="IBackend.FreeWeights"/>.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_decInW is not null) yield return _decInW;
        if (_decInB is not null) yield return _decInB;
        foreach (Tensor t in _generator.EnumerateWeights()) yield return t;
    }

    /// <summary>Decode latents <c>[1, LatentChannels, S, T]</c> into a waveform <c>[1, S, T · 800]</c> in [-1, 1] at
    /// <see cref="SampleRate"/> (S = 2 for the stereo H3 soundtrack). Caller owns the returned tensor.</summary>
    public Tensor Decode(IBackend backend, Tensor latent)
    {
        if (_decInW is null)
            throw new InvalidOperationException("MiniMaxH3AudioVaeDecoder.LoadWeights must run before Decode.");
        if (latent.Shape.Rank != 4 || latent.Shape[0] != 1)
            throw new ArgumentException($"expected a batch-1 [1, C, S, T] latent, got {latent.Shape}.", nameof(latent));
        if ((int)latent.Shape[1] != _config.LatentChannels)
            throw new ArgumentException(
                $"expected {_config.LatentChannels} latent channels, got {latent.Shape[1]}.", nameof(latent));

        Tensor src = DtypeCastHelper.EnsureF32(backend, latent, disposeSourceOnCast: false);
        int channels = (int)src.Shape[1], stereo = (int)src.Shape[2], frames = (int)src.Shape[3];
        Tensor perm = new(new TensorShape(1, stereo, channels, frames), DType.F32);
        backend.Permute0213(perm, src, channels, stereo, frames);
        if (!ReferenceEquals(src, latent)) src.Dispose();

        Tensor[] rendered = new Tensor[stereo];
        for (int s = 0; s < stereo; s++)
        {
            Tensor z = new(new TensorShape(1, channels, frames), DType.F32);
            backend.SliceRows(z, perm, s * channels);
            Tensor lifted = new(new TensorShape(1, _config.DecoderInputChannels, frames), DType.F32);
            backend.Conv1d(lifted, z, _decInW!, _decInB, 1, 0, 0, 1, 1);
            z.Dispose();
            rendered[s] = _generator.Forward(backend, lifted);
            lifted.Dispose();
        }
        perm.Dispose();

        int samples = (int)rendered[0].Shape[2];
        Tensor waveform = new(new TensorShape(1, stereo, samples), DType.F32);
        backend.Concat(waveform, rendered, 1);
        foreach (Tensor r in rendered) r.Dispose();
        backend.Clamp(waveform, waveform, -1f, 1f);
        return waveform;
    }

    /// <summary>Folds <c>latent · std + mean</c> into the kernel-1 <c>dec_in_proj</c>: <c>W'[o,c] = W[o,c]·std[c]</c>,
    /// <c>b' = b + W·mean</c>. Exact for kernel 1, and the reason no denormalization appears in <see cref="Decode"/>.</summary>
    private void FoldLatentNorm(IReadOnlyDictionary<string, Tensor> w)
    {
        Tensor srcW = AsF32(w["dec_in_proj.weight"], out bool ownW);
        int outCh = (int)srcW.Shape[0];
        int inCh = (int)srcW.Shape[1];
        if (inCh != _config.LatentChannels)
            throw new ArgumentException($"dec_in_proj expects {_config.LatentChannels} input channels, got {inCh}.");

        Tensor foldedW = new(new TensorShape(outCh, inCh, 1), DType.F32);
        Tensor foldedB = new(new TensorShape(outCh), DType.F32);
        float* sp = (float*)srcW.DataPointer;
        float* wp = (float*)foldedW.DataPointer;
        float* bp = (float*)foldedB.DataPointer;
        bool ownB = false;
        Tensor? srcB = w.TryGetValue("dec_in_proj.bias", out Tensor? rawB) ? AsF32(rawB, out ownB) : null;
        float* sbp = srcB is null ? null : (float*)srcB.DataPointer;
        for (int o = 0; o < outCh; o++)
        {
            double shift = sbp is null ? 0d : sbp[o];
            for (int c = 0; c < inCh; c++)
            {
                float v = sp[(long)o * inCh + c];
                wp[(long)o * inCh + c] = v * _config.LatentsStd[c];
                shift += (double)v * _config.LatentsMean[c];
            }
            bp[o] = (float)shift;
        }
        if (ownW) srcW.Dispose();
        if (ownB) srcB!.Dispose();
        _owned.Add(foldedW);
        _owned.Add(foldedB);
        _decInW = foldedW;
        _decInB = foldedB;
    }

    /// <summary>Builds the <see cref="LtxBigVganGenerator"/>-shaped weight dictionary from the H3 checkpoint names.
    /// Fused conv weights are added to <see cref="_owned"/>; biases and SnakeBeta parameters are borrowed.</summary>
    private Dictionary<string, Tensor> Remap(IReadOnlyDictionary<string, Tensor> w)
    {
        Dictionary<string, Tensor> m = [];
        m["decoder.conv_pre.weight"] = FuseOwned(w, "decoder.conv_pre");
        Borrow(w, m, "decoder.conv_pre.bias", "decoder.conv_pre.bias");

        int stages = _config.UpsampleRates.Length;
        int resPerUp = _config.ResblockKernels.Length;
        for (int i = 0; i < stages; i++)
        {
            m[$"decoder.ups.{i}.weight"] = FuseOwned(w, $"decoder.ups.{i}.0");
            Borrow(w, m, $"decoder.ups.{i}.0.bias", $"decoder.ups.{i}.bias");
        }
        for (int n = 0; n < stages * resPerUp; n++)
        {
            int[] dilations = _config.ResblockDilations[n % resPerUp];
            for (int j = 0; j < dilations.Length; j++)
            {
                m[$"decoder.resblocks.{n}.convs1.{j}.weight"] = FuseOwned(w, $"decoder.resblocks.{n}.convs1.{j}");
                m[$"decoder.resblocks.{n}.convs2.{j}.weight"] = FuseOwned(w, $"decoder.resblocks.{n}.convs2.{j}");
                Borrow(w, m, $"decoder.resblocks.{n}.convs1.{j}.bias", $"decoder.resblocks.{n}.convs1.{j}.bias");
                Borrow(w, m, $"decoder.resblocks.{n}.convs2.{j}.bias", $"decoder.resblocks.{n}.convs2.{j}.bias");
                // AMPBlock1 reads activations[::2] as acts1 and activations[1::2] as acts2.
                BorrowAct(w, m, $"decoder.resblocks.{n}.activations.{2 * j}", $"decoder.resblocks.{n}.acts1.{j}");
                BorrowAct(w, m, $"decoder.resblocks.{n}.activations.{2 * j + 1}", $"decoder.resblocks.{n}.acts2.{j}");
            }
        }
        BorrowAct(w, m, "decoder.activation_post", "decoder.act_post");
        m["decoder.conv_post.weight"] = FuseOwned(w, "decoder.conv_post");
        return m;
    }

    private Tensor FuseOwned(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        Tensor fused = AudioWeightNorm.Fuse(w, prefix, out bool allocated);
        if (allocated) _owned.Add(fused);
        return fused;
    }

    private static void Borrow(IReadOnlyDictionary<string, Tensor> w, Dictionary<string, Tensor> m, string from, string to)
    {
        if (w.TryGetValue(from, out Tensor? t)) m[to] = t;
    }

    private static void BorrowAct(IReadOnlyDictionary<string, Tensor> w, Dictionary<string, Tensor> m, string from, string to)
    {
        m[$"{to}.act.alpha"] = w[$"{from}.act.alpha"];
        m[$"{to}.act.beta"] = w[$"{from}.act.beta"];
    }

    private static Tensor AsF32(Tensor t, out bool owned)
    {
        owned = t.DType != DType.F32;
        return owned ? t.CastTo(DType.F32) : t;
    }

    public void Dispose()
    {
        _generator.Dispose();
        foreach (Tensor t in _owned) t.Dispose();
        _owned.Clear();
        _decInW = null;
        _decInB = null;
    }
}
