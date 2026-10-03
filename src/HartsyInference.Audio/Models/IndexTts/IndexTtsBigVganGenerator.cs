using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.IndexTts;

/// <summary>IndexTTS-1.5's custom-trained 24 kHz BigVGAN-v2 generator. Unlike a stock BigVGAN, the primary
/// time-aligned input is the T2S GPT's own final-layer hidden states ("latent"), not a decoded mel spectrogram —
/// verified against the real checkpoint (<c>conv_pre</c> takes 1280 = <c>gpt_dim</c> channels, not a mel-band
/// count) and the reference <c>infer()</c>'s <c>self.bigvgan(latent, auto_conditioning.transpose(1,2))</c> call.
/// An embedded <see cref="IndexTtsEcapaTdnn"/> speaker encoder computes a 512-dim d-vector from the raw reference
/// mel, injected once after <c>conv_pre</c> (<c>cond_layer</c>) and again before every upsampling stage's resblocks
/// (<c>conds[i]</c>, <c>cond_d_vector_in_each_upsampling_layer: true</c> in <c>config.yaml</c>).</summary>
/// <remarks><c>conv_pre</c>/<c>conv_post</c>/<c>ups[i]</c> carry PyTorch <c>weight_norm</c> (fused via
/// <see cref="WeightNormFusion.Compose"/>); <c>cond_layer</c>/<c>conds[i]</c> are plain convs. Assembly shape
/// follows <c>LtxBigVganGenerator</c>'s structure as a template (Audio has no dependency on Diffusion, so it
/// cannot be referenced directly), but every activation site is a standalone <see cref="AntiAliasedSnake"/>
/// instance rather than a ported copy of its anti-aliasing math.</remarks>
internal sealed unsafe class IndexTtsBigVganGenerator : IDisposable
{
    private readonly IndexTtsBigVganConfig _cfg;
    private readonly IndexTtsEcapaTdnn _speakerEncoder = new();
    private readonly int _numStages;
    private readonly Tensor?[] _upsW = [];
    private readonly Tensor?[] _upsB = [];
    private readonly Tensor?[] _condsW = [];
    private readonly Tensor?[] _condsB = [];
    private readonly IndexTtsBigVganResBlock[][] _resblocks;   // [stage][kernelIdx]
    private Tensor? _convPreW, _convPreB, _condLayerW, _condLayerB, _convPostW, _convPostB;
    private AntiAliasedSnake? _activationPost;
    private int _disposed;

    public IndexTtsBigVganGenerator(IndexTtsBigVganConfig cfg)
    {
        _cfg = cfg;
        _numStages = cfg.UpsampleRates.Length;
        _upsW = new Tensor?[_numStages]; _upsB = new Tensor?[_numStages];
        _condsW = new Tensor?[_numStages]; _condsB = new Tensor?[_numStages];
        _resblocks = new IndexTtsBigVganResBlock[_numStages][];
        for (int i = 0; i < _numStages; i++)
        {
            int ch = cfg.UpsampleInitialChannel / (1 << (i + 1));
            _resblocks[i] = new IndexTtsBigVganResBlock[cfg.ResblockKernelSizes.Length];
            for (int j = 0; j < cfg.ResblockKernelSizes.Length; j++)
                _resblocks[i][j] = new IndexTtsBigVganResBlock(ch, cfg.ResblockKernelSizes[j], cfg.ResblockDilations[j]);
        }
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix, string speakerEncoderPrefix)
    {
        _convPreW = WeightNormFusion.Compose(w, $"{prefix}.conv_pre");
        _convPreB = EnsureF32(w[$"{prefix}.conv_pre.bias"]);
        _condLayerW = EnsureF32(w[$"{prefix}.cond_layer.weight"]);
        _condLayerB = EnsureF32(w[$"{prefix}.cond_layer.bias"]);

        for (int i = 0; i < _numStages; i++)
        {
            _upsW[i] = WeightNormFusion.Compose(w, $"{prefix}.ups.{i}.0");
            _upsB[i] = EnsureF32(w[$"{prefix}.ups.{i}.0.bias"]);
            _condsW[i] = EnsureF32(w[$"{prefix}.conds.{i}.weight"]);
            _condsB[i] = EnsureF32(w[$"{prefix}.conds.{i}.bias"]);
            for (int j = 0; j < _resblocks[i].Length; j++)
                _resblocks[i][j].LoadWeights(w, $"{prefix}.resblocks.{i * _resblocks[i].Length + j}");
        }

        int finalCh = _cfg.UpsampleInitialChannel / (1 << _numStages);
        _activationPost = new AntiAliasedSnake(finalCh);
        _activationPost.LoadWeights(w, $"{prefix}.activation_post");
        _convPostW = WeightNormFusion.Compose(w, $"{prefix}.conv_post");
        _convPostB = EnsureF32(w[$"{prefix}.conv_post.bias"]);

        _speakerEncoder.LoadWeights(w, speakerEncoderPrefix);
    }

    /// <summary>Synthesizes 24 kHz PCM from the GPT latent <c>[1, T, gptDim]</c> and the reference mel
    /// <c>[1, Tref, nMels]</c> (channels-last). Returns <c>[1, 1, T·hop]</c> where <c>hop = Π upsampleRates</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, int latentLen, Tensor referenceMel, int refLen)
    {
        if (_convPreW is null) throw new InvalidOperationException("IndexTtsBigVganGenerator weights not loaded.");

        Tensor speakerEmb = _speakerEncoder.Forward(backend, referenceMel, refLen);   // [1, 512, 1]

        Tensor latentChFirst = new(new TensorShape(1, _cfg.GptDim, latentLen), DType.F32);
        backend.Transpose2D(latentChFirst, latent, latentLen, _cfg.GptDim);

        Tensor x = new(new TensorShape(1, _cfg.UpsampleInitialChannel, latentLen), DType.F32);
        backend.Conv1d(x, latentChFirst, _convPreW!, _convPreB, stride: 1, padLeft: 3, padRight: 3, dilation: 1, groups: 1);
        latentChFirst.Dispose();

        AddBroadcastCond(x, speakerEmb, _condLayerW!, _condLayerB!, _cfg.UpsampleInitialChannel, latentLen);

        int t = latentLen;
        for (int i = 0; i < _numStages; i++)
        {
            int inCh = _cfg.UpsampleInitialChannel / (1 << i);
            int outCh = _cfg.UpsampleInitialChannel / (1 << (i + 1));
            int rate = _cfg.UpsampleRates[i];
            int kernel = _cfg.UpsampleKernelSizes[i];
            int pad = (kernel - rate) / 2;
            int tUp = (t - 1) * rate + kernel - 2 * pad;

            Tensor up = new(new TensorShape(1, outCh, tUp), DType.F32);
            backend.ConvTranspose1d(up, x, _upsW[i]!, _upsB[i], rate, pad, pad, 1, 1);
            x.Dispose();
            x = up;
            t = tUp;

            AddBroadcastCond(x, speakerEmb, _condsW[i]!, _condsB[i]!, outCh, t);

            Tensor? sum = null;
            foreach (IndexTtsBigVganResBlock rb in _resblocks[i])
            {
                Tensor rbOut = rb.Forward(backend, x);
                if (sum is null) sum = rbOut;
                else { backend.Add(sum, sum, rbOut); rbOut.Dispose(); }
            }
            x.Dispose();
            x = sum!;
            float invK = 1f / _resblocks[i].Length;
            backend.Scale(x, x, invK);
        }
        speakerEmb.Dispose();

        Tensor activated = _activationPost!.Forward(backend, x);
        x.Dispose();

        int finalCh = _cfg.UpsampleInitialChannel / (1 << _numStages);
        Tensor outWave = new(new TensorShape(1, 1, (int)activated.Shape[2]), DType.F32);
        backend.Conv1d(outWave, activated, _convPostW!, _convPostB, stride: 1, padLeft: 3, padRight: 3, dilation: 1, groups: 1);
        activated.Dispose();
        backend.Tanh(outWave, outWave);
        return outWave;
    }

    /// <summary>Projects the time-invariant speaker embedding <c>[1, 512, 1]</c> through a 1×1 conv and adds it
    /// to every timestep of <paramref name="x"/> <c>[1, ch, t]</c> in place.</summary>
    private static void AddBroadcastCond(Tensor x, Tensor speakerEmb, Tensor condW, Tensor condB, int ch, int t)
    {
        float* xp = (float*)x.DataPointer;
        float* sp = (float*)speakerEmb.DataPointer;
        float* wp = (float*)condW.DataPointer;
        float* bp = (float*)condB.DataPointer;
        int spkDim = (int)speakerEmb.Shape[1];
        for (int oc = 0; oc < ch; oc++)
        {
            float acc = bp[oc];
            float* wRow = wp + (long)oc * spkDim;
            for (int e = 0; e < spkDim; e++) acc += wRow[e] * sp[e];
            float* row = xp + (long)oc * t;
            for (int i = 0; i < t; i++) row[i] += acc;
        }
    }

    private static Tensor EnsureF32(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] core = [_convPreW, _convPreB, _condLayerW, _condLayerB, _convPostW, _convPostB];
        foreach (Tensor? t in core) if (t is not null) yield return t;
        for (int i = 0; i < _numStages; i++)
        {
            yield return _upsW[i]!; yield return _upsB[i]!; yield return _condsW[i]!; yield return _condsB[i]!;
            foreach (IndexTtsBigVganResBlock rb in _resblocks[i]) foreach (Tensor t in rb.EnumerateWeights()) yield return t;
        }
        if (_activationPost is not null) foreach (Tensor t in _activationPost.EnumerateWeights()) yield return t;
        foreach (Tensor t in _speakerEncoder.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (IndexTtsBigVganResBlock[] stage in _resblocks) foreach (IndexTtsBigVganResBlock rb in stage) rb.Dispose();
        _activationPost?.Dispose();
        _speakerEncoder.Dispose();
        GC.SuppressFinalize(this);
    }
}
