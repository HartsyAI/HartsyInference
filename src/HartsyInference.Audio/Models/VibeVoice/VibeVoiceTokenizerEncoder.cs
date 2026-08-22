using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.VibeVoice;

/// <summary>Encoder side of the VibeVoice VAEs: 7 stages of (down-conv + ConvNeXt-block chain) on channels-first <c>[B, 1, T_pcm]</c> input, an optional pre-head norm, then a 1×1 projection to the latent dimension; used by both the acoustic VAE (vae_dim=64) and the semantic VAE (vae_dim=128).</summary>
/// <remarks>
/// <para>Channel chain on the published checkpoints (encoder_n_filters=32,
/// encoder_ratios=[8,5,5,4,2,2] reversed in-forward to [2,2,4,5,5,8]):
/// <code>
///   stage 0: stem SConv1d(1 → 32, k=7, s=1)         + 3 × Block1D(dim=32)
///   stage 1: down SConv1d(32 → 64, k=4, s=2)         + 3 × Block1D(dim=64)
///   stage 2: down SConv1d(64 → 128, k=4, s=2)        + 3 × Block1D(dim=128)
///   stage 3: down SConv1d(128 → 256, k=8, s=4)       + 3 × Block1D(dim=256)
///   stage 4: down SConv1d(256 → 512, k=10, s=5)      + 3 × Block1D(dim=512)
///   stage 5: down SConv1d(512 → 1024, k=10, s=5)     + 3 × Block1D(dim=1024)
///   stage 6: down SConv1d(1024 → 2048, k=16, s=8)    + 8 × Block1D(dim=2048)
///   head: SConv1d(2048 → vae_dim, k=7, s=1)
/// </code>
/// Total temporal downsample = <c>2*2*4*5*5*8 = 3200</c> → 7.5 Hz on 24 kHz input.</para></remarks>
internal sealed class VibeVoiceTokenizerEncoder
{
    private readonly VibeVoiceTokenizerConfig _config;
    private readonly string _prefix;
    private readonly int _vaeDim;
    private readonly int[] _ratios;     // reversed: [2,2,4,5,5,8]
    private readonly int[] _depths;     // parsed: [3,3,3,3,3,3,8]

    private readonly SConv1d[] _downsamples;       // length = depths.Count; index 0 is the stem.
    private readonly VibeVoiceConvNeXtBlock[][] _stageBlocks;
    private readonly SConv1d _head;
    private Tensor? _lastNormW;                    // null when disable_last_norm=true.

    public VibeVoiceTokenizerEncoder(VibeVoiceTokenizerConfig config, string prefix)
    {
        _config = config;
        _prefix = prefix;
        _vaeDim = config.VaeDim;
        _ratios = ReverseRatios(config.EncoderRatios);
        _depths = ParseDepths(config.EncoderDepths);

        int nFilters = config.EncoderNFilters;
        int kernel = 7;

        _downsamples = new SConv1d[_depths.Length];
        _stageBlocks = new VibeVoiceConvNeXtBlock[_depths.Length][];

        // Stem (stage 0).
        _downsamples[0] = new SConv1d(
            layerId: $"{prefix}.downsample_layers.0.0",
            inChannels: config.Channels, outChannels: nFilters,
            kernelSize: kernel, stride: 1, bias: config.ConvBias);

        // Downsample stages 1..N-1.
        for (int i = 1; i < _depths.Length; i++)
        {
            int inCh = nFilters * (1 << (i - 1));
            int outCh = nFilters * (1 << i);
            int stride = _ratios[i - 1];
            _downsamples[i] = new SConv1d(
                layerId: $"{prefix}.downsample_layers.{i}.0",
                inChannels: inCh, outChannels: outCh,
                kernelSize: stride * 2, stride: stride, bias: config.ConvBias);
        }

        // Block chains.
        for (int i = 0; i < _depths.Length; i++)
        {
            int dim = nFilters * (1 << i);
            _stageBlocks[i] = new VibeVoiceConvNeXtBlock[_depths[i]];
            for (int j = 0; j < _depths[i]; j++)
            {
                _stageBlocks[i][j] = new VibeVoiceConvNeXtBlock(
                    prefix: $"{prefix}.stages.{i}.{j}",
                    dim: dim,
                    ffnExpansion: 4,
                    kernelSize: kernel,
                    normEps: config.LayerNormEps);
            }
        }

        // Head: 1×1-style 1D conv projecting last stage channels → vae_dim.
        int lastCh = nFilters * (1 << (_depths.Length - 1));
        _head = new SConv1d(
            layerId: $"{prefix}.head",
            inChannels: lastCh, outChannels: _vaeDim,
            kernelSize: 7, stride: 1, bias: config.ConvBias);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        // SConv1d.LoadWeights already appends ".conv.conv.{weight,bias}" (outer
        // SConv1d.conv = NormConv1d, inner NormConv1d.conv = nn.Conv1d). We pass the
        // prefix for the SConv1d itself, not for its inner .conv attribute.
        for (int i = 0; i < _downsamples.Length; i++)
            _downsamples[i].LoadWeights(w, $"{_prefix}.downsample_layers.{i}.0");
        for (int i = 0; i < _stageBlocks.Length; i++)
            for (int j = 0; j < _stageBlocks[i].Length; j++)
                _stageBlocks[i][j].LoadWeights(w);
        _head.LoadWeights(w, $"{_prefix}.head");
        if (!_config.DisableLastNorm)
            _lastNormW = WhisperOps.EnsureF32(w[$"{_prefix}.norm.weight"]);
    }

    /// <summary>Forward, input <c>[batch, channels=1, T_pcm]</c> to output <c>[batch, vae_dim, T_pcm / hop]</c> where <c>hop = product(encoder_ratios) = 3200</c> for the published configs; streams when <paramref name="cache"/> is non-null.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int batch, int tIn,
        VibeVoiceTokenizerStreamingCache? cache = null, ReadOnlySpan<int> sampleIndices = default)
    {
        Tensor current = x;
        bool ownsCurrent = false;
        int t = tIn;

        for (int i = 0; i < _depths.Length; i++)
        {
            // Downsample / stem.
            Tensor down = cache is null
                ? _downsamples[i].Forward(backend, current, batch, t)
                : _downsamples[i].ForwardStreaming(backend, current, batch, t, cache, sampleIndices);
            if (ownsCurrent) current.Dispose();
            current = down;
            ownsCurrent = true;
            t = (int)current.Shape[2];

            // Block chain at this stage.
            int dim = (int)current.Shape[1];
            foreach (VibeVoiceConvNeXtBlock block in _stageBlocks[i])
            {
                Tensor next = block.Forward(backend, current, batch, t, cache, sampleIndices);
                current.Dispose();
                current = next;
            }
        }

        // Optional pre-head norm.
        if (_lastNormW is not null)
        {
            Tensor normed = VibeVoiceOps.RmsNormChannelsFirstGpu(backend, current, _lastNormW!, batch, (int)current.Shape[1], t, _config.LayerNormEps);
            current.Dispose();
            current = normed;
        }

        // Head projection (1×1-ish — kernel=7, stride=1, so T preserved).
        Tensor head = cache is null
            ? _head.Forward(backend, current, batch, t)
            : _head.ForwardStreaming(backend, current, batch, t, cache, sampleIndices);
        current.Dispose();
        return head;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (SConv1d ds in _downsamples)
            foreach (Tensor t in ds.EnumerateWeights()) yield return t;
        foreach (VibeVoiceConvNeXtBlock[] stage in _stageBlocks)
            foreach (VibeVoiceConvNeXtBlock b in stage)
                foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (Tensor t in _head.EnumerateWeights()) yield return t;
        if (_lastNormW is not null) yield return _lastNormW;
    }

    private static int[] ReverseRatios(IReadOnlyList<int> ratios)
    {
        int[] r = new int[ratios.Count];
        for (int i = 0; i < ratios.Count; i++) r[i] = ratios[ratios.Count - 1 - i];
        return r;
    }

    private static int[] ParseDepths(string depths)
    {
        string[] parts = depths.Split('-', StringSplitOptions.RemoveEmptyEntries);
        int[] result = new int[parts.Length];
        for (int i = 0; i < parts.Length; i++) result[i] = int.Parse(parts[i]);
        return result;
    }
}
