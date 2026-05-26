using SharpInference.Audio.Models.Whisper;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.VibeVoice;

/// <summary>Decoder side of the VibeVoice acoustic VAE. Mirrors
/// <see cref="VibeVoiceTokenizerEncoder"/> but with transposed convs for upsampling and
/// channel widths running in reverse. Used by the acoustic VAE only — the semantic VAE
/// has no decoder.
///
/// <para>Channel chain on the published checkpoints (decoder_n_filters=32, decoder_ratios
/// in forward order [8,5,5,4,2,2], decoder_depths = reverse(encoder_depths) = [8,3,3,3,3,3,3]):
/// <code>
///   stage 0: stem  SConv1d(vae_dim → 2048, k=7, s=1)                + 8 × Block1D(dim=2048)
///   stage 1: up   SConvTranspose1d(2048 → 1024, k=16, s=8)          + 3 × Block1D(dim=1024)
///   stage 2: up   SConvTranspose1d(1024 → 512,  k=10, s=5)          + 3 × Block1D(dim=512)
///   stage 3: up   SConvTranspose1d(512  → 256,  k=10, s=5)          + 3 × Block1D(dim=256)
///   stage 4: up   SConvTranspose1d(256  → 128,  k=8,  s=4)          + 3 × Block1D(dim=128)
///   stage 5: up   SConvTranspose1d(128  → 64,   k=4,  s=2)          + 3 × Block1D(dim=64)
///   stage 6: up   SConvTranspose1d(64   → 32,   k=4,  s=2)          + 3 × Block1D(dim=32)
///   head: SConv1d(32 → 1, k=7, s=1)
/// </code>
/// Total temporal upsample = <c>8*5*5*4*2*2 = 3200</c> — exact inverse of the encoder.</para></summary>
internal sealed class VibeVoiceTokenizerDecoder
{
    private readonly VibeVoiceTokenizerConfig _config;
    private readonly string _prefix;
    private readonly int _channels;
    private readonly int[] _ratios;          // forward order: [8,5,5,4,2,2]
    private readonly int[] _depths;          // reversed encoder: [8,3,3,3,3,3,3]

    private readonly SConv1d _stem;                                            // upsample_layers[0]
    private readonly SConvTranspose1d[] _upsamples;                            // upsample_layers[1..N-1]
    private readonly VibeVoiceConvNeXtBlock[][] _stageBlocks;
    private readonly SConv1d _head;
    private Tensor? _lastNormW;

    public VibeVoiceTokenizerDecoder(VibeVoiceTokenizerConfig config, string prefix)
    {
        if (config.DecoderRatios is null)
            throw new ArgumentException("Decoder requires non-null DecoderRatios — semantic VAE has no decoder.", nameof(config));
        _config = config;
        _prefix = prefix;
        _channels = config.Channels;
        _ratios = [.. config.DecoderRatios];

        int[] encDepths = ParseDepths(config.EncoderDepths);
        _depths = config.DecoderDepths is null
            ? ReverseArray(encDepths)
            : ParseDepths(config.DecoderDepths);

        int nFilters = config.DecoderNFilters;
        int kernel = 7;
        int lastIdx = _depths.Length - 1;     // 6 for the published checkpoint
        int peakChannels = nFilters * (1 << lastIdx);     // 32 * 64 = 2048

        // Stem: SConv1d from latent dim → highest channel count.
        _stem = new SConv1d(
            layerId: $"{prefix}.upsample_layers.0.0",
            inChannels: config.VaeDim, outChannels: peakChannels,
            kernelSize: kernel, stride: 1, bias: config.ConvBias);

        // Upsample stages 1..N-1.
        _upsamples = new SConvTranspose1d[_depths.Length - 1];
        for (int i = 0; i < _upsamples.Length; i++)
        {
            int inCh = nFilters * (1 << (lastIdx - i));
            int outCh = nFilters * (1 << (lastIdx - i - 1));
            int stride = _ratios[i];
            _upsamples[i] = new SConvTranspose1d(
                layerId: $"{prefix}.upsample_layers.{i + 1}.0",
                inChannels: inCh, outChannels: outCh,
                kernelSize: stride * 2, stride: stride, bias: config.ConvBias);
        }

        // Block chains.
        _stageBlocks = new VibeVoiceConvNeXtBlock[_depths.Length][];
        for (int i = 0; i < _depths.Length; i++)
        {
            int dim = nFilters * (1 << (lastIdx - i));
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

        // Head: SConv1d projecting smallest stage channels → output audio channels (= 1).
        _head = new SConv1d(
            layerId: $"{prefix}.head",
            inChannels: nFilters, outChannels: _channels,
            kernelSize: kernel, stride: 1, bias: config.ConvBias);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        // SConv1d / SConvTranspose1d already append ".conv.conv" / ".convtr.convtr"
        // internally — pass the prefix for the module itself, not its inner attribute.
        _stem.LoadWeights(w, $"{_prefix}.upsample_layers.0.0");
        for (int i = 0; i < _upsamples.Length; i++)
            _upsamples[i].LoadWeights(w, $"{_prefix}.upsample_layers.{i + 1}.0");
        for (int i = 0; i < _stageBlocks.Length; i++)
            for (int j = 0; j < _stageBlocks[i].Length; j++)
                _stageBlocks[i][j].LoadWeights(w);
        _head.LoadWeights(w, $"{_prefix}.head");
        if (!_config.DisableLastNorm)
            _lastNormW = WhisperOps.EnsureF32(w[$"{_prefix}.norm.weight"]);
    }

    /// <summary>Forward — input <c>[batch, vae_dim, T_latent]</c>, output
    /// <c>[batch, 1, T_latent * hop]</c>. Streaming when <paramref name="cache"/> is non-null.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, int batch, int tLatent,
        VibeVoiceTokenizerStreamingCache? cache = null, ReadOnlySpan<int> sampleIndices = default)
    {
        // Stem.
        Tensor current = cache is null
            ? _stem.Forward(latent, batch, tLatent)
            : _stem.ForwardStreaming(latent, batch, tLatent, cache, sampleIndices);
        int t = (int)current.Shape[2];

        // Stage 0 blocks (run BEFORE the first transpose — see Python's per-stage loop:
        // upsample_layers[i] then stages[i]; stage 0's upsample IS the stem).
        foreach (VibeVoiceConvNeXtBlock block in _stageBlocks[0])
        {
            Tensor next = block.Forward(backend, current, batch, t, cache, sampleIndices);
            current.Dispose();
            current = next;
        }

        // Stages 1..N-1: upsample → blocks.
        for (int i = 1; i < _depths.Length; i++)
        {
            Tensor up = cache is null
                ? _upsamples[i - 1].Forward(current, batch, t)
                : _upsamples[i - 1].ForwardStreaming(current, batch, t, cache, sampleIndices);
            current.Dispose();
            current = up;
            t = (int)current.Shape[2];

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
            Tensor normed = new(current.Shape, DType.F32);
            VibeVoiceOps.RmsNormChannelsFirst(normed, current, _lastNormW!, batch, (int)current.Shape[1], t, _config.LayerNormEps);
            current.Dispose();
            current = normed;
        }

        // Head projection.
        Tensor head = cache is null
            ? _head.Forward(current, batch, t)
            : _head.ForwardStreaming(current, batch, t, cache, sampleIndices);
        current.Dispose();
        return head;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _stem.EnumerateWeights()) yield return t;
        foreach (SConvTranspose1d up in _upsamples)
            foreach (Tensor t in up.EnumerateWeights()) yield return t;
        foreach (VibeVoiceConvNeXtBlock[] stage in _stageBlocks)
            foreach (VibeVoiceConvNeXtBlock b in stage)
                foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (Tensor t in _head.EnumerateWeights()) yield return t;
        if (_lastNormW is not null) yield return _lastNormW;
    }

    private static int[] ReverseArray(int[] arr)
    {
        int[] r = new int[arr.Length];
        for (int i = 0; i < arr.Length; i++) r[i] = arr[arr.Length - 1 - i];
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
