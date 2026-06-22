using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.Csm;
using HartsyInference.Audio.Models.HeartMula;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>HeartMuLa music-generation pipeline. The LM is the **already-built+verified** CSM dual-transformer
/// (<see cref="CsmModel"/>): a Llama-3B global backbone + Llama-300M depth decoder emit 8 RVQ codebooks per
/// 12.5 Hz frame, conditioned on lyrics tokens (+ a MuQ-MuLan style embedding, staged). Returns the codebook
/// grid; HeartCodec waveform decode (48 kHz flow-matching codec) is staged.</summary>
public sealed unsafe class HeartMulaPipeline : IDisposable
{
    private readonly HeartMulaConfig _cfg;
    private readonly CsmModel _lm;
    private readonly HeartCodecDecoder _codec;
    private int _disposed;

    public HeartMulaPipeline(HeartMulaConfig cfg)
    {
        _cfg = cfg;
        _lm = new CsmModel(cfg.Lm);
        _codec = new HeartCodecDecoder(cfg);
    }

    public int SampleRate => _cfg.Lm.SampleRate;

    public HeartCodecDecoder Codec => _codec;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w) => _lm.LoadWeights(w);

    /// <summary>Loads the HeartCodec decoder weights (separate checkpoint from the LM). Call before
    /// <see cref="Generate"/>; <see cref="GenerateCodes"/> alone needs only the LM weights.</summary>
    public void LoadCodecWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "") =>
        _codec.LoadWeights(w, prefix);

    /// <summary>Generates the 8-codebook grid <c>[NumCodebooks, T]</c> from lyrics token ids (the audio EOS
    /// frame stops generation), optionally prefixed with a MuQ style-conditioning embedding already projected
    /// into the LM hidden (<c>[1, hiddenSize]</c>, from <see cref="MuqEmbedder.ProjectToLmHidden"/>).</summary>
    public int[,] GenerateCodes(IBackend backend, ReadOnlySpan<int> lyricsTokens, int maxFrames, int seed = 0,
        Tensor? muqLmEmbed = null)
    {
        ThrowIfDisposed();
        int nb = _cfg.Lm.NumCodebooks;
        uint rng = DeterministicRng.Seed(seed);
        List<int[]> frames = new(maxFrames);

        // Context = optional MuQ style row + lyrics text embeddings; then AR audio frames re-fed.
        for (int f = 0; f < maxFrames; f++)
        {
            Tensor ctx = BuildContext(lyricsTokens, frames, muqLmEmbed);
            int[] codes = _lm.GenerateFrame(backend, ctx, ref rng);
            ctx.Dispose();
            if (codes[0] == _cfg.Lm.AudioEosToken) break;
            frames.Add(codes);
        }

        int t = frames.Count;
        int[,] grid = new int[nb, t];
        for (int j = 0; j < t; j++)
            for (int i = 0; i < nb; i++) grid[i, j] = frames[j][i];
        return grid;
    }

    /// <summary>End-to-end: lyrics tokens → CSM AR codes (8 codebooks) → HeartCodec flow-matching decode →
    /// 48 kHz mono waveform. <paramref name="muqLmEmbed"/> is the optional MuQ style conditioning projected
    /// into the LM hidden. Requires both the LM and codec weights to be loaded.</summary>
    public float[] Generate(IBackend backend, ReadOnlySpan<int> lyricsTokens, int maxFrames, int seed = 0,
        Tensor? muqLmEmbed = null)
    {
        ThrowIfDisposed();
        int[,] codes = GenerateCodes(backend, lyricsTokens, maxFrames, seed, muqLmEmbed);
        if (codes.GetLength(1) == 0) return [];
        return _codec.Decode(backend, codes, seed);
    }

    private Tensor BuildContext(ReadOnlySpan<int> lyrics, List<int[]> frames, Tensor? muqLmEmbed)
    {
        // Concatenate optional MuQ style row + lyrics text embeddings + prior audio-frame embeddings.
        int h = _cfg.Lm.Backbone.HiddenSize;
        int muqRows = muqLmEmbed is not null ? 1 : 0;
        int total = muqRows + lyrics.Length + frames.Count;
        Tensor ctx = new(new TensorShape(1, Math.Max(1, total), h), DType.F32);
        float* cp = (float*)ctx.DataPointer;
        int row = 0;
        if (muqLmEmbed is not null)
        {
            if (muqLmEmbed.ElementCount != h)
                throw new ArgumentException($"MuQ LM embedding must have {h} elements, got {muqLmEmbed.ElementCount}.", nameof(muqLmEmbed));
            Buffer.MemoryCopy((void*)muqLmEmbed.DataPointer, cp + (long)row++ * h, h * 4, h * 4);
        }
        for (int i = 0; i < lyrics.Length; i++)
        {
            Tensor e = _lm.EmbedText(lyrics[i]);
            Buffer.MemoryCopy((void*)e.DataPointer, cp + (long)row++ * h, h * 4, h * 4); e.Dispose();
        }
        foreach (int[] fr in frames)
        {
            Tensor e = _lm.EmbedAudioFrame(fr);
            Buffer.MemoryCopy((void*)e.DataPointer, cp + (long)row++ * h, h * 4, h * 4); e.Dispose();
        }
        return ctx;
    }

    public IEnumerable<Tensor> EnumerateWeights() => _lm.EnumerateWeights();

    /// <summary>Enumerates the HeartCodec decoder weights (for save/validation), separate from the LM set.</summary>
    public IEnumerable<Tensor> EnumerateCodecWeights() => _codec.EnumerateWeights();

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _lm.Dispose();
        _codec.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(HeartMulaPipeline));
    }
}
