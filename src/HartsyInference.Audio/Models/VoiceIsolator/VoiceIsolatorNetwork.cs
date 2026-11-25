using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Layers;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Audio.Models.VoiceIsolator;

/// <summary>The Inworld Voice Isolator network over one clip: STFT, band tokens, dual-path attention and Mamba blocks, per-band
/// decoder heads, a bounded complex mask plus residual, and the inverse STFT added to the input. Weights stay F32 and borrow the
/// memory map of their file.</summary>
public sealed class VoiceIsolatorNetwork : IDisposable
{
    private readonly SafeTensorsLoader _loader;
    private readonly IReadOnlyDictionary<string, Tensor> _weights;
    private readonly (int Start, int End)[] _edges;
    private readonly object?[] _blocks;
    private readonly Tensor _bandEmbedding;
    private readonly Tensor? _lookaheadEmbedding;
    private readonly float[] _decoderScale;
    private readonly float[] _decoderShift;
    private readonly VoiceIsolatorSpectral _spectral;
    private int _disposed;

    private VoiceIsolatorNetwork(SafeTensorsLoader loader, VoiceIsolatorConfig config)
    {
        _loader = loader;
        Config = config;
        _weights = loader.GetAllTensors();
        foreach (Tensor tensor in _weights.Values)
        {
            if (tensor.DType != DType.F32)
            {
                throw new InvalidDataException($"Voice isolator weights are F32; a tensor is {tensor.DType.Name}.");
            }
        }
        _edges = VoiceIsolatorBands.Edges(config);
        _spectral = new VoiceIsolatorSpectral(config.NFft, config.HopLength);
        _bandEmbedding = _weights["band_embedding"];
        if (_bandEmbedding.Shape[0] != _edges.Length || _bandEmbedding.Shape[1] != config.Width)
        {
            throw new InvalidDataException($"band_embedding is {_bandEmbedding.Shape}, expected [{_edges.Length}, {config.Width}] for this config.");
        }
        _lookaheadEmbedding = _weights.TryGetValue("lookahead_embedding.weight", out Tensor? embedding) ? embedding : null;
        if (config.LookaheadChoices.Count > 0 != (_lookaheadEmbedding is not null))
        {
            throw new InvalidDataException("lookahead_frames_choices and lookahead_embedding.weight must be present together.");
        }
        if (_lookaheadEmbedding is not null)
        {
            VoiceIsolatorOps.RequireF32(_lookaheadEmbedding, "lookahead_embedding.weight", (long)config.LookaheadChoices.Count * config.Width);
        }
        _blocks = new object?[config.BlockTypes.Count];
        for (int i = 0; i < _blocks.Length; i++)
        {
            _blocks[i] = config.BlockTypes[i] == "transformer"
                ? new VoiceIsolatorAttentionBlock(config, _weights, $"blocks.{i}")
                : new VoiceIsolatorMambaBlock(config, _weights, $"blocks.{i}");
        }
        (_decoderScale, _decoderShift) = DecoderCondition(_weights, config.Width);
    }

    /// <summary>Architecture of the loaded checkpoint.</summary>
    public VoiceIsolatorConfig Config { get; }

    /// <summary>Number of bands.</summary>
    public int BandCount => _edges.Length;

    /// <summary>The host STFT shared by whole-clip and streaming runs.</summary>
    internal VoiceIsolatorSpectral Spectral => _spectral;

    /// <summary>Weights by name, for the streaming session and tests.</summary>
    internal IReadOnlyDictionary<string, Tensor> Weights => _weights;

    /// <summary>Reads a checkpoint: the architecture comes from its metadata, never from arguments.</summary>
    public static VoiceIsolatorNetwork Load(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        SafeTensorsLoader loader = new();
        try
        {
            loader.Load(path);
            if (loader.Metadata is null || !loader.Metadata.TryGetValue("config", out string? json))
            {
                throw new InvalidDataException($"{path}: safetensors metadata lacks the 'config' entry; refusing to guess the architecture.");
            }
            return new VoiceIsolatorNetwork(loader, VoiceIsolatorConfig.FromJson(json));
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }

    /// <summary>Isolates the voice of <paramref name="audio"/> (mono, at <see cref="VoiceIsolatorConfig.SampleRate"/>) in one pass.
    /// The offline checkpoint pads the edges by reflection as the reference's <c>forward</c> does; the causal checkpoint pads with
    /// zeros, as its stream frames the start and end of its input, and reads the look-ahead frames of the decoder.</summary>
    internal float[] Isolate(IBackend backend, float[] audio, int lookahead)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(audio);
        if (Config.Causal)
        {
            ValidateLookahead(lookahead);
        }
        else if (lookahead != 0)
        {
            throw new ArgumentException("An offline checkpoint has no look-ahead; pass 0.", nameof(lookahead));
        }
        if (audio.Length == 0)
        {
            throw new ArgumentException("There is nothing to isolate in empty audio.", nameof(audio));
        }
        int frames = _spectral.FrameCount(audio.Length);
        int bins = _spectral.Bins;
        float[] re = new float[(long)frames * bins];
        float[] im = new float[(long)frames * bins];
        _spectral.Analyze(audio, re, im, reflect: !Config.Causal);
        float[] raw = Backbone(backend, re, im, frames, lookahead);
        if (Config.Causal && lookahead > 0)
        {
            raw = ReadAhead(raw, frames, bins, lookahead);
        }
        Correct(raw, re, im, frames, bins, Config.CorrectionBound);
        float[] delta = _spectral.Synthesize(re, im, frames, audio.Length);
        for (int i = 0; i < delta.Length; i++)
        {
            delta[i] += audio[i];
        }
        return delta;
    }

    /// <summary>The backbone and decoder over spectra <c>[frames, bins]</c>: raw decoder output <c>[frames, bins, 4]</c>.</summary>
    internal float[] Backbone(IBackend backend, float[] re, float[] im, int frames, int lookahead)
    {
        int width = Config.Width;
        int bands = _edges.Length;
        (float[] cos, float[] sin) = VoiceIsolatorOps.RopeTables(frames, Config.HeadDim);
        Tensor x = Encode(backend, re, im, frames, lookahead);
        bool causal = Config.Causal;
        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i] is VoiceIsolatorAttentionBlock attention
                ? attention.Forward(backend, x, bands, frames, causal, cos, sin)
                : ((VoiceIsolatorMambaBlock)_blocks[i]!).Forward(backend, x, bands, frames, null);
            x.Dispose();
            x = next;
        }
        float[] raw = Decode(backend, x, frames);
        x.Dispose();
        return raw;
    }

    /// <summary>A fresh lane for a stream: empty attention pasts and zero Mamba carries.</summary>
    internal VoiceIsolatorLane NewLane()
    {
        VoiceIsolatorLane lane = new(_blocks.Length);
        for (int i = 0; i < _blocks.Length; i++)
        {
            if (_blocks[i] is VoiceIsolatorAttentionBlock)
            {
                lane.Attention[i] = new VoiceIsolatorAttentionLane();
            }
            else
            {
                lane.Carry[i] = new VoiceIsolatorMambaCarry(
                    VoiceIsolatorLane.Zeros(_edges.Length, Config.MambaConv - 1, Config.MambaInner),
                    VoiceIsolatorLane.Zeros(_edges.Length, Config.MambaInner, Config.MambaState));
            }
        }
        return lane;
    }

    /// <summary>The blocks over the new frames' tokens <paramref name="features"/> (not disposed) with <paramref name="lane"/>'s carried state.</summary>
    internal Tensor StreamBlocks(IBackend backend, Tensor features, int frames, VoiceIsolatorLane lane, VoiceIsolatorStreamAttention attention)
    {
        Tensor x = features;
        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i] is VoiceIsolatorAttentionBlock block
                ? block.Forward(backend, x, _edges.Length, frames, true, attention.Cos, attention.Sin, lane.Attention[i], attention)
                : ((VoiceIsolatorMambaBlock)_blocks[i]!).Forward(backend, x, _edges.Length, frames, lane.Carry[i]);
            if (!ReferenceEquals(x, features))
            {
                x.Dispose();
            }
            x = next;
        }
        return x;
    }

    /// <summary>Band tokens <c>[bands * frames, width]</c> of the spectra.</summary>
    internal unsafe Tensor Encode(IBackend backend, float[] re, float[] im, int frames, int lookahead)
    {
        int width = Config.Width;
        int bins = _spectral.Bins;
        Tensor tokens = new(new TensorShape(1, (long)_edges.Length * frames, width), DType.F32);
        float* tokenP = (float*)tokens.DataPointer;
        float* embeddingP = (float*)_bandEmbedding.DataPointer;
        float* lookaheadP = _lookaheadEmbedding is null ? null
            : (float*)_lookaheadEmbedding.DataPointer + (long)LookaheadRow(lookahead) * width;
        for (int band = 0; band < _edges.Length; band++)
        {
            (int start, int end) = _edges[band];
            int count = 2 * (end - start);
            Tensor input = new(new TensorShape(1, frames, count), DType.F32);
            float* inP = (float*)input.DataPointer;
            for (int f = 0; f < frames; f++)
            {
                for (int j = start; j < end; j++)
                {
                    inP[(long)f * count + 2 * (j - start)] = re[(long)f * bins + j];
                    inP[(long)f * count + 2 * (j - start) + 1] = im[(long)f * bins + j];
                }
            }
            Tensor normed = new(input.Shape, DType.F32);
            backend.LayerNorm(normed, input, _weights[$"encoders.{band}.0.weight"], _weights[$"encoders.{band}.0.bias"], 1e-5f);
            input.Dispose();
            Tensor projected = WhisperOps.ProjectLinear(backend, normed, _weights[$"encoders.{band}.1.weight"],
                _weights[$"encoders.{band}.1.bias"], 1, frames, count, width);
            normed.Dispose();
            float* projP = (float*)projected.DataPointer;
            float* bandEmbedding = embeddingP + (long)band * width;
            for (int f = 0; f < frames; f++)
            {
                float* dst = tokenP + ((long)band * frames + f) * width;
                float* src = projP + (long)f * width;
                for (int c = 0; c < width; c++)
                {
                    dst[c] = src[c] + bandEmbedding[c] + (lookaheadP is null ? 0f : lookaheadP[c]);
                }
            }
            projected.Dispose();
        }
        return tokens;
    }

    /// <summary>The per-band heads over <paramref name="features"/> <c>[bands * frames, width]</c>: <c>[frames, bins, 4]</c>.</summary>
    internal unsafe float[] Decode(IBackend backend, Tensor features, int frames)
    {
        int width = Config.Width;
        int bins = _spectral.Bins;
        float[] raw = new float[(long)frames * bins * 4];
        float* featureP = (float*)features.DataPointer;
        for (int band = 0; band < _edges.Length; band++)
        {
            (int start, int end) = _edges[band];
            int bandBins = end - start;
            Tensor rows = new(new TensorShape(1, frames, width), DType.F32);
            float* rowP = (float*)rows.DataPointer;
            float* source = featureP + (long)band * frames * width;
            for (int f = 0; f < frames; f++)
            {
                for (int c = 0; c < width; c++)
                {
                    rowP[(long)f * width + c] = source[(long)f * width + c] * (1f + _decoderScale[c]) + _decoderShift[c];
                }
            }
            Tensor normed = new(rows.Shape, DType.F32);
            backend.LayerNorm(normed, rows, _weights[$"denoise_decoder.heads.{band}.0.weight"], _weights[$"denoise_decoder.heads.{band}.0.bias"], 1e-5f);
            rows.Dispose();
            Tensor hidden = WhisperOps.ProjectLinear(backend, normed, _weights[$"denoise_decoder.heads.{band}.1.weight"],
                _weights[$"denoise_decoder.heads.{band}.1.bias"], 1, frames, width, width);
            normed.Dispose();
            Activations.ErfGelu(hidden);
            Tensor head = WhisperOps.ProjectLinear(backend, hidden, _weights[$"denoise_decoder.heads.{band}.3.weight"],
                _weights[$"denoise_decoder.heads.{band}.3.bias"], 1, frames, width, 4 * bandBins);
            hidden.Dispose();
            float* headP = (float*)head.DataPointer;
            for (int f = 0; f < frames; f++)
            {
                for (int j = 0; j < bandBins; j++)
                {
                    for (int c = 0; c < 4; c++)
                    {
                        raw[((long)f * bins + start + j) * 4 + c] = headP[(long)f * 4 * bandBins + j * 4 + c];
                    }
                }
            }
            head.Dispose();
        }
        return raw;
    }

    /// <summary>The decoder's output for frame <c>t</c> taken from frame <c>t + lookahead</c>, the last frame serving the tail.</summary>
    internal static float[] ReadAhead(float[] raw, int frames, int bins, int lookahead)
    {
        float[] shifted = new float[raw.Length];
        long frameSize = (long)bins * 4;
        for (int f = 0; f < frames; f++)
        {
            int source = Math.Min(f + lookahead, frames - 1);
            Array.Copy(raw, source * frameSize, shifted, f * frameSize, frameSize);
        }
        return shifted;
    }

    /// <summary>Turns the raw decoder output into the corrected spectra, overwriting <paramref name="re"/> and <paramref name="im"/>:
    /// <c>spec * 2 tanh(mask) + bound * |spec| * tanh(residual) / max(1, |tanh(residual)|)</c>.</summary>
    internal static void Correct(float[] raw, float[] re, float[] im, int frames, int bins, float bound)
    {
        long total = (long)frames * bins;
        CpuParallel.ForRanges(total, 4096, 16, (start, length) =>
        {
            for (long i = start; i < start + length; i++)
            {
                float maskRe = 2f * MathF.Tanh(raw[i * 4]);
                float maskIm = 2f * MathF.Tanh(raw[i * 4 + 1]);
                float resRe = MathF.Tanh(raw[i * 4 + 2]);
                float resIm = MathF.Tanh(raw[i * 4 + 3]);
                float norm = MathF.Max(MathF.Sqrt(resRe * resRe + resIm * resIm), 1f);
                float magnitude = MathF.Sqrt(re[i] * re[i] + im[i] * im[i]);
                float scale = bound * magnitude / norm;
                float outRe = re[i] * maskRe - im[i] * maskIm + resRe * scale;
                float outIm = re[i] * maskIm + im[i] * maskRe + resIm * scale;
                re[i] = outRe;
                im[i] = outIm;
            }
        });
    }

    private static (float[] Scale, float[] Shift) DecoderCondition(IReadOnlyDictionary<string, Tensor> weights, int width)
    {
        Tensor w = weights["denoise_decoder.condition.weight"];
        Tensor b = weights["denoise_decoder.condition.bias"];
        VoiceIsolatorOps.RequireF32(w, "denoise_decoder.condition.weight", 2L * 2 * width);
        VoiceIsolatorOps.RequireF32(b, "denoise_decoder.condition.bias", 2L * width);
        float[] scale = new float[width];
        float[] shift = new float[width];
        unsafe
        {
            float* wp = (float*)w.DataPointer;
            float* bp = (float*)b.DataPointer;
            // controls are the constant [1, 0], so only the first weight column contributes
            for (int c = 0; c < width; c++)
            {
                scale[c] = wp[(long)c * 2] + bp[c];
                shift[c] = wp[(long)(width + c) * 2] + bp[width + c];
            }
        }
        return (scale, shift);
    }

    /// <summary>Checks a look-ahead against the positions the checkpoint trained at; each stream passes its own value.</summary>
    internal int ValidateLookahead(int frames)
    {
        IReadOnlyList<int> choices = Config.LookaheadChoices;
        bool trained = choices.Count > 0 ? choices.Contains(frames) : frames == Config.LookaheadFrames;
        if (!Config.Causal || !trained)
        {
            throw new ArgumentException($"Look-ahead {frames} is not a trained position of this checkpoint.");
        }
        return frames;
    }

    /// <summary>Row of the look-ahead embedding for a trained position.</summary>
    private int LookaheadRow(int lookahead)
    {
        IReadOnlyList<int> choices = Config.LookaheadChoices;
        for (int row = 0; row < choices.Count; row++)
        {
            if (choices[row] == lookahead)
            {
                return row;
            }
        }
        throw new ArgumentException($"Look-ahead {lookahead} is not a trained position of this checkpoint.");
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        foreach (object? block in _blocks)
        {
            (block as IDisposable)?.Dispose();
        }
        _loader.Dispose();
    }
}
