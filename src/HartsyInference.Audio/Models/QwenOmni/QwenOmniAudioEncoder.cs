using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.QwenOmni;

/// <summary>Qwen2.5-Omni thinker audio tower: per-chunk conv stem, block-diagonal Whisper layers, 2x pool, ln_post and the LLM projection.</summary>
/// <remarks>Mel frames split into <c>2 * NWindow</c>-frame chunks that are convolved and attended independently (a short tail chunk at its own length) with positions restarting at 0; the outputs are concatenated, pooled by 2 over the whole sequence, normed and projected.</remarks>
public sealed unsafe class QwenOmniAudioEncoder
{
    private readonly QwenOmniConfig _cfg;
    private readonly WhisperConfig _layerCfg;
    private readonly WhisperEncoderLayer[] _layers;
    private readonly Tensor _positions;
    private Tensor[] _deviceWeights = [];
    private Tensor? _conv1Weight;
    private Tensor? _conv1Bias;
    private Tensor? _conv2Weight;
    private Tensor? _conv2Bias;
    private Tensor? _lnPostWeight;
    private Tensor? _lnPostBias;
    private Tensor? _projWeight;
    private Tensor? _projBias;
    private bool _loaded;
    private int _streamWarned;

    /// <summary>Creates the tower with generated sinusoidal positions and no weights; call <see cref="LoadWeights"/>.</summary>
    public QwenOmniAudioEncoder(QwenOmniConfig cfg)
    {
        ArgumentNullException.ThrowIfNull(cfg);
        cfg.Validate();
        _cfg = cfg;
        _layerCfg = cfg.ToWhisperLayerConfig();
        _layers = new WhisperEncoderLayer[cfg.Layers];
        for (int i = 0; i < _layers.Length; i++) _layers[i] = new WhisperEncoderLayer(_layerCfg);
        _positions = new Tensor(new TensorShape(cfg.NWindow, cfg.DModel), DType.F32);
        FillSinusoids(new Span<float>((float*)_positions.DataPointer, cfg.NWindow * cfg.DModel), cfg.NWindow, cfg.DModel);
    }

    /// <summary>The tower configuration.</summary>
    public QwenOmniConfig Config => _cfg;

    /// <summary>Writes the sinusoid table <c>[length, channels]</c>: <c>sin(t * s_i) | cos(t * s_i)</c>, <c>s_i = exp(-i * ln(10000) / (channels / 2 - 1))</c>.</summary>
    public static void FillSinusoids(Span<float> table, int length, int channels)
    {
        if (channels % 2 != 0 || channels < 4) throw new ArgumentException("channels must be even and at least 4.", nameof(channels));
        if (table.Length < length * channels) throw new ArgumentException("table is too small.", nameof(table));
        int half = channels / 2;
        double increment = Math.Log(10_000.0) / (half - 1);
        for (int t = 0; t < length; t++)
        {
            for (int i = 0; i < half; i++)
            {
                double angle = t * Math.Exp(-increment * i);
                table[t * channels + i] = (float)Math.Sin(angle);
                table[t * channels + half + i] = (float)Math.Cos(angle);
            }
        }
    }

    /// <summary>Every checkpoint key (relative to the prefix) with its expected shape; the unused <c>audio_bos_eos_token</c> is not required.</summary>
    public IEnumerable<(string Key, long[] Shape)> ExpectedWeights()
    {
        int d = _cfg.DModel;
        yield return ("conv1.weight", [d, _cfg.NumMelBins, 3]);
        yield return ("conv1.bias", [d]);
        yield return ("conv2.weight", [d, d, 3]);
        yield return ("conv2.bias", [d]);
        for (int i = 0; i < _cfg.Layers; i++)
        {
            string p = $"layers.{i}";
            yield return ($"{p}.self_attn_layer_norm.weight", [d]);
            yield return ($"{p}.self_attn_layer_norm.bias", [d]);
            yield return ($"{p}.self_attn.q_proj.weight", [d, d]);
            yield return ($"{p}.self_attn.q_proj.bias", [d]);
            yield return ($"{p}.self_attn.k_proj.weight", [d, d]);
            yield return ($"{p}.self_attn.v_proj.weight", [d, d]);
            yield return ($"{p}.self_attn.v_proj.bias", [d]);
            yield return ($"{p}.self_attn.out_proj.weight", [d, d]);
            yield return ($"{p}.self_attn.out_proj.bias", [d]);
            yield return ($"{p}.fc1.weight", [_cfg.FfnDim, d]);
            yield return ($"{p}.fc1.bias", [_cfg.FfnDim]);
            yield return ($"{p}.fc2.weight", [d, _cfg.FfnDim]);
            yield return ($"{p}.fc2.bias", [d]);
            yield return ($"{p}.final_layer_norm.weight", [d]);
            yield return ($"{p}.final_layer_norm.bias", [d]);
        }
        yield return ("ln_post.weight", [d]);
        yield return ("ln_post.bias", [d]);
        yield return ("proj.weight", [_cfg.OutputDim, d]);
        yield return ("proj.bias", [_cfg.OutputDim]);
    }

    /// <summary>Loads the tower from a safetensors dictionary, failing before any state changes if a key is missing or mis-shaped.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "thinker.audio_tower")
    {
        ArgumentNullException.ThrowIfNull(weights);
        List<string> problems = [];
        foreach ((string key, long[] shape) in ExpectedWeights())
        {
            string full = $"{prefix}.{key}";
            if (!weights.TryGetValue(full, out Tensor? t))
            {
                problems.Add($"missing {full}");
                continue;
            }
            if (!new TensorShape(shape).Equals(t.Shape))
            {
                problems.Add($"{full} has shape {t.Shape}, expected [{string.Join(", ", shape)}]");
            }
        }
        if (problems.Count > 0)
        {
            int shown = Math.Min(problems.Count, 8);
            throw new InvalidOperationException(
                $"Qwen2.5-Omni audio tower weights are invalid ({problems.Count} problems): {string.Join("; ", problems.GetRange(0, shown))}");
        }

        int d = _cfg.DModel;
        _conv1Weight = WhisperOps.EnsureF32(weights[$"{prefix}.conv1.weight"]).Reshape(new TensorShape(d, _cfg.NumMelBins, 1, 3));
        _conv1Bias = WhisperOps.EnsureF32(weights[$"{prefix}.conv1.bias"]);
        _conv2Weight = WhisperOps.EnsureF32(weights[$"{prefix}.conv2.weight"]).Reshape(new TensorShape(d, d, 1, 3));
        _conv2Bias = WhisperOps.EnsureF32(weights[$"{prefix}.conv2.bias"]);
        for (int i = 0; i < _layers.Length; i++) _layers[i].LoadWeights(weights, $"{prefix}.layers.{i}");
        _lnPostWeight = WhisperOps.EnsureF32(weights[$"{prefix}.ln_post.weight"]);
        _lnPostBias = WhisperOps.EnsureF32(weights[$"{prefix}.ln_post.bias"]);
        _projWeight = WhisperOps.EnsureF32(weights[$"{prefix}.proj.weight"]);
        _projBias = WhisperOps.EnsureF32(weights[$"{prefix}.proj.bias"]);

        List<Tensor> device = [_conv1Weight, _conv1Bias, _conv2Weight, _conv2Bias, _lnPostWeight, _lnPostBias, _projWeight, _projBias];
        foreach (WhisperEncoderLayer layer in _layers) device.AddRange(layer.EnumerateWeights());
        _deviceWeights = [.. device];
        _loaded = true;
    }

    /// <summary>Encodes the first <paramref name="frames"/> columns of <paramref name="mel"/> <c>[NumMelBins, T]</c> to <c>[AudioTokens(frames), OutputDim]</c>; the caller owns the result.</summary>
    public Tensor Forward(IBackend backend, Tensor mel, int frames)
    {
        if (!_loaded) throw new InvalidOperationException("Call LoadWeights before Forward.");
        if (mel.DType != DType.F32 || mel.Shape.Rank != 2 || (int)mel.Shape[0] != _cfg.NumMelBins)
        {
            throw new ArgumentException($"mel must be F32 [{_cfg.NumMelBins}, T]; got {mel.Shape}.", nameof(mel));
        }
        int rowStride = (int)mel.Shape[1];
        if (frames < 1 || frames > rowStride) throw new ArgumentOutOfRangeException(nameof(frames), $"frames {frames} outside [1, {rowStride}].");
        int tokens = QwenOmniProcessor.AudioTokens(frames);
        if (tokens < 1) throw new ArgumentException($"{frames} mel frames are too short to yield an audio token (need at least 3).", nameof(frames));

        WhisperOps.PreloadOrStream(backend, _deviceWeights, "qwen-omni-audio", ref _streamWarned);
        int chunk = _cfg.ChunkFrames;
        int fullCount = frames / chunk;
        int tailLength = frames % chunk;
        int convTotal = QwenOmniProcessor.ConvFrames(frames);

        Tensor? full = fullCount > 0 ? RunChunks(backend, mel, rowStride, 0, fullCount, chunk) : null;
        Tensor? tail = tailLength > 0 ? RunChunks(backend, mel, rowStride, fullCount * chunk, 1, tailLength) : null;
        Tensor hidden = Join(backend, full, tail, convTotal);

        int pooled = convTotal / 2;
        int[] even = new int[pooled];
        int[] odd = new int[pooled];
        for (int j = 0; j < pooled; j++)
        {
            even[j] = 2 * j;
            odd[j] = 2 * j + 1;
        }
        int d = _cfg.DModel;
        Tensor a = new(new TensorShape(pooled, d), DType.F32);
        Tensor b = new(new TensorShape(pooled, d), DType.F32);
        backend.GatherRows(a, hidden, even);
        backend.GatherRows(b, hidden, odd);
        hidden.Dispose();
        Tensor sum = new(a.Shape, DType.F32);
        backend.Add(sum, a, b);
        a.Dispose();
        b.Dispose();
        Tensor avg = new(a.Shape, DType.F32);
        backend.Scale(avg, sum, 0.5f);
        sum.Dispose();

        Tensor normed = new(avg.Shape, DType.F32);
        backend.LayerNorm(normed, avg, _lnPostWeight!, _lnPostBias!, _cfg.LayerNormEps);
        avg.Dispose();
        Tensor output = new(new TensorShape(pooled, _cfg.OutputDim), DType.F32);
        backend.Linear(output, normed, _projWeight!, _projBias);
        normed.Dispose();
        return output;
    }

    private Tensor Join(IBackend backend, Tensor? full, Tensor? tail, int convTotal)
    {
        int d = _cfg.DModel;
        if (full is not null && tail is not null)
        {
            Tensor joined = new(new TensorShape(convTotal, d), DType.F32);
            backend.Concat(joined, [full, tail], 0);
            full.Dispose();
            tail.Dispose();
            return joined;
        }
        return full ?? tail!;
    }

    /// <summary>Runs <paramref name="count"/> chunks of <paramref name="length"/> mel frames as one batch and returns the frames as <c>[count * convLength, DModel]</c>.</summary>
    private Tensor RunChunks(IBackend backend, Tensor mel, int rowStride, int start, int count, int length)
    {
        int d = _cfg.DModel;
        int bins = _cfg.NumMelBins;
        int convLength = QwenOmniProcessor.ConvFrames(length);

        Tensor input = new(new TensorShape(count, bins, 1, length), DType.F32);
        float* src = (float*)mel.DataPointer;
        float* dst = (float*)input.DataPointer;
        for (int c = 0; c < count; c++)
        {
            for (int m = 0; m < bins; m++)
            {
                Buffer.MemoryCopy(src + (long)m * rowStride + start + (long)c * length,
                    dst + ((long)c * bins + m) * length, length * 4L, length * 4L);
            }
        }

        Tensor conv1 = new(new TensorShape(count, d, 1, length), DType.F32);
        backend.Conv2D(conv1, input, _conv1Weight!, _conv1Bias, strideH: 1, strideW: 1, padH: 0, padW: 1);
        input.Dispose();
        Tensor act1 = new(conv1.Shape, DType.F32);
        backend.GeluErf(act1, conv1);
        conv1.Dispose();
        Tensor conv2 = new(new TensorShape(count, d, 1, convLength), DType.F32);
        backend.Conv2D(conv2, act1, _conv2Weight!, _conv2Bias, strideH: 1, strideW: 2, padH: 0, padW: 1);
        act1.Dispose();
        Tensor act2 = new(conv2.Shape, DType.F32);
        backend.GeluErf(act2, conv2);
        conv2.Dispose();
        Tensor seq = new(new TensorShape(count, convLength, d), DType.F32);
        backend.Transpose2D(seq, act2, d, convLength);
        act2.Dispose();

        Tensor pos = new(seq.Shape, DType.F32);
        float* pSrc = (float*)_positions.DataPointer;
        float* pDst = (float*)pos.DataPointer;
        for (int c = 0; c < count; c++)
        {
            Buffer.MemoryCopy(pSrc, pDst + (long)c * convLength * d, (long)convLength * d * 4L, (long)convLength * d * 4L);
        }
        Tensor hidden = new(seq.Shape, DType.F32);
        backend.Add(hidden, seq, pos);
        seq.Dispose();
        pos.Dispose();

        foreach (WhisperEncoderLayer layer in _layers)
        {
            Tensor next = layer.Forward(backend, hidden);
            hidden.Dispose();
            hidden = next;
        }
        Tensor flat = hidden.Reshape(new TensorShape((long)count * convLength, d));
        return flat;
    }
}
