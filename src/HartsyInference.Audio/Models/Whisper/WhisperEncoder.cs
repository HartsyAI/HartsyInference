using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Whisper;

/// <summary>Whisper audio encoder — takes a log-mel spectrogram and produces per-frame hidden states used as cross-attention keys/values by the decoder.</summary>
/// <remarks>Architecture (per <c>WHISPER_ARCHITECTURE.md</c>):
/// <code>
/// mel [B, n_mels, 3000]
///   → Conv1D(k=3, s=1, p=1) + GELU   → [B, d_model, 3000]
///   → Conv1D(k=3, s=2, p=1) + GELU   → [B, d_model, 1500]
///   → transpose to [B, 1500, d_model]
///   → + sinusoidal_pos_embed [1500, d_model]
///   → N × ResidualAttentionBlock (pre-norm self-attn → MLP)
///   → final_layer_norm
///   → [B, 1500, d_model]
/// </code>
///
/// <para>Conv1D is dispatched through <see cref="IBackend.Conv2D"/> by treating the
/// 1-D weight <c>[out, in, k]</c> as a 2-D weight <c>[out, in, 1, k]</c> with
/// <c>strideH = 1, padH = 0</c>. This matches the existing diffusion pattern for
/// running 1-D ops on a 2-D backend.</para>
///
/// <para><b>Attention scale</b>: Whisper's original Python code multiplies Q and K
/// each by <c>head_dim^(-0.25)</c>, which is algebraically identical to the standard
/// <c>head_dim^(-0.5)</c> on QK^T. We use the standard form via
/// <see cref="IBackend.ScaledDotProductAttention"/> and the weights load unchanged.</para>
///
/// <para><b>Key projection has no bias</b>: HF safetensors omit
/// <c>self_attn.k_proj.bias</c>; <see cref="WhisperOps.ProjectLinear"/> handles a null
/// bias by skipping the broadcast-add.</para></remarks>
public sealed unsafe class WhisperEncoder : IDisposable
{
    private readonly WhisperConfig _cfg;
    private readonly WhisperEncoderLayer[] _layers;

    // Conv stem
    private Tensor? _conv1Weight;   // [d_model, n_mels, 3] in the file; reshaped to [d_model, n_mels, 1, 3] for Conv2D
    private Tensor? _conv1Bias;
    private Tensor? _conv2Weight;   // [d_model, d_model, 3] → [d_model, d_model, 1, 3]
    private Tensor? _conv2Bias;
    private Tensor? _embedPositions; // [max_audio_positions, d_model] sinusoidal materialized

    // Final layer norm
    private Tensor? _finalLnWeight;
    private Tensor? _finalLnBias;

    // Everything a device op reads, kept resident across forwards. The position table is read through a row view
    // (see PositionRows), which auto-promotes and demotes with the view's own lifetime.
    private Tensor[] _deviceWeights = [];
    private Tensor? _positionRows;

    private bool _weightsLoaded;
    private int _disposed;

    public WhisperConfig Config => _cfg;

    /// <summary>Creates an encoder with no weights loaded — call <see cref="LoadWeights"/> after construction.</summary>
    public WhisperEncoder(WhisperConfig cfg)
    {
        _cfg = cfg;
        _layers = new WhisperEncoderLayer[cfg.EncoderLayers];
        for (int i = 0; i < cfg.EncoderLayers; i++)
            _layers[i] = new WhisperEncoderLayer(cfg);
    }

    /// <summary>Loads weights from a HuggingFace safetensors dictionary; the default prefix matches the standard HF transformers Whisper save layout (<c>model.encoder.*</c>) — pass a different prefix if your conversion script strips or remaps it.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "model.encoder")
    {
        // Conv weights ship as 3-D [out, in, k] in HF safetensors. Reshape to 4-D
        // for our Conv2D op without copying — Reshape returns a view onto the same
        // memory.
        Tensor conv1Raw = WhisperOps.EnsureF32(weights[$"{prefix}.conv1.weight"]);
        Tensor conv2Raw = WhisperOps.EnsureF32(weights[$"{prefix}.conv2.weight"]);
        _conv1Weight = conv1Raw.Reshape(new TensorShape(_cfg.HiddenSize, _cfg.NumMelBins, 1, 3));
        _conv2Weight = conv2Raw.Reshape(new TensorShape(_cfg.HiddenSize, _cfg.HiddenSize, 1, 3));
        _conv1Bias = WhisperOps.EnsureF32(weights[$"{prefix}.conv1.bias"]);
        _conv2Bias = WhisperOps.EnsureF32(weights[$"{prefix}.conv2.bias"]);

        _embedPositions = WhisperOps.EnsureF32(weights[$"{prefix}.embed_positions.weight"]);

        for (int i = 0; i < _layers.Length; i++)
            _layers[i].LoadWeights(weights, $"{prefix}.layers.{i}");

        _finalLnWeight = WhisperOps.EnsureF32(weights[$"{prefix}.layer_norm.weight"]);
        _finalLnBias = WhisperOps.EnsureF32(weights[$"{prefix}.layer_norm.bias"]);

        List<Tensor> device = [_conv1Weight, _conv1Bias, _conv2Weight, _conv2Bias];
        foreach (WhisperEncoderLayer layer in _layers) device.AddRange(layer.EnumerateWeights());
        device.Add(_finalLnWeight);
        device.Add(_finalLnBias);
        _deviceWeights = [.. device];
        _weightsLoaded = true;
    }

    /// <summary>Forward pass: <c>mel [B, n_mels, n_frames]</c> → <c>features [B, n_frames/2, d_model]</c>; <paramref name="mel"/> typically has <c>n_frames = 3000</c> (Whisper's 30-second chunk) but the encoder runs on any frame count for ablations / unit tests.</summary>
    public Tensor Forward(IBackend backend, Tensor mel) => Forward(backend, mel, timer: null);

    /// <summary><see cref="Forward(IBackend, Tensor)"/> with the stem and each layer attributed to <paramref name="timer"/>.</summary>
    internal Tensor Forward(IBackend backend, Tensor mel, WhisperStageTimer? timer)
    {
        ThrowIfDisposed();
        if (!_weightsLoaded) throw new InvalidOperationException("Call LoadWeights before Forward.");
        if (mel.Shape.Rank != 3) throw new ArgumentException($"mel must be 3-D [B, n_mels, n_frames]; got rank {mel.Shape.Rank}");
        if ((int)mel.Shape[1] != _cfg.NumMelBins)
            throw new ArgumentException($"mel n_mels {(int)mel.Shape[1]} != config NumMelBins {_cfg.NumMelBins}");

        int batch = (int)mel.Shape[0];
        int nFrames = (int)mel.Shape[2];
        int d = _cfg.HiddenSize;

        // Idempotent, so an eviction since the last forward is simply undone; the biases and norms are below the
        // auto-promotion floor and would otherwise upload on every op.
        backend.PreloadWeights(_deviceWeights);

        // Stage 1: Conv1 (stride=1, pad=1) + GELU. Treat 1-D conv as Conv2D with H=1.
        Tensor mel4d = mel.Reshape(new TensorShape(batch, _cfg.NumMelBins, 1, nFrames));
        Tensor conv1Out = new(new TensorShape(batch, d, 1, nFrames), DType.F32);
        backend.Conv2D(conv1Out, mel4d, _conv1Weight!, _conv1Bias, strideH: 1, strideW: 1, padH: 0, padW: 1);
        mel4d.Dispose();
        Tensor gelu1 = new(conv1Out.Shape, DType.F32);
        backend.Gelu(gelu1, conv1Out);
        conv1Out.Dispose();

        // Stage 2: Conv2 (stride=2, pad=1) + GELU.
        int nFrames2 = (nFrames + 2 * 1 - 3) / 2 + 1;
        Tensor conv2Out = new(new TensorShape(batch, d, 1, nFrames2), DType.F32);
        backend.Conv2D(conv2Out, gelu1, _conv2Weight!, _conv2Bias, strideH: 1, strideW: 2, padH: 0, padW: 1);
        gelu1.Dispose();
        Tensor gelu2 = new(conv2Out.Shape, DType.F32);
        backend.Gelu(gelu2, conv2Out);
        conv2Out.Dispose();

        // Stage 3: transpose [B, d, 1, T] → [B, T, d]. Transpose2D reads the flat buffer with explicit dims, so the
        // H=1 axis needs no Reshape view — which would read the device-resident activation back to the host.
        Tensor transposed = new(new TensorShape(batch, nFrames2, d), DType.F32);
        backend.Transpose2D(transposed, gelu2, d, nFrames2);
        gelu2.Dispose();

        // Stage 4: add sinusoidal positional embedding [n_frames2, d], broadcast over batch.
        // The HF checkpoint stores 1500 positions; we slice the leading nFrames2 rows.
        Tensor hidden = AddPositions(backend, transposed, batch, nFrames2, d);
        timer?.Accumulate("enc.stem");

        // Stage 5: N residual attention blocks.
        for (int i = 0; i < _layers.Length; i++)
        {
            Tensor next = _layers[i].Forward(backend, hidden);
            hidden.Dispose();
            hidden = next;
            timer?.Accumulate("enc.layer");
        }

        // Stage 6: final layer norm.
        Tensor normed = new(hidden.Shape, DType.F32);
        backend.LayerNorm(normed, hidden, _finalLnWeight!, _finalLnBias!, _cfg.LayerNormEps);
        hidden.Dispose();
        return normed;
    }

    /// <summary>Adds positions <c>[0, seqLen)</c> of the sinusoidal table to <paramref name="x"/>, disposing it and
    /// returning the sum: one backend add at batch 1 (every Whisper decode), the host loop for larger batches.</summary>
    private Tensor AddPositions(IBackend backend, Tensor x, int batch, int seqLen, int d)
    {
        if (batch != 1)
        {
            AddSinusoidalPosBroadcast(x, _embedPositions!, batch, seqLen, d);
            return x;
        }
        Tensor sum = new(x.Shape, DType.F32);
        backend.Add(sum, x, PositionRows(seqLen));
        x.Dispose();
        return sum;
    }

    /// <summary>A view of the table's leading <paramref name="seqLen"/> rows, kept so its device copy is reused by
    /// every forward at that length (1499 for a 30 s window); a different length replaces it.</summary>
    private Tensor PositionRows(int seqLen)
    {
        if (_positionRows is not null && _positionRows.Shape[0] == seqLen)
        {
            return _positionRows;
        }
        if ((long)seqLen > _embedPositions!.Shape[0])
        {
            throw new ArgumentException(
                $"The encoder has {_embedPositions.Shape[0]} positions; {seqLen} frames after the conv stem need more.");
        }
        _positionRows?.Dispose();
        _positionRows = _embedPositions.SliceRows(0, seqLen);
        return _positionRows;
    }

    private static void AddSinusoidalPosBroadcast(Tensor target, Tensor pos, int batch, int seqLen, int hiddenSize)
    {
        float* tPtr = (float*)target.DataPointer;
        float* pPtr = (float*)pos.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
            {
                int tOff = (b * seqLen + s) * hiddenSize;
                int pOff = s * hiddenSize;
                for (int d = 0; d < hiddenSize; d++) tPtr[tOff + d] += pPtr[pOff + d];
            }
    }

    /// <summary>Enumerates every loaded weight tensor for backend preload.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_conv1Weight is not null) yield return _conv1Weight;
        if (_conv1Bias is not null) yield return _conv1Bias;
        if (_conv2Weight is not null) yield return _conv2Weight;
        if (_conv2Bias is not null) yield return _conv2Bias;
        if (_embedPositions is not null) yield return _embedPositions;
        foreach (WhisperEncoderLayer layer in _layers)
            foreach (Tensor t in layer.EnumerateWeights()) yield return t;
        if (_finalLnWeight is not null) yield return _finalLnWeight;
        if (_finalLnBias is not null) yield return _finalLnBias;
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(WhisperEncoder));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            // Cast-allocated tensors (when source was non-F32) are owned by EnsureF32's caller.
            // Pass-through F32 tensors are owned by the safetensors loader. We don't double-dispose.
            // The row view borrows the table's bytes; disposing it only drops its device copy.
            _positionRows?.Dispose();
        }
    }
}

/// <summary>Single Whisper encoder block: pre-norm self-attention → residual → pre-norm MLP → residual; identical structure to GPT-2 / CLIP — Whisper diverges only in the lack of a key bias and (in OpenAI's reference) the split Q/K scaling, both handled at the projection / SDPA level.</summary>
internal sealed unsafe class WhisperEncoderLayer
{
    private readonly WhisperConfig _cfg;

    private Tensor? _attnLnWeight;
    private Tensor? _attnLnBias;
    private Tensor? _qWeight;
    private Tensor? _qBias;
    private Tensor? _kWeight;
    private Tensor? _vWeight;
    private Tensor? _vBias;
    private Tensor? _outWeight;
    private Tensor? _outBias;

    private Tensor? _fc1Weight;
    private Tensor? _fc1Bias;
    private Tensor? _fc2Weight;
    private Tensor? _fc2Bias;
    private Tensor? _finalLnWeight;
    private Tensor? _finalLnBias;

    public WhisperEncoderLayer(WhisperConfig cfg) { _cfg = cfg; }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _attnLnWeight = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn_layer_norm.weight"]);
        _attnLnBias = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn_layer_norm.bias"]);
        _qWeight = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn.q_proj.weight"]);
        _qBias = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn.q_proj.bias"]);
        _kWeight = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn.k_proj.weight"]);
        // k_proj has no bias in Whisper.
        _vWeight = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn.v_proj.weight"]);
        _vBias = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn.v_proj.bias"]);
        _outWeight = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn.out_proj.weight"]);
        _outBias = WhisperOps.EnsureF32(weights[$"{prefix}.self_attn.out_proj.bias"]);

        _fc1Weight = WhisperOps.EnsureF32(weights[$"{prefix}.fc1.weight"]);
        _fc1Bias = WhisperOps.EnsureF32(weights[$"{prefix}.fc1.bias"]);
        _fc2Weight = WhisperOps.EnsureF32(weights[$"{prefix}.fc2.weight"]);
        _fc2Bias = WhisperOps.EnsureF32(weights[$"{prefix}.fc2.bias"]);

        _finalLnWeight = WhisperOps.EnsureF32(weights[$"{prefix}.final_layer_norm.weight"]);
        _finalLnBias = WhisperOps.EnsureF32(weights[$"{prefix}.final_layer_norm.bias"]);
    }

    public Tensor Forward(IBackend backend, Tensor hidden)
    {
        int batch = (int)hidden.Shape[0];
        int seqLen = (int)hidden.Shape[1];
        int d = _cfg.HiddenSize;
        TensorShape shape = new(batch, seqLen, d);

        // --- Self-attention sub-block ---
        Tensor normed = new(shape, DType.F32);
        backend.LayerNorm(normed, hidden, _attnLnWeight!, _attnLnBias!, _cfg.LayerNormEps);

        Tensor q = WhisperOps.ProjectLinear(backend, normed, _qWeight!, _qBias, batch, seqLen, d, d);
        Tensor k = WhisperOps.ProjectLinear(backend, normed, _kWeight!, bias: null, batch, seqLen, d, d);
        Tensor v = WhisperOps.ProjectLinear(backend, normed, _vWeight!, _vBias, batch, seqLen, d, d);
        normed.Dispose();

        // Head split and merge are backend permutes: the projections and the attention stay on the device.
        TensorShape mh = new(batch, _cfg.NumHeads, seqLen, _cfg.HeadDim);
        Tensor qMh = new(mh, DType.F32);
        Tensor kMh = new(mh, DType.F32);
        Tensor vMh = new(mh, DType.F32);
        backend.Permute0213(qMh, q, seqLen, _cfg.NumHeads, _cfg.HeadDim);
        backend.Permute0213(kMh, k, seqLen, _cfg.NumHeads, _cfg.HeadDim);
        backend.Permute0213(vMh, v, seqLen, _cfg.NumHeads, _cfg.HeadDim);
        q.Dispose(); k.Dispose(); v.Dispose();

        float scale = 1f / MathF.Sqrt(_cfg.HeadDim);
        Tensor attnOut = new(mh, DType.F32);
        backend.ScaledDotProductAttention(attnOut, qMh, kMh, vMh, mask: null, scale);
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        Tensor merged = new(shape, DType.F32);
        backend.Permute0213(merged, attnOut, _cfg.NumHeads, seqLen, _cfg.HeadDim);
        attnOut.Dispose();

        Tensor projected = WhisperOps.ProjectLinear(backend, merged, _outWeight!, _outBias, batch, seqLen, d, d);
        merged.Dispose();

        Tensor residual1 = new(shape, DType.F32);
        backend.Add(residual1, hidden, projected);
        projected.Dispose();

        // --- MLP sub-block ---
        Tensor normed2 = new(shape, DType.F32);
        backend.LayerNorm(normed2, residual1, _finalLnWeight!, _finalLnBias!, _cfg.LayerNormEps);

        Tensor fc1 = WhisperOps.ProjectLinear(backend, normed2, _fc1Weight!, _fc1Bias, batch, seqLen, d, _cfg.IntermediateSize);
        normed2.Dispose();
        Tensor activated = new(new TensorShape(batch, seqLen, _cfg.IntermediateSize), DType.F32);
        backend.Gelu(activated, fc1);
        fc1.Dispose();

        Tensor fc2 = WhisperOps.ProjectLinear(backend, activated, _fc2Weight!, _fc2Bias, batch, seqLen, _cfg.IntermediateSize, d);
        activated.Dispose();

        Tensor residual2 = new(shape, DType.F32);
        backend.Add(residual2, residual1, fc2);
        residual1.Dispose();
        fc2.Dispose();
        return residual2;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_attnLnWeight is not null) yield return _attnLnWeight;
        if (_attnLnBias is not null) yield return _attnLnBias;
        if (_qWeight is not null) yield return _qWeight;
        if (_qBias is not null) yield return _qBias;
        if (_kWeight is not null) yield return _kWeight;
        if (_vWeight is not null) yield return _vWeight;
        if (_vBias is not null) yield return _vBias;
        if (_outWeight is not null) yield return _outWeight;
        if (_outBias is not null) yield return _outBias;
        if (_fc1Weight is not null) yield return _fc1Weight;
        if (_fc1Bias is not null) yield return _fc1Bias;
        if (_fc2Weight is not null) yield return _fc2Weight;
        if (_fc2Bias is not null) yield return _fc2Bias;
        if (_finalLnWeight is not null) yield return _finalLnWeight;
        if (_finalLnBias is not null) yield return _finalLnBias;
    }
}
