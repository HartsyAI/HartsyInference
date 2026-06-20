using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Chatterbox;

/// <summary>Chatterbox voice encoder — a GE2E/Resemblyzer-style 3-layer LSTM over 40-bin mel + a linear
/// projection + L2 normalization, producing the 256-d speaker embedding that conditions T3. Reuses the
/// shared <see cref="UnidirectionalLstm"/>.</summary>
public sealed unsafe class ChatterboxVoiceEncoder : IDisposable
{
    private const int NumMels = 40;
    private const int Hidden = 256;
    private const int Layers = 3;

    private readonly UnidirectionalLstm _lstm;
    private Tensor? _projW, _projB;
    private int _disposed;

    public ChatterboxVoiceEncoder() => _lstm = new UnidirectionalLstm(NumMels, Hidden, Layers);

    public int EmbeddingDim => Hidden;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        _lstm.LoadWeights(w, $"{prefix}lstm");
        _projW = WhisperOps.EnsureF32(w[$"{prefix}proj.weight"]);
        _projB = w.TryGetValue($"{prefix}proj.bias", out Tensor? b) ? WhisperOps.EnsureF32(b) : null;
    }

    /// <summary>Embeds a 40-bin mel <c>[1, T, 40]</c> → L2-normalized 256-d speaker embedding <c>[1, 256]</c>
    /// (the projection of the final LSTM hidden state).</summary>
    public Tensor Forward(IBackend backend, Tensor mel, int t)
    {
        Tensor seq = _lstm.Forward(backend, mel, 1, t);     // [1, T, Hidden]
        Tensor last = new(new TensorShape(1, 1, Hidden), DType.F32);
        Buffer.MemoryCopy((float*)seq.DataPointer + (long)(t - 1) * Hidden, (void*)last.DataPointer, Hidden * 4, Hidden * 4);
        seq.Dispose();
        Tensor proj = WhisperOps.ProjectLinear(backend, last, _projW!, _projB, 1, 1, Hidden, Hidden);
        last.Dispose();

        float* p = (float*)proj.DataPointer;
        double norm = 0;
        for (int i = 0; i < Hidden; i++) norm += (double)p[i] * p[i];
        float inv = (float)(1.0 / (Math.Sqrt(norm) + 1e-8));
        for (int i = 0; i < Hidden; i++) p[i] *= inv;
        return proj.Reshape(new TensorShape(1, Hidden));
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _lstm.EnumerateWeights()) yield return t;
        if (_projW is not null) yield return _projW;
        if (_projB is not null) yield return _projB;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
