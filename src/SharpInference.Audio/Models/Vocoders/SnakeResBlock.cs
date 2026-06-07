using SharpInference.Audio.Layers;
using SharpInference.Audio.Models.Whisper;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.Vocoders;

/// <summary>HiFiGAN-style <c>ResBlock1</c> with plain (non-style) Snake activations — three sequential
/// dilated branches, each <c>Snake → dilated Conv1d → Snake → Conv1d</c> added back as a residual. The
/// canonical MRF block shared by every HiFiGAN/DAC-family wave generator in the package (CosyVoice
/// HiFTNet, Spark-TTS BiCodec, …) — one implementation, parameterized by channels / kernel / dilations,
/// loading the standard weight-normed <c>convs1/convs2</c> + per-channel Snake <c>alpha</c> keys.</summary>
public sealed unsafe class SnakeResBlock
{
    private readonly int _channels;
    private readonly int _kernel;
    private readonly int[] _dilations;
    private readonly Tensor?[] _convs1W;
    private readonly Tensor?[] _convs1B;
    private readonly Tensor?[] _convs2W;
    private readonly Tensor?[] _convs2B;
    private readonly Tensor?[] _alpha1;
    private readonly Tensor?[] _alpha2;

    public SnakeResBlock(int channels, int kernel, int[] dilations)
    {
        _channels = channels;
        _kernel = kernel;
        _dilations = dilations;
        int n = dilations.Length;
        _convs1W = new Tensor[n];
        _convs1B = new Tensor[n];
        _convs2W = new Tensor[n];
        _convs2B = new Tensor[n];
        _alpha1 = new Tensor[n];
        _alpha2 = new Tensor[n];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        for (int i = 0; i < _dilations.Length; i++)
        {
            _convs1W[i] = WeightNorm.Compose(w, $"{prefix}.convs1.{i}");
            _convs1B[i] = WhisperOps.EnsureF32(w[$"{prefix}.convs1.{i}.bias"]);
            _convs2W[i] = WeightNorm.Compose(w, $"{prefix}.convs2.{i}");
            _convs2B[i] = WhisperOps.EnsureF32(w[$"{prefix}.convs2.{i}.bias"]);
            _alpha1[i] = WhisperOps.EnsureF32(w[$"{prefix}.activations1.{i}.alpha"]);
            _alpha2[i] = WhisperOps.EnsureF32(w[$"{prefix}.activations2.{i}.alpha"]);
        }
    }

    public Tensor Forward(IBackend backend, Tensor x)
    {
        int t = (int)x.Shape[2];
        Tensor cur = new(x.Shape, DType.F32);
        Buffer.MemoryCopy((void*)x.DataPointer, (void*)cur.DataPointer, x.ElementCount * 4, x.ElementCount * 4);
        for (int i = 0; i < _dilations.Length; i++)
        {
            int pad1 = (_kernel - 1) * _dilations[i] / 2;
            int pad2 = (_kernel - 1) / 2;
            Tensor a1 = new(cur.Shape, DType.F32);
            backend.Snake(a1, cur, _alpha1[i]!, null);
            Tensor c1 = new(new TensorShape(1, _channels, t), DType.F32);
            backend.Conv1d(c1, a1, _convs1W[i]!, _convs1B[i], 1, pad1, pad1, _dilations[i], 1);
            a1.Dispose();
            Tensor a2 = new(c1.Shape, DType.F32);
            backend.Snake(a2, c1, _alpha2[i]!, null);
            c1.Dispose();
            Tensor c2 = new(new TensorShape(1, _channels, t), DType.F32);
            backend.Conv1d(c2, a2, _convs2W[i]!, _convs2B[i], 1, pad2, pad2, 1, 1);
            a2.Dispose();
            float* cp = (float*)cur.DataPointer;
            float* c2p = (float*)c2.DataPointer;
            long n = cur.ElementCount;
            for (long k = 0; k < n; k++) cp[k] += c2p[k];
            c2.Dispose();
        }
        return cur;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        for (int i = 0; i < _dilations.Length; i++)
        {
            Tensor?[] all = [_convs1W[i], _convs1B[i], _convs2W[i], _convs2B[i], _alpha1[i], _alpha2[i]];
            foreach (Tensor? t in all) if (t is not null) yield return t;
        }
    }
}
