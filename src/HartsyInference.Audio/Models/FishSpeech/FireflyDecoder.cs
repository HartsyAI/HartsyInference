using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.FishSpeech;

/// <summary>Firefly-gan-vq decoder (Fish-Speech 1.5). Two stages: the real <see cref="FireflyQuantizer"/>
/// (grouped-residual FSQ levels <c>[8,5,5,5]</c> + ConvNeXt up-sample <c>[2,2]</c>) turns the codebook grid into a
/// 1024-channel decoder feature at <c>T·4</c> frames, then a <b>SiLU</b> HiFi-GAN generator
/// (<see cref="FireflySiluGenerator"/>, upsample <c>[8,8,2,2,2]</c> + tanh) renders 44.1 kHz mono PCM.
///
/// <para>The firefly generator uses SiLU (not LeakyReLU) so it cannot reuse <c>VitsHiFiGan</c>; the self-contained
/// SiLU generator lives alongside this decoder.</para></summary>
public sealed unsafe class FireflyDecoder : IDisposable
{
    private const int DecoderInputChannels = 1_024;
    private readonly int _numCodebooks;
    private readonly int _sampleRate;
    private readonly FireflyQuantizer _quantizer;
    private readonly FireflySiluGenerator _gen;
    private int _disposed;

    public FireflyDecoder(int numCodebooks, int codebookSize, int inputDim = 128, FireflyConfig? config = null)
    {
        FireflyConfig cfg = config ?? FireflyConfig.V1_5;
        _numCodebooks = numCodebooks;
        _sampleRate = cfg.SampleRate;
        _quantizer = new FireflyQuantizer(cfg.FsqLevels, numCodebooks, cfg.QuantizerUpsampleFactors);
        _gen = new FireflySiluGenerator(cfg.UpsampleInitialChannel, cfg.UpsampleRates, cfg.UpsampleKernelSizes,
            cfg.ResBlockKernelSizes, cfg.ResBlockDilations);
    }

    public int SampleRate => _sampleRate;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "head", string quantizerPrefix = "quantizer")
    {
        _quantizer.LoadWeights(w, quantizerPrefix);
        _gen.LoadWeights(w, prefix);
    }

    /// <summary>Decodes <c>[numCodebooks, T]</c> codebook indices to 44.1 kHz mono PCM.</summary>
    public float[] Decode(IBackend backend, int[,] codes, int t)
    {
        if (codes.GetLength(0) != _numCodebooks)
            throw new ArgumentException($"codes must have {_numCodebooks} rows, got {codes.GetLength(0)}.");
        Tensor decoderInput = _quantizer.DequantToDecoderInput(backend, codes, t);
        int genT = (int)decoderInput.Shape[2];
        float[] audio = _gen.Forward(backend, decoderInput, genT);
        decoderInput.Dispose();
        return audio;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _quantizer.EnumerateWeights()) yield return t;
        foreach (Tensor t in _gen.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _quantizer.Dispose();
        GC.SuppressFinalize(this);
    }
}
