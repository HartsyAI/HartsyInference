using HartsyInference.Audio.Models.Vits;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.FishSpeech;

/// <summary>Firefly-gan-vq decoder (Fish-Speech 1.5): dequantizes the 8 audio codebooks to the decoder input
/// channels and runs a HiFi-GAN generator (upsample [8,8,2,2,2] → 44.1 kHz). Reuses <see cref="VitsHiFiGan"/>
/// for the generator.
///
/// <para>Structural: the real firefly quantizer is a grouped-<b>residual FSQ</b> (levels [8,5,5,5]) with a
/// ConvNeXt down/up-sample stack and SiLU activations; this build uses per-codebook learned embeddings (summed)
/// + the LeakyReLU HiFi-GAN — the FSQ grouped-residual path, ConvNeXt resampling, and SiLU activation are
/// checkpoint-gated reconcile items.</para></summary>
public sealed unsafe class FireflyDecoder : IDisposable
{
    private readonly int _numCodebooks, _codebookSize, _inputDim;
    private readonly VitsConfig _genCfg;
    private readonly VitsHiFiGan _gen;
    private readonly Tensor?[] _codebookEmb;
    private int _disposed;

    public FireflyDecoder(int numCodebooks, int codebookSize, int inputDim = 128, VitsConfig? genConfig = null)
    {
        _numCodebooks = numCodebooks; _codebookSize = codebookSize; _inputDim = inputDim;
        _genCfg = genConfig ?? new VitsConfig
        {
            InterChannels = inputDim, UpsampleInitialChannel = 512,
            UpsampleRates = [8, 8, 2, 2, 2], UpsampleKernelSizes = [16, 16, 8, 2, 2],
            ResBlock = "1", ResBlockKernelSizes = [3, 7, 11], ResBlockDilations = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            SampleRate = 44_100,
        };
        _gen = new VitsHiFiGan(_genCfg);
        _codebookEmb = new Tensor?[numCodebooks];
    }

    public int SampleRate => _genCfg.SampleRate;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "head")
    {
        for (int i = 0; i < _numCodebooks; i++)
            _codebookEmb[i] = WhisperOps.EnsureF32(w[$"quantizer.codebook_emb.{i}.weight"]);
        _gen.LoadWeights(w, prefix);
    }

    /// <summary>Decodes <c>[numCodebooks, T]</c> codebook indices to 44.1 kHz mono PCM.</summary>
    public float[] Decode(IBackend backend, int[,] codes, int t)
    {
        Tensor input = new(new TensorShape(1, _inputDim, t), DType.F32);
        float* ip = (float*)input.DataPointer;
        for (int i = 0; i < _numCodebooks; i++)
        {
            float* emb = (float*)_codebookEmb[i]!.DataPointer;
            for (int j = 0; j < t; j++)
            {
                float* row = emb + (long)codes[i, j] * _inputDim;
                for (int c = 0; c < _inputDim; c++) ip[(long)c * t + j] += row[c];
            }
        }
        float[] audio = _gen.Forward(backend, input, t);
        input.Dispose();
        return audio;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in _codebookEmb) if (t is not null) yield return t;
        foreach (Tensor t in _gen.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
