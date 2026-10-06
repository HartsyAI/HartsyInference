using HartsyInference.Audio.Layers;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>Code-to-waveform path of fish-speech's ModifiedDAC (<c>DownsampleResidualVectorQuantize.decode</c> followed
/// by the causal DAC <c>Decoder</c>): per-codebook lookup + 1×1 out-projection summed into the latent, the post-module
/// window-limited transformer, the causal ConvTranspose/ConvNeXt upsampler, then Snake / causal transposed-conv stages
/// with residual units and a final tanh. The encoder half is <see cref="ModifiedDacEncoder"/>.</summary>
public sealed unsafe class ModifiedDacDecoder : IDisposable
{
    private readonly ModifiedDacConfig _cfg;
    private Tensor[] _codebook = [], _projW = [], _projB = [];
    private DacOps.Transformer? _post;
    private Tensor[] _upW = [], _upB = [];
    private DacOps.ConvNeXt[] _upNext = [];
    private Tensor? _stemW, _stemB, _finalAlpha, _finalW, _finalB;
    private DecoderStage[] _stages = [];
    private int _disposed;

    private sealed record DecoderStage(Tensor Alpha, Tensor UpW, Tensor UpB, DacOps.Unit[] Units);

    public ModifiedDacDecoder(ModifiedDacConfig cfg) { _cfg = cfg; }

    public int SampleRate => _cfg.SampleRate;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        int books = _cfg.TotalCodebooks;
        _codebook = new Tensor[books]; _projW = new Tensor[books]; _projB = new Tensor[books];
        for (int i = 0; i < books; i++)
        {
            string p = i == 0 ? "quantizer.semantic_quantizer.quantizers.0" : $"quantizer.quantizer.quantizers.{i - 1}";
            _codebook[i] = DacOps.F(w, $"{p}.codebook.weight");
            _projW[i] = WeightNorm.Compose(w, $"{p}.out_proj");
            _projB[i] = DacOps.F(w, $"{p}.out_proj.bias");
        }

        _post = DacOps.LoadTransformer(w, "quantizer.post_module", _cfg.TransformerLayers, _cfg.TransformerHeads,
            _cfg.TransformerHeadDim, _cfg.TransformerIntermediate, _cfg.TransformerWindow, _cfg.TransformerRopeBase,
            _cfg.TransformerNormEps);

        int stages = _cfg.UpsampleFactors.Length;
        _upW = new Tensor[stages]; _upB = new Tensor[stages]; _upNext = new DacOps.ConvNeXt[stages];
        for (int i = 0; i < stages; i++)
        {
            string p = $"quantizer.upsample.{i}";
            _upW[i] = DacOps.F(w, $"{p}.0.conv.weight"); _upB[i] = DacOps.F(w, $"{p}.0.conv.bias");
            _upNext[i] = DacOps.LoadConvNeXt(w, $"{p}.1");
        }

        _stemW = WeightNorm.Compose(w, "decoder.model.0.conv"); _stemB = DacOps.F(w, "decoder.model.0.conv.bias");
        _stages = new DecoderStage[_cfg.DecoderRates.Length];
        for (int i = 0; i < _stages.Length; i++)
        {
            string p = $"decoder.model.{i + 1}.block";
            DacOps.Unit[] units = new DacOps.Unit[_cfg.ResidualDilations.Length];
            for (int j = 0; j < units.Length; j++) units[j] = DacOps.LoadUnit(w, $"{p}.{j + 2}.block");
            _stages[i] = new DecoderStage(DacOps.F(w, $"{p}.0.alpha"), WeightNorm.Compose(w, $"{p}.1.conv"),
                DacOps.F(w, $"{p}.1.conv.bias"), units);
        }
        int last = _stages.Length + 1;
        _finalAlpha = DacOps.F(w, $"decoder.model.{last}.alpha");
        _finalW = WeightNorm.Compose(w, $"decoder.model.{last + 1}.conv");
        _finalB = DacOps.F(w, $"decoder.model.{last + 1}.conv.bias");
    }

    /// <summary>Decodes <c>[1 + numResidual, T]</c> codes (row 0 = semantic) to mono PCM in [-1, 1],
    /// <c>T × SamplesPerFrame</c> samples long. Out-of-range codes are clamped, as upstream does.
    /// <paramref name="tap"/> receives intermediate activations for parity checks.</summary>
    public float[] Decode(IBackend backend, int[,] codes, int t, Action<string, float[]>? tap = null)
    {
        int books = _cfg.TotalCodebooks, d = _cfg.LatentDim;
        if (codes.GetLength(0) != books) throw new ArgumentException($"codes must have {books} rows.", nameof(codes));
        if (t <= 0) throw new ArgumentException("t must be positive.", nameof(t));

        // Σ out_proj_i(codebook_i[code_i]) → [T, D]
        float[] z = new float[t * d];
        for (int i = 0; i < books; i++)
        {
            int size = i == 0 ? _cfg.SemanticCodebookSize : _cfg.ResidualCodebookSize, cb = _cfg.CodebookDim;
            float* book = (float*)_codebook[i].DataPointer, pw = (float*)_projW[i].DataPointer, pb = (float*)_projB[i].DataPointer;
            for (int j = 0; j < t; j++)
            {
                float* vec = book + (long)Math.Clamp(codes[i, j], 0, size - 1) * cb;
                for (int c = 0; c < d; c++)
                {
                    float acc = pb[c];
                    float* row = pw + (long)c * cb;
                    for (int k = 0; k < cb; k++) acc += row[k] * vec[k];
                    z[j * d + c] += acc;
                }
            }
        }

        tap?.Invoke("sum", z);
        float[] h = DacOps.RunTransformer(backend, _post!, z, t, d);
        tap?.Invoke("post", h);

        Tensor x = DacOps.FromTimeMajor(h, t, d);
        int curT = t;
        for (int i = 0; i < _upW.Length; i++)
        {
            int s = _cfg.UpsampleFactors[_cfg.UpsampleFactors.Length - 1 - i];   // reversed(downsample_factor)
            x = DacOps.CausalConvTranspose(backend, x, _upW[i], _upB[i], s, kernel: s);
            curT *= s;
            tap?.Invoke($"up{i}_conv", DacOps.ToHost(x));
            x = DacOps.ConvNeXtBlock(backend, x, _upNext[i]);
            tap?.Invoke($"up{i}", DacOps.ToHost(x));
        }

        x = DacOps.CausalConv(backend, x, _stemW!, _stemB);
        tap?.Invoke("stem", DacOps.ToHost(x));
        for (int i = 0; i < _stages.Length; i++)
        {
            DecoderStage st = _stages[i];
            int stride = _cfg.DecoderRates[i];
            x = DacOps.SnakeOp(backend, x, st.Alpha);
            x = DacOps.CausalConvTranspose(backend, x, st.UpW, st.UpB, stride, kernel: 2 * stride);
            curT *= stride;
            tap?.Invoke($"stage{i}_up", DacOps.ToHost(x));
            for (int j = 0; j < st.Units.Length; j++) x = DacOps.ResidualUnit(backend, x, st.Units[j], _cfg.ResidualDilations[j]);
            tap?.Invoke($"stage{i}", DacOps.ToHost(x));
        }
        x = DacOps.SnakeOp(backend, x, _finalAlpha!);
        x = DacOps.CausalConv(backend, x, _finalW!, _finalB);
        float[] audio = new float[curT];
        float* ap = (float*)x.DataPointer;
        for (int j = 0; j < curT; j++) audio[j] = MathF.Tanh(ap[j]);
        x.Dispose();
        return audio;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
