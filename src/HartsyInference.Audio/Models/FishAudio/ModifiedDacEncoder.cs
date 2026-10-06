using HartsyInference.Audio.Layers;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>Waveform-to-code path of fish-speech's ModifiedDAC (<c>DAC.encode</c>): the causal strided encoder with a
/// windowed transformer in its last block, the downsampling ConvNeXt stack, the pre-module transformer, and the
/// residual vector quantizer (one semantic codebook, then nine residual ones, each a cosine nearest-neighbour search
/// in the 8-dim projected space). Used to turn a reference clip into the codes a voice-cloning prompt carries.</summary>
public sealed unsafe class ModifiedDacEncoder : IDisposable
{
    private readonly ModifiedDacConfig _cfg;
    private Tensor? _stemW, _stemB, _finalAlpha, _finalW, _finalB;
    private EncoderBlock[] _blocks = [];
    private Tensor[] _downW = [], _downB = [];
    private DacOps.ConvNeXt[] _downNext = [];
    private DacOps.Transformer? _pre;
    private Quantizer[] _quantizers = [];
    private int _disposed;

    private sealed record EncoderBlock(DacOps.Unit[] Units, Tensor Alpha, Tensor ConvW, Tensor ConvB, int Stride,
        DacOps.Transformer? Transformer);

    private sealed record Quantizer(Tensor InW, Tensor InB, Tensor Codebook, Tensor CodebookNormalized, Tensor OutW, Tensor OutB);

    public ModifiedDacEncoder(ModifiedDacConfig cfg) { _cfg = cfg; }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _stemW = WeightNorm.Compose(w, "encoder.block.0.conv"); _stemB = DacOps.F(w, "encoder.block.0.conv.bias");
        _blocks = new EncoderBlock[_cfg.EncoderRates.Length];
        int dim = _cfg.EncoderDim;
        for (int i = 0; i < _blocks.Length; i++)
        {
            dim *= 2;
            string p = $"encoder.block.{i + 1}.block";
            DacOps.Unit[] units = new DacOps.Unit[_cfg.ResidualDilations.Length];
            for (int j = 0; j < units.Length; j++) units[j] = DacOps.LoadUnit(w, $"{p}.{j}.block");
            int layers = _cfg.EncoderTransformerLayers[i];
            DacOps.Transformer? tr = layers == 0 ? null : DacOps.LoadTransformer(w, $"{p}.5", layers, dim / 64, 64, dim * 3,
                _cfg.EncoderTransformerWindow, _cfg.TransformerRopeBase, _cfg.TransformerNormEps);
            _blocks[i] = new EncoderBlock(units, DacOps.F(w, $"{p}.3.alpha"), WeightNorm.Compose(w, $"{p}.4.conv"),
                DacOps.F(w, $"{p}.4.conv.bias"), _cfg.EncoderRates[i], tr);
        }
        int last = _blocks.Length + 1;
        _finalAlpha = DacOps.F(w, $"encoder.block.{last}.alpha");
        _finalW = WeightNorm.Compose(w, $"encoder.block.{last + 1}.conv");
        _finalB = DacOps.F(w, $"encoder.block.{last + 1}.conv.bias");

        int stages = _cfg.DownsampleFactors.Length;
        _downW = new Tensor[stages]; _downB = new Tensor[stages]; _downNext = new DacOps.ConvNeXt[stages];
        for (int i = 0; i < stages; i++)
        {
            string p = $"quantizer.downsample.{i}";
            _downW[i] = DacOps.F(w, $"{p}.0.conv.weight"); _downB[i] = DacOps.F(w, $"{p}.0.conv.bias");
            _downNext[i] = DacOps.LoadConvNeXt(w, $"{p}.1");
        }
        _pre = DacOps.LoadTransformer(w, "quantizer.pre_module", _cfg.TransformerLayers, _cfg.TransformerHeads,
            _cfg.TransformerHeadDim, _cfg.TransformerIntermediate, _cfg.TransformerWindow, _cfg.TransformerRopeBase,
            _cfg.TransformerNormEps);

        int books = _cfg.TotalCodebooks;
        _quantizers = new Quantizer[books];
        for (int i = 0; i < books; i++)
        {
            string p = i == 0 ? "quantizer.semantic_quantizer.quantizers.0" : $"quantizer.quantizer.quantizers.{i - 1}";
            Tensor cb = DacOps.F(w, $"{p}.codebook.weight");
            _quantizers[i] = new Quantizer(
                WeightNorm.Compose(w, $"{p}.in_proj").Reshape(new TensorShape(_cfg.CodebookDim, _cfg.LatentDim)),
                DacOps.F(w, $"{p}.in_proj.bias"), cb, Normalize(cb),
                WeightNorm.Compose(w, $"{p}.out_proj").Reshape(new TensorShape(_cfg.LatentDim, _cfg.CodebookDim)),
                DacOps.F(w, $"{p}.out_proj.bias"));
        }
    }

    private static Tensor Normalize(Tensor codebook)
    {
        int n = (int)codebook.Shape[0], d = (int)codebook.Shape[1];
        Tensor o = new(codebook.Shape, DType.F32);
        float* src = (float*)codebook.DataPointer, dst = (float*)o.DataPointer;
        for (int i = 0; i < n; i++)
        {
            float norm = 0f;
            for (int k = 0; k < d; k++) norm += src[i * d + k] * src[i * d + k];
            norm = MathF.Max(MathF.Sqrt(norm), 1e-12f);   // F.normalize eps
            for (int k = 0; k < d; k++) dst[i * d + k] = src[i * d + k] / norm;
        }
        return o;
    }

    /// <summary>Encodes 44.1 kHz mono audio (right-padded to a whole number of frames, as upstream) into
    /// <c>[1 + numResidual, T]</c> codes.</summary>
    public int[,] Encode(IBackend backend, ReadOnlySpan<float> audio, Action<string, float[]>? tap = null)
    {
        int frame = _cfg.EncodeSamplesPerFrame;
        int padded = (audio.Length + frame - 1) / frame * frame;
        if (padded == 0) throw new ArgumentException("audio is empty.", nameof(audio));
        Tensor x = new(new TensorShape(1, 1, padded), DType.F32);
        audio.CopyTo(new Span<float>((void*)x.DataPointer, audio.Length));

        x = DacOps.CausalConv(backend, x, _stemW!, _stemB);
        for (int i = 0; i < _blocks.Length; i++)
        {
            EncoderBlock b = _blocks[i];
            for (int j = 0; j < b.Units.Length; j++) x = DacOps.ResidualUnit(backend, x, b.Units[j], _cfg.ResidualDilations[j]);
            x = DacOps.SnakeOp(backend, x, b.Alpha);
            x = DacOps.CausalConv(backend, x, b.ConvW, b.ConvB, stride: b.Stride);
            if (b.Transformer is { } tr)
            {
                int c = (int)x.Shape[1], t0 = (int)x.Shape[2];
                float[] h = DacOps.RunTransformer(backend, tr, DacOps.ToTimeMajor(x), t0, c);
                x.Dispose();
                x = DacOps.FromTimeMajor(h, t0, c);
            }
            tap?.Invoke($"enc{i}", DacOps.ToHost(x));
        }
        x = DacOps.SnakeOp(backend, x, _finalAlpha!);
        x = DacOps.CausalConv(backend, x, _finalW!, _finalB);
        tap?.Invoke("latent", DacOps.ToHost(x));

        for (int i = 0; i < _downW.Length; i++)
        {
            x = DacOps.CausalConv(backend, x, _downW[i], _downB[i], stride: _cfg.DownsampleFactors[i]);
            x = DacOps.ConvNeXtBlock(backend, x, _downNext[i]);
        }
        int t = (int)x.Shape[2], d = _cfg.LatentDim;
        float[] z = DacOps.RunTransformer(backend, _pre!, DacOps.ToTimeMajor(x), t, d);
        x.Dispose();
        tap?.Invoke("pre", z);

        // semantic codebook first, then the residual stack on what it leaves behind
        int books = _cfg.TotalCodebooks;
        int[,] codes = new int[books, t];
        float[] residual = z;
        for (int i = 0; i < books; i++)
        {
            Quantizer q = _quantizers[i];
            float[] ze = DacOps.Linear(backend, residual, q.InW, t, d, _cfg.CodebookDim, q.InB);
            float[] quantized = new float[t * _cfg.CodebookDim];
            float* book = (float*)q.Codebook.DataPointer, unit = (float*)q.CodebookNormalized.DataPointer;
            int size = (int)q.Codebook.Shape[0], cb = _cfg.CodebookDim;
            for (int j = 0; j < t; j++)
            {
                float norm = 0f;
                for (int k = 0; k < cb; k++) norm += ze[j * cb + k] * ze[j * cb + k];
                norm = MathF.Max(MathF.Sqrt(norm), 1e-12f);
                int best = 0; float bestScore = float.NegativeInfinity;
                for (int n = 0; n < size; n++)
                {
                    float s = 0f;
                    for (int k = 0; k < cb; k++) s += ze[j * cb + k] / norm * unit[n * cb + k];
                    if (s > bestScore) { bestScore = s; best = n; }
                }
                codes[i, j] = best;
                for (int k = 0; k < cb; k++) quantized[j * cb + k] = book[best * cb + k];
            }
            float[] zq = DacOps.Linear(backend, quantized, q.OutW, t, cb, d, q.OutB);
            float[] next = new float[residual.Length];
            for (int n = 0; n < next.Length; n++) next[n] = residual[n] - zq[n];
            residual = next;
        }
        return codes;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
