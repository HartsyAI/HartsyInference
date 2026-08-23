using HartsyInference.Audio.Models.LanguageModels.Gpt;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Bark;

/// <summary>Bark fine acoustics — a non-causal (bidirectional) GPT that fills EnCodec codebooks 2..7
/// given the 2 coarse codebooks, in six refinement passes. Eight input codebook embeddings are summed
/// per timestep; seven output heads predict codebooks 1..7. Reuses the shared <see cref="GptBackbone"/>
/// in non-causal mode. Faithful to upstream <c>generate_fine</c>: 1024-frame windows (512 stride),
/// CODEBOOK_SIZE placeholder for unfilled slots, temperature-0.5 sampling.</summary>
public sealed unsafe class BarkFineModel : IDisposable
{
    private readonly BarkConfig _cfg;
    private readonly GptBackbone _backbone;
    private readonly Tensor?[] _inputEmbeds;   // 8 × [fineVocab, hidden]
    private readonly Tensor?[] _outHeads;      // 7 × [fineVocab, hidden]
    private int _disposed;

    public BarkFineModel(BarkConfig cfg)
    {
        _cfg = cfg;
        _backbone = new GptBackbone(cfg.Stage);
        _inputEmbeds = new Tensor?[cfg.NumCodebooks];
        _outHeads = new Tensor?[cfg.NumCodebooks - 1];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "fine_acoustics")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        for (int i = 0; i < _cfg.NumCodebooks; i++)
            _inputEmbeds[i] = WhisperOps.EnsureF32(w[$"{p}input_embeds_layers.{i}.weight"]);
        _backbone.LoadWeights(w, $"{p}position_embeds_layer.weight", $"{p}layers",
            $"{p}layernorm_final.weight", $"{p}layernorm_final.bias");
        for (int i = 0; i < _cfg.NumCodebooks - 1; i++)
            _outHeads[i] = WhisperOps.EnsureF32(w[$"{p}lm_heads.{i}.weight"]);
    }

    /// <summary>Fills codebooks 2..7 from the 2 coarse codebooks. <paramref name="coarse"/> is
    /// <c>[numCoarse, T]</c>; returns all 8 codebooks <c>[8, T]</c>.</summary>
    public int[,] Refine(IBackend backend, int[,] coarse, int seed = 0)
    {
        if (_inputEmbeds[0] is null) throw new InvalidOperationException("BarkFineModel weights not loaded.");
        int t = coarse.GetLength(1);
        int h = _cfg.Stage.Hidden;
        int padT = Math.Max(1024, t);
        uint rng = Dsp.DeterministicRng.Seed(seed);

        // Upstream generate_fine: unfilled slots (codebooks ≥ numCoarse, and frames beyond T) hold the
        // CODEBOOK_SIZE placeholder id (1024) — NOT 0 — and the grid is padded to ≥1024 frames because the
        // non-causal model was trained on absolute positions within 1024-frame windows.
        int[,] grid = new int[_cfg.NumCodebooks, padT];
        for (int cb = 0; cb < _cfg.NumCodebooks; cb++)
        {
            for (int j = 0; j < padT; j++)
            {
                grid[cb, j] = cb < _cfg.NumCoarseCodebooks && j < t ? coarse[cb, j] : _cfg.CodebookSize;
            }
        }

        // 1024-frame windows sliding by 512; each predicts codebooks numCoarse..7 in order, sampling at
        // FineTemperature over the real codebook range, writing back so later windows condition on results.
        int nLoops = Math.Max(0, (int)Math.Ceiling((t - 1024) / 512.0)) + 1;
        for (int n = 0; n < nLoops; n++)
        {
            int startIdx = Math.Min(n * 512, padT - 1024);
            int startFill = Math.Min(n * 512, padT - 512);
            int relStart = startFill - startIdx;
            for (int pred = _cfg.NumCoarseCodebooks; pred < _cfg.NumCodebooks; pred++)
            {
                Tensor input = SumEmbedsWindow(grid, pred + 1, startIdx, 1024, h);
                Tensor hidden = _backbone.Forward(backend, input, nonCausal: true);
                input.Dispose();
                Tensor logits = WhisperOps.ProjectLinear(backend, hidden, _outHeads[pred - 1]!, bias: null, 1, 1024, h, _cfg.FineVocab);
                hidden.Dispose();
                float* lp = (float*)logits.DataPointer;
                for (int j = relStart; j < 1024; j++)
                {
                    Span<float> slice = new(lp + (long)j * _cfg.FineVocab, _cfg.CodebookSize);
                    grid[pred, startIdx + j] = Sampling.NucleusSampler.Draw(
                        slice, _cfg.CodebookSize, _cfg.FineTemperature, 0, 1f, ref rng);
                }
                logits.Dispose();
            }
        }

        int[,] codes = new int[_cfg.NumCodebooks, t];
        for (int cb = 0; cb < _cfg.NumCodebooks; cb++)
        {
            for (int j = 0; j < t; j++) codes[cb, j] = grid[cb, j];
        }
        return codes;
    }

    /// <summary>Parity-debug only: teacher-forced logits for codebook <paramref name="codebookIdx"/> given the full
    /// <paramref name="codes"/> grid <c>[NumCodebooks, T]</c>. Matches HF <c>BarkFineModel.forward(codebook_idx,
    /// input_ids)</c>: sum embeds of codebooks <c>[0, codebookIdx]</c> inclusive → non-causal body →
    /// <c>lm_heads[codebookIdx - NCodesGiven]</c>. Returns <c>[1, T, fineVocab]</c>.</summary>
    /// <remarks>Kept unwired as the C# half of the <c>tests/python-reference/bark_reference/</c> oracle; the inclusive-sum convention it encodes is the fix recorded in PARITY_VERIFICATION.md.</remarks>
    public Tensor DebugLogits(IBackend backend, int[,] codes, int codebookIdx, int t)
    {
        if (_inputEmbeds[0] is null) throw new InvalidOperationException("BarkFineModel weights not loaded.");
        int h = _cfg.Stage.Hidden;
        Tensor input = SumEmbeds(codes, codebookIdx + 1, t, h);
        Tensor hidden = _backbone.Forward(backend, input, nonCausal: true);
        input.Dispose();
        Tensor logits = WhisperOps.ProjectLinear(backend, hidden, _outHeads[codebookIdx - _cfg.NCodesGiven]!,
            bias: null, 1, t, h, _cfg.FineVocab);
        hidden.Dispose();
        return logits;
    }

    /// <summary>Sums the embeddings of codebooks <c>[0, upTo)</c> over a <paramref name="len"/>-frame window of
    /// the padded grid starting at <paramref name="start"/> (upstream's <c>in_buffer</c> slice).</summary>
    private Tensor SumEmbedsWindow(int[,] grid, int upTo, int start, int len, int h)
    {
        Tensor outT = new(new TensorShape(1, len, h), DType.F32);
        float* op = (float*)outT.DataPointer;
        for (int cb = 0; cb < upTo; cb++)
        {
            float* tab = (float*)_inputEmbeds[cb]!.DataPointer;
            for (int j = 0; j < len; j++)
            {
                int id = grid[cb, start + j];
                if ((uint)id >= (uint)_cfg.FineVocab) id = 0;
                float* row = tab + (long)id * h;
                long dst = (long)j * h;
                for (int c = 0; c < h; c++) op[dst + c] += row[c];
            }
        }
        return outT;
    }

    private Tensor SumEmbeds(int[,] codes, int upTo, int t, int h)
    {
        Tensor outT = new(new TensorShape(1, t, h), DType.F32);
        float* op = (float*)outT.DataPointer;
        for (int cb = 0; cb < upTo; cb++)
        {
            float* tab = (float*)_inputEmbeds[cb]!.DataPointer;
            for (int j = 0; j < t; j++)
            {
                int id = codes[cb, j];
                if ((uint)id >= (uint)_cfg.FineVocab) id = 0;
                float* row = tab + (long)id * h;
                long dst = (long)j * h;
                for (int c = 0; c < h; c++) op[dst + c] += row[c];
            }
        }
        return outT;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? e in _inputEmbeds) if (e is not null) yield return e;
        foreach (Tensor t in _backbone.EnumerateWeights()) yield return t;
        foreach (Tensor? hd in _outHeads) if (hd is not null) yield return hd;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _backbone.Dispose();
        GC.SuppressFinalize(this);
    }
}
