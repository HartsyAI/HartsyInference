using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ZipVoice;

/// <summary>One stage of the 5-stage <c>TTSZipformer</c> pyramid — either a bare <c>Zipformer2Encoder</c>
/// (downsample factor 1: stages 0 and 4) or a <c>DownsampledZipformer2Encoder</c> (factor != 1: stages 1-3,
/// which run their inner layer stack at a reduced frame rate then upsample and recombine with the
/// pre-downsample input via a learned <see cref="ZipformerBypass"/>).
///
/// <para>Each stage owns its own <see cref="ZipformerRelPosEncoding"/> table (rebuilt per forward at that
/// stage's effective sequence length) and, when time-conditioned (fm_decoder only), its own
/// <c>SwooshR → Linear(time_embed_dim → dim)</c> re-projection of the single shared 192-dim time embedding —
/// note the SwooshR runs BEFORE the stage-local Linear, not after (checkpoint key <c>time_emb.1.*</c>: index 0
/// is the parameter-free SwooshR, index 1 is the Linear).</para></summary>
internal sealed unsafe class ZipformerEncoderStage
{
    private readonly int _dim, _numLayers, _downsampleFactor, _posDim, _timeEmbedDim;
    private readonly bool _useTimeEmb;
    private readonly ZipformerEncoderLayer[] _layers;
    private readonly ZipformerSimpleDownsample? _downsample;
    private readonly ZipformerBypass? _outCombiner;
    private Tensor? _timeEmbW, _timeEmbB;

    /// <summary>Cache for <see cref="ZipformerRelPosEncoding.Build"/> keyed by the stage's effective sequence
    /// length. Within one <c>GenerateFromAudio</c> call, T is identical across all 16-step × 2-CFG-branch Euler
    /// forward passes, so rebuilding this pure function of T from scratch on every single forward call (as the
    /// original code did) was redundant CPU work repeated ~32× per stage for no reason — cache the one table
    /// that's actually ever needed for a given generation.</summary>
    private int _cachedPosEmbT = -1;
    private Tensor? _cachedPosEmb;

    public ZipformerEncoderStage(int dim, int numLayers, int downsampleFactor, int cnnKernel,
        int numHeads, int queryHeadDim, int posHeadDim, int posDim, int valueHeadDim, int feedforwardDim,
        int timeEmbedDim, bool useTimeEmb)
    {
        _dim = dim; _numLayers = numLayers; _downsampleFactor = downsampleFactor;
        _posDim = posDim; _timeEmbedDim = timeEmbedDim; _useTimeEmb = useTimeEmb;

        _layers = new ZipformerEncoderLayer[numLayers];
        for (int i = 0; i < numLayers; i++)
            _layers[i] = new ZipformerEncoderLayer(dim, numHeads, queryHeadDim, posHeadDim, posDim, valueHeadDim, feedforwardDim, cnnKernel);

        if (downsampleFactor != 1)
        {
            _downsample = new ZipformerSimpleDownsample(dim, downsampleFactor);
            _outCombiner = new ZipformerBypass(dim);
        }
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        string layerRoot = _downsampleFactor != 1 ? $"{prefix}.encoder" : prefix;
        for (int i = 0; i < _numLayers; i++)
            _layers[i].LoadWeights(w, $"{layerRoot}.layers.{i}");

        if (_downsampleFactor != 1)
        {
            _downsample!.LoadWeights(w, $"{prefix}.downsample");
            _outCombiner!.LoadWeights(w, $"{prefix}.out_combiner.bypass_scale");
        }

        if (_useTimeEmb)
        {
            _timeEmbW = WhisperOps.EnsureF32(w[$"{layerRoot}.time_emb.1.weight"]);
            _timeEmbB = WhisperOps.EnsureF32(w[$"{layerRoot}.time_emb.1.bias"]);
        }
    }

    /// <summary><paramref name="x"/> is <c>[1, T, dim]</c> (NOT disposed — caller owns it).
    /// <paramref name="sharedTimeEmb"/> is the top-level MLP's shared <c>[time_embed_dim]</c> output, or null
    /// for the text encoder. Returns a new <c>[1, T, dim]</c> tensor (same T as input).</summary>
    public Tensor Forward(IBackend backend, Tensor x, Tensor? sharedTimeEmb, int t)
    {
        Tensor working;
        int workT;
        bool ownWorking;
        if (_downsample is not null)
        {
            working = _downsample.Forward(x, t, out workT);
            ownWorking = true;
        }
        else
        {
            working = x;
            workT = t;
            ownWorking = false;
        }

        Tensor? stageTimeEmb = _useTimeEmb ? ComputeStageTimeEmb(backend, sharedTimeEmb!) : null;
        Tensor posEmb = GetOrBuildPosEmb(workT);

        Tensor cur = working;
        for (int i = 0; i < _numLayers; i++)
        {
            Tensor next = _layers[i].Forward(backend, cur, posEmb, stageTimeEmb, workT);
            if (ownWorking) cur.Dispose();
            cur = next;
            ownWorking = true;
        }
        stageTimeEmb?.Dispose();

        if (_downsample is null)
            return cur;

        Tensor upsampled = ZipformerSimpleUpsample.Forward(cur, workT, _dim, _downsampleFactor);
        cur.Dispose();
        Tensor trimmed = TrimToLength(upsampled, t, _dim);
        upsampled.Dispose();
        Tensor combined = _outCombiner!.Forward(backend, x, trimmed, t);
        trimmed.Dispose();
        return combined;
    }

    /// <summary>Returns the cached <see cref="ZipformerRelPosEncoding.Build"/> table for <paramref name="t"/>,
    /// rebuilding only when <paramref name="t"/> differs from the last call (see <see cref="_cachedPosEmbT"/>
    /// doc). Owned by this stage — callers must NOT dispose the returned tensor.</summary>
    private Tensor GetOrBuildPosEmb(int t)
    {
        if (_cachedPosEmb is null || _cachedPosEmbT != t)
        {
            _cachedPosEmb?.Dispose();
            _cachedPosEmb = ZipformerRelPosEncoding.Build(t, _posDim);
            _cachedPosEmbT = t;
        }
        return _cachedPosEmb;
    }

    /// <summary>Returns the stage-projected time embedding, shape <c>[1, 1, dim]</c> (a single per-channel
    /// vector, broadcast across all frames by the calling layer).</summary>
    private Tensor ComputeStageTimeEmb(IBackend backend, Tensor sharedTimeEmb)
    {
        Tensor activated = new(new TensorShape(1, 1, _timeEmbedDim), DType.F32);
        float* sp = (float*)sharedTimeEmb.DataPointer;
        float* ap = (float*)activated.DataPointer;
        for (int d = 0; d < _timeEmbedDim; d++) ap[d] = ZipformerScaling.SwooshR(sp[d]);
        Tensor projected = WhisperOps.ProjectLinear(backend, activated, _timeEmbW!, _timeEmbB, 1, 1, _timeEmbedDim, _dim);
        activated.Dispose();
        return projected;
    }

    private static Tensor TrimToLength(Tensor x, int t, int dim)
    {
        Tensor output = new(new TensorShape(1, t, dim), DType.F32);
        Buffer.MemoryCopy((float*)x.DataPointer, (float*)output.DataPointer,
            (long)t * dim * sizeof(float), (long)t * dim * sizeof(float));
        return output;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (ZipformerEncoderLayer l in _layers) foreach (Tensor t in l.EnumerateWeights()) yield return t;
        if (_downsample is not null) foreach (Tensor t in _downsample.EnumerateWeights()) yield return t;
        if (_outCombiner is not null) foreach (Tensor t in _outCombiner.EnumerateWeights()) yield return t;
        if (_timeEmbW is not null) yield return _timeEmbW;
        if (_timeEmbB is not null) yield return _timeEmbB;
    }
}
