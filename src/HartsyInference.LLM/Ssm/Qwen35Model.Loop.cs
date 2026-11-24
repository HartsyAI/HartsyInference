using HartsyInference.Core.Backends;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.LLM.Transformer.Mhc;

namespace HartsyInference.LLM.Ssm;

public sealed unsafe partial class Qwen35Model
{
    /// <summary>The two F32 routers of one physical layer's hyper-connections (<c>mhc_attn</c> and <c>mhc_ffn</c>): a <c>[2 * streams + streams^2, streams * dim]</c> projection and its bias each.</summary>
    internal sealed record MhcRouterWeights(float[] AttnProjection, float[] AttnBias, float[] FfnProjection, float[] FfnBias);

    private RopeScaling _ropeScaling = RopeScaling.None;
    private SenseNovaLoopConfig? _loop;
    private IReadOnlyDictionary<int, MhcRouterWeights>? _mhcWeights;
    private readonly Dictionary<int, (HyperConnection Attn, HyperConnection Ffn)> _mhcRouters = [];
    private IBackend? _mhcBackend;

    /// <summary>The RoPE frequency scaling the full-attention layers apply (YaRN on SenseNova-Loop; none otherwise).</summary>
    public RopeScaling RopeScaling => _ropeScaling;

    /// <summary>The Loop execution plan when this model is a SenseNova-Loop checkpoint; null for a plain Qwen3.5.</summary>
    /// <remarks>Layers of the model are then execution slots: <see cref="NumLayers"/> is the slot count, state and caches are per slot, and
    /// the weights of a repeated physical layer are the same tensors under every slot that runs it.</remarks>
    public SenseNovaLoopConfig? LoopPlan => _loop;

    /// <summary>Installs the Loop plan, its routers and the RoPE scaling a loader read from the checkpoint.</summary>
    internal void ConfigureLoop(SenseNovaLoopConfig loop, IReadOnlyDictionary<int, MhcRouterWeights> routers, RopeScaling ropeScaling)
    {
        ArgumentNullException.ThrowIfNull(loop);
        ArgumentNullException.ThrowIfNull(routers);
        if (loop.Slots != NumLayers)
        {
            throw new ArgumentException($"The Loop plan has {loop.Slots} slots; the model has {NumLayers} layers.", nameof(loop));
        }
        for (int p = loop.MhcStart; p < loop.MhcEnd; p++)
        {
            if (!routers.ContainsKey(p)) throw new ArgumentException($"No mHC routers for physical layer {p}.", nameof(routers));
        }
        _loop = loop;
        _mhcWeights = routers;
        _ropeScaling = ropeScaling;
    }

    /// <summary>Sets the RoPE scaling of a model built without a Loop plan.</summary>
    internal void ConfigureRope(RopeScaling ropeScaling) => _ropeScaling = ropeScaling;

    private (HyperConnection Attn, HyperConnection Ffn) RoutersFor(IBackend backend, int physical)
    {
        if (!ReferenceEquals(_mhcBackend, backend))
        {
            _mhcRouters.Clear();
            _mhcBackend = backend;
        }
        if (_mhcRouters.TryGetValue(physical, out (HyperConnection Attn, HyperConnection Ffn) cached)) return cached;
        SenseNovaLoopConfig loop = _loop!;
        MhcRouterWeights w = _mhcWeights![physical];
        HyperConnectionConfig config = loop.HyperConnection(DModel, Eps);
        float[] unitScale = [1f, 1f, 1f];
        (HyperConnection, HyperConnection) pair =
            (new HyperConnection(backend, config, w.AttnProjection, unitScale, w.AttnBias),
             new HyperConnection(backend, config, w.FfnProjection, unitScale, w.FfnBias));
        _mhcRouters[physical] = pair;
        return pair;
    }

    /// <summary>One decoder layer of a Loop checkpoint's hyper-connection range: two residual streams, a router before each sublayer that
    /// collapses them into the sublayer input and then mixes and writes the output back (<c>MHCDecoderLayer.forward</c>).</summary>
    /// <param name="h">The hidden state entering the range; read only at the first layer of the range.</param>
    /// <param name="streams">The streams <c>[seq, streams, DModel]</c>, created at the first layer and carried between layers.</param>
    /// <returns>At the last layer of the range the streams averaged back into a hidden state (caller owns); null elsewhere.</returns>
    private Tensor? MhcBlock(IBackend backend, Tensor h, ref float[]? streams, int i, int seq, Tensor? ropeCos, Tensor? ropeSin)
    {
        SenseNovaLoopConfig loop = _loop!;
        int hc = loop.Streams, d = DModel, physical = loop.Plan[i];
        string p = $"blk.{i}";
        if (loop.EntersStreams(i))
        {
            backend.Sync();
            streams = new float[(long)seq * hc * d];
            ReadOnlySpan<float> hidden = new((void*)h.DataPointer, seq * d);
            for (int t = 0; t < seq; t++)
                for (int s = 0; s < hc; s++)
                    hidden.Slice(t * d, d).CopyTo(streams.AsSpan((t * hc + s) * d, d));
        }
        else if (streams is null)
        {
            throw new InvalidOperationException($"Slot {i} (layer {physical}) is inside the hyper-connection range but no layer entered it.");
        }
        (HyperConnection attnRouter, HyperConnection ffnRouter) = RoutersFor(backend, physical);

        // Attention half: collapse, norm (inside the core), attention, mix and write back.
        Span<float> state = streams;
        (float[] pre, float[] post, float[] comb) = Mixes(attnRouter, state, seq);
        using (Tensor branchIn = Collapse(attnRouter, state, pre, seq, loop.PreScale))
        using (Tensor branch = _isLinear[i] ? LinearAttnCore(backend, branchIn, i, seq) : FullAttnCore(backend, branchIn, i, seq, ropeCos, ropeSin))
        {
            backend.Sync();
            attnRouter.Expand(branch.AsReadOnlySpan<float>(), state, post, comb, seq, state);
        }

        // Feed-forward half.
        (pre, post, comb) = Mixes(ffnRouter, state, seq);
        using (Tensor collapsed = Collapse(ffnRouter, state, pre, seq, loop.PreScale))
        using (Tensor preFfn = new(new TensorShape(1, seq, d), DType.F32))
        {
            backend.RmsNorm(preFfn, collapsed, W($"{p}.post_attention_norm.weight"), Eps);
            using Tensor branch = Ffn(backend, preFfn, p, i, seq);
            backend.Sync();
            ffnRouter.Expand(branch.AsReadOnlySpan<float>(), state, post, comb, seq, state);
        }

        if (!loop.LeavesStreams(i)) return null;
        Tensor left = new(new TensorShape(1, seq, d), DType.F32);
        Span<float> output = left.AsSpan<float>();
        for (int t = 0; t < seq; t++)
        {
            for (int c = 0; c < d; c++)
            {
                float sum = 0f;
                for (int s = 0; s < hc; s++) sum += state[(t * hc + s) * d + c];
                output[t * d + c] = sum / hc;
            }
        }
        streams = null;
        return left;
    }

    private (float[] Pre, float[] Post, float[] Comb) Mixes(HyperConnection router, ReadOnlySpan<float> streams, int seq)
    {
        int hc = router.Hc;
        float[] pre = new float[seq * hc], post = new float[seq * hc], comb = new float[seq * hc * hc];
        router.Mixes(streams, seq, pre, post, comb);
        return (pre, post, comb);
    }

    private Tensor Collapse(HyperConnection router, ReadOnlySpan<float> streams, float[] pre, int seq, float preScale)
    {
        Tensor y = new(new TensorShape(1, seq, router.Dim), DType.F32);
        Span<float> row = y.AsSpan<float>();
        router.Collapse(streams, pre, seq, row);
        if (preScale != 1f)
        {
            for (int i = 0; i < row.Length; i++) row[i] *= preScale;
        }
        return y;
    }
}
