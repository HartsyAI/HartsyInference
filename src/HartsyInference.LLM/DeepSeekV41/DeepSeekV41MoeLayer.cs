using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>One V4.1 feed-forward layer on the host reference path: gate, <see cref="IBackend.MoeRoute"/>, routed experts and the shared expert.</summary>
public sealed class DeepSeekV41MoeLayer
{
    private readonly IBackend _backend;
    private readonly float[] _gateWeight;
    private readonly float[]? _gateBias;
    private readonly float[]? _gateBiasVl;
    private readonly IDeepSeekV41ExpertSource _experts;
    private readonly DeepSeekV41SwigluWeights _shared;
    private readonly MoeRouteArgs _route;
    private readonly float _swigluLimit;

    /// <summary>Creates the layer from F32 gate weights and an expert source.</summary>
    /// <param name="backend">Provides <see cref="IBackend.MoeRoute"/>.</param>
    /// <param name="gateWeight">Router matrix, <c>[numExperts, dim]</c>.</param>
    /// <param name="gateBias">Selection-only correction bias, <c>[numExperts]</c>, or null.</param>
    /// <param name="gateBiasVl">Bias used for image-span tokens, or null when the model has no vision path.</param>
    /// <param name="route">Scoring recipe; its <c>NumExperts</c> must match the gate.</param>
    /// <param name="experts">Routed experts, fetched lazily.</param>
    /// <param name="shared">The shared expert.</param>
    /// <param name="swigluLimit">Config <c>swiglu_limit</c>.</param>
    public DeepSeekV41MoeLayer(IBackend backend, float[] gateWeight, float[]? gateBias, float[]? gateBiasVl, in MoeRouteArgs route,
        IDeepSeekV41ExpertSource experts, DeepSeekV41SwigluWeights shared, float swigluLimit)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(gateWeight);
        ArgumentNullException.ThrowIfNull(experts);
        ArgumentNullException.ThrowIfNull(shared);
        shared.Validate();
        if (gateWeight.Length != (long)route.NumExperts * shared.Dim) throw new ArgumentException("gateWeight must be numExperts x dim.", nameof(gateWeight));
        if (gateBias is not null && gateBias.Length != route.NumExperts) throw new ArgumentException("gateBias must hold one value per expert.", nameof(gateBias));
        if (gateBiasVl is not null && (gateBias is null || gateBiasVl.Length != route.NumExperts))
            throw new ArgumentException("gateBiasVl needs gateBias and one value per expert.", nameof(gateBiasVl));
        _backend = backend;
        _gateWeight = gateWeight;
        _gateBias = gateBias;
        _gateBiasVl = gateBiasVl;
        _route = route;
        _experts = experts;
        _shared = shared;
        _swigluLimit = swigluLimit;
    }

    /// <summary>Diagnostic tap: receives the chosen expert ids, <c>[tokens, k]</c> row-major, after routing. Null in normal use.</summary>
    public Action<int[]>? RouteProbe { get; set; }

    /// <summary>Runs the layer over <paramref name="tokens"/> rows of <paramref name="x"/> and writes F32 results to <paramref name="y"/>.</summary>
    /// <param name="x">Normalized hidden states, <c>[tokens, dim]</c>.</param>
    /// <param name="tokens">Row count.</param>
    /// <param name="imageTokens">Per-token flag, nonzero inside an image span; null when there are none or no vision bias.</param>
    /// <param name="y">Receives <c>[tokens, dim]</c>.</param>
    public void Forward(ReadOnlySpan<float> x, int tokens, ReadOnlySpan<byte> imageTokens, Span<float> y)
    {
        int dim = _shared.Dim, experts = _route.NumExperts, k = _route.TopK;
        if (x.Length != (long)tokens * dim) throw new ArgumentException("x must hold tokens x dim values.", nameof(x));
        if (!imageTokens.IsEmpty && imageTokens.Length != tokens) throw new ArgumentException("imageTokens must hold one flag per token.", nameof(imageTokens));

        using Tensor logits = new(new TensorShape(tokens, experts), DType.F32);
        Span<float> l = logits.AsSpan<float>();
        for (int t = 0; t < tokens; t++)
            for (int e = 0; e < experts; e++)
            {
                float sum = 0f;
                ReadOnlySpan<float> row = x.Slice(t * dim, dim), g = _gateWeight.AsSpan(e * dim, dim);
                for (int d = 0; d < dim; d++) sum += row[d] * g[d];
                l[t * experts + e] = sum;
            }

        using Tensor idx = new(new TensorShape(tokens, k), DType.I32);
        using Tensor weights = new(new TensorShape(tokens, k), DType.F32);
        using Tensor? bias = _gateBias is null ? null : F32(_gateBias);
        using Tensor? altBias = _gateBiasVl is null || imageTokens.IsEmpty ? null : F32(_gateBiasVl);
        using Tensor? kinds = altBias is null ? null : Kinds(imageTokens);
        _backend.MoeRoute(idx, weights, logits, _route, bias, altBias, kinds);
        RouteProbe?.Invoke(idx.AsReadOnlySpan<int>().ToArray());

        DeepSeekV41MoeExecutor.Run(x, tokens, idx.AsReadOnlySpan<int>(), weights.AsReadOnlySpan<float>(), k, experts, _experts, _shared,
            _swigluLimit, y);
    }

    private static Tensor F32(float[] values)
    {
        Tensor t = new(new TensorShape(values.Length), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    private static Tensor Kinds(ReadOnlySpan<byte> flags)
    {
        Tensor t = new(new TensorShape(flags.Length), DType.I32);
        Span<int> kinds = t.AsSpan<int>();
        for (int i = 0; i < flags.Length; i++) kinds[i] = flags[i] != 0 ? 1 : 0;
        return t;
    }
}
