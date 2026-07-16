using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Adapters;

/// <summary>FLUX.1 Redux image projector: maps SigLIP-so400m patch tokens <c>[B, 729, 1152]</c> into the T5 conditioning space <c>[B, 729, 4096]</c> via <c>redux_down(silu(redux_up(x)))</c> (BFL / diffusers <c>ReduxImageEncoder</c>). The projected tokens are appended after the T5 text tokens in the Flux joint attention stream (they carry the same all-zero RoPE position as text).</summary>
public sealed class ReduxImageEncoder : IDisposable
{
    /// <summary>SigLIP-so400m hidden size — the projector's input feature dim.</summary>
    public const int ReduxDim = 1152;

    /// <summary>Flux T5 conditioning dim — the projector's output feature dim.</summary>
    public const int TxtInFeatures = 4096;

    private Tensor? _upWeight;
    private Tensor? _upBias;
    private Tensor? _downWeight;
    private Tensor? _downBias;
    private int _disposed;

    /// <summary>Loads the four projector tensors (<c>redux_up.weight/bias</c>, <c>redux_down.weight/bias</c>) from a checkpoint dict, casting to F32. Accepts both the BFL single-file names and a <c>image_embedder.</c>-prefixed variant.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        Tensor Get(string name)
        {
            if (weights.TryGetValue(name, out Tensor? t)) return t;
            if (weights.TryGetValue("image_embedder." + name, out t)) return t;
            throw new HartsyInferenceException($"Redux checkpoint is missing tensor '{name}'.");
        }
        _upWeight = EnsureF32(Get("redux_up.weight"));
        _upBias = EnsureF32(Get("redux_up.bias"));
        _downWeight = EnsureF32(Get("redux_down.weight"));
        _downBias = EnsureF32(Get("redux_down.bias"));
        if (_upWeight.Shape[1] != ReduxDim || _downWeight.Shape[0] != TxtInFeatures)
        {
            throw new HartsyInferenceException(
                $"Redux projector shape mismatch: redux_up.weight {_upWeight.Shape} (expected [*, {ReduxDim}]), " +
                $"redux_down.weight {_downWeight.Shape} (expected [{TxtInFeatures}, *]).");
        }
    }

    /// <summary>Projects SigLIP patch tokens <c>[B, N, 1152]</c> → Flux text-space tokens <c>[B, N, 4096]</c>.</summary>
    public Tensor Project(IBackend backend, Tensor siglipHiddenStates)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        if (_upWeight is null)
        {
            throw new InvalidOperationException("Redux projector weights not loaded. Call LoadWeights first.");
        }
        if (siglipHiddenStates.Shape.Rank != 3 || siglipHiddenStates.Shape[2] != ReduxDim)
        {
            throw new ArgumentException(
                $"Redux input must be [B, N, {ReduxDim}] SigLIP hidden states; got {siglipHiddenStates.Shape}.",
                nameof(siglipHiddenStates));
        }
        long batch = siglipHiddenStates.Shape[0];
        long tokens = siglipHiddenStates.Shape[1];
        long innerDim = _upWeight.Shape[0];
        Tensor up = new Tensor(new TensorShape(batch, tokens, innerDim), DType.F32);
        backend.Linear(up, siglipHiddenStates, _upWeight, _upBias);
        backend.Silu(up, up);
        Tensor output = new Tensor(new TensorShape(batch, tokens, TxtInFeatures), DType.F32);
        backend.Linear(output, up, _downWeight!, _downBias);
        up.Dispose();
        return output;
    }

    private static Tensor EnsureF32(Tensor t) => t.DType != DType.F32 ? t.CastTo(DType.F32) : t;

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _upWeight?.Dispose();
        _upBias?.Dispose();
        _downWeight?.Dispose();
        _downBias?.Dispose();
    }
}
