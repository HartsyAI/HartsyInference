using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.TextEncoders.Bert;

namespace HartsyInference.Vision.Detection.GroundingDino;

/// <summary>Grounding DINO text tower: the block-masked BERT-base encoder (<c>model.text_backbone.*</c>) followed by
/// the linear projection to <c>d_model</c> (<c>model.text_projection.*</c>). Produces the projected caption token
/// features <c>[1, T, d_model]</c> consumed by the cross-modality encoder and the contrastive class head.</summary>
public sealed unsafe class GroundingDinoTextTower : IDisposable
{
    private const float MaskNeg = -1e30f;

    private readonly GroundingDinoConfig _cfg;
    private readonly BertModel _bert;
    private Tensor? _projW;   // [d_model, bert_hidden]
    private Tensor? _projB;   // [d_model]
    private int _disposed;

    public GroundingDinoTextTower(GroundingDinoConfig cfg)
    {
        _cfg = cfg;
        _bert = new BertModel(cfg.Text);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "model")
    {
        _bert.LoadWeights(weights, $"{prefix}.text_backbone");
        _projW = weights[$"{prefix}.text_projection.weight"];
        _projB = weights[$"{prefix}.text_projection.bias"];
    }

    /// <summary>Encodes caption ids to projected features <c>[1, T, d_model]</c>. Applies the special-token block
    /// self-attention mask and reset position ids; token-type is 0 everywhere (single caption).</summary>
    public Tensor Encode(IBackend backend, ReadOnlySpan<int> inputIds)
    {
        if (_projW is null) throw new InvalidOperationException("GroundingDinoTextTower weights not loaded.");
        int t = inputIds.Length;
        (bool[] attend, int[] positionIds) = GroundingDinoTextPrompt.Build(inputIds);

        int nh = _cfg.Text.NumHeads;
        Tensor mask = new(new TensorShape(1, nh, t, t), DType.F32);
        float* mp = (float*)mask.DataPointer;
        for (int h = 0; h < nh; h++)
            for (int i = 0; i < t; i++)
                for (int j = 0; j < t; j++)
                    mp[((long)h * t + i) * t + j] = attend[(long)i * t + j] ? 0f : MaskNeg;

        Tensor hidden = _bert.Forward(backend, inputIds, positionIds, ReadOnlySpan<int>.Empty, mask);
        mask.Dispose();

        Tensor projected = new(new TensorShape(1, t, _cfg.DModel), DType.F32);
        backend.Linear(projected, hidden, _projW!, _projB);
        hidden.Dispose();
        return projected;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor tt in _bert.EnumerateWeights()) yield return tt;
        if (_projW is not null) yield return _projW;
        if (_projB is not null) yield return _projB;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _bert.Dispose();
        GC.SuppressFinalize(this);
    }
}
