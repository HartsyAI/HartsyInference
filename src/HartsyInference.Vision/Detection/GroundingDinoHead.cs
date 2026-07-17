using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection;

/// <summary>Grounding DINO's detection heads and the open-vocabulary scoring math.
/// <list type="bullet">
///   <item><b>Box head</b> — a 3-layer MLP (Linear→ReLU→Linear→ReLU→Linear) predicts a 4-vector delta per query
///   that is added (in inverse-sigmoid space) to the query's reference box, then sigmoid'd back to
///   <c>cxcywh ∈ [0,1]</c>.</item>
///   <item><b>Contrastive class score</b> — there are no fixed class weights. A query's logit against text token
///   <c>t</c> is the dot product of the decoder query embedding with the (projected) text token embedding
///   (computed by the model, which owns the text→hidden projection). A phrase's score aggregates the logits over
///   its sub-token span via <see cref="AggregateSpanScore"/>.</item>
/// </list></summary>
public sealed class GroundingDinoHead
{
    private readonly int _hidden;

    private Tensor? _fc1W, _fc1B, _fc2W, _fc2B, _fc3W, _fc3B;

    public GroundingDinoHead(int hidden)
    {
        _hidden = hidden;
    }

    public void Build(IGroundingWeightBag bag, string prefix)
    {
        _fc1W = bag.Get($"{prefix}.box.fc1.weight", _hidden, _hidden);
        _fc1B = bag.Get($"{prefix}.box.fc1.bias", _hidden);
        _fc2W = bag.Get($"{prefix}.box.fc2.weight", _hidden, _hidden);
        _fc2B = bag.Get($"{prefix}.box.fc2.bias", _hidden);
        _fc3W = bag.Get($"{prefix}.box.fc3.weight", 4, _hidden);
        _fc3B = bag.Get($"{prefix}.box.fc3.bias", 4);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_fc1W, _fc1B, _fc2W, _fc2B, _fc3W, _fc3B];
        foreach (Tensor? t in all)
            if (t is not null) yield return t;
    }

    /// <summary>Predicts boxes from <paramref name="queries"/> <c>[1, Nq, hidden]</c> and per-query
    /// <paramref name="referenceLogit"/> <c>[1, Nq, 4]</c> (reference boxes already in inverse-sigmoid space).
    /// Returns <c>[1, Nq, 4]</c> cxcywh in <c>[0,1]</c>; caller owns it.</summary>
    public Tensor ForwardBoxes(IBackend backend, Tensor queries, Tensor referenceLogit)
    {
        if (_fc1W is null)
            throw new InvalidOperationException("GroundingDinoHead weights not loaded.");
        int nq = (int)queries.Shape[1];

        Tensor h1 = new Tensor(new TensorShape(1, nq, _hidden), DType.F32);
        backend.Linear(h1, queries, _fc1W!, _fc1B);
        backend.LeakyRelu(h1, h1, 0f);
        Tensor h2 = new Tensor(new TensorShape(1, nq, _hidden), DType.F32);
        backend.Linear(h2, h1, _fc2W!, _fc2B);
        backend.LeakyRelu(h2, h2, 0f);
        h1.Dispose();
        Tensor delta = new Tensor(new TensorShape(1, nq, 4), DType.F32);
        backend.Linear(delta, h2, _fc3W!, _fc3B);
        h2.Dispose();

        backend.Add(delta, delta, referenceLogit);
        Tensor boxes = new Tensor(new TensorShape(1, nq, 4), DType.F32);
        backend.Sigmoid(boxes, delta);
        delta.Dispose();
        return boxes;
    }

    /// <summary>Aggregates a phrase's contrastive score from its sub-token logits. <paramref name="tokenScores"/> is
    /// the length-T score row for one query; the phrase spans <c>[start, end)</c> (end exclusive). Max or mean over
    /// the span per <paramref name="aggregation"/>. Returns <see cref="float.NegativeInfinity"/> for an empty span.</summary>
    public static float AggregateSpanScore(ReadOnlySpan<float> tokenScores, int start, int end, GroundingDinoPhraseAggregation aggregation)
    {
        int lo = Math.Max(0, start);
        int hi = Math.Min(tokenScores.Length, end);
        if (hi <= lo)
            return float.NegativeInfinity;

        if (aggregation == GroundingDinoPhraseAggregation.Mean)
        {
            double sum = 0;
            for (int i = lo; i < hi; i++)
                sum += tokenScores[i];
            return (float)(sum / (hi - lo));
        }

        float max = float.NegativeInfinity;
        for (int i = lo; i < hi; i++)
            if (tokenScores[i] > max)
                max = tokenScores[i];
        return max;
    }
}
