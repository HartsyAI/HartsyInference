using HartsyInference.Core.Exceptions;

namespace HartsyInference.Audio.Models.Wake;

/// <summary>Trains a wake-word head: the small classifier that turns 16 frozen backbone embeddings into one
/// probability. Everything upstream of it — mel front-end and speech-embedding backbone — is frozen and shared,
/// so a new wake word costs only this.
///
/// <para>The architecture matches the shipped <c>alexa</c> head exactly (Flatten → Linear(1536→H) → ReLU →
/// Linear(H→H) → ReLU → Linear(H→1) → Sigmoid), <b>without</b> the LayerNorm that <c>hey_mycroft</c> and
/// <c>hey_jarvis</c> carry. Both variants ship and both work; picking the simpler one means the backward pass
/// has no normalization statistics to differentiate through, which is the part of a hand-written trainer most
/// likely to be subtly wrong. Weights are emitted under the same <c>nn.Sequential</c> index names the ONNX
/// heads use, so trained and shipped heads load through one code path.</para>
///
/// <para>Plain float arrays and an explicit backward pass rather than any autograd: the model is ~213k
/// parameters and trains on CPU in seconds, so the machinery would cost more than it saves.</para></summary>
public sealed class WakeHeadTrainer
{
    private const float Epsilon = 1e-8f;

    private readonly int _hidden;
    private readonly int _inputDim;
    private readonly Random _random;

    private readonly float[] _w1, _b1, _w2, _b2, _w3, _b3;
    private readonly float[] _mw1, _vw1, _mb1, _vb1, _mw2, _vw2, _mb2, _vb2, _mw3, _vw3, _mb3, _vb3;
    private int _step;

    /// <summary>Hidden width; 128 matches every shipped openWakeWord head.</summary>
    public int Hidden => _hidden;

    public WakeHeadTrainer(int hidden = 128, int inputDim = WakeHead.InputDim, int seed = 1234)
    {
        if (hidden <= 0) throw new ArgumentOutOfRangeException(nameof(hidden));
        _hidden = hidden;
        _inputDim = inputDim;
        _random = new Random(seed);

        _w1 = new float[hidden * inputDim]; _b1 = new float[hidden];
        _w2 = new float[hidden * hidden]; _b2 = new float[hidden];
        _w3 = new float[hidden]; _b3 = new float[1];
        HeInit(_w1, inputDim);
        HeInit(_w2, hidden);
        HeInit(_w3, hidden);

        _mw1 = new float[_w1.Length]; _vw1 = new float[_w1.Length];
        _mb1 = new float[_b1.Length]; _vb1 = new float[_b1.Length];
        _mw2 = new float[_w2.Length]; _vw2 = new float[_w2.Length];
        _mb2 = new float[_b2.Length]; _vb2 = new float[_b2.Length];
        _mw3 = new float[_w3.Length]; _vw3 = new float[_w3.Length];
        _mb3 = new float[_b3.Length]; _vb3 = new float[_b3.Length];
    }

    private void HeInit(float[] weights, int fanIn)
    {
        float scale = MathF.Sqrt(2f / fanIn);
        for (int i = 0; i < weights.Length; i++)
        {
            // Box-Muller: Random has no Gaussian, and uniform init visibly slows convergence here.
            double u1 = 1.0 - _random.NextDouble(), u2 = _random.NextDouble();
            weights[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2)) * scale;
        }
    }

    /// <summary>Probability in <c>[0, 1]</c> for one flattened <c>16 × 96</c> feature window — the same value
    /// <see cref="WakeHead.Score"/> produces once these weights are exported.</summary>
    public float Predict(ReadOnlySpan<float> features)
    {
        Span<float> h1 = stackalloc float[_hidden];
        Span<float> h2 = stackalloc float[_hidden];
        return 1f / (1f + MathF.Exp(-Forward(features, h1, h2)));
    }

    private float Forward(ReadOnlySpan<float> x, Span<float> h1, Span<float> h2)
    {
        for (int j = 0; j < _hidden; j++)
        {
            float sum = _b1[j];
            int row = j * _inputDim;
            for (int i = 0; i < _inputDim; i++) sum += _w1[row + i] * x[i];
            h1[j] = sum > 0f ? sum : 0f;
        }
        for (int j = 0; j < _hidden; j++)
        {
            float sum = _b2[j];
            int row = j * _hidden;
            for (int i = 0; i < _hidden; i++) sum += _w2[row + i] * h1[i];
            h2[j] = sum > 0f ? sum : 0f;
        }
        float logit = _b3[0];
        for (int i = 0; i < _hidden; i++) logit += _w3[i] * h2[i];
        return logit;
    }

    /// <summary>Runs one Adam step over a batch and returns its mean binary cross-entropy.
    /// <paramref name="batch"/> rows are flattened feature windows; <paramref name="labels"/> are 1 for the
    /// wake word and 0 otherwise. <paramref name="positiveWeight"/> scales the positive class, which matters
    /// because a usable training set is heavily negative-dominated.</summary>
    public float TrainBatch(IReadOnlyList<float[]> batch, IReadOnlyList<float> labels, float learningRate, float positiveWeight = 1f)
    {
        WakeHeadGradients gradients = ComputeGradients(batch, labels, positiveWeight);
        _step++;
        Adam(_w1, gradients.W1, _mw1, _vw1, learningRate);
        Adam(_b1, gradients.B1, _mb1, _vb1, learningRate);
        Adam(_w2, gradients.W2, _mw2, _vw2, learningRate);
        Adam(_b2, gradients.B2, _mb2, _vb2, learningRate);
        Adam(_w3, gradients.W3, _mw3, _vw3, learningRate);
        Adam(_b3, gradients.B3, _mb3, _vb3, learningRate);
        return gradients.Loss;
    }

    /// <summary>Gradients for a batch without applying them. Exposed so a finite-difference check can compare
    /// the analytic backward pass directly instead of inferring it through Adam's normalization, which hides
    /// magnitude entirely and leaves only the sign to test.</summary>
    public WakeHeadGradients ComputeGradients(IReadOnlyList<float[]> batch, IReadOnlyList<float> labels, float positiveWeight = 1f)
    {
        if (batch.Count != labels.Count) throw new ArgumentException("Batch and label counts differ.", nameof(labels));
        if (batch.Count == 0) throw new ArgumentException("Empty batch.", nameof(batch));

        float[] gw1 = new float[_w1.Length], gb1 = new float[_b1.Length];
        float[] gw2 = new float[_w2.Length], gb2 = new float[_b2.Length];
        float[] gw3 = new float[_w3.Length], gb3 = new float[_b3.Length];
        Span<float> h1 = stackalloc float[_hidden];
        Span<float> h2 = stackalloc float[_hidden];
        Span<float> dh1 = stackalloc float[_hidden];
        Span<float> dh2 = stackalloc float[_hidden];

        float loss = 0f;
        for (int n = 0; n < batch.Count; n++)
        {
            float[] x = batch[n];
            if (x.Length != _inputDim) throw new HartsyInferenceException($"Feature window {n} has {x.Length} values, expected {_inputDim}.");
            float label = labels[n];
            float weight = label > 0.5f ? positiveWeight : 1f;

            float logit = Forward(x, h1, h2);
            float p = 1f / (1f + MathF.Exp(-logit));
            loss += weight * -(label * MathF.Log(p + Epsilon) + (1f - label) * MathF.Log(1f - p + Epsilon));

            // d(BCE)/d(logit) collapses to (p - y) through the sigmoid.
            float dLogit = weight * (p - label) / batch.Count;

            for (int i = 0; i < _hidden; i++)
            {
                gw3[i] += dLogit * h2[i];
                dh2[i] = dLogit * _w3[i] * (h2[i] > 0f ? 1f : 0f);
            }
            gb3[0] += dLogit;

            dh1.Clear();
            for (int j = 0; j < _hidden; j++)
            {
                float d = dh2[j];
                if (d == 0f) continue;
                int row = j * _hidden;
                for (int i = 0; i < _hidden; i++)
                {
                    gw2[row + i] += d * h1[i];
                    dh1[i] += d * _w2[row + i];
                }
                gb2[j] += d;
            }

            for (int j = 0; j < _hidden; j++)
            {
                float d = dh1[j] * (h1[j] > 0f ? 1f : 0f);
                if (d == 0f) continue;
                int row = j * _inputDim;
                for (int i = 0; i < _inputDim; i++) gw1[row + i] += d * x[i];
                gb1[j] += d;
            }
        }

        return new WakeHeadGradients(gw1, gb1, gw2, gb2, gw3, gb3, loss / batch.Count);
    }

    private void Adam(float[] parameters, float[] gradients, float[] m, float[] v, float learningRate)
    {
        const float Beta1 = 0.9f, Beta2 = 0.999f;
        float correction1 = 1f - MathF.Pow(Beta1, _step);
        float correction2 = 1f - MathF.Pow(Beta2, _step);
        for (int i = 0; i < parameters.Length; i++)
        {
            m[i] = Beta1 * m[i] + (1f - Beta1) * gradients[i];
            v[i] = Beta2 * v[i] + (1f - Beta2) * gradients[i] * gradients[i];
            float mHat = m[i] / correction1;
            float vHat = v[i] / correction2;
            parameters[i] -= learningRate * mHat / (MathF.Sqrt(vHat) + 1e-8f);
        }
    }

    /// <summary>Mutable view of the live parameter arrays, in the same order <see cref="WakeHeadGradients"/>
    /// reports gradients. For tests that perturb a parameter and re-measure the loss.</summary>
    public IReadOnlyList<float[]> Parameters => [_w1, _b1, _w2, _b2, _w3, _b3];

    /// <summary>The trained weights under the <c>nn.Sequential</c> index names <see cref="WakeHead"/> reads,
    /// ready to write as safetensors. Indices skip the activation layers, matching the shipped heads.</summary>
    public IReadOnlyDictionary<string, (int[] Shape, float[] Data)> ExportWeights() => new Dictionary<string, (int[], float[])>
    {
        ["1.weight"] = ([_hidden, _inputDim], _w1),
        ["1.bias"] = ([_hidden], _b1),
        ["3.weight"] = ([_hidden, _hidden], _w2),
        ["3.bias"] = ([_hidden], _b2),
        ["5.weight"] = ([1, _hidden], _w3),
        ["5.bias"] = ([1], _b3),
    };
}

/// <summary>Gradients for one batch, in the order <see cref="WakeHeadTrainer.Parameters"/> reports.</summary>
public sealed record WakeHeadGradients(float[] W1, float[] B1, float[] W2, float[] B2, float[] W3, float[] B3, float Loss)
{
    /// <summary>Gradient arrays in parameter order.</summary>
    public IReadOnlyList<float[]> All => [W1, B1, W2, B2, W3, B3];
}
