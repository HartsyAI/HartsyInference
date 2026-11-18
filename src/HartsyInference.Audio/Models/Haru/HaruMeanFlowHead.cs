using HartsyInference.Audio.Layers;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Haru;

/// <summary>Haru's MeanFlow head: one audio latent per language-model hidden state. The network predicts the mean velocity
/// <c>u(z, t, h | cond, w)</c> of the flow from noise (<c>t = 0</c>) to data over an interval <c>h</c>, so a sample is
/// <c>z += h * u</c>; the guidance weight <c>w</c> is an input, so one forward pass gives a guided latent.</summary>
internal sealed class HaruMeanFlowHead : IDisposable
{
    private const string Prefix = "head";

    private readonly HaruHeadConfig _config;
    private readonly ScalarEmbed _time;
    private readonly ScalarEmbed _interval;
    private readonly ScalarEmbed _step;
    private readonly ScalarEmbed _omega;
    private readonly List<Tensor> _owned = [];
    private AdaLnSwiGluStack? _stack;
    private Tensor? _latentProj;
    private Tensor? _condProj;

    internal HaruMeanFlowHead(HaruHeadConfig config)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _time = new ScalarEmbed(config.HiddenSize);
        _interval = new ScalarEmbed(config.HiddenSize);
        _step = new ScalarEmbed(config.HiddenSize);
        _omega = new ScalarEmbed(config.HiddenSize);
    }

    /// <summary>Reads the <c>head.*</c> tensors of <paramref name="weights"/> as stored; they must outlive the head.</summary>
    internal void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        if (_stack is not null)
        {
            throw new InvalidOperationException("The Haru head weights are already loaded.");
        }
        int h = _config.HiddenSize;
        _latentProj = Weight(weights, "latent_proj.weight", h, _config.LatentDim);
        _condProj = Weight(weights, "cond_proj.weight", h, h);
        _time.LoadWeights(Weight(weights, "t_embed.fc1.weight", h, ScalarEmbed.DefaultFrequencyDim), Weight(weights, "t_embed.fc2.weight", h, h));
        _interval.LoadWeights(Weight(weights, "dt_embed.fc1.weight", h, ScalarEmbed.DefaultFrequencyDim), Weight(weights, "dt_embed.fc2.weight", h, h));
        _step.LoadWeights(Weight(weights, "h_embed.fc1.weight", h, ScalarEmbed.DefaultFrequencyDim), Weight(weights, "h_embed.fc2.weight", h, h));
        _omega.LoadWeights(Weight(weights, "omega_embed.fc1.weight", h, ScalarEmbed.DefaultFrequencyDim), Weight(weights, "omega_embed.fc2.weight", h, h));

        int ffn = _config.FfnSize;
        List<AdaLnSwiGluStack.Layer> layers = [];
        for (int i = 0; i < _config.NumLayers; i++)
        {
            string p = $"layers.{i}";
            layers.Add(new AdaLnSwiGluStack.Layer(
                Norm: NormWeight(weights, $"{p}.norm.weight", h),
                Ada: Weight(weights, $"{p}.ada.weight", 3 * h, h),
                Gate: Weight(weights, $"{p}.ffn.gate_proj.weight", ffn, h),
                Up: Weight(weights, $"{p}.ffn.up_proj.weight", ffn, h),
                Down: Weight(weights, $"{p}.ffn.down_proj.weight", h, ffn)));
        }
        _stack = new AdaLnSwiGluStack(h, ffn, _config.NormEps, _config.LatentDim, layers,
            Weight(weights, "final.ada.weight", 2 * h, h), Weight(weights, "final.proj.weight", _config.LatentDim, h));
    }

    /// <summary>Samples one normalized latent per hidden state: <paramref name="hidden"/> (<c>[1, N, H]</c>) and
    /// <paramref name="noise"/> (<c>[1, N, D]</c>) in, a fresh <c>[1, N, D]</c> out. <paramref name="guidance"/> and
    /// <paramref name="steps"/> default to the config's.</summary>
    internal Tensor Sample(IBackend backend, Tensor hidden, Tensor noise, float? guidance = null, int? steps = null)
    {
        if (_stack is null || _latentProj is null || _condProj is null)
        {
            throw new InvalidOperationException("The Haru head weights are not loaded.");
        }
        if (hidden.Shape.Rank != 3 || hidden.Shape[0] != 1 || hidden.Shape[2] != _config.HiddenSize)
        {
            throw new ArgumentException($"hidden must be [1, N, {_config.HiddenSize}], got {hidden.Shape}.", nameof(hidden));
        }
        int n = (int)hidden.Shape[1];
        if (noise.Shape.Rank != 3 || noise.Shape[0] != 1 || noise.Shape[1] != n || noise.Shape[2] != _config.LatentDim)
        {
            throw new ArgumentException($"noise must be [1, {n}, {_config.LatentDim}], got {noise.Shape}.", nameof(noise));
        }
        float w = guidance ?? _config.GuidanceScale;
        int count = steps ?? _config.NumSteps;
        if (w <= 0 || count < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(guidance), "Guidance must be positive and the step count at least one.");
        }

        Tensor cond = new(hidden.Shape, DType.F32);
        backend.Linear(cond, hidden, _condProj, null);
        Tensor z = new(noise.Shape, DType.F32);
        backend.Scale(z, noise, 1f);
        float interval = 1f / count;
        for (int i = 0; i < count; i++)
        {
            Tensor velocity = Velocity(backend, z, cond, i * interval, interval, w, n);
            Tensor step = new(z.Shape, DType.F32);
            backend.Scale(step, velocity, interval);
            velocity.Dispose();
            Tensor next = new(z.Shape, DType.F32);
            backend.Add(next, z, step);
            step.Dispose();
            z.Dispose();
            z = next;
        }
        cond.Dispose();
        return z;
    }

    internal IEnumerable<Tensor> EnumerateWeights()
    {
        if (_latentProj is not null)
        {
            yield return _latentProj;
        }
        if (_condProj is not null)
        {
            yield return _condProj;
        }
        foreach (ScalarEmbed embed in new[] { _time, _interval, _step, _omega })
        {
            foreach (Tensor t in embed.EnumerateWeights())
            {
                yield return t;
            }
        }
        if (_stack is not null)
        {
            foreach (Tensor t in _stack.EnumerateWeights())
            {
                yield return t;
            }
        }
    }

    public void Dispose()
    {
        _stack?.Dispose();
        foreach (Tensor t in _owned)
        {
            t.Dispose();
        }
        _owned.Clear();
    }

    private Tensor Velocity(IBackend backend, Tensor z, Tensor cond, float t, float h, float guidance, int n)
    {
        float[] rows = new float[n];
        Tensor c = Embedded(backend, _time, rows, t, cond);
        Tensor withInterval = Embedded(backend, _interval, rows, _config.IntervalEmbed, c);
        c.Dispose();
        Tensor withStep = Embedded(backend, _step, rows, h, withInterval);
        withInterval.Dispose();
        Tensor withOmega = Embedded(backend, _omega, rows, 1f - 1f / guidance, withStep);
        withStep.Dispose();
        Tensor active = new(withOmega.Shape, DType.F32);
        backend.Silu(active, withOmega);
        withOmega.Dispose();

        Tensor x = new(new TensorShape(1, n, _config.HiddenSize), DType.F32);
        backend.Linear(x, z, _latentProj!, null);
        Tensor result = _stack!.Forward(backend, x, active);
        x.Dispose();
        active.Dispose();
        return result;
    }

    /// <summary><paramref name="sum"/> plus the embedding of <paramref name="value"/> in every row, as a fresh tensor.</summary>
    private static Tensor Embedded(IBackend backend, ScalarEmbed embed, float[] rows, float value, Tensor sum)
    {
        Array.Fill(rows, value);
        Tensor embedding = embed.Forward(backend, rows);
        Tensor result = new(sum.Shape, DType.F32);
        backend.Add(result, sum, embedding);
        embedding.Dispose();
        return result;
    }

    private static Tensor Weight(IReadOnlyDictionary<string, Tensor> weights, string name, int rows, int cols)
    {
        string key = $"{Prefix}.{name}";
        if (!weights.TryGetValue(key, out Tensor? tensor))
        {
            throw new KeyNotFoundException($"The Haru checkpoint has no tensor '{key}'.");
        }
        if (tensor.Shape.Rank != 2 || tensor.Shape[0] != rows || tensor.Shape[1] != cols)
        {
            throw new InvalidDataException($"'{key}' must be [{rows}, {cols}], got {tensor.Shape}.");
        }
        return tensor;
    }

    private Tensor NormWeight(IReadOnlyDictionary<string, Tensor> weights, string name, int size)
    {
        string key = $"{Prefix}.{name}";
        if (!weights.TryGetValue(key, out Tensor? tensor) || tensor.ElementCount != size)
        {
            throw new InvalidDataException($"The Haru checkpoint needs '{key}' with {size} values.");
        }
        if (tensor.DType == DType.F32)
        {
            return tensor;
        }
        Tensor widened = tensor.CastTo(DType.F32);
        _owned.Add(widened);
        return widened;
    }
}
