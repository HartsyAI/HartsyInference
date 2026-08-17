using HartsyInference.Audio.Models.Wake;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Correctness for the hand-written wake-head trainer.
///
/// <para>A wrong backward pass does not throw — it trains to something slightly worse and produces a wake word
/// that mysteriously misses. So the load-bearing test here is the finite-difference gradient check; the rest
/// confirm the optimizer actually converges and that a trained head round-trips through the production loader
/// rather than only through the trainer's own <c>Predict</c>.</para></summary>
public sealed class WakeHeadTrainerTests
{
    [Fact]
    public void Backward_MatchesNumericalGradients()
    {
        // Small dims: a finite-difference check over the full 1536-wide head would be 200k forward passes.
        const int inputDim = 12, hidden = 5;
        WakeHeadTrainer trainer = new(hidden, inputDim, seed: 7);
        Random random = new(99);

        List<float[]> batch = [];
        List<float> labels = [];
        for (int i = 0; i < 64; i++)
        {
            float[] x = new float[inputDim];
            for (int j = 0; j < inputDim; j++) x[j] = (float)(random.NextDouble() * 2 - 1);
            batch.Add(x);
            labels.Add(i % 2);
        }

        WakeHeadGradients analytic = trainer.ComputeGradients(batch, labels);
        IReadOnlyList<float[]> parameters = trainer.Parameters;
        IReadOnlyList<float[]> gradients = analytic.All;

        // Large enough that the loss difference clears float32 cancellation noise, small enough that the
        // second-order truncation error stays well under the tolerance.
        // Small on purpose. A ReLU network's loss is piecewise linear, so a wide probe straddles kinks and the
        // two-sided difference measures a slope the true gradient does not have. Sweeping h shows the estimate
        // converging on the analytic value (17.0% at 5e-3, 15.8% at 1e-3, 9.6% at 2e-4, agreement at 5e-5),
        // which is truncation error, not a backward-pass error. Double-precision loss keeps this h usable.
        const float h = 5e-5f;
        int compared = 0, dead = 0;
        float worst = 0f;
        string worstDetail = "";
        for (int layer = 0; layer < parameters.Count; layer++)
        {
            for (int i = 0; i < parameters[layer].Length; i += Math.Max(1, parameters[layer].Length / 8))
            {
                float original = parameters[layer][i];
                parameters[layer][i] = original + h;
                double up = Loss(trainer, batch, labels);
                parameters[layer][i] = original - h;
                double down = Loss(trainer, batch, labels);
                parameters[layer][i] = original;

                float numerical = (float)((up - down) / (2 * h));
                float expected = gradients[layer][i];
                // An exactly-zero analytic gradient means the unit is off for every sample in the batch. The
                // loss has a kink there, so a +/-h probe can switch the unit on and report a slope the true
                // subgradient does not have. Those are skipped as undefined, not counted as disagreement.
                if (expected == 0f) { dead++; continue; }
                if (MathF.Abs(numerical) < 1e-5f) continue;
                compared++;
                float relative = MathF.Abs(numerical - expected) / MathF.Max(MathF.Abs(numerical), 1e-6f);
                if (relative > worst) { worst = relative; worstDetail = $"layer {layer} idx {i}: numerical {numerical:G6} analytic {expected:G6}"; }
            }
        }

        Assert.True(compared >= 10, $"only {compared} parameters had a usable gradient to compare ({dead} dead)");
        Assert.True(dead < compared, $"{dead} dead parameters vs {compared} live — the network is mostly inactive, so this check proves little");
        Assert.True(worst < 0.02f, $"analytic gradient differs from finite differences by {worst:P1} at worst over {compared} parameters ({worstDetail})");
    }

    [Fact]
    public void Training_SeparatesTwoClusters()
    {
        const int inputDim = 96, hidden = 16;
        WakeHeadTrainer trainer = new(hidden, inputDim, seed: 3);
        Random random = new(5);

        List<float[]> batch = [];
        List<float> labels = [];
        for (int i = 0; i < 64; i++)
        {
            bool positive = i % 2 == 0;
            float[] x = new float[inputDim];
            for (int j = 0; j < inputDim; j++)
                x[j] = (float)(random.NextDouble() * 0.5 - 0.25) + (positive ? 1f : -1f);
            batch.Add(x);
            labels.Add(positive ? 1f : 0f);
        }

        float first = trainer.TrainBatch(batch, labels, 3e-3f);
        for (int epoch = 0; epoch < 300; epoch++) trainer.TrainBatch(batch, labels, 3e-3f);
        float last = trainer.TrainBatch(batch, labels, 3e-3f);

        Assert.True(last < first * 0.2f, $"loss only fell from {first} to {last}");
        for (int i = 0; i < batch.Count; i++)
        {
            float p = trainer.Predict(batch[i]);
            Assert.True(labels[i] > 0.5f ? p > 0.5f : p < 0.5f, $"sample {i} (label {labels[i]}) predicted {p}");
        }
    }

    [Fact]
    public void ExportedWeights_LoadThroughTheProductionHead()
    {
        WakeHeadTrainer trainer = new(hidden: 8);
        Random random = new(11);
        List<float[]> batch = [];
        List<float> labels = [];
        for (int i = 0; i < 8; i++)
        {
            float[] x = new float[WakeHead.InputDim];
            for (int j = 0; j < x.Length; j++) x[j] = (float)(random.NextDouble() * 2 - 1);
            batch.Add(x);
            labels.Add(i % 2);
        }
        for (int epoch = 0; epoch < 20; epoch++) trainer.TrainBatch(batch, labels, 1e-3f);

        string path = Path.Combine(Path.GetTempPath(), $"wake-head-{Guid.NewGuid():N}.safetensors");
        try
        {
            Dictionary<string, Tensor> tensors = [];
            foreach ((string name, (int[] shape, float[] data)) in trainer.ExportWeights())
            {
                Tensor t = new(new TensorShape([.. shape.Select(static d => (long)d)]), DType.F32);
                data.CopyTo(t.AsSpan<float>());
                tensors[name] = t;
            }
            SafeTensorsWriter.Save(path, tensors);
            foreach (Tensor t in tensors.Values) t.Dispose();

            using SafeTensorsLoader loader = new();
            loader.Load(path);
            using WakeHead head = new("trained");
            head.LoadWeights(loader.GetAllTensors());

            using CpuBackend backend = new();
            foreach (float[] sample in batch)
            {
                float expected = trainer.Predict(sample);
                Assert.True(MathF.Abs(head.Score(backend, sample) - expected) < 1e-4f,
                    "the production head disagrees with the trainer that produced its weights");
            }
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    /// <summary>Mean BCE from an independent double-precision forward pass over the same parameters.
    ///
    /// <para>Deliberately not <c>trainer.Predict</c>: the finite-difference numerator is a difference of two
    /// nearly equal losses, and the trainer's float32 forward loses that signal to cancellation, which shows up
    /// as a false gradient mismatch. Re-implementing the forward here in double also means the check compares
    /// the analytic gradient against an independent implementation rather than against the same arithmetic that
    /// produced it.</para></summary>
    private static double Loss(WakeHeadTrainer trainer, IReadOnlyList<float[]> batch, IReadOnlyList<float> labels)
    {
        IReadOnlyList<float[]> p = trainer.Parameters;
        float[] w1 = p[0], b1 = p[1], w2 = p[2], b2 = p[3], w3 = p[4], b3 = p[5];
        int hidden = b1.Length, inputDim = w1.Length / hidden;

        double total = 0;
        double[] h1 = new double[hidden], h2 = new double[hidden];
        for (int n = 0; n < batch.Count; n++)
        {
            float[] x = batch[n];
            for (int j = 0; j < hidden; j++)
            {
                double sum = b1[j];
                for (int i = 0; i < inputDim; i++) sum += w1[j * inputDim + i] * (double)x[i];
                h1[j] = sum > 0 ? sum : 0;
            }
            for (int j = 0; j < hidden; j++)
            {
                double sum = b2[j];
                for (int i = 0; i < hidden; i++) sum += w2[j * hidden + i] * h1[i];
                h2[j] = sum > 0 ? sum : 0;
            }
            double logit = b3[0];
            for (int i = 0; i < hidden; i++) logit += w3[i] * h2[i];
            double prob = Math.Clamp(1.0 / (1.0 + Math.Exp(-logit)), 1e-12, 1 - 1e-12);
            total += -(labels[n] * Math.Log(prob) + (1 - labels[n]) * Math.Log(1 - prob));
        }
        return total / batch.Count;
    }
}
