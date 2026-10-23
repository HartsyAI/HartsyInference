using System.Globalization;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>Teacher-forced denoiser probe (<c>diagnostics.denoiserProbe</c>), the diffusion counterpart of the LLM's teacher-forced
/// log-probs: the record arm writes the conditioning and, at the probed steps, the denoiser's inputs and outputs; the replay arm
/// runs one more forward there on the recorded inputs and writes its distance from the recorded outputs to <c>probe.tsv</c>.</summary>
/// <remarks>One denoise loop per process (the CLI). Step graphs stay off while it is set: a captured step replays fixed buffers. A
/// normalized CFG combine (Lumina 2's default) is not <c>g·out0 + (1 − g)·out1</c>, so such a loop records no CFG row.</remarks>
public sealed class DenoiserProbe
{
    private const string TsvName = "probe.tsv";
    private static readonly HashSet<string> s_started = new(StringComparer.Ordinal);
    private readonly string _dir;
    private readonly string? _outDir;
    private readonly HashSet<int> _steps;
    private readonly HashSet<int> _taken = [];
    private Tensor[]? _pending;
    private int _pendingStep;
    private float _pendingSigma;

    private DenoiserProbe(string dir, string? outDir, HashSet<int> steps)
    {
        _dir = dir;
        _outDir = outDir;
        _steps = steps;
    }

    /// <summary>True while <c>diagnostics.denoiserProbe</c> is set.</summary>
    public static bool Active => !string.IsNullOrEmpty(EngineKnobs.DenoiserProbe.Value);

    /// <summary>True on the replay arm.</summary>
    public bool Replays => _outDir is not null;

    /// <summary>True while <c>diagnostics.denoiserProbe</c> is set to replay.</summary>
    public static bool Replaying => Active && IsReplay();

    /// <summary>The probe for a denoise loop of <paramref name="stepCount"/> steps, or null when the probe is off.</summary>
    public static DenoiserProbe? Open(int stepCount)
    {
        if (!Active)
            return null;
        if (stepCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(stepCount), $"The denoiser probe needs at least one step; got {stepCount}.");
        bool replay = IsReplay();
        string dir = RecordDir();
        HashSet<int> steps = ParseSteps(EngineKnobs.DenoiserProbeSteps.Value, stepCount);
        if (!replay)
        {
            Directory.CreateDirectory(dir);
            Logs.Info($"[DenoiserProbe] recording steps {string.Join(",", steps.Order())} of {stepCount} to {dir}");
            return new DenoiserProbe(dir, null, steps);
        }
        string outDir = EngineKnobs.DenoiserProbeOut.Value is { Length: > 0 } o ? o : Path.Combine(dir, "replay");
        Directory.CreateDirectory(outDir);
        // A pipeline may build more than one sampler per generation; the first starts the table.
        lock (s_started)
            if (s_started.Add(Path.GetFullPath(outDir)))
                File.WriteAllText(Path.Combine(outDir, TsvName), "step\tsigma\toutput\trelL2\tcos\tmaxAbs\trmsRecorded\n");
        Logs.Info($"[DenoiserProbe] replaying steps {string.Join(",", steps.Order())} of {stepCount} against {dir} into {outDir}");
        return new DenoiserProbe(dir, outDir, steps);
    }

    /// <summary>True the first time a probed step is reached, so a sampler's second evaluation inside a step passes through.</summary>
    public bool Takes(int step) => _steps.Contains(step) && _taken.Add(step);

    /// <summary>Record arm: takes a copy of the inputs of the forward the loop is about to run at <paramref name="step"/>.</summary>
    public void RecordInputs(int step, float sigma, Tensor[] inputs)
    {
        if (Replays)
            throw new InvalidOperationException("The replay arm does not record.");
        if (_pending is not null)
            throw new InvalidOperationException($"Step {_pendingStep}'s record was never completed.");
        Tensor[] copies = new Tensor[inputs.Length];
        for (int i = 0; i < inputs.Length; i++) copies[i] = Copy(inputs[i]);
        _pending = copies;
        _pendingStep = step;
        _pendingSigma = sigma;
    }

    /// <summary>Record arm: writes the inputs taken by <see cref="RecordInputs"/> and the outputs of the forward that consumed them.</summary>
    /// <param name="guidance">The CFG scale that combines outputs 0 and 1 as <c>g·out0 + (1 − g)·out1</c>; 0 when they are not
    /// such a pair.</param>
    public void RecordOutputs(int step, float guidance, string[] names, Tensor?[] outputs)
    {
        if (_pending is null || _pendingStep != step)
            throw new InvalidOperationException($"Step {step}'s outputs arrived without its inputs.");
        if (names.Length != outputs.Length)
            throw new ArgumentException($"{names.Length} output names for {outputs.Length} outputs.", nameof(names));
        Dictionary<string, Tensor> tensors = new(StringComparer.Ordinal);
        List<Tensor> copies = [];
        try
        {
            for (int i = 0; i < _pending.Length; i++)
                tensors[$"in.{i}"] = _pending[i];
            for (int i = 0; i < outputs.Length; i++)
            {
                if (outputs[i] is not { } output)
                    continue;
                Tensor copy = Copy(output);
                copies.Add(copy);
                tensors[$"out.{names[i]}"] = copy;
            }
            SafeTensorsWriter.Save(StepPath(_dir, step), tensors, new Dictionary<string, string>
            {
                ["sigma"] = Bits(_pendingSigma),
                ["guidance"] = Bits(guidance),
                ["inputs"] = _pending.Length.ToString(CultureInfo.InvariantCulture),
                ["outputs"] = string.Join(",", names),
            });
        }
        finally
        {
            foreach (Tensor t in _pending) t.Dispose();
            foreach (Tensor t in copies) t.Dispose();
            _pending = null;
        }
    }

    /// <summary>Replay arm: runs <paramref name="forward"/> on the inputs recorded at <paramref name="step"/> and writes how far
    /// its outputs, which the probe disposes, land from the recorded ones. <paramref name="live"/> are this arm's own inputs, for
    /// the geometry check.</summary>
    public void Replay(int step, float sigma, Tensor[] live, Func<Tensor[], Tensor?[]> forward) =>
        ReplayOutputs(step, forward(ReplayInputs(step, sigma, live)));

    /// <summary>Replay arm, for a loop that runs the forward itself: the inputs recorded at <paramref name="step"/>, which the
    /// probe owns until <see cref="ReplayOutputs"/>. <paramref name="live"/> are this arm's own inputs, for the geometry check.</summary>
    public Tensor[] ReplayInputs(int step, float sigma, Tensor[] live)
    {
        if (!Replays)
            throw new InvalidOperationException("The record arm does not replay.");
        if (_pending is not null)
            throw new InvalidOperationException($"Step {_pendingStep}'s replay was never completed.");
        string path = StepPath(_dir, step);
        if (!File.Exists(path))
            throw new HartsyInferenceException($"The denoiser probe has no record of step {step} at {path}.");
        using SafeTensorsLoader record = new();
        record.Load(path);
        IReadOnlyDictionary<string, string> meta = record.Metadata
            ?? throw new HartsyInferenceException($"{path} carries no denoiser-probe metadata.");
        if (meta["sigma"] != Bits(sigma))
            throw new HartsyInferenceException($"Step {step} ran at sigma {sigma:R} but was recorded at {FromBits(meta["sigma"]):R}: "
                + "the two arms do not share a schedule, so the probe cannot compare them.");
        int count = int.Parse(meta["inputs"], CultureInfo.InvariantCulture);
        if (count != live.Length)
            throw new HartsyInferenceException($"Step {step} was recorded with {count} denoiser inputs; this arm has {live.Length}.");
        Tensor[] inputs = new Tensor[count];
        try
        {
            for (int i = 0; i < count; i++)
            {
                inputs[i] = Copy(record.GetTensor($"in.{i}"));
                if (inputs[i].Shape != live[i].Shape || inputs[i].DType != live[i].DType)
                    throw new HartsyInferenceException($"Step {step} input {i} was recorded as {inputs[i].DType} {inputs[i].Shape}; "
                        + $"this arm has {live[i].DType} {live[i].Shape}.");
            }
        }
        catch
        {
            foreach (Tensor? input in inputs) input?.Dispose();
            throw;
        }
        _pending = inputs;
        _pendingStep = step;
        _pendingSigma = sigma;
        return inputs;
    }

    /// <summary>Replay arm: compares the outputs of the forward run on <see cref="ReplayInputs"/> with the recorded ones, writes
    /// them, and disposes them and the recorded inputs.</summary>
    public void ReplayOutputs(int step, Tensor?[] outputs)
    {
        if (_pending is null || _pendingStep != step)
            throw new InvalidOperationException($"Step {step}'s replay outputs arrived without its inputs.");
        try
        {
            using SafeTensorsLoader record = new();
            record.Load(StepPath(_dir, step));
            Compare(record, step, _pendingSigma, FromBits(record.Metadata!["guidance"]), record.Metadata!["outputs"].Split(','), outputs);
        }
        finally
        {
            foreach (Tensor input in _pending) input.Dispose();
            _pending = null;
            for (int i = 0; i < outputs.Length; i++)
                if (outputs[i] is { } output && Array.IndexOf(outputs, output) == i)
                    output.Dispose();
        }
    }

    /// <summary>Record arm: writes <paramref name="tensor"/> as the conditioning <paramref name="name"/>. Replay arm: overwrites
    /// it in place with the recorded one, so this arm's denoiser sees the record's conditioning. No-op when the probe is off.</summary>
    public static unsafe void Condition(string name, Tensor? tensor)
    {
        if (tensor is null || !Active)
            return;
        string dir = RecordDir();
        string path = Path.Combine(dir, $"cond-{name}.safetensors");
        if (!IsReplay())
        {
            Directory.CreateDirectory(dir);
            using Tensor copy = Copy(tensor);
            SafeTensorsWriter.Save(path, new Dictionary<string, Tensor>(StringComparer.Ordinal) { ["t"] = copy });
            return;
        }
        if (!File.Exists(path))
            throw new HartsyInferenceException($"The denoiser probe has no recorded conditioning '{name}' at {path}.");
        using SafeTensorsLoader record = new();
        record.Load(path);
        using Tensor recorded = record.GetTensor("t");
        if (recorded.Shape != tensor.Shape || recorded.DType != tensor.DType)
            throw new HartsyInferenceException($"Conditioning '{name}' was recorded as {recorded.DType} {recorded.Shape}; this arm has "
                + $"{tensor.DType} {tensor.Shape}.");
        long bytes = Tensor.ComputeByteSize(tensor.Shape, tensor.DType);
        Buffer.MemoryCopy(recorded.DataPointer, tensor.DataPointer, bytes, bytes);
    }

    /// <summary>A host copy of <paramref name="t"/>'s current contents, in its own dtype.</summary>
    internal static unsafe Tensor Copy(Tensor t)
    {
        long bytes = Tensor.ComputeByteSize(t.Shape, t.DType);
        Tensor copy = new(t.Shape, t.DType);
        Buffer.MemoryCopy(t.DataPointer, copy.DataPointer, bytes, bytes);
        return copy;
    }

    private unsafe void Compare(SafeTensorsLoader record, int step, float sigma, float guidance, string[] names, Tensor?[] outputs)
    {
        if (outputs.Length != names.Length)
            throw new HartsyInferenceException($"Step {step} was recorded with outputs {string.Join(",", names)}; the forward returned "
                + $"{outputs.Length}.");
        Dictionary<string, Tensor> written = new(StringComparer.Ordinal);
        List<Tensor> owned = [];
        List<string> rows = [];
        float*[] candidate = new float*[outputs.Length], recorded = new float*[outputs.Length];
        long[] counts = new long[outputs.Length];
        try
        {
            for (int i = 0; i < outputs.Length; i++)
            {
                string key = $"out.{names[i]}";
                bool hasRecord = record.Descriptors.ContainsKey(key);
                if ((outputs[i] is not null) != hasRecord)
                    throw new HartsyInferenceException($"Step {step} output '{names[i]}' is {(hasRecord ? "recorded but not produced" : "produced but not recorded")}.");
                if (outputs[i] is not { } output)
                    continue;
                Tensor stored = record.GetTensor(key);
                owned.Add(stored);
                if (stored.Shape != output.Shape)
                    throw new HartsyInferenceException($"Step {step} output '{names[i]}' is {output.Shape}; it was recorded as {stored.Shape}.");
                candidate[i] = F32(output, owned);
                recorded[i] = F32(stored, owned);
                counts[i] = output.ElementCount;
                written[key] = Copy(output);
                rows.Add(Row(step, sigma, names[i], candidate[i], recorded[i], null, null, 0f, counts[i]));
            }
            if (guidance != 0f && outputs.Length >= 2 && candidate[0] != null && candidate[1] != null)
                rows.Add(Row(step, sigma, "cfg", candidate[0], recorded[0], candidate[1], recorded[1], guidance, counts[0]));
            SafeTensorsWriter.Save(StepPath(_outDir!, step), written);
        }
        finally
        {
            foreach (Tensor t in written.Values) t.Dispose();
            foreach (Tensor t in owned) t.Dispose();
        }
        File.AppendAllLines(Path.Combine(_outDir!, TsvName), rows);
    }

    /// <summary>One pass over the outputs: relL2, cosine and max |d|; with a second pair, of the CFG combine <c>g·out0 + (1 − g)·out1</c>.</summary>
    private static unsafe string Row(int step, float sigma, string name, float* cand0, float* rec0, float* cand1, float* rec1, float guidance,
        long count)
    {
        double diff = 0, norm = 0, dot = 0, cand = 0, max = 0;
        for (long i = 0; i < count; i++)
        {
            double c = cand0[i], r = rec0[i];
            if (cand1 != null)
            {
                c = guidance * c + (1.0 - guidance) * cand1[i];
                r = guidance * r + (1.0 - guidance) * rec1[i];
            }
            double d = c - r;
            diff += d * d;
            norm += r * r;
            cand += c * c;
            dot += c * r;
            max = Math.Max(max, Math.Abs(d));
        }
        double rel = norm > 0 ? Math.Sqrt(diff / norm) : Math.Sqrt(diff);
        double cos = norm > 0 && cand > 0 ? dot / Math.Sqrt(norm * cand) : double.NaN;
        Logs.Info($"[DenoiserProbe] step {step} {name}: relL2 {rel:E3}, cos {cos:F9}, max|d| {max:E3}");
        return string.Create(CultureInfo.InvariantCulture,
            $"{step}\t{sigma:R}\t{name}\t{rel:E6}\t{cos:F12}\t{max:E6}\t{Math.Sqrt(norm / count):E6}");
    }

    /// <summary>The tensor's values as F32 on the host; a widened copy, when one is needed, joins <paramref name="owned"/>.</summary>
    private static unsafe float* F32(Tensor t, List<Tensor> owned)
    {
        if (t.DType == DType.F32) return (float*)t.DataPointer;
        Tensor wide = t.CastTo(DType.F32);
        owned.Add(wide);
        return (float*)wide.DataPointer;
    }

    private static bool IsReplay()
    {
        string mode = EngineKnobs.DenoiserProbe.Value!;
        if (mode.Equals("record", StringComparison.OrdinalIgnoreCase)) return false;
        if (mode.Equals("replay", StringComparison.OrdinalIgnoreCase)) return true;
        throw new HartsyInferenceException($"diagnostics.denoiserProbe must be record or replay; got '{mode}'.");
    }

    private static string RecordDir() => EngineKnobs.DenoiserProbeDir.Value is { Length: > 0 } dir ? dir
        : throw new HartsyInferenceException("diagnostics.denoiserProbe needs diagnostics.denoiserProbeDir.");

    private static HashSet<int> ParseSteps(string? spec, int stepCount)
    {
        HashSet<int> steps = [];
        foreach (string token in (string.IsNullOrWhiteSpace(spec) ? "first,mid,last" : spec)
                     .Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
        {
            switch (token)
            {
                case "first": steps.Add(0); break;
                case "mid": steps.Add(stepCount / 2); break;
                case "last": steps.Add(stepCount - 1); break;
                case "all": for (int i = 0; i < stepCount; i++) steps.Add(i); break;
                default:
                    if (!int.TryParse(token, NumberStyles.None, CultureInfo.InvariantCulture, out int step) || step >= stepCount)
                        throw new HartsyInferenceException($"diagnostics.denoiserProbeSteps: '{token}' is not a step of a {stepCount}-step loop.");
                    steps.Add(step);
                    break;
            }
        }
        return steps;
    }

    private static string StepPath(string dir, int step) =>
        Path.Combine(dir, string.Create(CultureInfo.InvariantCulture, $"step-{step:D4}.safetensors"));

    private static string Bits(float value) => BitConverter.SingleToUInt32Bits(value).ToString("X8", CultureInfo.InvariantCulture);

    private static float FromBits(string bits) => BitConverter.UInt32BitsToSingle(uint.Parse(bits, NumberStyles.HexNumber, CultureInfo.InvariantCulture));
}
