using System.Text.Json;
using HartsyInference.Audio.Models.SheetSage2;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.Parity;

/// <summary>Gates S4 and S5: SheetSage2's decode side against the released implementation on the real checkpoint.
///
/// <para>The ladder is ordered so a failure names its own cause. <b>S3</b> compares raw logits under teacher
/// forcing, which is decoder math alone. <b>S4</b> compares the masked argmax per step, still teacher-forced, so
/// one disagreement stays one disagreement instead of cascading — that is the gate the grammar mask is in.
/// <b>S5</b> is the free-running greedy decode, token for token; it is the only gate that can fail for a reason
/// the first two cannot see, and the only one where a single flipped argmax rewrites everything after it.</para>
///
/// <para>Four reference cases, each able to fail for something the others cannot see. <c>audio</c> is a real
/// clip run to its own EOS. <c>prefixed</c> seeds the same clip from a prefix that leaves the grammar owing a
/// partner token, as an overlap prefix does to the next window. <c>synthetic</c> decodes a seeded memory at the
/// encoder's own scale, which the model never resolves, so it runs into the token limit. <c>shuffled</c> is the
/// clip's own memory with its rows permuted — the same values and scale, no temporal structure, and the only
/// case measured to put the checkpoint's unmasked argmax outside the grammar, which makes it the gate on
/// whether the mask is applied inside the loop at all.</para>
///
/// <para>The checkpoint has internalised its own grammar: on every realistic input tried here the unmasked
/// argmax was already legal, so a port that dropped the mask entirely would still pass the first three cases.
/// That is what <c>shuffled</c> is for, and why its <c>maskChangedTheArgmax</c> count is reported.</para>
///
/// <para>Reference and port both run float32 over the same float32-upcast BF16 weights, so S5's token-exactness
/// is a real criterion rather than a dtype coincidence. Gated on <c>SHEETSAGE2_CHECKPOINT</c> and
/// <c>SHEETSAGE2_REF_DIR</c> (<c>dump_sheetsage2_decoder_reference.py</c> output).</para></summary>
public sealed class SheetSage2DecoderParityTests
{
    /// <summary>A coarse guard on the logits, not the precision claim — S4 and S5 carry that, and both are
    /// exact comparisons. It is set well above the drift measured on any backend so that it fails for a
    /// structural mistake rather than for arithmetic order, and is reported alongside the share of the top-2
    /// margin that drift actually consumed, which is the number that says how close a gate came to flipping.</summary>
    private const float LogitTolerance = 2e-2f;

    private readonly ITestOutputHelper _out;

    public SheetSage2DecoderParityTests(ITestOutputHelper output) => _out = output;

    /// <summary>The dumped cases, each read from its own subdirectory of the reference dir.</summary>
    public static TheoryData<string> Cases => new("audio", "prefixed", "synthetic", "shuffled");

    /// <summary>Gate S3: the logits themselves, teacher-forced over the reference's own tokens.</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    [Trait("Category", "Integration")]
    public void Logits_MatchTheReference(string caseName)
    {
        if (!TryLoadCase(caseName, out ReferenceCase reference)) return;

        using SafeTensorsLoader loader = new();
        loader.Load(reference.Checkpoint);
        using IBackend backend = CreateBackend(out string backendName);
        using SheetSage2Decoder decoder = LoadDecoder(loader);
        using Tensor memory = reference.ReadMemory();
        using SheetSage2DecodeState state = decoder.StartDecode(backend, memory, reference.MaxTokens);

        float worst = 0f;
        int worstStep = -1;
        double worstShare = 0.0;
        int worstShareStep = -1;
        decoder.Decode(backend, reference.Prompt, state);
        for (int step = 0; step < reference.LogitRows; step++)
        {
            ReadOnlySpan<float> expected = reference.LogitRow(step);
            ReadOnlySpan<float> actual = state.Logits.AsReadOnlySpan<float>();
            for (int i = 0; i < expected.Length; i++)
            {
                float delta = MathF.Abs(expected[i] - actual[i]);
                if (delta > worst) { worst = delta; worstStep = step; }
            }
            // What the drift is worth where it can change the outcome: the two contenders at this step, against
            // the gap between them. A worst-case delta somewhere out in the tail moves no argmax.
            int chosen = reference.Tokens[reference.Prompt.Length + step];
            int runnerUp = reference.RunnerUp(step);
            double share = (MathF.Abs(expected[chosen] - actual[chosen]) + MathF.Abs(expected[runnerUp] - actual[runnerUp]))
                / reference.Margin(step);
            if (share > worstShare) { worstShare = share; worstShareStep = step; }
            decoder.Decode(backend, reference.Tokens.AsSpan(reference.Prompt.Length + step, 1), state);
        }
        _out.WriteLine($"[{backendName}/{caseName}] worst |delta logit| {worst:G4} over {reference.LogitRows} steps (step {worstStep}); "
            + $"drift ate at most {worstShare:P1} of the top-2 margin (step {worstShareStep})");
        Assert.True(worst < LogitTolerance, $"Logit drift {worst:G4} at step {worstStep} passes {LogitTolerance}.");
    }

    /// <summary>Gate S4: the masked argmax at every step, teacher-forced so a disagreement cannot cascade.</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    [Trait("Category", "Integration")]
    public void MaskedArgmax_MatchesTheReferenceStepForStep(string caseName)
    {
        if (!TryLoadCase(caseName, out ReferenceCase reference)) return;

        using SafeTensorsLoader loader = new();
        loader.Load(reference.Checkpoint);
        using IBackend backend = CreateBackend(out string backendName);
        using SheetSage2Decoder decoder = LoadDecoder(loader);
        using Tensor memory = reference.ReadMemory();
        using SheetSage2DecodeState state = decoder.StartDecode(backend, memory, reference.MaxTokens);

        ScoreTokenizer tokenizer = decoder.Tokenizer;
        PromptGrammar grammar = new(tokenizer);
        int outIndex = Array.IndexOf(reference.Prompt, ScoreTokenizer.OutToken);
        for (int i = outIndex + 1; i < reference.Prompt.Length; i++) grammar.Update(reference.Prompt[i]);

        bool[] allowed = new bool[tokenizer.TokenCount];
        List<int> disagreements = [];
        decoder.Decode(backend, reference.Prompt, state);
        for (int step = 0; step < reference.Steps; step++)
        {
            grammar.Allowed(allowed);
            Assert.Equal(reference.AllowedCount(step), allowed.Count(a => a));

            int expected = reference.Tokens[reference.Prompt.Length + step];
            if (Argmax(state.Logits.AsReadOnlySpan<float>(), allowed) != expected) disagreements.Add(step);
            if (grammar.Update(expected)) break;
            decoder.Decode(backend, reference.Tokens.AsSpan(reference.Prompt.Length + step, 1), state);
        }
        _out.WriteLine($"[{backendName}/{caseName}] teacher-forced argmax disagreed at {disagreements.Count}/{reference.Steps} steps; "
            + $"the mask decided {reference.MaskChangedTheArgmax} of them in the reference");
        Assert.Empty(disagreements);
    }

    /// <summary>Gate S5: the free-running greedy decode, token for token.</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    [Trait("Category", "Integration")]
    public void GreedyDecode_MatchesTheReferenceTokenForToken(string caseName)
    {
        if (!TryLoadCase(caseName, out ReferenceCase reference)) return;

        using SafeTensorsLoader loader = new();
        loader.Load(reference.Checkpoint);
        using IBackend backend = CreateBackend(out string backendName);
        using SheetSage2Decoder decoder = LoadDecoder(loader);
        using Tensor memory = reference.ReadMemory();

        List<int> produced = decoder.GenerateTokens(backend, memory, stopSeconds: 300.0,
            prefix: reference.Prompt, maxTokens: reference.MaxTokens);

        int first = FirstDivergence(reference.Tokens, produced);
        if (first >= 0)
        {
            int step = first - reference.Prompt.Length;
            string margin = step >= 0 && step < reference.Steps ? $"{reference.Margin(step):G4}" : "n/a";
            _out.WriteLine($"[{backendName}/{caseName}] diverged at token {first} of {reference.Tokens.Length} "
                + $"(step {step}); the reference's top-2 margin there was {margin}");
        }
        else
        {
            _out.WriteLine($"[{backendName}/{caseName}] token-exact over {produced.Count} tokens; "
                + $"the reference's smallest top-2 margin was {reference.MarginMin:G4}");
        }
        Assert.Equal(reference.Tokens, produced);
    }

    /// <summary>The time stop: a decode ends at the first timestamp at or past the window's stop, which is what
    /// keeps a lookahead window from writing events the next window owns.</summary>
    /// <remarks>EOS is the obvious ending. This rule and the token limit are the two a port drops without a
    /// symptom — every window then runs to its full budget and the stitcher quietly gets events it must discard.</remarks>
    [Theory]
    [MemberData(nameof(Cases))]
    [Trait("Category", "Integration")]
    public void TheTimeStop_CutsTheDecodeWhereTheReferenceWould(string caseName)
    {
        if (!TryLoadCase(caseName, out ReferenceCase reference)) return;

        ScoreTokenizer tokenizer = new();
        // The third timestamp the reference wrote, so there is a decode on either side of the cut.
        double stop = double.NaN;
        int seen = 0;
        foreach (int token in reference.Tokens)
        {
            if (IsTime(tokenizer, token) && ++seen == 3) { stop = tokenizer.TokenToSeconds(token); break; }
        }
        if (double.IsNaN(stop)) return;

        List<int> expected = [];
        foreach (int token in reference.Tokens)
        {
            expected.Add(token);
            if (IsTime(tokenizer, token) && tokenizer.TokenToSeconds(token) >= stop)
            {
                expected.Add(ScoreTokenizer.EosToken);
                break;
            }
        }

        using SafeTensorsLoader loader = new();
        loader.Load(reference.Checkpoint);
        using IBackend backend = CreateBackend(out string backendName);
        using SheetSage2Decoder decoder = LoadDecoder(loader);
        using Tensor memory = reference.ReadMemory();
        List<int> produced = decoder.GenerateTokens(backend, memory, stop, reference.Prompt, reference.MaxTokens);
        _out.WriteLine($"[{backendName}/{caseName}] stopping at {stop:0.00}s cut the decode to {produced.Count} tokens "
            + $"of the reference's {reference.Tokens.Length}");
        Assert.Equal(expected, produced);
    }

    /// <summary>The position table is two rows longer than the token budget because a token at index <c>i</c>
    /// reads row <c>i + 2</c>. A checkpoint that disagrees is refused rather than read off by two.</summary>
    [Fact]
    [Trait("Category", "Integration")]
    public void APositionTableOfTheWrongHeight_IsRefused()
    {
        string? checkpoint = Environment.GetEnvironmentVariable("SHEETSAGE2_CHECKPOINT");
        if (checkpoint is null || !File.Exists(checkpoint)) return;

        using SafeTensorsLoader loader = new();
        loader.Load(checkpoint);
        SheetSage2Config config = SheetSage2Config.Released with { MaxTokens = 4096 };
        using SheetSage2Decoder decoder = new(config, new ScoreTokenizer());
        Assert.Throws<ArgumentException>(() => decoder.LoadWeights(loader.GetAllTensors()));
    }

    private static bool IsTime(ScoreTokenizer tokenizer, int token) => token >= tokenizer.TimeStart && token < tokenizer.TimeEnd;

    private static int Argmax(ReadOnlySpan<float> logits, ReadOnlySpan<bool> allowed)
    {
        int best = -1;
        float bestScore = float.NegativeInfinity;
        for (int i = 0; i < allowed.Length; i++)
        {
            if (allowed[i] && logits[i] > bestScore) { bestScore = logits[i]; best = i; }
        }
        return best;
    }

    private static int FirstDivergence(int[] expected, List<int> actual)
    {
        int shared = Math.Min(expected.Length, actual.Count);
        for (int i = 0; i < shared; i++)
        {
            if (expected[i] != actual[i]) return i;
        }
        return expected.Length == actual.Count ? -1 : shared;
    }

    /// <summary>The checkpoint's tensors borrow the loader's mmap, so they are left to it rather than disposed
    /// here — a weight the decoder did not have to cast is the same object, and freeing it would dangle.</summary>
    private static SheetSage2Decoder LoadDecoder(SafeTensorsLoader loader)
    {
        SheetSage2Decoder decoder = new(SheetSage2Config.Released, new ScoreTokenizer());
        decoder.LoadWeights(loader.GetAllTensors());
        return decoder;
    }

    /// <summary>CUDA when a device and PTX are present, since that is the path that ships and its cross-attention
    /// runs in F16; <c>SHEETSAGE2_FORCE_CPU=1</c> pins the host path.</summary>
    private static IBackend CreateBackend(out string name)
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (Environment.GetEnvironmentVariable("SHEETSAGE2_FORCE_CPU") != "1" && Directory.Exists(ptxDir))
        {
            try
            {
                name = "CUDA";
                return new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);
            }
            catch (Exception)
            {
                // fall through to the host path
            }
        }
        name = "CPU";
        return new CpuBackend();   // tier-lint: guarded
    }

    private static bool TryLoadCase(string caseName, out ReferenceCase reference)
    {
        reference = null!;
        string? checkpoint = Environment.GetEnvironmentVariable("SHEETSAGE2_CHECKPOINT");
        string? referenceDir = Environment.GetEnvironmentVariable("SHEETSAGE2_REF_DIR");
        if (checkpoint is null || !File.Exists(checkpoint) || referenceDir is null) return false;
        string dir = Path.Combine(referenceDir, caseName);
        if (!File.Exists(Path.Combine(dir, "protocol.json"))) return false;
        reference = new ReferenceCase(checkpoint, dir);
        return true;
    }

    /// <summary>One dumped case: its protocol, the memory the decode ran against, and the logits it produced.</summary>
    private sealed class ReferenceCase
    {
        private readonly string _dir;
        private readonly JsonElement _protocol;
        private readonly float[] _logits;

        public string Checkpoint { get; }

        public int[] Prompt { get; }

        public int[] Tokens { get; }

        public int Steps { get; }

        public int LogitRows { get; }

        public int MaxTokens { get; }

        public int MemoryTokens { get; }

        public int MaskChangedTheArgmax { get; }

        public double MarginMin { get; }

        public ReferenceCase(string checkpoint, string dir)
        {
            Checkpoint = checkpoint;
            _dir = dir;
            _protocol = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "protocol.json"))).RootElement.Clone();
            Prompt = Ints(_protocol.GetProperty("promptPrefix"));
            Tokens = Ints(_protocol.GetProperty("tokens"));
            Steps = _protocol.GetProperty("steps").GetInt32();
            LogitRows = _protocol.GetProperty("logitRows").GetInt32();
            MaxTokens = _protocol.GetProperty("maxTokens").GetInt32();
            MemoryTokens = _protocol.GetProperty("memoryTokens").GetInt32();
            MaskChangedTheArgmax = _protocol.GetProperty("maskChangedTheArgmax").GetInt32();
            MarginMin = _protocol.GetProperty("marginMin").GetDouble();
            _logits = ReadFloats(Path.Combine(dir, "logits.bin"));
        }

        public int TokenCount => _protocol.GetProperty("nTokens").GetInt32();

        public int AllowedCount(int step) => _protocol.GetProperty("trace")[step].GetProperty("allowedCount").GetInt32();

        public double Margin(int step) => _protocol.GetProperty("trace")[step].GetProperty("margin").GetDouble();

        /// <summary>The id the reference's masked argmax came second on — the only rival a drift has to beat.</summary>
        public int RunnerUp(int step) => _protocol.GetProperty("trace")[step].GetProperty("top2Id").GetInt32();

        public ReadOnlySpan<float> LogitRow(int step) => _logits.AsSpan(step * TokenCount, TokenCount);

        public Tensor ReadMemory()
        {
            float[] values = ReadFloats(Path.Combine(_dir, "memory.bin"));
            int dim = _protocol.GetProperty("dim").GetInt32();
            Assert.Equal(MemoryTokens * dim, values.Length);
            Tensor memory = new(new TensorShape(1, MemoryTokens, dim), DType.F32);
            values.CopyTo(memory.AsSpan<float>());
            return memory;
        }

        private static float[] ReadFloats(string path)
        {
            byte[] bytes = File.ReadAllBytes(path);
            float[] values = new float[bytes.Length / sizeof(float)];
            Buffer.BlockCopy(bytes, 0, values, 0, values.Length * sizeof(float));
            return values;
        }

        private static int[] Ints(JsonElement element)
        {
            int[] values = new int[element.GetArrayLength()];
            int i = 0;
            foreach (JsonElement entry in element.EnumerateArray()) values[i++] = entry.GetInt32();
            return values;
        }
    }
}
