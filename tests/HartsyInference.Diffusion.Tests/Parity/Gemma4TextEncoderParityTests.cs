using System.Text.Json;
using System.Text.Json.Serialization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Per-hidden-state parity gate for <see cref="Gemma4TextEncoder"/> against ComfyUI's
/// <c>comfy/text_encoders/gemma4.py</c> on a tiny random-weight config that keeps Gemma 4's real per-layer-type
/// geometry (5 sliding + 1 global cycle, different head dims, GQA, <c>k_eq_v</c> and partial rotary on the global
/// layers). A sliding layer and a global layer are asserted separately because they exercise disjoint code paths.
/// Two cases cover both sides of the sliding-window threshold. Skips cleanly when the reference dump is missing
/// (its <c>.bin</c> files are gitignored); regenerate with
/// <c>tests/python-reference/dump_gemma4_tiny_reference.py</c>.</summary>
[Trait("Category", "Integration")]
public sealed unsafe class Gemma4TextEncoderParityTests
{
    private const float RelL2Tolerance = 1e-5f;
    private const int SlidingStateIndex = 1;    // output of block 0 (sliding)
    private const int GlobalStateIndex = 6;     // output of block 5 (global)

    private static string ReferenceDir =>
        Environment.GetEnvironmentVariable("GEMMA4_TINY_REFERENCE_DIR")
        ?? Path.Combine(RepoRoot.Path, "tests", "python-reference", "gemma4_tiny_reference");

    private readonly ITestOutputHelper _output;
    public Gemma4TextEncoderParityTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Gemma4_TinyConfig_MatchesReference_SequenceWithinSlidingWindow() => Run("within_window");

    [Fact]
    public void Gemma4_TinyConfig_MatchesReference_SequenceBeyondSlidingWindow() => Run("beyond_window");

    private void Run(string caseName)
    {
        string caseDir = Path.Combine(ReferenceDir, caseName);
        string metaPath = Path.Combine(caseDir, "meta.json");
        if (!File.Exists(metaPath))
        {
            _output.WriteLine($"SKIPPED: Gemma 4 reference case not found at {caseDir}.");
            _output.WriteLine("Generate it with ComfyUI's venv python: tests/python-reference/dump_gemma4_tiny_reference.py");
            return;
        }

        Meta meta = JsonSerializer.Deserialize<Meta>(File.ReadAllText(metaPath))
            ?? throw new InvalidDataException("Gemma 4 reference meta.json malformed.");

        Gemma4TextEncoderConfig config = new()
        {
            HiddenSize = meta.HiddenSize,
            NumLayers = meta.NumHiddenLayers,
            NumQueryHeads = meta.NumAttentionHeads,
            NumKvHeads = meta.NumKeyValueHeads,
            NumGlobalKvHeads = meta.NumGlobalKeyValueHeads,
            HeadDim = meta.HeadDim,
            GlobalHeadDim = meta.GlobalHeadDim,
            IntermediateSize = meta.IntermediateSize,
            VocabSize = meta.VocabSize,
            RmsNormEps = meta.RmsNormEps,
            SlidingRopeTheta = meta.SlidingRopeTheta,
            GlobalRopeTheta = meta.GlobalRopeTheta,
            PartialRotaryFactor = meta.PartialRotaryFactor,
            SlidingWindow = meta.SlidingWindow,
            MaxPositionEmbeddings = 4096,
        };

        List<Tensor> owned = [];
        Dictionary<string, Tensor> weights = new();
        try
        {
            foreach (KeyValuePair<string, List<long>> entry in meta.WeightShapes)
            {
                Tensor tensor = ReadTensor(Path.Combine(caseDir, "weights", entry.Key + ".bin"), entry.Value);
                owned.Add(tensor);
                weights[entry.Key] = tensor;
            }

            int seqLen = meta.SeqLen;
            int[] tokenIds = ReadInt32(Path.Combine(caseDir, "input_ids.bin"), seqLen);
            int stateCount = config.NumLayers + 1;
            float[] expected = ReadFloats(Path.Combine(caseDir, "hidden_states.bin"),
                (long)stateCount * seqLen * config.HiddenSize);

            using IBackend backend = new CpuBackend();
            using Gemma4TextEncoder encoder = new(config);
            encoder.LoadWeights(weights);

            int[] layerIndices = new int[stateCount];
            for (int i = 0; i < stateCount; i++) layerIndices[i] = i;
            using Tensor actual = encoder.EncodeMultiLayer(backend, [tokenIds], layerIndices);
            Assert.Equal(new TensorShape(1, seqLen, stateCount * config.HiddenSize), actual.Shape);

            float worst = 0f;
            int worstState = -1;
            float[] perState = new float[stateCount];
            float* actualPtr = (float*)actual.DataPointer;
            for (int state = 0; state < stateCount; state++)
            {
                double diffSq = 0.0, refSq = 0.0;
                for (int s = 0; s < seqLen; s++)
                {
                    for (int c = 0; c < config.HiddenSize; c++)
                    {
                        float reference = expected[((long)state * seqLen + s) * config.HiddenSize + c];
                        float mine = actualPtr[((long)s * stateCount + state) * config.HiddenSize + c];
                        double delta = mine - reference;
                        diffSq += delta * delta;
                        refSq += (double)reference * reference;
                    }
                }
                perState[state] = (float)Math.Sqrt(diffSq / Math.Max(refSq, 1e-30));
                if (perState[state] > worst) { worst = perState[state]; worstState = state; }
                _output.WriteLine($"[{caseName}] state {state,2}: relL2 = {perState[state]:E3}");
            }
            _output.WriteLine($"[{caseName}] worst relL2 {worst:E3} at state {worstState}");

            Assert.True(perState[SlidingStateIndex] < RelL2Tolerance,
                $"sliding layer state {SlidingStateIndex} relL2 {perState[SlidingStateIndex]:E3} exceeds {RelL2Tolerance:E1}");
            Assert.True(perState[GlobalStateIndex] < RelL2Tolerance,
                $"global layer state {GlobalStateIndex} relL2 {perState[GlobalStateIndex]:E3} exceeds {RelL2Tolerance:E1}");
            Assert.True(worst < RelL2Tolerance,
                $"state {worstState} relL2 {worst:E3} exceeds {RelL2Tolerance:E1}");

            // The single-state convenience must agree with the packed path's last state.
            using Tensor last = encoder.Encode(backend, [tokenIds]);
            Assert.Equal(new TensorShape(1, seqLen, config.HiddenSize), last.Shape);
            float* lastPtr = (float*)last.DataPointer;
            for (long i = 0; i < last.ElementCount; i++)
            {
                long s = i / config.HiddenSize;
                long c = i % config.HiddenSize;
                Assert.Equal(actualPtr[(s * stateCount + (stateCount - 1)) * config.HiddenSize + c], lastPtr[i], 4);
            }
        }
        finally
        {
            foreach (Tensor tensor in owned) tensor.Dispose();
        }
    }

    private static Tensor ReadTensor(string path, List<long> shape)
    {
        long count = 1;
        foreach (long dim in shape) count *= dim;
        Tensor tensor = new(new TensorShape([.. shape]), DType.F32);
        float[] values = ReadFloats(path, count);
        float* p = (float*)tensor.DataPointer;
        for (long i = 0; i < count; i++) p[i] = values[i];
        return tensor;
    }

    private static float[] ReadFloats(string path, long count)
    {
        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.Length != count * sizeof(float))
            throw new InvalidDataException($"{path}: expected {count * sizeof(float)} bytes, got {bytes.Length}.");
        float[] values = new float[count];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    private static int[] ReadInt32(string path, int count)
    {
        byte[] bytes = File.ReadAllBytes(path);
        if (bytes.Length != count * sizeof(int))
            throw new InvalidDataException($"{path}: expected {count * sizeof(int)} bytes, got {bytes.Length}.");
        int[] values = new int[count];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    private sealed record Meta
    {
        [JsonPropertyName("vocab_size")] public int VocabSize { get; init; }
        [JsonPropertyName("hidden_size")] public int HiddenSize { get; init; }
        [JsonPropertyName("intermediate_size")] public int IntermediateSize { get; init; }
        [JsonPropertyName("num_hidden_layers")] public int NumHiddenLayers { get; init; }
        [JsonPropertyName("num_attention_heads")] public int NumAttentionHeads { get; init; }
        [JsonPropertyName("num_key_value_heads")] public int NumKeyValueHeads { get; init; }
        [JsonPropertyName("num_global_key_value_heads")] public int NumGlobalKeyValueHeads { get; init; }
        [JsonPropertyName("head_dim")] public int HeadDim { get; init; }
        [JsonPropertyName("global_head_dim")] public int GlobalHeadDim { get; init; }
        [JsonPropertyName("sliding_window")] public int SlidingWindow { get; init; }
        [JsonPropertyName("partial_rotary_factor")] public float PartialRotaryFactor { get; init; }
        [JsonPropertyName("rms_norm_eps")] public float RmsNormEps { get; init; }
        [JsonPropertyName("global_rope_theta")] public float GlobalRopeTheta { get; init; }
        [JsonPropertyName("sliding_rope_theta")] public float SlidingRopeTheta { get; init; }
        [JsonPropertyName("seq_len")] public int SeqLen { get; init; }
        [JsonPropertyName("weight_shapes")] public Dictionary<string, List<long>> WeightShapes { get; init; } = [];
    }
}
