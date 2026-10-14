using System.Text.Json;
using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Sampling;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary><see cref="BeamSearch"/> against Hugging Face transformers 4.52.1 <c>generate()</c> on a toy model whose
/// logits are a fixed function of the sequence (<c>Fixtures/beam_search_reference.json</c>, from
/// tools/parity/beam_search_reference.py): greedy beam search must return HF's tokens exactly, with repetition
/// penalty, length penalty, min_new_tokens and the length cap. Beam sampling is checked for determinism and validity.</summary>
public sealed class BeamSearchTests
{
    private const int Vocab = 40;
    private const int Eos = 39;
    private static readonly int[] Prompt = [1, 1, 1, 38];

    private sealed class ToyModel(int beams) : IBeamSearchModel
    {
        private List<int>[] _sequences = [.. Enumerable.Range(0, beams).Select(_ => Prompt[..^1].ToList())];

        public int Steps { get; private set; }

        public void Step(int beam, int token, int step, Span<float> logits)
        {
            Assert.Equal(Prompt.Length - 1 + step, _sequences[beam].Count);
            _sequences[beam].Add(token);
            Steps++;
            List<int> seq = _sequences[beam];
            int last = seq[^1], n = seq.Count;
            for (int v = 0; v < Vocab; v++)
            {
                double value = ((v * 7 + last * 13 + n * 5) % 31) * 0.25 + v * 0.001;
                if (v == Eos) value += (n - 12) * 0.6;
                logits[v] = (float)value;
            }
        }

        public void Reorder(ReadOnlySpan<int> parents)
        {
            List<int>[] next = new List<int>[parents.Length];
            for (int b = 0; b < parents.Length; b++) next[b] = [.. _sequences[parents[b]]];
            _sequences = next;
        }
    }

    [Fact]
    public void GreedyBeamSearch_MatchesHuggingFace()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "beam_search_reference.json");
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(path));
        foreach (JsonElement c in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            int beams = c.GetProperty("num_beams").GetInt32();
            BeamSearchSettings settings = new()
            {
                NumBeams = beams,
                MaxNewTokens = c.GetProperty("max_new_tokens").GetInt32(),
                EosToken = Eos,
                RepetitionPenalty = c.GetProperty("repetition_penalty").GetSingle(),
                LengthPenalty = c.GetProperty("length_penalty").GetSingle(),
                MinNewTokens = c.GetProperty("min_new_tokens").GetInt32(),
            };
            uint rng = 1;
            int[] actual = BeamSearch.Generate(new ToyModel(beams), settings, Prompt, Prompt[^1], Vocab, ref rng);
            int[] expected = [.. c.GetProperty("tokens").EnumerateArray().Select(static e => e.GetInt32())];
            Assert.True(expected.SequenceEqual(actual),
                $"{c.GetProperty("name").GetString()}: expected {string.Join(',', expected)}, got {string.Join(',', actual)}");
        }
    }

    [Fact]
    public void BeamSampling_IsDeterministicPerSeed_AndStaysInVocab()
    {
        BeamSearchSettings settings = new()
        {
            NumBeams = 3, MaxNewTokens = 30, EosToken = Eos, DoSample = true, Temperature = 0.8f, TopK = 5, TopP = 0.8f,
            RepetitionPenalty = 10f, LengthPenalty = 0f,
        };
        uint a = DeterministicRng.Seed(7), b = DeterministicRng.Seed(7);
        int[] first = BeamSearch.Generate(new ToyModel(3), settings, Prompt, Prompt[^1], Vocab, ref a);
        int[] second = BeamSearch.Generate(new ToyModel(3), settings, Prompt, Prompt[^1], Vocab, ref b);
        Assert.Equal(first, second);
        Assert.NotEmpty(first);
        Assert.All(first, t => Assert.InRange(t, 0, Vocab - 1));
        Assert.True(first[^1] == Eos || first.Length == 30);
    }

    [Fact]
    public void StableTopIndices_IsDescending_WithLowerIndexFirstOnTies()
    {
        int[] top = new int[3];
        BeamSearch.StableTopIndices([1f, 3f, 3f, float.NegativeInfinity, 2f], top);
        Assert.Equal([1, 2, 4], top);
    }
}
