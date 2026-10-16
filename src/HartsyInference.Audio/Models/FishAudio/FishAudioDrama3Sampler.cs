using HartsyInference.Audio.Dsp;

namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>Drama 3's per-request sampling state (fish-speech <c>decode_step</c>): the main token is drawn over eos plus the semantic
/// band, and when that draw repeats a semantic token among the latest <see cref="FishAudioDrama3Config.RasWindow"/> main tokens it is
/// redrawn with the fixed repetition-aware parameters; each codebook code is then drawn from its head. Every draw tempers first and
/// filters the tempered distribution (<see cref="FishAudioSampling"/>).</summary>
public sealed class FishAudioDrama3Sampler
{
    private readonly FishAudioDrama3Config _config;
    private readonly FishAudioSampling.Scratch _scratch;
    private readonly int[] _window = new int[FishAudioDrama3Config.RasWindow];
    private int _windowCount, _windowNext;
    private uint _rng;

    /// <summary>Creates the sampler; null parameters take the model defaults.</summary>
    public FishAudioDrama3Sampler(FishAudioDrama3Config config, int seed, float? temperature = null, float? topP = null, int? topK = null)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        Temperature = temperature ?? config.Temperature;
        TopP = topP ?? config.TopP;
        TopK = topK ?? config.TopK;
        if (TopK < 1) throw new ArgumentOutOfRangeException(nameof(topK), "Drama 3 top-k must be at least 1.");
        if (TopP <= 0 || TopP > 1) throw new ArgumentOutOfRangeException(nameof(topP), "Drama 3 top-p must lie in (0, 1].");
        _scratch = new FishAudioSampling.Scratch(Math.Max(config.CodebookSize + 1, 1));
        _rng = DeterministicRng.Seed(seed);
    }

    public float Temperature { get; }

    public float TopP { get; }

    public int TopK { get; }

    /// <summary>Starts the repetition window from the main tokens of the last prompt rows (upstream's window holds them too).</summary>
    public void Prime(ReadOnlySpan<int> promptMainTokens)
    {
        _windowCount = _windowNext = 0;
        int start = Math.Max(0, promptMainTokens.Length - _window.Length);
        foreach (int token in promptMainTokens[start..]) Remember(token);
    }

    /// <summary>Draws the main token from <paramref name="candidateLogits"/>, the logits of <paramref name="candidates"/> in order,
    /// and records it in the repetition window.</summary>
    public int SampleMain(ReadOnlySpan<float> candidateLogits, ReadOnlySpan<int> candidates)
    {
        int main = candidates[FishAudioSampling.Sample(candidateLogits, Temperature, TopP, TopK, ref _rng, _scratch, temperatureFirst: true)];
        if (_config.IsSemantic(main) && InWindow(main))
        {
            main = candidates[FishAudioSampling.Sample(candidateLogits, _config.RasTemperature, _config.RasTopP, _config.RasTopK, ref _rng,
                _scratch, temperatureFirst: true)];
        }
        Remember(main);
        return main;
    }

    /// <summary>Draws one codebook code from its head's logits.</summary>
    public int SampleCode(ReadOnlySpan<float> logits) =>
        FishAudioSampling.Sample(logits, Temperature, TopP, TopK, ref _rng, _scratch, temperatureFirst: true);

    private bool InWindow(int token)
    {
        for (int i = 0; i < _windowCount; i++)
        {
            if (_window[i] == token) return true;
        }
        return false;
    }

    private void Remember(int token)
    {
        _window[_windowNext] = token;
        _windowNext = (_windowNext + 1) % _window.Length;
        if (_windowCount < _window.Length) _windowCount++;
    }
}
