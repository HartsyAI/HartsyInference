namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>YaRN rope extension parameters from the config's <c>rope_scaling</c> block.</summary>
public sealed record DeepSeekV41RopeScaling(double Factor, double BetaFast, double BetaSlow, int OriginalMaxPositionEmbeddings);
