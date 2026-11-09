namespace HartsyInference.LLM.Transformer;

/// <summary>Checkpoint key names a <see cref="MoeFeedForward"/> block reads, relative to its layer prefix. The defaults are the engine
/// names every existing mapper produces; a checkpoint loaded without a key remap supplies its own set.</summary>
public sealed record MoeTensorNames
{
    /// <summary>The engine names (<c>mlp.gate.weight</c>, <c>mlp.experts.N.gate_proj.weight</c>, ...).</summary>
    public static MoeTensorNames Default { get; } = new();

    /// <summary>ERNIE-5.5's own names (<c>block_sparse_moe.*</c>, <c>w1</c>/<c>w3</c>/<c>w2</c> experts), checked against its weight index.</summary>
    public static MoeTensorNames Ernie55 { get; } = new()
    {
        Block = "block_sparse_moe",
        CorrectionBias = "e_score_correction_bias",
        GateProj = "w1",
        UpProj = "w3",
        DownProj = "w2",
        SharedExpert = "shared_experts",
        LatentDown = "fc1_latent_proj.weight",
        LatentNorm = "latent_norm.weight",
        LatentUp = "fc2_latent_proj.weight",
    };

    /// <summary>Module under the layer prefix that holds the whole block.</summary>
    public string Block { get; init; } = "mlp";

    /// <summary>Router matrix <c>[E, hidden]</c>.</summary>
    public string RouterWeight { get; init; } = "gate.weight";

    /// <summary>Second router matrix of <see cref="MoeConfig.SplitFeatureRouting"/>.</summary>
    public string RouterWeight2 { get; init; } = "gate.weight_1";

    /// <summary>Selection-only <c>e_score_correction_bias</c> <c>[E]</c>.</summary>
    public string CorrectionBias { get; init; } = "gate.e_score_correction_bias";

    /// <summary>Per-expert routed scale <c>[E]</c> of <see cref="MoeConfig.LearnedRoutedScale"/>.</summary>
    public string RoutedScale { get; init; } = "gate.routed_scaling_factor_param";

    /// <summary>Module holding the routed experts, indexed <c>.N.</c> below it.</summary>
    public string Experts { get; init; } = "experts";

    /// <summary>Gate projection of a routed or shared expert, without the <c>.weight</c> suffix.</summary>
    public string GateProj { get; init; } = "gate_proj";

    /// <summary>Up projection of a routed or shared expert, without the <c>.weight</c> suffix.</summary>
    public string UpProj { get; init; } = "up_proj";

    /// <summary>Down projection of a routed or shared expert, without the <c>.weight</c> suffix.</summary>
    public string DownProj { get; init; } = "down_proj";

    /// <summary>Module holding the always-on shared expert.</summary>
    public string SharedExpert { get; init; } = "shared_expert";

    /// <summary>Optional sigmoid gate <c>[1, hidden]</c> on the shared expert.</summary>
    public string SharedExpertGate { get; init; } = "shared_expert_gate.weight";

    /// <summary>Latent entry projection <c>[latent, hidden]</c>.</summary>
    public string LatentDown { get; init; } = "pre_expert_proj.weight";

    /// <summary>Latent RMSNorm weight <c>[latent]</c>.</summary>
    public string LatentNorm { get; init; } = "pre_expert_norm.weight";

    /// <summary>Latent exit projection <c>[hidden, latent]</c>.</summary>
    public string LatentUp { get; init; } = "post_expert_proj.weight";

    /// <summary>Per-expert output norm weight of <see cref="MoeConfig.PostExpertNorm"/>.</summary>
    public string PostExpertNorm { get; init; } = "post_expert_norm.weight";
}
