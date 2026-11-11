namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>One modality of the structure flow: its name in the latent dictionary and its channel count (<c>latent_mapping</c> of <c>ss_generator.yaml</c>).</summary>
public sealed record SamMotLatent(string Name, int Channels);
