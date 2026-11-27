namespace HartsyInference.ThreeD.Models.HairDeck;

/// <summary>The files of one HairDeck model set, as the engine resolves them. Optional components are null when absent.</summary>
/// <param name="ImageModel">Image DiT safetensors (<c>ckpt/image_model/model.safetensors</c>).</param>
/// <param name="CardVae">Card VAE safetensors (<c>ckpt/card_vae/model.safetensors</c>).</param>
/// <param name="CardVaeStats">Card VAE output statistics (<c>ckpt/card_vae/stats.json</c>).</param>
/// <param name="TokenStats">Head-frame root statistics (<c>data/token_stats_head.json</c>).</param>
/// <param name="LatentStats">Head-frame latent statistics (<c>ckpt/card_vae/latent_stats_head.json</c>).</param>
/// <param name="ColorModel">Colour model safetensors (<c>ckpt/color_model/model.safetensors</c>), or null for one hair colour.</param>
/// <param name="ChainModel">Chain DiT safetensors (<c>ckpt/chain_model/model.safetensors</c>), or null for no chains.</param>
/// <param name="ChainStats">Chain statistics (<c>data/spring/chain_stats_head.json</c>); required with <paramref name="ChainModel"/>.</param>
public sealed record HairDeckModelFiles(
    string ImageModel,
    string CardVae,
    string CardVaeStats,
    string TokenStats,
    string LatentStats,
    string? ColorModel = null,
    string? ChainModel = null,
    string? ChainStats = null);
