namespace HartsyInference.ModelAssets.Gguf.KeyMappers;

/// <summary>Every diffusion family whose GGUF tensor names are already the names its checkpoint converter expects.</summary>
/// <remarks><para>A GGUF of a diffusion model is a repack, not a re-export: city96, QuantStack and unsloth quantize the
/// published safetensors file and keep its tensor names. So the "key mapper" every one of these families needs is the
/// identity, and the only thing that actually differs between them is how to recognize a file whose metadata declares
/// no architecture. Fourteen classes said that fourteen times; this says it once and carries the recognition rules as
/// data, so adding a family is a row rather than a file.</para>
/// <para>LLM GGUFs are the opposite case and keep their own classes: llama.cpp re-exports a text model into its own
/// <c>blk.N.attn_q</c> dialect, so those mappers do real rewriting.</para></remarks>
public static class DiffusionGgufFamilies
{
    /// <summary>Whether one tensor name carries some marker.</summary>
    public delegate bool KeyMarker(string tensorName);

    /// <summary>The families, in the order heuristic detection must probe them.</summary>
    /// <remarks>Order is load-bearing wherever one family's key signature is a superset of another's: the more specific
    /// variant has to be asked first, or the broader family claims the file and the model loads as garbage weights.
    /// Each such pairing is called out on the row that depends on it.</remarks>
    public static IReadOnlyList<DiffusionGgufKeyMapper> InDetectionOrder { get; } =
    [
        // Radiance is classic Chroma plus a pixel-space NeRF head, so it matches Chroma's signature too.
        new DiffusionGgufKeyMapper("chroma-radiance", [Has("distilled_guidance_layer."), Has("nerf_blocks.")]),
        // Zeta is the Z-Image S3-DiT retrained for pixel space with a dec_net head; it matches Z-Image's signature too.
        new DiffusionGgufKeyMapper("zeta-chroma", [Has("noise_refiner."), Has("dec_net.")]),
        // Two shipped layouts: a diffusers dump, and a Tencent repack whose double_blocks naming also matches Flux.
        // byt5_in is what separates Hunyuan Image 2.1 from HunyuanVideo, which shares that naming.
        DiffusionGgufKeyMapper.MatchingAny("hunyuan_image",
            [And(Has("transformer_blocks."), Or(Has(".dual_attention"), Has(".moe.")))],
            [And(Has("double_blocks."), Has("img_attn_qkv.")), HasPrefix("byt5_in.")]),
        // Flux.2 keeps Flux's double+single block naming and adds top-level shared modulation linears, so it matches
        // Flux's signature as well; it has to be asked first. (It was registered after Flux before this table, which
        // made a Flux.2 GGUF with no declared architecture detect as Flux.1 — latent only because every shipped Flux.2
        // GGUF does declare one, so the heuristic never ran.) Flux.1 has no double_stream_modulation_* and names its
        // block MLPs .mlp.0/.mlp.2, so this cannot claim a Flux.1 file.
        new DiffusionGgufKeyMapper("flux2",
        [
            Or(Has("double_stream_modulation_img."), Has("double_stream_modulation_txt."), Has("single_stream_modulation.")),
            And(Has("double_blocks."), Has(".mlp.linear_in.")),
        ]),
        // Flux's signature is only double+single blocks, which the whole Chroma family and the Tencent Hunyuan repack
        // satisfy as well — hence all of those above it.
        new DiffusionGgufKeyMapper("flux", [Has("double_blocks."), Has("single_blocks.")]),
        // SDXL and SD1.5 share the LDM input_blocks naming; the class-conditioning label_emb tree is SDXL's alone.
        new DiffusionGgufKeyMapper("sdxl", [Has("input_blocks."), Has("label_emb.")]),
        new DiffusionGgufKeyMapper("sd3", [Or(Has("joint_blocks."), And(Has("x_block."), Has("attn.")))]),
        new DiffusionGgufKeyMapper("sd15", [Has("input_blocks.")], [Has("label_emb.")]),
        new DiffusionGgufKeyMapper("flite", [Is("register_tokens"), And(Has("blocks."), Has(".self_attn."))]),
        // Wan's whole family — T2V/I2V/TI2V and the VACE, Animate and S2V conditioning builds, which all keep the
        // backbone's naming. Asked after F-Lite, the only other row whose blocks carry a .self_attn.: F-Lite is
        // claimed first by its register_tokens, and cannot reach here anyway since it has no cross-attention and
        // spells its patch embed patch_embed. (no -ing). The dots in .self_attn. / .cross_attn. are load-bearing in
        // the other direction — HunyuanVideo's token refiner has blocks.N.self_attn_qkv, which they exclude.
        // vace_/pose_patch_embedding contain patch_embedding., which is fine: those files are Wan too.
        DiffusionGgufKeyMapper.MatchingAny("wan",
            [And(Has("blocks."), Has(".self_attn.")), And(Has("blocks."), Has(".cross_attn.")), Has("patch_embedding.")],
            [Has("patch_embedding."), Has("condition_embedder.time_embedder.")]),
        new DiffusionGgufKeyMapper("chroma",
            [Has("distilled_guidance_layer."), Or(Has("double_blocks."), Has("single_blocks."))]),
        new DiffusionGgufKeyMapper("auraflow",
            [Or(Has("double_layers."), Has("modF."), Has("cond_seq_linear."))]),
        new DiffusionGgufKeyMapper("zimage", [Has("noise_refiner."), Has("context_refiner.")]),
        // ERNIE-Image's giveaway is its shared AdaLN modulation: one top-level linear broadcast across all 36 layers.
        new DiffusionGgufKeyMapper("ernie_image",
            [Has("shared_adaLN_modulation."), And(Has("transformer_blocks."), Has(".mlp.gate_proj."))]),
        new DiffusionGgufKeyMapper("qwen_image", [And(Has("transformer_blocks."), Has(".attn.add_q_proj."))]),
    ];

    /// <summary>A name containing <paramref name="fragment"/>.</summary>
    public static KeyMarker Has(string fragment) => name => name.Contains(fragment, StringComparison.Ordinal);

    /// <summary>A name starting with <paramref name="prefix"/>.</summary>
    public static KeyMarker HasPrefix(string prefix) => name => name.StartsWith(prefix, StringComparison.Ordinal);

    /// <summary>A name equal to <paramref name="exact"/>.</summary>
    public static KeyMarker Is(string exact) => name => name.Equals(exact, StringComparison.Ordinal);

    /// <summary>One name carrying all of <paramref name="parts"/> — distinct from listing them separately in a signature, which any mix of names may satisfy between them.</summary>
    public static KeyMarker And(params KeyMarker[] parts) => name =>
    {
        foreach (KeyMarker part in parts)
        {
            if (!part(name)) return false;
        }
        return true;
    };

    /// <summary>A name carrying any of <paramref name="parts"/>.</summary>
    public static KeyMarker Or(params KeyMarker[] parts) => name =>
    {
        foreach (KeyMarker part in parts)
        {
            if (part(name)) return true;
        }
        return false;
    };
}

/// <summary>One family's key signature: markers that must each appear on some tensor name, and markers that must appear on none.</summary>
/// <remarks>The two halves are not symmetric. A required marker is satisfied by any name in the file, so listing two of
/// them does not mean one name carries both — use <see cref="DiffusionGgufFamilies.And"/> for that. A forbidden marker
/// is how a family excludes a relative whose signature is otherwise a superset of its own, and it is the reason a match
/// cannot be declared before every name has been read.</remarks>
public sealed record GgufKeySignature(
    IReadOnlyList<DiffusionGgufFamilies.KeyMarker> Required,
    IReadOnlyList<DiffusionGgufFamilies.KeyMarker>? Forbidden = null)
{
    /// <summary>Whether <paramref name="tensorNames"/> satisfies this signature.</summary>
    public bool Matches(IEnumerable<string> tensorNames)
    {
        bool[] seen = new bool[Required.Count];
        int found = 0;
        foreach (string name in tensorNames)
        {
            if (Forbidden is not null)
            {
                foreach (DiffusionGgufFamilies.KeyMarker marker in Forbidden)
                {
                    if (marker(name)) return false;
                }
            }
            for (int i = 0; i < Required.Count; i++)
            {
                if (seen[i] || !Required[i](name)) continue;
                seen[i] = true;
                found++;
            }
            if (found == Required.Count && Forbidden is null) return true;
        }
        return found == Required.Count;
    }
}

/// <summary>Recognizes one diffusion family's GGUF and passes its tensor names through unchanged.</summary>
/// <param name="architecture">The <c>general.architecture</c> value this family's files declare.</param>
/// <param name="required">Markers each carried by some tensor name.</param>
/// <param name="forbidden">Markers carried by none.</param>
public sealed class DiffusionGgufKeyMapper(
    string architecture,
    IReadOnlyList<DiffusionGgufFamilies.KeyMarker> required,
    IReadOnlyList<DiffusionGgufFamilies.KeyMarker>? forbidden = null) : IGgufKeyMapper
{
    private readonly IReadOnlyList<GgufKeySignature> _signatures = [new GgufKeySignature(required, forbidden)];

    private DiffusionGgufKeyMapper(string architecture, IReadOnlyList<GgufKeySignature> signatures)
        : this(architecture, signatures[0].Required, signatures[0].Forbidden)
    {
        _signatures = signatures;
    }

    /// <summary>A family recognized by any one of several signatures, for the architectures that ship under more than one layout.</summary>
    public static DiffusionGgufKeyMapper MatchingAny(string architecture,
        params IReadOnlyList<DiffusionGgufFamilies.KeyMarker>[] signatures) =>
        new(architecture, signatures.Select(markers => new GgufKeySignature(markers)).ToArray());

    public string Architecture => architecture;

    public bool MatchesByKeys(IEnumerable<string> tensorNames)
    {
        ArgumentNullException.ThrowIfNull(tensorNames);
        foreach (GgufKeySignature signature in _signatures)
        {
            if (signature.Matches(tensorNames)) return true;
        }
        return false;
    }

    /// <summary>The identity: a diffusion GGUF is a repack of the safetensors build and keeps its tensor names.</summary>
    public string? MapKey(string ggufKey) => ggufKey;
}
