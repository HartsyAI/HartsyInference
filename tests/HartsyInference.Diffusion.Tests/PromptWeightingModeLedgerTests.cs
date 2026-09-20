using HartsyInference.Diffusion.Prompting;
using HartsyInference.Engine.Recipes;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pins which of SwarmUI's two prompt-weighting mechanisms each family must use, read out of ComfyUI's own
/// tokenizers rather than inferred. SwarmUI decides with a single runtime probe —
/// <c>use_attn_token_weights = not token_batches_have_weights(clip.tokenize("(x:2)"))</c>
/// (<c>SwarmText.py:553</c>, helper at <c>:216-225</c>) — so a family is <see cref="PromptWeightingMode.CondScale"/>
/// only when EVERY tokenizer arm passes ComfyUI's <c>disable_weights=True</c>; a single weight-keeping arm (Kandinsky5's
/// CLIP-L, HiDream's CLIP-L/G) puts the whole family back on <see cref="PromptWeightingMode.ComfyBlend"/>.
/// <para>Each entry carries the evidence chain as a comment: the ComfyUI config that names the tokenizer, the tokenizer
/// class, and the line that does (or does not) disable weights. Paths are relative to ComfyUI's <c>comfy/</c>, except
/// <c>WorkflowGenerator.cs</c>/<c>SwarmText.py</c> which are SwarmUI's. Once an entry is here a wrong mode is permanent,
/// so families with no ComfyUI support at all live in <see cref="Unresolved"/> and are NOT guessed.</para></summary>
public sealed class PromptWeightingModeLedgerTests
{
    /// <summary>The verified family → mechanism table. Keys are engine family ids (<c>IArchitectureRecipe.Name</c> /
    /// <c>IVideoRecipe.Name</c>), not SwarmUI compat-class ids.</summary>
    private static readonly Dictionary<string, PromptWeightingMode> Ledger = new(StringComparer.Ordinal)
    {
        // --- Image families ---

        // supported_models.py:94 -> sd1_clip.SD1Tokenizer; SDTokenizer's disable_weights defaults false (sd1_clip.py:487).
        ["sd15"] = PromptWeightingMode.ComfyBlend,
        // supported_models.py:273 -> sdxl_clip.SDXLTokenizer (sdxl_clip.py:24): clip_l + clip_g, neither disables weights.
        ["sdxl"] = PromptWeightingMode.ComfyBlend,
        // supported_models.py:201 -> the same sdxl_clip.SDXLTokenizer; only the model half differs on the refiner.
        ["sdxl-refiner"] = PromptWeightingMode.ComfyBlend,
        // supported_models.py:584 -> sd3_clip.SD3Tokenizer (sd3_clip.py:41-45): clip_l + clip_g + t5xxl, none disable.
        ["sd3"] = PromptWeightingMode.ComfyBlend,
        // supported_models.py:661 -> aura_t5.AuraT5Tokenizer (aura_t5.py:16) -> PT5XlTokenizer (:11-14), Pile-T5, no disable.
        ["auraflow"] = PromptWeightingMode.ComfyBlend,
        // supported_models.py:771 -> flux.FluxTokenizer (flux.py:17) = clip_l + T5XXLTokenizer (flux.py:11-14), no disable.
        // Kontext/Fill/Canny/Depth/Redux are the same Flux config (FluxInpaint :773 inherits clip_target), same answer.
        ["flux1"] = PromptWeightingMode.ComfyBlend,
        // supported_models.py:827/832/838 -> KleinTokenizer (flux.py:169), KleinTokenizer8B (:172), Flux2Tokenizer (:104);
        // the Mistral arm also disables in its ctor (flux.py:88). All three Flux.2 text stacks discard weights.
        ["flux2"] = PromptWeightingMode.CondScale,
        // supported_models.py:1797 -> pixart_t5.PixArtTokenizer (pixart_t5.py:29) -> T5XXLTokenizer (:24-27), no disable.
        ["chroma"] = PromptWeightingMode.ComfyBlend,
        // supported_models.py:1833 ChromaRadiance(Chroma) adds no clip_target, so it inherits Chroma's T5 stack (:1794-1797).
        ["chroma-radiance"] = PromptWeightingMode.ComfyBlend,
        // Zeta-Chroma is the Z-Image S3-DiT retrained for pixel space: its dec_net.* head makes ComfyUI stamp
        // image_model="zimage_pixel" (model_detection.py:618-620) -> ZImagePixelSpace(ZImage) (supported_models.py:1230),
        // which inherits ZImage.clip_target (:1225-1228) -> z_image.ZImageTokenizer -> disable_weights (z_image.py:23).
        ["zeta-chroma"] = PromptWeightingMode.CondScale,
        // supported_models.py:1228 -> z_image.ZImageTokenizer (z_image.py:12) -> disable_weights at z_image.py:23.
        ["zimage"] = PromptWeightingMode.CondScale,
        // supported_models.py:1202 -> lumina2.LuminaTokenizer (lumina2.py:22-24) -> Gemma2BTokenizer (:7-11), which does
        // NOT disable weights; sd.py:1843-1850 routes a Gemma-2-2B TE here. lumina2.py:20's disable_weights belongs to
        // Gemma3_4BTokenizer, reached only through NTokenizer (:26-28, sd.py:1851-1858) — a different model's TE.
        // Lumina2Recipe.cs:54 pins SideModels.Gemma2_2B and :127 rejects the Gemma-3 tokenizer by hash, so only this arm applies.
        ["lumina2"] = PromptWeightingMode.ComfyBlend,
        // supported_models.py:2052 -> qwen_image.QwenImageTokenizer (:14) -> disable_weights at qwen_image.py:39.
        // Qwen-Image Edit rides the same tokenizer (the llama_template_images branch, :18/:20-38), so same mode.
        ["qwen-image"] = PromptWeightingMode.CondScale,
        // supported_models.py:2081 -> qwen_image21.QwenImage21Tokenizer (:23), a Qwen3VLTokenizer subclass whose
        // tokenize_with_weights (qwen3vl.py:186) passes disable_weights=True. One arm, and it disables, so the
        // probe selects CondScale.
        ["qwen-image-2.1"] = PromptWeightingMode.CondScale,
        // supported_models.py:2109 -> hunyuan_image.HunyuanImageTokenizer (:13), a QwenImageTokenizer subclass, so the
        // Qwen arm disables (qwen_image.py:39). Its byt5 arm (:8-11) keeps weights but is only populated for QUOTED text
        // (:24-38), and SwarmUI's discriminator probes the unquoted literal "(x:2)" — byt5 never enters the probe.
        ["hunyuan-image"] = PromptWeightingMode.CondScale,
        // supported_models.py:1906 -> omnigen2.Omnigen2Tokenizer (:13) -> Qwen25_3BTokenizer (:7-10); :18-23 wraps the
        // llama template WITHOUT passing disable_weights, so weights survive.
        ["omnigen2"] = PromptWeightingMode.ComfyBlend,
        // supported_models.py:1927 -> boogu.BooguTokenizer (:20), a qwen3vl.Qwen3VLTokenizer subclass -> qwen3vl.py:187.
        ["boogu"] = PromptWeightingMode.CondScale,
        // supported_models.py:1994 -> krea2.Krea2Tokenizer (:23), a Qwen3VLTokenizer subclass -> qwen3vl.py:187. The extra
        // attention patch is SwarmUI's, not ComfyUI's: WorkflowGenerator.cs:965-972 inserts SwarmAttnTokenWeights for
        // IsKrea2() && ModelSpecificEnhancements only, implemented at SwarmText.py:281-312.
        ["krea2"] = PromptWeightingMode.CondScaleWithAttention,
        // supported_models.py:2023 -> mage_flow.MageFlowTokenizer (:23), a Qwen3VLTokenizer subclass -> qwen3vl.py:187.
        ["mage-flow"] = PromptWeightingMode.CondScale,
        // HiDream's own clip_target returns None (supported_models.py:1679, "TODO"); the TE is loaded through
        // sd.py:1995 -> hidream.HiDreamTokenizer (hidream.py:10-15). clip_l/clip_g keep weights, so the probe sees them
        // even though the llama arm (hunyuan_video.LLAMA3Tokenizer) and t5 arm are along for the ride.
        ["hidream"] = PromptWeightingMode.ComfyBlend,
        // supported_models.py:1965 -> ideogram4.Ideogram4Tokenizer (:28) -> disable_weights at ideogram4.py:42.
        ["ideogram4"] = PromptWeightingMode.CondScale,
        // supported_models.py:2404 -> ernie.ErnieTokenizer (:9) -> disable_weights at ernie.py:14 (Mistral3 arm).
        ["ernie-image"] = PromptWeightingMode.CondScale,
        // supported_models.py:2227 -> kandinsky5.Kandinsky5TokenizerImage (:19), a Kandinsky5Tokenizer subclass whose
        // clip_l arm (kandinsky5.py:10, emitted at :14) is a plain SDTokenizer and KEEPS weights, which is what selects
        // the ComfyBlend path; the Qwen arm inherits QwenImageTokenizer's hard disable (qwen_image.py:39).
        // But the blend only rewrites hidden states (sd1_clip.py:54-63; first_pooled is taken at :47, before the loop),
        // and Kandinsky5TEModel.encode_token_weights (kandinsky5.py:39-43) returns the Qwen cond plus CLIP-L's POOLED
        // vector, discarding l_out entirely. So on Kandinsky5 SwarmUI's weighting is a no-op end to end, and parity is
        // to leave the conditioning alone. Do NOT "fix" it by blending the Qwen arm — that diverges from the
        // reference. See NotYetWired for why the recipe stays undeclared until strip-and-blend-nothing exists.
        ["kandinsky5"] = PromptWeightingMode.ComfyBlend,
        // supported_models.py:1157 -> anima.AnimaTokenizer (:18-21): qwen3_06b (:8-11) + t5xxl (:13-16), neither disables.
        ["anima"] = PromptWeightingMode.ComfyBlend,
        // supported_models.py:877 -> gpt_oss.LensTokenizer (:496) -> LensGptOssTokenizer (:456) disable_weights at :475.
        ["lens"] = PromptWeightingMode.CondScale,

        // --- Video families ---

        // supported_models.py:1341 -> wan.WanT5Tokenizer (:20-22) -> UMT5XXlTokenizer (:11-14), no disable_weights. Every
        // Wan variant class (WAN21_I2V :1362, WAN21_Vace :1406, WAN22_Animate :1443, WAN_Animate2 :1456, WAN22_S2V :1430,
        // WAN22_T2V :1470) subclasses WAN21_T2V and adds no clip_target, so all of them share this one answer.
        ["wan"] = PromptWeightingMode.ComfyBlend,
        ["wan-22-5b"] = PromptWeightingMode.ComfyBlend,
        ["wan-21-1_3b"] = PromptWeightingMode.ComfyBlend,
        ["wan-21-14b"] = PromptWeightingMode.ComfyBlend,
        ["wan-vace"] = PromptWeightingMode.ComfyBlend,
        ["wan-animate"] = PromptWeightingMode.ComfyBlend,
        ["wan-animate-2"] = PromptWeightingMode.ComfyBlend,
        ["wan-s2v"] = PromptWeightingMode.ComfyBlend,
        // supported_models.py:1038 -> hunyuan_video.HunyuanVideoTokenizer (:47-51) = clip_l + LLAMA3Tokenizer (:26-29),
        // neither disables. This is HunyuanVideo 1.0 (LLaVA-Llama-3 + CLIP-L, HunyuanVideoRecipe.cs:90-95); the 1.5
        // tokenizer (hunyuan_video.py:79, a HunyuanImageTokenizer subclass) would be CondScale and is not this recipe.
        ["hunyuan-video"] = PromptWeightingMode.ComfyBlend,
        // supported_models.py:945 -> lt.LTXVT5Tokenizer (:17-19) -> T5XXLTokenizer (:11-14), no disable_weights.
        ["ltx-video"] = PromptWeightingMode.ComfyBlend,
        // LTX-2 loads its TE through sd.py's CLIPType.LTXV branch, not the checkpoint's clip_target: sd.py:2019 ->
        // lt.LTXAVGemmaTokenizer (:79-81) -> Gemma3_12BTokenizer disable_weights (lt.py:76). A Gemma-4 TE takes the
        // sibling branch (sd.py:2022-2030) -> gemma4.py:1610, also disabled. Both arms agree, so the family is CondScale.
        ["ltx-video-2"] = PromptWeightingMode.CondScale,
        ["ltx-2.5-distilled"] = PromptWeightingMode.CondScale,
        // supported_models.py:988 -> minimax.MiniMaxH3Tokenizer (:136) -> disable_weights at minimax.py:158.
        ["minimax-h3"] = PromptWeightingMode.CondScale,
        // supported_models.py:2203 -> kandinsky5.Kandinsky5Tokenizer (:6); same CLIP-L arm, same discarded-l_out no-op
        // as the image variant above (both go through Kandinsky5TEModel, kandinsky5.py:39-43).
        ["kandinsky5-video"] = PromptWeightingMode.ComfyBlend,
    };

    /// <summary>Families with NO ComfyUI support, so SwarmUI has no behaviour to match and no mode can be read from a
    /// primary source. They are listed rather than guessed: a wrong entry in <see cref="Ledger"/> becomes permanent.
    /// <para><c>lance-image</c>/<c>lance-video</c> — ByteDance Lance is absent from ComfyUI entirely (no
    /// <c>supported_models.py</c> class, no <c>CLIPType</c>, no tokenizer module).</para>
    /// <para><c>f-lite</c> — Freepik F-Lite is likewise absent; its T5-XXL stack makes ComfyBlend the likely answer but
    /// there is no ComfyUI tokenizer to read it off, so it stays out.</para></summary>
    private static readonly string[] Unresolved = ["f-lite", "lance-image", "lance-video"];

    /// <summary><b>EMPTY as of alpha.135 — every registered family consumes its weights.</b> This listed the
    /// families whose recipe still declared <see cref="PromptWeightingMode.None"/> because their pipeline could not
    /// act on weights yet. A recipe may NOT declare a mode it cannot act on: the declaration is what keeps the
    /// <c>(text:N)</c> grammar in the prompt, so an unwired pipeline would hand the parens and digits to its
    /// encoder as prose — a regression on a family that works today. The list could only shrink, and shrinking it
    /// meant editing this test, which was the point; it stays here, with the notes below, so a NEW recipe that
    /// cannot weight yet has somewhere honest to go rather than silently declaring a mode.
    /// <para>The notes that follow are kept as the record of what each family turned out to need. Several of them
    /// contradict what this entry predicted before the work was done, which is the reason to keep them.</para>
    /// <para><b>Kandinsky5 and kandinsky5-video came off this list by declaring a no-op.</b> They are ComfyBlend
    /// because CLIP-L keeps weights, but <c>Kandinsky5TEModel.encode_token_weights</c> (<c>kandinsky5.py:39-43</c>)
    /// returns the Qwen cond plus CLIP-L's POOLED vector and discards the blended hidden states, so SwarmUI's
    /// weighting does nothing end to end and parity is to leave the conditioning alone. What they still owe is the
    /// STRIP: <c>Kandinsky5TextEncoding.StripEmphasis</c> takes the grammar off before either arm tokenizes, because
    /// declaring the mode is what stops <c>ImagesService</c> collapsing the tag and would otherwise hand Qwen the
    /// parens as prose. Blending the Qwen arm to "fix" the no-op would BREAK parity, not achieve it.</para>
    /// <para><b>hunyuan-image came off this list by returning the UNPADDED weights.</b> It pads to 1034 and the
    /// encoder then slices <c>[34, 34 + keep)</c> (<c>HunyuanImageQwenTextEncoder.cs:17,60-63</c>), so handing the
    /// padded array through would give <c>offset = keep − 1034</c> and drop every prompt weight off the front — a
    /// silent no-op rather than an error. The weights are cut to the mask's real length instead, which makes the
    /// right-alignment offset −34 exactly. Gating it also surfaced a PRE-EXISTING defect, recorded as a TODO and
    /// not fixed: the chat template is 33 ids on our tokenizer, not 34, because <c>EncodeRaw("\n")</c> returns
    /// nothing where HF emits id 198 — so the encoder's slice drops the prompt's first token.</para>
    /// <para><b>minimax-h3 came off this list, and needed less than this entry predicted.</b> It tokenizes INSIDE
    /// its encoder (<c>MiniMaxH3TextEncoding.Build</c>), so the weights are built there — but they do NOT need to
    /// be a full-length array with non-text runs forced to 1. <c>Build</c> appends the user prompt LAST, after
    /// every condition label and vision block, so the prompt is contiguous at the tail and a prompt-length weight
    /// array right-aligns onto exactly those rows. Its cond is rank-2 <c>[seq, hidden]</c> and F32
    /// (<c>MiniMaxH3TextEncoder.cs:215</c>), which is what <c>ScaleRightAligned</c> requires.</para>
    /// <para><b>The whole Wan family is wired.</b> The four variant recipe classes came off this list once the
    /// blend was hoisted into <c>VideoRecipeUtils</c>; Animate-2's driving stream is weighted as its own leaf.</para>
    /// <para><b>ltx-video-2 and ltx-2.5-distilled: the seam is now located, and it is inside the connector.</b>
    /// Read off the reference rather than inferred. <c>LTXAVTEModel.encode_token_weights</c> (<c>lt.py:163-189</c>)
    /// runs the embeddings connectors ONLY under <c>compat_mode</c>; its default path returns un-connected,
    /// token-length embeddings tagged <c>unprocessed_ltxav_embeds</c> and the DiT runs the connectors. SwarmUI
    /// right-aligns unconditionally — <c>offset = cond_len − len(batch)</c>, <c>SwarmText.py:269-271</c> — so on
    /// that default path the scale is token-aligned and correct.
    /// <para>Our pipeline mirrors <c>compat_mode</c> instead: <c>LtxVideo2Pipeline.EncodeText</c> pads to a
    /// 128-register multiple and calls the connectors itself (<c>:682-689,726</c>). Right-aligning a
    /// <c>real</c>-length weight array against that <c>seq</c>-length conditioning would put every weight on
    /// REGISTER rows, since the real tokens sit at the FRONT — a scale applied to learnable padding.</para>
    /// <para>The fix is not to scale <c>feats</c> before the connector either. <c>lt.py:174-176</c> normalizes by a
    /// global min/max over the whole sequence before <c>text_embedding_projection</c>, so scaling one token ahead of
    /// that moves the statistics every other token is divided by. The scale belongs after the projection and before
    /// the register concat, which is inside <c>LtxVideo2TextConnectors</c> — real work, not a recipe-layer call, so
    /// it stays unwired rather than half-done.</para>
    /// <b>krea2</b> came off this list once the joint-attention patch landed; it is the only family that needs
    /// both halves, so the partial declaration it carried first was refused here rather than accepted.</para></summary>
    private static readonly string[] NotYetWired = [];

    private readonly ITestOutputHelper _output;

    public PromptWeightingModeLedgerTests(ITestOutputHelper output) => _output = output;

    /// <summary>Every registered family must declare a verified mode or be explicitly unresolved, so a newly landed
    /// recipe fails here until someone reads its ComfyUI tokenizer instead of inheriting a default.</summary>
    [Fact]
    public void EveryRegisteredFamilyIsLedgeredOrExplicitlyUnresolved()
    {
        List<string> missing = [];
        foreach (string family in RegisteredFamilies())
        {
            if (!Ledger.ContainsKey(family) && !Unresolved.Contains(family, StringComparer.Ordinal))
            {
                missing.Add(family);
            }
        }
        _output.WriteLine($"registered: {string.Join(", ", RegisteredFamilies())}");
        Assert.True(missing.Count == 0,
            $"No verified prompt-weighting mode for: {string.Join(", ", missing)}. Read the family's ComfyUI "
            + "tokenizer (disable_weights on every arm => CondScale) and add it, or list it as unresolved.");
    }

    /// <summary>The reverse direction: a ledger key that no longer resolves is a rename nobody carried across, and it
    /// would silently stop pinning anything.</summary>
    [Fact]
    public void EveryLedgeredFamilyResolvesToARegisteredRecipe()
    {
        HashSet<string> registered = new(RegisteredFamilies(), StringComparer.Ordinal);
        foreach (string family in Ledger.Keys.Concat(Unresolved))
        {
            Assert.True(registered.Contains(family), $"'{family}' is ledgered but no longer registered.");
        }
    }

    /// <summary>A family cannot be both pinned and unresolved.</summary>
    [Fact]
    public void UnresolvedFamiliesAreAbsentFromTheLedger()
    {
        foreach (string family in Unresolved)
        {
            Assert.False(Ledger.ContainsKey(family), $"'{family}' is listed unresolved but also ledgered.");
        }
    }

    /// <summary>Krea2 is the only family whose workflow gets <c>SwarmAttnTokenWeights</c>
    /// (<c>WorkflowGenerator.cs:965-972</c>). Flux, Chroma, Qwen-Image and HunyuanVideo expose the same <c>img_slice</c>
    /// hook the patch needs (<c>SwarmText.py:284</c>), so the tempting generalization is wrong: SwarmUI does not wire
    /// the node for them, and doing it here would diverge from the reference rather than match it.</summary>
    [Fact]
    public void Krea2IsTheOnlyFamilyUsingTheAttentionPatch()
    {
        string[] withAttention = [.. Ledger
            .Where(entry => entry.Value == PromptWeightingMode.CondScaleWithAttention)
            .Select(entry => entry.Key)
            .Order(StringComparer.Ordinal)];
        Assert.Equal(["krea2"], withAttention);
    }

    /// <summary><see cref="PromptWeightingMode.None"/> means "never reaches <c>SwarmTextEncodeAdvanced</c>"
    /// (<c>WorkflowGenerator.cs:2584-2599</c>), which is true of the music families only. No image or video family may
    /// claim it — that would turn a missing implementation into a declared non-feature.</summary>
    [Fact]
    public void NoImageOrVideoFamilyDeclaresNoWeighting()
    {
        foreach (KeyValuePair<string, PromptWeightingMode> entry in Ledger)
        {
            Assert.True(entry.Value != PromptWeightingMode.None,
                $"'{entry.Key}' declares None; only music families bypass SwarmTextEncodeAdvanced.");
        }
    }

    /// <summary>The mechanisms split roughly in half across the catalogue; a mode that collapsed to one value would mean
    /// the table was filled in by default rather than read, which is exactly the failure this ledger exists to prevent.</summary>
    [Fact]
    public void BothMechanismsAreRepresented()
    {
        Assert.Contains(PromptWeightingMode.ComfyBlend, Ledger.Values);
        Assert.Contains(PromptWeightingMode.CondScale, Ledger.Values);
        _output.WriteLine($"ComfyBlend: {Ledger.Values.Count(m => m == PromptWeightingMode.ComfyBlend)}, "
            + $"CondScale: {Ledger.Values.Count(m => m == PromptWeightingMode.CondScale)}, "
            + $"CondScaleWithAttention: {Ledger.Values.Count(m => m == PromptWeightingMode.CondScaleWithAttention)}, "
            + $"unresolved: {Unresolved.Length}");
    }

    /// <summary>The join between the ledger and the code: a recipe that declares a mode must declare the one read off
    /// ComfyUI. This is the assertion the whole ledger exists for — a wrong mode still generates an image, just with the
    /// emphasis applied to the wrong thing, so nothing else would catch it.</summary>
    [Fact]
    public void EveryRecipeThatDeclaresAModeDeclaresTheLedgeredOne()
    {
        foreach ((string family, PromptWeightingMode declared) in DeclaredModes())
        {
            if (declared == PromptWeightingMode.None)
            {
                continue;
            }
            Assert.True(Ledger.TryGetValue(family, out PromptWeightingMode ledgered),
                $"'{family}' declares {declared} but has no ledger entry; read its ComfyUI tokenizer first.");
            Assert.True(declared == ledgered,
                $"'{family}' declares {declared} but the ledger reads {ledgered} off ComfyUI's tokenizer.");
        }
    }

    /// <summary>The reverse: a family may not declare a mode its pipeline cannot act on, and the list of the ones that
    /// still cannot is pinned exactly so wiring one is a deliberate edit rather than a silent drift.</summary>
    [Fact]
    public void TheUnwiredFamiliesAreExactlyTheOnesListed()
    {
        List<string> stillNone = [];
        foreach ((string family, PromptWeightingMode declared) in DeclaredModes())
        {
            if (declared == PromptWeightingMode.None && Ledger.ContainsKey(family))
            {
                stillNone.Add(family);
            }
        }
        _output.WriteLine($"wired: {Ledger.Count - stillNone.Count} of {Ledger.Count} ledgered families");
        string[] expected = [.. NotYetWired.Order(StringComparer.Ordinal)];
        string[] actual = [.. stillNone.Order(StringComparer.Ordinal)];
        Assert.Equal(expected, actual);
    }

    /// <summary>A family with no verified mode must not be wired on a guess: unresolved means unresolved.</summary>
    [Fact]
    public void UnresolvedFamiliesDeclareNoMode()
    {
        foreach ((string family, PromptWeightingMode declared) in DeclaredModes())
        {
            if (Unresolved.Contains(family, StringComparer.Ordinal))
            {
                Assert.True(declared == PromptWeightingMode.None,
                    $"'{family}' has no ComfyUI tokenizer to read, but its recipe declares {declared}.");
            }
        }
    }

    /// <summary>What each registered recipe declares, image first.</summary>
    private static IEnumerable<(string Family, PromptWeightingMode Mode)> DeclaredModes()
    {
        foreach (string family in RecipeRegistry.DefaultNames)
        {
            IArchitectureRecipe recipe = RecipeRegistry.Resolve(family)
                ?? throw new InvalidOperationException($"'{family}' is registered but does not resolve to a recipe.");
            yield return (family, recipe.PromptWeighting);
        }
        foreach (string family in VideoRecipeRegistry.DefaultNames)
        {
            IVideoRecipe recipe = VideoRecipeRegistry.Resolve(family)
                ?? throw new InvalidOperationException($"'{family}' is registered but does not resolve to a recipe.");
            yield return (family, recipe.PromptWeighting);
        }
    }

    /// <summary>Both registries' SHIPPED families, image first, as a stable ordered list. Deliberately not
    /// <c>RegisteredNames</c>: tests register throwaway recipes into these static registries and never remove them, so
    /// the runtime list depends on which other suite ran first in the same process.</summary>
    private static IReadOnlyList<string> RegisteredFamilies() =>
        [.. RecipeRegistry.DefaultNames.Concat(VideoRecipeRegistry.DefaultNames).Order(StringComparer.Ordinal)];
}
