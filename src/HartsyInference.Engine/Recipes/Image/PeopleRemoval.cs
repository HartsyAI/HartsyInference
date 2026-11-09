using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>
/// The people-removal recipe: OneReward's FLUX.1 Fill fine-tune with its object-removal LoRA, aimed at people. The CLI
/// (<c>hartsy remove-people</c>) and the native route (<c>POST /v1/native/images/remove-people</c>) both build their request
/// here, so the two cannot drift apart. The caller supplies the photo and a mask that covers the people, because the engine
/// does not detect people.
/// </summary>
/// <remarks>
/// On the bus scene (fp8, 15 px two-person mask, unmasked T5), the people negative removed both people on 24 of 24 seeds, where
/// the README's <c>nsfw</c> negative removed them on 3 of 12. The negative prompt also changes the background fill: on the README
/// example it leaves a faint lighter patch where the object was.
/// </remarks>
public static class PeopleRemoval
{
    /// <summary>The fixed prompt the removal LoRA was trained on (the README's removal example).</summary>
    public const string Prompt = "remove";

    /// <summary>What the fill must not paint back in. The README uses <c>nsfw</c>, which does not keep people out.</summary>
    public const string NegativePrompt = "person, people, human, man, woman, pedestrian";

    /// <summary>True-CFG scale: the negative pass is combined as <c>neg + scale·(pos − neg)</c>.</summary>
    public const float TrueCfgScale = 4f;

    /// <summary>Distilled guidance, as in the README's removal example.</summary>
    public const float FluxGuidance = 1f;

    /// <summary>Denoising steps, as in the README's removal example.</summary>
    public const int Steps = 50;

    /// <summary>Creativity 1.0: the masked region is generated from noise, which is what the Fill model expects.</summary>
    public const double FillCreativity = 1.0;

    /// <summary>The object-removal LoRA, by catalog id.</summary>
    public const string LoraId = "flux.1-fill-dev-object-removal-lora";

    /// <summary>Blend weight of the removal LoRA.</summary>
    public const double LoraWeight = 1.0;

    /// <summary>Catalog id of the checkpoint for <paramref name="precision"/>: <c>fp8</c> (the default) or <c>bf16</c>.</summary>
    public static string ModelId(string? precision) => precision switch
    {
        null or "" or "fp8" => "onereward-fill-fp8",
        "bf16" => "onereward-fill",
        _ => throw new ArgumentException($"precision must be fp8 or bf16, not '{precision}'.", nameof(precision)),
    };

    /// <summary>The image request that removes the people under <paramref name="mask"/> from <paramref name="image"/>.</summary>
    public static ImageRequest Build(ImageData image, ImageData mask, long seed = -1) => new()
    {
        Prompt = Prompt,
        NegativePrompt = NegativePrompt,
        Width = image.Width,
        Height = image.Height,
        Steps = Steps,
        FluxGuidance = FluxGuidance,
        TrueCfgScale = TrueCfgScale,
        Seed = seed,
        Loras = new LoraStack { Entries = [new LoraEntry { Model = LoraId, Weight = LoraWeight }] },
        Img2Img = new Img2Img { InitImage = image, Creativity = FillCreativity },
        Inpaint = new Inpaint { Mask = mask },
    };
}
