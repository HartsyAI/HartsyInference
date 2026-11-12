using HartsyInference.Core.Imaging;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Codec;
using HartsyInference.Vision.Matting.BiRefNet;
using HartsyInference.Vision.Segmentation;
using HartsyInference.Vision.Segmentation.Sam3;

namespace HartsyInference.ThreeD.Models.HairDeck.Preprocessing;

/// <summary>HairDeck's image preprocessing, a port of hairdeck.prep (preset "best") and hairdeck.infer.image.Pipeline.prep: SAM 3
/// text grounding and BiRefNet-matting at the working resolution, the preset's mask recipe, the head box, and the 448 px model
/// input (the crop on white with its matte). The segmenter and the matting model are owned by the caller.</summary>
public sealed class HairDeckPreprocessor
{
    /// <summary>The longest side of the working resolution the masks are computed at.</summary>
    public const int WorkingResolution = 1024;

    /// <summary>Side of the square input of the matting model.</summary>
    public const int MatteInputSize = 768;

    /// <summary>The stage name reported while SAM 3 segments the image.</summary>
    public const string SegmentStage = "SAM 3 segmentation";

    /// <summary>The stage name reported while BiRefNet mattes the image.</summary>
    public const string MatteStage = "BiRefNet matting";

    private const float ScoreFloor = 0.2f;
    private const float UncertainAgreement = 0.8f;
    private const int Resolution = HairDeckInput.Resolution;

    private readonly Sam3Pipeline _segmenter;
    private readonly BiRefNetMatter _matter;
    private readonly IBackend _backend;

    /// <summary>Binds the models the preprocessing runs; both must already hold their weights.</summary>
    public HairDeckPreprocessor(Sam3Pipeline segmenter, BiRefNetMatter matter, IBackend backend)
    {
        _segmenter = segmenter ?? throw new ArgumentNullException(nameof(segmenter));
        _matter = matter ?? throw new ArgumentNullException(nameof(matter));
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
    }

    /// <summary>Preprocesses one character image.</summary>
    /// <exception cref="HairDeckInputException">Neither hair nor face was found, so no head crop can be made.</exception>
    /// <param name="cancel">Checked between the stages (loading, SAM 3, BiRefNet): a model run in progress finishes first.</param>
    /// <param name="onStage">Called with the name of each stage as it starts.</param>
    public HairDeckPreprocessResult Process(HairDeckSourceImage source, CancellationToken cancel = default, Action<string>? onStage = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        cancel.ThrowIfCancellationRequested();
        HairDeckLoadedImage image = HairDeckImageRules.Load(source);
        cancel.ThrowIfCancellationRequested();
        onStage?.Invoke(SegmentStage);
        IReadOnlyList<TextGroundedLogitResult> segments = Segment(image);
        cancel.ThrowIfCancellationRequested();
        onStage?.Invoke(MatteStage);
        float[] matte = MatteOf(image);
        cancel.ThrowIfCancellationRequested();
        return FromModels(image, segments, matte);
    }

    /// <summary>The SAM 3 queries of the preset prompts above the working floor, for the loaded image.</summary>
    internal IReadOnlyList<TextGroundedLogitResult> Segment(HairDeckLoadedImage image)
    {
        using Tensor tensor = new(new TensorShape(1, 3, image.Height, image.Width), DType.F32);
        Span<float> values = tensor.AsSpan<float>();
        int pixels = image.Width * image.Height;
        for (int i = 0; i < pixels; i++)
        {
            for (int c = 0; c < 3; c++)
            {
                values[c * pixels + i] = image.ModelRgb[i * 3 + c] / 255f;
            }
        }

        return _segmenter.SegmentLogits(_backend, tensor, HairDeckMaskRecipe.Prompts, ScoreFloor);
    }

    /// <summary>The whole-character matte at the source size, in [0, 1], for the loaded image.</summary>
    internal float[] MatteOf(HairDeckLoadedImage image) =>
        _matter.MatteFloat(_backend, image.ModelRgb, image.Width, image.Height, MatteInputSize);

    /// <summary>The rest of the preprocessing, from the model outputs of <paramref name="image"/>.</summary>
    internal HairDeckPreprocessResult FromModels(HairDeckLoadedImage image, IReadOnlyList<TextGroundedLogitResult> sam, float[] matteSource)
    {
        int width = image.Width, height = image.Height;
        int pixels = width * height;
        (int shapeHeight, int shapeWidth) = WorkingShape(width, height);

        float[] matte = new float[shapeHeight * shapeWidth];
        Resample.BilinearPlaneF32(matteSource, width, height, matte, shapeWidth, shapeHeight);
        HairDeckRecipeResult recipe = HairDeckMaskRecipe.Run(sam, shapeHeight, shapeWidth, matte);

        float[] personFloat = new float[pixels];
        Resample.BilinearPlaneF32(recipe.Alpha, shapeWidth, shapeHeight, personFloat, width, height);
        byte[] person = new byte[pixels];
        for (int i = 0; i < pixels; i++)
        {
            person[i] = (byte)MathF.Round(Math.Clamp(personFloat[i], 0f, 1f) * 255f);
        }

        bool[] hair = FullResolutionMask(sam, recipe.HairKeys, width, height);
        bool[] face = FullResolutionMask(sam, recipe.FaceKeys, width, height);

        List<string> notes = [];
        if (recipe.Agreement is { } agreement && agreement < UncertainAgreement)
        {
            notes.Add("matte_uncertain");
        }

        if (recipe.Status == HairDeckPrepStatus.NoPersonMask)
        {
            notes.Add("no_person");
        }

        if (recipe.BoxFallback)
        {
            notes.Add("box_from_low_scores");
        }

        if (recipe.UnselectedPersonCount > 0)
        {
            notes.Add("multiple_persons");
        }

        if (image.InputAlpha is { } inputAlpha)
        {
            for (int i = 0; i < pixels; i++)
            {
                person[i] = Math.Min(person[i], inputAlpha[i]);
                hair[i] &= inputAlpha[i] > 127;
                face[i] &= inputAlpha[i] > 127;
            }

            notes.Add("input_alpha");
        }

        (double X0, double Y0, double X1, double Y1)? hairBox = HairDeckCrop.BoundingBox(hair, width, height);
        (double X0, double Y0, double X1, double Y1)? faceBox = HairDeckCrop.BoundingBox(face, width, height);
        (double X0, double Y0, double X1, double Y1)? union = HairDeckCrop.Union(hairBox, faceBox);
        if (union is null)
        {
            throw new HairDeckInputException(HairDeckInputFailure.NoHairOrFace,
                "No hair or face was found in the image, so no head crop can be made. Use an image in which the head is clearly visible.");
        }

        HairDeckBox box = HairDeckCrop.HeadBox(union.Value);
        HairDeckPrepStatus status = recipe.Status;
        if (hairBox is null)
        {
            status = HairDeckPrepStatus.NoHair;
        }

        bool emptyMatte = !person.Any(value => value > 127);
        if (emptyMatte)
        {
            notes.Add("empty_matte");
        }

        byte[] compositeMatte = emptyMatte ? Filled(pixels, 255) : person;
        byte[] composed = HairDeckCrop.ComposeOnWhite(image.StoredRgb, compositeMatte, pixels);
        byte[] crop = HairDeckCrop.CropOnWhite(composed, width, height, box, Resolution);
        byte[] matteCrop = HairDeckCrop.CropMatte(person, width, height, box, Resolution);
        double[] hairColor = HairDeckCrop.HairColor(image.StoredRgb, hair, pixels);
        HairDeckInput input = HairDeckInput.Create(ToChannelMajor(crop, Resolution), matteCrop, hairColor);

        return new HairDeckPreprocessResult
        {
            Input = input,
            Box = box,
            Status = status,
            Notes = notes,
            Person = person,
            Hair = hair,
            Face = face,
            Width = width,
            Height = height,
            HairColor = hairColor,
            MatteAgreement = recipe.Agreement,
        };
    }

    /// <summary>The working size of the masks: the longest side at most <see cref="WorkingResolution"/>, rounded as Python's round does.</summary>
    internal static (int Height, int Width) WorkingShape(int width, int height)
    {
        double scale = Math.Min(1.0, (double)WorkingResolution / Math.Max(width, height));
        int shapeHeight = Math.Max(1, (int)Math.Round(height * scale, MidpointRounding.ToEven));
        int shapeWidth = Math.Max(1, (int)Math.Round(width * scale, MidpointRounding.ToEven));
        return (shapeHeight, shapeWidth);
    }

    /// <summary>Union of the selected instances, each upsampled from its logits to the source size (hairdeck.prep full()).</summary>
    private static bool[] FullResolutionMask(IReadOnlyList<TextGroundedLogitResult> sam, IReadOnlyList<HairDeckInstanceKey> keys, int width, int height)
    {
        bool[] mask = new bool[width * height];
        foreach (HairDeckInstanceKey key in keys)
        {
            TextGroundedLogitResult prompt = sam.Single(result => result.Prompt == key.Prompt);
            bool[] instance = HairDeckMorphology.LogitsToMask(prompt.Instances[key.Index].Logits, prompt.LogitSide, height, width);
            for (int i = 0; i < mask.Length; i++)
            {
                mask[i] |= instance[i];
            }
        }

        return mask;
    }

    private static byte[] Filled(int count, byte value)
    {
        byte[] result = new byte[count];
        Array.Fill(result, value);
        return result;
    }

    /// <summary>Interleaved RGB to the planar layout the image model reads: <c>[c][y][x]</c>.</summary>
    private static byte[] ToChannelMajor(byte[] interleaved, int side)
    {
        int plane = side * side;
        byte[] planar = new byte[3 * plane];
        for (int i = 0; i < plane; i++)
        {
            for (int c = 0; c < 3; c++)
            {
                planar[c * plane + i] = interleaved[i * 3 + c];
            }
        }

        return planar;
    }
}
