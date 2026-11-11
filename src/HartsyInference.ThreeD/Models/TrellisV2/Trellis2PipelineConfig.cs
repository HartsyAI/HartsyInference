using System.Text.Json;

namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>The constants of a TRELLIS.2 <c>pipeline.json</c>: the structure and shape samplers, the shape-latent normalization and, when
/// the file has them, the texture sampler and texture-latent normalization.</summary>
public sealed class Trellis2PipelineConfig
{
    /// <summary>Latent channels the shape normalization covers.</summary>
    public const int ShapeLatentChannels = 32;

    private Trellis2PipelineConfig(
        Trellis2SamplerParams structure, Trellis2SamplerParams shape, float[] mean, float[] std, string defaultPipelineType,
        Trellis2SamplerParams? texture, float[]? textureMean, float[]? textureStd)
    {
        SparseStructureSampler = structure;
        ShapeSampler = shape;
        ShapeLatentMean = mean;
        ShapeLatentStd = std;
        DefaultPipelineType = defaultPipelineType;
        TextureSampler = texture;
        TextureLatentMean = textureMean;
        TextureLatentStd = textureStd;
    }

    public Trellis2SamplerParams SparseStructureSampler { get; }

    public Trellis2SamplerParams ShapeSampler { get; }

    /// <summary>Per-channel mean of <c>shape_slat_normalization</c>; sampled latents are denormalized as <c>x · std + mean</c>.</summary>
    public float[] ShapeLatentMean { get; }

    public float[] ShapeLatentStd { get; }

    /// <summary>The pipeline type upstream runs when none is named (<c>default_pipeline_type</c>).</summary>
    public string DefaultPipelineType { get; }

    /// <summary><c>tex_slat_sampler</c>, or null when the file names none.</summary>
    public Trellis2SamplerParams? TextureSampler { get; }

    /// <summary>Per-channel mean of <c>tex_slat_normalization</c>; the sampled texture latent is denormalized as <c>x · std + mean</c>. Null without one.</summary>
    public float[]? TextureLatentMean { get; }

    public float[]? TextureLatentStd { get; }

    /// <summary>Whether the file carries everything the texture path reads.</summary>
    public bool HasTexture => TextureSampler is not null && TextureLatentMean is not null && TextureLatentStd is not null;

    public static Trellis2PipelineConfig Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
            throw new FileNotFoundException($"TRELLIS.2 pipeline.json not found: '{path}'.", path);
        return FromJson(File.ReadAllText(path));
    }

    public static Trellis2PipelineConfig FromJson(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using JsonDocument doc = JsonDocument.Parse(json);
        JsonElement root = doc.RootElement;
        string name = root.GetProperty("name").GetString() ?? string.Empty;
        if (name != "Trellis2ImageTo3DPipeline")
            throw new NotSupportedException($"'{name}' is not a Trellis2ImageTo3DPipeline config.");
        JsonElement args = root.GetProperty("args");
        JsonElement norm = args.GetProperty("shape_slat_normalization");
        (float[] mean, float[] std) = Normalization(norm, "shape_slat_normalization");
        Trellis2SamplerParams? texture = null;
        float[]? textureMean = null, textureStd = null;
        bool hasSampler = args.TryGetProperty("tex_slat_sampler", out JsonElement textureNode);
        bool hasNorm = args.TryGetProperty("tex_slat_normalization", out JsonElement textureNorm);
        if (hasSampler != hasNorm)
            throw new ArgumentException("pipeline.json needs both tex_slat_sampler and tex_slat_normalization for texture, or neither; it has one.");
        if (hasSampler)
        {
            texture = Trellis2SamplerParams.FromNode(textureNode);
            (textureMean, textureStd) = Normalization(textureNorm, "tex_slat_normalization");
        }

        string type = args.TryGetProperty("default_pipeline_type", out JsonElement t) ? t.GetString() ?? "1024_cascade" : "1024_cascade";
        return new Trellis2PipelineConfig(
            Trellis2SamplerParams.FromNode(args.GetProperty("sparse_structure_sampler")),
            Trellis2SamplerParams.FromNode(args.GetProperty("shape_slat_sampler")),
            mean,
            std,
            type,
            texture,
            textureMean,
            textureStd);
    }

    private static (float[] Mean, float[] Std) Normalization(JsonElement node, string name)
    {
        float[] mean = Channels(node.GetProperty("mean"), name + ".mean");
        float[] std = Channels(node.GetProperty("std"), name + ".std");
        foreach (float s in std)
        {
            if (!(s > 0f))
                throw new ArgumentException($"{name}.std must be positive.");
        }

        return (mean, std);
    }

    private static float[] Channels(JsonElement array, string what)
    {
        if (array.GetArrayLength() != ShapeLatentChannels)
            throw new ArgumentException($"{what} has {array.GetArrayLength()} entries; expected {ShapeLatentChannels}.");
        float[] values = new float[ShapeLatentChannels];
        for (int i = 0; i < values.Length; i++)
            values[i] = (float)array[i].GetDouble();
        return values;
    }
}
