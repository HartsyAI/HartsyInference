using System.Text.Json;

namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>The shape-stage constants of a TRELLIS.2 <c>pipeline.json</c>: the two samplers and the shape-latent normalization.
/// Texture entries are ignored.</summary>
public sealed class Trellis2PipelineConfig
{
    /// <summary>Latent channels the shape normalization covers.</summary>
    public const int ShapeLatentChannels = 32;

    private Trellis2PipelineConfig(
        Trellis2SamplerParams structure, Trellis2SamplerParams shape, float[] mean, float[] std, string defaultPipelineType)
    {
        SparseStructureSampler = structure;
        ShapeSampler = shape;
        ShapeLatentMean = mean;
        ShapeLatentStd = std;
        DefaultPipelineType = defaultPipelineType;
    }

    public Trellis2SamplerParams SparseStructureSampler { get; }

    public Trellis2SamplerParams ShapeSampler { get; }

    /// <summary>Per-channel mean of <c>shape_slat_normalization</c>; sampled latents are denormalized as <c>x · std + mean</c>.</summary>
    public float[] ShapeLatentMean { get; }

    public float[] ShapeLatentStd { get; }

    /// <summary>The pipeline type upstream runs when none is named (<c>default_pipeline_type</c>).</summary>
    public string DefaultPipelineType { get; }

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
        float[] mean = Channels(norm.GetProperty("mean"), "mean");
        float[] std = Channels(norm.GetProperty("std"), "std");
        foreach (float s in std)
        {
            if (!(s > 0f))
                throw new ArgumentException("shape_slat_normalization.std must be positive.");
        }

        string type = args.TryGetProperty("default_pipeline_type", out JsonElement t) ? t.GetString() ?? "1024_cascade" : "1024_cascade";
        return new Trellis2PipelineConfig(
            Trellis2SamplerParams.FromNode(args.GetProperty("sparse_structure_sampler")),
            Trellis2SamplerParams.FromNode(args.GetProperty("shape_slat_sampler")),
            mean,
            std,
            type);
    }

    private static float[] Channels(JsonElement array, string what)
    {
        if (array.GetArrayLength() != ShapeLatentChannels)
            throw new ArgumentException($"shape_slat_normalization.{what} has {array.GetArrayLength()} entries; expected {ShapeLatentChannels}.");
        float[] values = new float[ShapeLatentChannels];
        for (int i = 0; i < values.Length; i++)
            values[i] = (float)array[i].GetDouble();
        return values;
    }
}
