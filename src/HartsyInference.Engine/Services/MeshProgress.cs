using HartsyInference.Diffusion.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Turns a 3D pipeline's progress into a <see cref="StepPreview"/> with a stage and an overall fraction that does not reset between stages. The fraction weighs each stage by how long it usually takes (the shape flows dwarf everything else), so a bar driven by it moves at the pace the user waits at, not at the pace of the tick count.</summary>
internal static class MeshProgress
{
    /// <summary>Relative cost of each stage in pipeline order: the structure, coarse and fine flows, the decode and, with texture, the texture flow and bake. Approximate: taken from the 12 GB card timings recorded in the model status document, and the same for every grid.</summary>
    private static readonly (string Name, double Weight)[] ShapeStages =
    [
        ("Image conditioning", 3), ("Structure conditioning", 3), ("Structure flow", 40), ("Coarse conditioning", 1), ("Coarse shape flow", 30),
        ("Shape flow 512", 30), ("Upsample", 7), ("Fine conditioning", 5), ("Fine shape flow", 260), ("Shape flow 1024", 260),
        ("Shape decode", 30), ("Mesh extraction", 3),
    ];

    private static readonly (string Name, double Weight)[] TextureStages =
    [
        ("Texture conditioning", 4), ("Texture flow", 150), ("Texture decode", 35), ("Texture bake", 250),
    ];

    /// <summary>A tick of a model that has no stages: the steps, and optionally a named stage with its own steps.</summary>
    public static StepPreview Plain(int step, int total, string? stage = null, int stageStep = 0, int stageSteps = 0, int stageIndex = 0, int stageCount = 0) => new()
    {
        Step = step,
        TotalSteps = total,
        Stage = stage,
        StageIndex = stageIndex,
        StageCount = stageCount,
        OverallFraction = total > 0 ? Math.Clamp((float)step / total, 0f, 1f) : null,
    };

    /// <summary>A tick of a staged TRELLIS.2 or Pixal3D run. A tick whose stage is not in the table falls back to the step count.</summary>
    public static StepPreview Map(string modelId, bool textured, int grid, GenerationProgress progress)
    {
        (string Name, double Weight)[] stages = [.. PresentStages(modelId, grid), .. (textured ? TextureStages : [])];
        int index = Array.FindIndex(stages, s => s.Name == progress.Stage);
        StepPreview fallback = Plain(progress.Step, progress.TotalSteps, progress.Stage, progress.StageStep, progress.StageSteps);
        if (index < 0 || progress.TotalSteps <= 0)
            return fallback;
        double total = stages.Sum(s => s.Weight);
        double before = stages.Take(index).Sum(s => s.Weight);
        double within = progress.StageSteps > 0 ? Math.Clamp((double)progress.StageStep / progress.StageSteps, 0, 1) : 1;
        return fallback with
        {
            StageIndex = index + 1,
            StageCount = stages.Length,
            OverallFraction = (float)Math.Clamp((before + stages[index].Weight * within) / total, 0, 1),
        };
    }

    /// <summary>The shape stages one model runs, in order; names a pipeline does not use are left out so they do not stretch the bar.</summary>
    private static IEnumerable<(string Name, double Weight)> PresentStages(string modelId, int grid)
    {
        bool pixal = modelId == MeshModelIds.Pixal3D;
        bool cascade = grid != 512;
        foreach ((string name, double weight) in ShapeStages)
        {
            bool pixalOnly = name is "Structure conditioning" or "Coarse conditioning" or "Coarse shape flow" or "Fine conditioning" or "Fine shape flow";
            bool trellisOnly = name is "Image conditioning" or "Shape flow 512" or "Shape flow 1024";
            if (!cascade && name is "Upsample" or "Shape flow 1024")
                continue;
            if (pixal ? trellisOnly : pixalOnly)
                continue;
            yield return (name, name == "Shape flow 512" && !cascade ? 120 : weight);
        }
    }
}
