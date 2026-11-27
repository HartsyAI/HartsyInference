using HartsyInference.Diffusion.Requests;
using HartsyInference.ThreeD.Pipelines;

namespace HartsyInference.Engine.Services;

/// <summary>Turns a 3D pipeline's progress into a <see cref="StepPreview"/> with a stage and an overall fraction that does not reset between stages. For TRELLIS.2 and Pixal3D the fraction weighs each stage by how long it usually takes (the shape flows dwarf everything else), so a bar driven by it moves at the pace the user waits at, not at the pace of the tick count; the other models report the plain step ratio.</summary>
internal static class MeshProgress
{
    /// <summary>Relative cost of each shape stage in pipeline order. Approximate: taken from the 12 GB card timings recorded in the model status document, and the same for every grid except the TRELLIS.2 512 type.</summary>
    private static readonly (string Name, double Weight)[] ShapeStages =
    [
        (MeshStageNames.ImageConditioning, 3), (MeshStageNames.StructureConditioning, 3), (MeshStageNames.StructureFlow, 40),
        (MeshStageNames.CoarseConditioning, 1), (MeshStageNames.CoarseShapeFlow, 30), (MeshStageNames.ShapeFlow512, 30),
        (MeshStageNames.Upsample, 7), (MeshStageNames.FineConditioning, 5), (MeshStageNames.FineShapeFlow, 260),
        (MeshStageNames.ShapeFlow1024, 260), (MeshStageNames.ShapeDecode, 30), (MeshStageNames.MeshExtraction, 3),
    ];

    private static readonly (string Name, double Weight)[] TextureStages =
    [
        (MeshStageNames.TextureConditioning, 4), (MeshStageNames.TextureFlow, 150), (MeshStageNames.TextureDecode, 35), (MeshStageNames.TextureBake, 250),
    ];

    private static readonly HashSet<string> PixalOnly =
    [
        MeshStageNames.StructureConditioning, MeshStageNames.CoarseConditioning, MeshStageNames.CoarseShapeFlow,
        MeshStageNames.FineConditioning, MeshStageNames.FineShapeFlow,
    ];

    private static readonly HashSet<string> TrellisOnly = [MeshStageNames.ImageConditioning, MeshStageNames.ShapeFlow512, MeshStageNames.ShapeFlow1024];

    /// <summary>A tick of a model that has no weighted stages: the steps, and optionally a named stage with its own steps and position.</summary>
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
        (string Name, double Weight)[] stages = [.. Stages(modelId, textured, grid)];
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

    /// <summary>The stages one run reports, in order: only those its pipeline ticks, so the stage count matches the stages a host sees.</summary>
    internal static IEnumerable<(string Name, double Weight)> Stages(string modelId, bool textured, int grid)
    {
        bool pixal = modelId == MeshModelIds.Pixal3D;
        bool cascade = grid != 512;
        foreach ((string name, double weight) in ShapeStages)
        {
            if (pixal ? TrellisOnly.Contains(name) : PixalOnly.Contains(name))
                continue;
            if (!cascade && name is MeshStageNames.Upsample or MeshStageNames.ShapeFlow1024)
                continue;
            // A textured run goes from the decode straight to the texture stages: nothing ticks the meshing.
            if (textured && name == MeshStageNames.MeshExtraction)
                continue;
            yield return (name, name == MeshStageNames.ShapeFlow512 && !cascade ? 120 : weight);
        }
        if (!textured)
            yield break;
        foreach ((string name, double weight) in TextureStages)
        {
            // TRELLIS.2 reuses the shape conditioning for its texture flow.
            if (!pixal && name == MeshStageNames.TextureConditioning)
                continue;
            yield return (name, weight);
        }
    }
}
