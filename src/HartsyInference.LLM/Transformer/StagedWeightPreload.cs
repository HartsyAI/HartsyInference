using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Per-stage headroom-guarded weight preload for a layer-split <see cref="LlmPlacement"/> — shared by every staged generation driver (<see cref="Generation.TextGenerationPipeline"/>, the VLM generators) so the budget/logging loop isn't copy-pasted per driver; without it, auto-promotion's size floor leaves every small weight (norms, gates) host-side forever on a stage, and each eager step re-uploads them.</summary>
internal static class StagedWeightPreload
{
    private const long BaseHeadroomBytes = 2L << 30;

    /// <summary>Uploads each stage's weight slice up to its own free-VRAM headroom (base 2 GB per stage, plus <paramref name="lastStageExtraHeadroom"/> reserved on the LAST stage only — e.g. the F32 embed table graph decode later materializes there; 0 for drivers that never enter graph decode). The lazy per-op upload path covers anything skipped, landing on the right device by construction since each layer's ops run on its own stage backend.</summary>
    public static void Preload(GenericTransformer model, LlmPlacement placement, long lastStageExtraHeadroom = 0)
    {
        for (int s = 0; s < placement.Stages.Count; s++)
        {
            LlmStage stage = placement.Stages[s];
            bool last = s == placement.Stages.Count - 1;
            long stageHeadroom = BaseHeadroomBytes + (last ? lastStageExtraHeadroom : 0);
            long stageFree = stage.Backend.FreeMemoryBytes();
            if (stageFree <= stageHeadroom) continue;
            long stageBudget = stageFree - stageHeadroom;
            List<Tensor> stageLoad = [];
            foreach (Tensor t in model.EnumerateStageWeights(stage.StartLayer, stage.EndLayer,
                isFirstStage: s == 0, isLastStage: last, includeRedundantSplits: false))
            {
                long bytes = Tensor.ComputeByteSize(t.Shape, t.DType);
                if (stageBudget - bytes < 0) continue;
                stageBudget -= bytes;
                stageLoad.Add(t);
            }
            HartsyInference.Core.Logging.Logs.Info(
                $"[preload] stage {s} [{stage.StartLayer},{stage.EndLayer}) kept={stageLoad.Count} on its device.");
            stage.Backend.PreloadWeights(stageLoad);
        }
    }
}
