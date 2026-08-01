using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Config for the SeedVR2 NaDiT (one-step video restoration, ByteDance ICLR 2026). MM-DiT with
/// per-layer windowed joint attention (<see cref="SeedVr2Windowing"/>), window-local 3-axis lang-RoPE,
/// SwiGLU MLPs, per-branch RMS qk-norm, and "single" AdaLN (per-block learned vectors combined with the
/// shared timestep embedding). Blocks below <see cref="MmLayers"/> hold separate video/text weights
/// (<c>.vid.</c>/<c>.txt.</c> keys); the rest share one set (<c>.all.</c>). The last layer computes the
/// video branch only. All dims are validation-gated against the real checkpoint via <see cref="Detect"/>.</summary>
public sealed record SeedVr2Config
{
    /// <summary>Transformer hidden width (3B: 2560).</summary>
    public required int VidDim { get; init; }

    /// <summary>Raw text-embedding width before <c>txt_in</c> projection (frozen pos/neg embeddings, 5120).</summary>
    public required int TxtInDim { get; init; }

    /// <summary>Timestep-embedding output width (6 chunks of <see cref="VidDim"/>: shift/scale/gate × attn/mlp).</summary>
    public required int EmbDim { get; init; }

    /// <summary>Attention heads (3B: 20; head_dim fixed at 128).</summary>
    public required int Heads { get; init; }

    /// <summary>Per-head dim — also the RMS qk-norm width and the RoPE dim.</summary>
    public required int HeadDim { get; init; }

    /// <summary>SwiGLU hidden width (3B: 6912).</summary>
    public required int MlpDim { get; init; }

    /// <summary>Total transformer blocks (3B: 32).</summary>
    public required int NumLayers { get; init; }

    /// <summary>Blocks with separate video/text weights; blocks ≥ this share one weight set (3B: 10).</summary>
    public required int MmLayers { get; init; }

    /// <summary>Patchify input channels: 16 noisy latent + 16 conditioning latent + 1 mask = 33.</summary>
    public required int InChannels { get; init; }

    /// <summary>Latent channels of the velocity prediction (16).</summary>
    public required int OutChannels { get; init; }

    /// <summary>Patchify size (T,H,W) — spatial-only (1,2,2).</summary>
    public (int T, int H, int W) PatchSize { get; init; } = (1, 2, 2);

    /// <summary>Window counts (T,H,W) for the alternating regular/shifted partition.</summary>
    public (int T, int H, int W) WindowCounts { get; init; } = (4, 3, 3);

    /// <summary>True = SwiGLU (3B, bias-free, gated); false = plain GELU-tanh MLP with biases (7B).</summary>
    public bool SwiGluMlp { get; init; } = true;

    /// <summary>Whether the tail RMS norm + attn-slice modulation exists (3B yes; 7B has a bare vid_out).</summary>
    public bool HasTailNorm { get; init; } = true;

    /// <summary>SeedVR2-3B dims (configs_3b/main.yaml).</summary>
    public static SeedVr2Config Seedvr2_3B => new()
    {
        VidDim = 2560, TxtInDim = 5120, EmbDim = 15360, Heads = 20, HeadDim = 128,
        MlpDim = 6912, NumLayers = 32, MmLayers = 10, InChannels = 33, OutChannels = 16,
    };

    /// <summary>Infers every dim from checkpoint tensor shapes and key layout, throwing on anything that
    /// contradicts the NaDiT structure. The <see cref="MmLayers"/> boundary is detected from the first block
    /// using shared (<c>.all.</c>) keys — the single most bug-prone loading detail, so it is derived from the
    /// checkpoint itself rather than trusted from a preset.</summary>
    public static SeedVr2Config Detect(IReadOnlyDictionary<string, Tensor> weights)
    {
        Tensor txtIn = Require(weights, "txt_in.weight");
        Tensor vidIn = Require(weights, "vid_in.proj.weight");
        Tensor vidOut = Require(weights, "vid_out.proj.weight");
        Tensor embOut = Require(weights, "emb_in.proj_out.weight");
        Tensor normQ = Require(weights, "blocks.0.attn.norm_q.vid.weight");
        bool swiglu = weights.ContainsKey("blocks.0.mlp.vid.proj_in_gate.weight");
        Tensor mlpIn = swiglu
            ? Require(weights, "blocks.0.mlp.vid.proj_in_gate.weight")
            : Require(weights, "blocks.0.mlp.vid.proj_in.weight");

        int vidDim = (int)txtIn.Shape[0];
        int txtInDim = (int)txtIn.Shape[1];
        int headDim = (int)normQ.Shape[0];
        if (vidDim % headDim != 0)
            throw new HartsyInferenceException($"vid_dim {vidDim} not divisible by head_dim {headDim}.");

        int numLayers = 0, mmLayers = -1;
        foreach (string key in weights.Keys)
        {
            if (!key.StartsWith("blocks.", StringComparison.Ordinal))
                continue;
            int end = key.IndexOf('.', 7);
            int idx = int.Parse(key.AsSpan(7, end - 7));
            numLayers = Math.Max(numLayers, idx + 1);
            if (mmLayers < 0 && key.Contains(".all.", StringComparison.Ordinal))
                mmLayers = idx;
        }
        // No `.all.` keys = full MM-DiT (7B: all 36 blocks split) → boundary at numLayers.
        if (mmLayers < 0)
            mmLayers = numLayers;
        if (numLayers == 0 || mmLayers <= 0)
            throw new HartsyInferenceException(
                $"Block layout not recognized: layers={numLayers}, first shared block={mmLayers}.");
        for (int i = 0; i < numLayers; i++)
        {
            bool shared = weights.ContainsKey($"blocks.{i}.attn.proj_qkv.all.weight");
            bool split = weights.ContainsKey($"blocks.{i}.attn.proj_qkv.vid.weight");
            if (shared == split || shared != i >= mmLayers)
                throw new HartsyInferenceException(
                    $"Block {i} weight layout contradicts mm boundary {mmLayers} (shared={shared}, split={split}).");
        }

        (int pT, int pH, int pW) = (1, 2, 2);
        int patchVolume = pT * pH * pW;
        int inChannels = (int)vidIn.Shape[1] / patchVolume;
        int outChannels = (int)vidOut.Shape[0] / patchVolume;
        if (inChannels * patchVolume != vidIn.Shape[1] || outChannels * patchVolume != vidOut.Shape[0])
            throw new HartsyInferenceException(
                $"Patch layout mismatch: vid_in {vidIn.Shape[1]}, vid_out {vidOut.Shape[0]}, patch {patchVolume}.");

        bool hasTailNorm = weights.ContainsKey("vid_out_norm.weight");
        // Plain-MLP + no tail norm = the 7B checkpoint, which configs_7b/main.yaml builds from
        // models.dit.nadit — the V1 architecture, not the dit_v2 this class implements. The key names
        // coincide, so it LOADS and silently produces mud (verified 2026-08-01: GT SSIM 0.71 vs the 3B's
        // 0.88, visually smeared). Fail loudly until the v1 NaDiT is ported with its own parity chain.
        if (!swiglu && !hasTailNorm)
            throw new HartsyInferenceException(
                "This SeedVR2 checkpoint is the 7B (v1 NaDiT: models/dit — qk_rope/shared_qkv layout); " +
                "only the dit_v2 architecture (3B) is ported. See MODEL_STATUS_VIDEO.md SeedVR2 follow-ups.");

        return new SeedVr2Config
        {
            VidDim = vidDim,
            TxtInDim = txtInDim,
            EmbDim = (int)embOut.Shape[0],
            Heads = vidDim / headDim,
            HeadDim = headDim,
            MlpDim = (int)mlpIn.Shape[0],
            SwiGluMlp = swiglu,
            HasTailNorm = hasTailNorm,
            NumLayers = numLayers,
            MmLayers = mmLayers,
            InChannels = inChannels,
            OutChannels = outChannels,
            PatchSize = (pT, pH, pW),
        };
    }

    private static Tensor Require(IReadOnlyDictionary<string, Tensor> weights, string key)
        => weights.TryGetValue(key, out Tensor? t)
            ? t
            : throw new HartsyInferenceException($"SeedVR2 checkpoint missing required tensor '{key}'.");
}
