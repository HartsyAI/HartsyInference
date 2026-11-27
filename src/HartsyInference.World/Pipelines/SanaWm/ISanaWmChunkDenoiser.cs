using HartsyInference.Core.Tensors;

namespace HartsyInference.World.Pipelines.SanaWm;

/// <summary>The SANA-WM streaming student as the self-forcing sampler drives it: one forward over one chunk for one guidance
/// branch, continuing from the chunks already committed. The denoiser keeps one stream state per guidance branch.</summary>
public interface ISanaWmChunkDenoiser
{
    /// <summary>Latent frames committed so far: the temporal position of the next chunk's first frame.</summary>
    int CommittedFrames { get; }

    /// <summary>Predicts the flow velocity for host F32 latents <c>[1, 128, T, H, W]</c> of the next chunk.</summary>
    /// <param name="latents">The chunk's latents.</param>
    /// <param name="timesteps">One timestep per latent frame of the chunk.</param>
    /// <param name="branch">The guidance branch whose caption and stream state the forward uses.</param>
    /// <param name="cameraConditions">Per latent frame of the chunk, 16 camera-to-world floats then fx, fy, cx, cy.</param>
    /// <param name="chunkPlucker">Host F32 Plücker features <c>[1, 48, T, H, W]</c> of the chunk's frames.</param>
    /// <param name="commit">Stages the branch's state after this chunk: set on the pass over the clean chunk at timestep 0.</param>
    /// <returns>A host F32 tensor shaped like <paramref name="latents"/>, owned by the caller.</returns>
    Tensor Predict(Tensor latents, ReadOnlySpan<float> timesteps, SanaWmGuidanceBranch branch, ReadOnlySpan<float> cameraConditions,
        Tensor chunkPlucker, bool commit);

    /// <summary>Makes the staged state of every branch current: the stream now stands after a chunk of
    /// <paramref name="latentFrames"/> frames.</summary>
    void Commit(int latentFrames);

    /// <summary>Drops the state a failed or cancelled chunk staged.</summary>
    void Discard();
}
