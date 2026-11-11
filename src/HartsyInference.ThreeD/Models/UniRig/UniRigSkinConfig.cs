namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>Hyper-parameters of the skin network outside its two point encoders (<c>unirig_skin.yaml</c>).</summary>
public sealed record UniRigSkinConfig
{
    /// <summary>Token width of the bone and mesh features.</summary>
    public int FeatDim { get; init; } = 768;

    /// <summary>Attention heads of the cross-attention blocks and of the final bone scoring.</summary>
    public int NumHeads { get; init; } = 16;

    /// <summary>Cross-attention blocks of the bone encoder.</summary>
    public int NumBoneAttn { get; init; } = 8;

    /// <summary>Cross-attention blocks letting mesh points read the bones.</summary>
    public int NumMeshBoneAttn { get; init; } = 16;

    /// <summary>Width of the first bone MLP layer.</summary>
    public int BoneEmbedDim { get; init; } = 1024;

    /// <summary>Width of the per-(point, bone) scoring MLP.</summary>
    public int MlpDim { get; init; } = 512;

    /// <summary>Exponent applied to the voxel prior before the output is renormalized.</summary>
    public float VoxelMask { get; init; } = 3.0f;

    /// <summary>Points scored per pass over all bones.</summary>
    public int PointChunk { get; init; } = 512;

    /// <summary>Points pushed through the mesh-bone attention per pass; the passes are independent.</summary>
    public int AttentionChunk { get; init; } = 4096;

    /// <summary>Frequencies of the bone-position embedding (2^i · π).</summary>
    public int BoneFreqs { get; init; } = 6;
}
