namespace HartsyInference.Core.Configuration;

/// <summary>Filesystem locations for native libraries and assets.</summary>
/// <remarks>Generated from the pre-migration call sites; defaults and grammars are those the code already had.</remarks>
public static partial class EngineKnobs
{
    /// <summary>Directory prepended to the CUDA userspace library probe list (cuBLAS/cuBLASLt/cuDNN/cudart).</summary>
    public static readonly Knob<string?> CudaLibDir =
        Str("paths.cudaLibDir", null, KnobScope.Construction, KnobDomain.Paths, "Directory prepended to the CUDA userspace library probe list (cuBLAS/cuBLASLt/cuDNN/cudart).");

    /// <summary>Lets a model load fetch a missing side model (text encoder, VAE, CLIP) from its catalog entry. On by
    /// default, which is what SwarmUI's own ComfyUI backend does; turn it off for an air-gapped install and a missing
    /// asset becomes an error naming the repo instead.</summary>
    public static readonly Knob<bool> SideModelAutofetch =
        Bool("paths.sideModelAutofetch", true, KnobScope.Runtime, KnobDomain.Paths, "Lets a model load fetch a missing side model (text encoder, VAE, CLIP) from its catalog entry; off makes a missing one an error.");

    /// <summary>Lets the cuDNN probe download the NVIDIA cuDNN 9 redist into the per-user cache when none is found locally.</summary>
    public static readonly Knob<bool> CudnnAutofetch =
        Bool("paths.cudnnAutofetch", false, KnobScope.Construction, KnobDomain.Paths, "Lets the cuDNN probe download the NVIDIA cuDNN 9 redist into the per-user cache when none is found locally.");

    /// <summary>First directory searched for libcudnn.so.9/cudnn64_9.dll, ahead of the per-user cache and bundled copy.</summary>
    public static readonly Knob<string?> CudnnDir =
        Str("paths.cudnnDir", null, KnobScope.Runtime, KnobDomain.Paths, "First directory searched for libcudnn.so.9/cudnn64_9.dll, ahead of the per-user cache and bundled copy.");

    /// <summary>Direct .tar.xz/.zip URL the cuDNN auto-fetch downloads from instead of NVIDIA's public redist.</summary>
    public static readonly Knob<string?> CudnnUrl =
        Str("paths.cudnnUrl", null, KnobScope.Construction, KnobDomain.Paths, "Direct .tar.xz/.zip URL the cuDNN auto-fetch downloads from instead of NVIDIA's public redist.");

    /// <summary>Overrides the cuDNN redist version (default 9.21.0.82) used in the autofetch download URL.</summary>
    public static readonly Knob<string?> CudnnVersion =
        Str("paths.cudnnVersion", null, KnobScope.Construction, KnobDomain.Paths, "Overrides the cuDNN redist version (default 9.21.0.82) used in the autofetch download URL.");

    /// <summary>Filename of the LTX-2.5 latent x2 upsampler to load for the two-stage flow instead of the shipped default.</summary>
    public static readonly Knob<string?> Ltx2Upsampler =
        Str("paths.ltx2Upsampler", null, KnobScope.Construction, KnobDomain.Paths, "Filename of the LTX-2.5 latent x2 upsampler to load for the two-stage flow instead of the shipped default.");

    /// <summary>Extra directory added to the native-library probe path so libnccl resolves (e.g. inside a torch venv).</summary>
    public static readonly Knob<string?> NcclDir =
        Str("paths.ncclDir", null, KnobScope.Construction, KnobDomain.Paths, "Extra directory added to the native-library probe path so libnccl resolves (e.g. inside a torch venv).");

}
