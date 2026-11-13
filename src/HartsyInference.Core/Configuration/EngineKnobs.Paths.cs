namespace HartsyInference.Core.Configuration;

/// <summary>Filesystem locations for native libraries and assets.</summary>
/// <remarks>Generated from the pre-migration call sites; defaults and grammars are those the code already had.</remarks>
public static partial class EngineKnobs
{
    /// <summary>Directory prepended to the CUDA userspace library probe list (cuBLAS/cuBLASLt/cuDNN/cudart).</summary>
    public static readonly Knob<string?> CudaLibDir =
        Str("paths.cudaLibDir", null, KnobScope.Construction, KnobDomain.Paths, "Directory prepended to the CUDA userspace library probe list (cuBLAS/cuBLASLt/cuDNN/cudart).");

    /// <summary>Directory holding the <c>ffmpeg</c> and <c>ffprobe</c> binaries the engine launches for video decode and encode.</summary>
    public static readonly Knob<string?> FfmpegDir =
        Str("paths.ffmpegDir", null, KnobScope.Runtime, KnobDomain.Paths, "Directory holding the ffmpeg and ffprobe binaries used for video input and output. Unset (or a folder without them) uses the ones on PATH.");

    /// <summary>Lets a model load fetch a missing side model (text encoder, VAE, CLIP) from its catalog entry; off makes a missing one an error naming the repo.</summary>
    public static readonly Knob<bool> SideModelAutofetch =
        Bool("paths.sideModelAutofetch", true, KnobScope.Runtime, KnobDomain.Paths, "Lets a model load fetch a missing side model (text encoder, VAE, CLIP) from its catalog entry; off makes a missing one an error.");

    /// <summary>Base URL of a weights mirror (for example HartsyWeb) that hosts safetensors repacks of pickle-only checkpoints; empty (the default) disables it.</summary>
    public static readonly Knob<string?> WeightsMirrorUrl =
        Str("paths.weightsMirrorUrl", null, KnobScope.Runtime, KnobDomain.Paths, "Base https URL of a weights mirror that hosts safetensors repacks of pickle-only 3D checkpoints. A catalog file that lists a repack is asked for there first and falls back to its upstream file on any failure (a dropped connection is not resumed: the fallback starts the upstream download over). A base URL with a query is ignored. Empty disables the mirror.");

    /// <summary>Lets the first call that needs cuDNN download the NVIDIA cuDNN 9 redist into the per-user cache when none is installed.</summary>
    public static readonly Knob<bool> CudnnAutofetch =
        Bool("paths.cudnnAutofetch", true, KnobScope.Construction, KnobDomain.Paths, "Lets the first conv or attention call that needs cuDNN download the NVIDIA cuDNN 9 redist (about 1 GB) into the per-user cache when none is installed. Off keeps the fallbacks.");

    /// <summary>First directory searched for libcudnn.so.9/cudnn64_9.dll, ahead of the per-user cache and bundled copy.</summary>
    public static readonly Knob<string?> CudnnDir =
        Str("paths.cudnnDir", null, KnobScope.Runtime, KnobDomain.Paths, "First directory searched for libcudnn.so.9/cudnn64_9.dll, ahead of the per-user cache and bundled copy.");

    /// <summary>Direct .tar.xz/.zip URL the cuDNN auto-fetch downloads from instead of NVIDIA's public redist.</summary>
    public static readonly Knob<string?> CudnnUrl =
        Str("paths.cudnnUrl", null, KnobScope.Construction, KnobDomain.Paths, "Direct .tar.xz/.zip URL the cuDNN auto-fetch downloads from instead of NVIDIA's public redist.");

    /// <summary>Overrides the cuDNN redist version (default 9.21.0.82) used in the autofetch download URL.</summary>
    public static readonly Knob<string?> CudnnVersion =
        Str("paths.cudnnVersion", null, KnobScope.Construction, KnobDomain.Paths, "Overrides the cuDNN redist version (default 9.21.0.82) used in the autofetch download URL.");

    /// <summary>Lets the first CUDA backend construction download NVIDIA's cuBLAS redist into the per-user cache when none is installed.</summary>
    public static readonly Knob<bool> CublasAutofetch =
        Bool("paths.cublasAutofetch", true, KnobScope.Construction, KnobDomain.Paths, "Lets the first CUDA backend construction download NVIDIA's cuBLAS redist (about 0.4 GB on Windows, 0.9 GB on Linux) into the per-user cache when no cuBLAS is installed. Off leaves CUDA unavailable without the CUDA Toolkit.");

    /// <summary>First directory searched for cublas64_N.dll/libcublas.so.N, ahead of the per-user cache and bundled copy.</summary>
    public static readonly Knob<string?> CublasDir =
        Str("paths.cublasDir", null, KnobScope.Construction, KnobDomain.Paths, "First directory searched for cublas64_N.dll/libcublas.so.N (and cuBLASLt), ahead of the per-user cache and bundled copy.");

    /// <summary>Direct .tar.xz/.zip URL (or local archive path) the cuBLAS auto-fetch uses instead of NVIDIA's public redist.</summary>
    public static readonly Knob<string?> CublasUrl =
        Str("paths.cublasUrl", null, KnobScope.Construction, KnobDomain.Paths, "Direct .tar.xz/.zip URL, or a local archive path, the cuBLAS auto-fetch uses instead of NVIDIA's public redist. Skips the sha256 check.");

    /// <summary>Overrides the cuBLAS redist version used in the autofetch download URL.</summary>
    public static readonly Knob<string?> CublasVersion =
        Str("paths.cublasVersion", null, KnobScope.Construction, KnobDomain.Paths, "Overrides the cuBLAS redist version (default 13.10.0.20 for CUDA 13, 12.9.2.10 for CUDA 12) used in the autofetch URL. Skips the sha256 check.");

    /// <summary>Filename of the LTX-2.5 latent x2 upsampler to load for the two-stage flow instead of the shipped default.</summary>
    public static readonly Knob<string?> Ltx2Upsampler =
        Str("paths.ltx2Upsampler", null, KnobScope.Construction, KnobDomain.Paths, "Filename of the LTX-2.5 latent x2 upsampler to load for the two-stage flow instead of the shipped default.");

    /// <summary>Extra directory added to the native-library probe path so libnccl resolves (e.g. inside a torch venv).</summary>
    public static readonly Knob<string?> NcclDir =
        Str("paths.ncclDir", null, KnobScope.Construction, KnobDomain.Paths, "Extra directory added to the native-library probe path so libnccl resolves (e.g. inside a torch venv).");

}
