namespace HartsyInference.Core.Configuration;

/// <summary>Tracing, dumps, probes and profiling. None of these change generated output.</summary>
/// <remarks>Generated from the pre-migration call sites; defaults and grammars are those the code already had.</remarks>
public static partial class EngineKnobs
{
    /// <summary>Prints per-stage wall-clock timings (with a device sync) for the Hunyuan3D and TripoSR mesh pipelines.</summary>
    public static readonly Knob<bool> ThreeDPhase =
        Bool("diagnostics.threeDPhase", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Prints per-stage wall-clock timings (with a device sync) for the Hunyuan3D and TripoSR mesh pipelines.");

    /// <summary>Directory where Wan-Animate's first forward writes per-stage tensor dumps; unset disables dumping.</summary>
    public static readonly Knob<string?> AnimateDump =
        Str("diagnostics.animateDump", null, KnobScope.Runtime, KnobDomain.Diagnostics, "Directory where Wan-Animate's first forward writes per-stage tensor dumps; unset disables dumping.");

    /// <summary>Directory to write the Wan-Animate rendered pose-skeleton frames as PPMs before they are VAE-encoded.</summary>
    public static readonly Knob<string?> AnimatePoseDump =
        Str("diagnostics.animatePoseDump", null, KnobScope.Runtime, KnobDomain.Diagnostics, "Directory to write the Wan-Animate rendered pose-skeleton frames as PPMs before they are VAE-encoded.");

    /// <summary>Throws instead of falling back when a CUDA helper resolves state ambient-lessly with 2+ backends live.</summary>
    public static readonly Knob<bool> AssertAmbient =
        Bool("diagnostics.assertAmbient", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Throws instead of falling back when a CUDA helper resolves state ambient-lessly with 2+ backends live.");

    /// <summary>Logs min/max/NaN/Inf of Chroma F16 block intermediates (D2H-draining) to locate F16 overflow sites.</summary>
    public static readonly Knob<bool> ChromaF16trace =
        Bool("diagnostics.chromaF16trace", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Logs min/max/NaN/Inf of Chroma F16 block intermediates (D2H-draining) to locate F16 overflow sites.");

    /// <summary>Adds per-block device syncs around the Ideogram 4 attention/MLP sublayers and accumulates their timings.</summary>
    public static readonly Knob<bool> DitProfile =
        Bool("diagnostics.ditProfile", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Adds per-block device syncs around the Ideogram 4 attention/MLP sublayers and accumulates their timings.");

    /// <summary>Enables sync-bracketed attn/qkv/ffn timers in F5-TTS DiT blocks plus a per-generation sample-loop timing line.</summary>
    public static readonly Knob<bool> F5Profile =
        Bool("diagnostics.f5Profile", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Enables sync-bracketed attn/qkv/ffn timers in F5-TTS DiT blocks plus a per-generation sample-loop timing line.");

    /// <summary>Directory for raw F32 per-stage .bin dumps of the FLite transformer's first forward, for the Python oracle.</summary>
    public static readonly Knob<string?> FliteDump =
        Str("diagnostics.fliteDump", null, KnobScope.Runtime, KnobDomain.Diagnostics, "Directory for raw F32 per-stage .bin dumps of the FLite transformer's first forward, for the Python oracle.");

    /// <summary>Logs per-stage absmax of the F-Lite transformer on the first forward, host-syncing at each probe.</summary>
    public static readonly Knob<bool> FliteProbe =
        Bool("diagnostics.fliteProbe", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Logs per-stage absmax of the F-Lite transformer on the first forward, host-syncing at each probe.");

    /// <summary>Re-enables Flux's per-tensor min/max/mean/NaN host scans, each a forced D2H sync in the denoise loop.</summary>
    public static readonly Knob<bool> FluxStats =
        Bool("diagnostics.fluxStats", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Re-enables Flux's per-tensor min/max/mean/NaN host scans, each a forced D2H sync in the denoise loop.");

    /// <summary>File path for a cuGraphDebugDotPrint dump of each captured CUDA graph; unset skips the dump.</summary>
    public static readonly Knob<string?> GraphDot =
        Str("diagnostics.graphDot", null, KnobScope.Runtime, KnobDomain.Diagnostics, "File path for a cuGraphDebugDotPrint dump of each captured CUDA graph; unset skips the dump.");

    /// <summary>Logs the captured CUDA graph's node-type histogram and any H2D cache miss that happens inside capture.</summary>
    public static readonly Knob<bool> GraphDump =
        Bool("diagnostics.graphDump", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Logs the captured CUDA graph's node-type histogram and any H2D cache miss that happens inside capture.");

    /// <summary>Logs which architecture predicate disqualified a model from CUDA-graph decode capture.</summary>
    public static readonly Knob<bool> GraphGateLog =
        Bool("diagnostics.graphGateLog", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Logs which architecture predicate disqualified a model from CUDA-graph decode capture.");

    /// <summary>Logs the first host-to-device weight-cache misses with shape, dtype and byte count.</summary>
    public static readonly Knob<bool> H2dTrace =
        Bool("diagnostics.h2dTrace", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Logs the first host-to-device weight-cache misses with shape, dtype and byte count.");

    /// <summary>Directory for raw per-stage tensor dumps from the MiniMax-H3 video denoise loop.</summary>
    public static readonly Knob<string?> H3Dump =
        Str("diagnostics.h3Dump", null, KnobScope.Runtime, KnobDomain.Diagnostics, "Directory for raw per-stage tensor dumps from the MiniMax-H3 video denoise loop.");

    /// <summary>Logs min/max/mean/rms of MiniMax-H3 tensors at each conditioning and denoise stage.</summary>
    public static readonly Knob<bool> H3Probe =
        Bool("diagnostics.h3Probe", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Logs min/max/mean/rms of MiniMax-H3 tensors at each conditioning and denoise stage.");

    /// <summary>Reports max|V| per MiniMax-H3 block against F16's 65504 ceiling via a host-side synchronizing scan.</summary>
    public static readonly Knob<bool> H3Vprobe =
        Bool("diagnostics.h3Vprobe", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Reports max|V| per MiniMax-H3 block against F16's 65504 ceiling via a host-side synchronizing scan.");

    /// <summary>Logs first-forward absmax/rms of HiDream residual streams (host sync) to judge F16-activation viability.</summary>
    public static readonly Knob<bool> HidreamProbe =
        Bool("diagnostics.hidreamProbe", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Logs first-forward absmax/rms of HiDream residual streams (host sync) to judge F16-activation viability.");

    /// <summary>Directory for raw F32 dumps of LTX-2 audio-stage tensors plus shape sidecars, for reference decoding.</summary>
    public static readonly Knob<string?> Ltx2AudioDump =
        Str("diagnostics.ltx2AudioDump", null, KnobScope.Runtime, KnobDomain.Diagnostics, "Directory for raw F32 dumps of LTX-2 audio-stage tensors plus shape sidecars, for reference decoding.");

    /// <summary>Logs absmax/rms/NaN stats for LTX-2 transformer blocks, VAE decode stages and pipeline stage outputs.</summary>
    public static readonly Knob<bool> Ltx2Probe =
        Bool("diagnostics.ltx2Probe", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Logs absmax/rms/NaN stats for LTX-2 transformer blocks, VAE decode stages and pipeline stage outputs.");

    /// <summary>Directory for raw f32 dumps of Mllama vision-encoder stage tensors; unset skips them.</summary>
    public static readonly Knob<string?> MllamaDump =
        Str("diagnostics.mllamaDump", null, KnobScope.Runtime, KnobDomain.Diagnostics, "Directory for raw f32 dumps of Mllama vision-encoder stage tensors; unset skips them.");

    /// <summary>Suppresses the CLI's inline half-block terminal image preview.</summary>
    public static readonly Knob<bool> NoImage =
        Bool("diagnostics.noImage", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Suppresses the CLI's inline half-block terminal image preview.");

    /// <summary>Sync-bracketed per-phase GPU timers (SDPA, attention rest, MLP, mod/norm) for the Oasis DiT eager path.</summary>
    public static readonly Knob<bool> OasisPhase =
        Bool("diagnostics.oasisPhase", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Sync-bracketed per-phase GPU timers (SDPA, attention rest, MLP, mod/norm) for the Oasis DiT eager path.");

    /// <summary>Times the Orpheus TTS decode loop's logits/sample/forward phases with per-step device syncs.</summary>
    public static readonly Knob<bool> OrpheusProf =
        Bool("diagnostics.orpheusProf", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Times the Orpheus TTS decode loop's logits/sample/forward phases with per-step device syncs.");

    /// <summary>Enables the per-label CPU wall-time accumulator over every NvtxRange-wrapped op.</summary>
    public static readonly Knob<bool> Profile =
        Bool("diagnostics.profile", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Enables the per-label CPU wall-time accumulator over every NvtxRange-wrapped op.");

    /// <summary>Dumps the accumulated per-op profile table at every backend Sync (per generation) to diagnostics.profileOut.</summary>
    public static readonly Knob<bool> ProfileEach =
        Bool("diagnostics.profileEach", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Dumps the accumulated per-op profile table at every backend Sync (per generation) to diagnostics.profileOut.");

    /// <summary>Enables the sub-op NVTX/wall-time ranges (PushFine) nested inside already-profiled ops.</summary>
    public static readonly Knob<bool> ProfileFine =
        Bool("diagnostics.profileFine", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Enables the sub-op NVTX/wall-time ranges (PushFine) nested inside already-profiled ops.");

    /// <summary>Base file path for NVTX profile dumps, replacing the default /tmp/hartsy_profile.txt.</summary>
    public static readonly Knob<string?> ProfileOut =
        Str("diagnostics.profileOut", null, KnobScope.Runtime, KnobDomain.Diagnostics, "Base file path for NVTX profile dumps, replacing the default /tmp/hartsy_profile.txt.");

    /// <summary>Splits selected NVTX/profiler op labels by tensor shape so one label's total breaks down by call regime.</summary>
    public static readonly Knob<bool> ProfileShapes =
        Bool("diagnostics.profileShapes", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Splits selected NVTX/profiler op labels by tensor shape so one label's total breaks down by call regime.");

    /// <summary>Syncs the compute stream at each NVTX range Dispose so per-op timing is GPU time; serializes the pipeline.</summary>
    public static readonly Knob<bool> ProfileSync =
        Bool("diagnostics.profileSync", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Syncs the compute stream at each NVTX range Dispose so per-op timing is GPU time; serializes the pipeline.");

    /// <summary>Logs inpaint mask-blend agreement stats per step in the SDXL pipeline (host scan of the latent).</summary>
    public static readonly Knob<bool> SdxlDebug =
        Bool("diagnostics.sdxlDebug", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Logs inpaint mask-blend agreement stats per step in the SDXL pipeline (host scan of the latent).");

    /// <summary>Logs per-layer absmax of the Llama-style text encoder's hidden stream, forcing a host sync per layer.</summary>
    public static readonly Knob<bool> TeProbe =
        Bool("diagnostics.teProbe", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Logs per-layer absmax of the Llama-style text encoder's hidden stream, forcing a host sync per layer.");

    /// <summary>Forces the CLI's assumed terminal background (light/dark) instead of sniffing COLORFGBG.</summary>
    public static readonly Knob<string?> Theme =
        Str("diagnostics.theme", null, KnobScope.Runtime, KnobDomain.Diagnostics, "Forces the CLI's assumed terminal background (light/dark) instead of sniffing COLORFGBG.");

    /// <summary>Enables per-tile/per-image min/max/mean host scans in the VAE decoder (black-tile bring-up instrumentation).</summary>
    public static readonly Knob<bool> VaeStats =
        Bool("diagnostics.vaeStats", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Enables per-tile/per-image min/max/mean host scans in the VAE decoder (black-tile bring-up instrumentation).");

    /// <summary>Logs per-stage min/max/mean/NaN stats through the NormalBAE and UperNet segmentation forwards.</summary>
    public static readonly Knob<bool> VisionProbe =
        Bool("diagnostics.visionProbe", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Logs per-stage min/max/mean/NaN stats through the NormalBAE and UperNet segmentation forwards.");

    /// <summary>Prints mean/maxabs stats for VLM vision-encoder stages and the spliced image embeddings to stderr.</summary>
    public static readonly Knob<bool> VlmDebug =
        Bool("diagnostics.vlmDebug", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Prints mean/maxabs stats for VLM vision-encoder stages and the spliced image embeddings to stderr.");

    /// <summary>Directory for raw f32 per-stage dumps of the Qwen2.5-VL and LLaVA-NeXT vision encoders.</summary>
    public static readonly Knob<string?> VlmDump =
        Str("diagnostics.vlmDump", null, KnobScope.Runtime, KnobDomain.Diagnostics, "Directory for raw f32 per-stage dumps of the Qwen2.5-VL and LLaVA-NeXT vision encoders.");

    /// <summary>Logs per-step velocity/latent stats and free-VRAM for the Wan video and Wan-S2V pipelines.</summary>
    public static readonly Knob<bool> WanDebug =
        Bool("diagnostics.wanDebug", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Logs per-step velocity/latent stats and free-VRAM for the Wan video and Wan-S2V pipelines.");

    /// <summary>File path to append per-block free-VRAM readings to during the Wan video transformer forward.</summary>
    public static readonly Knob<string?> WanVram =
        Str("diagnostics.wanVram", null, KnobScope.Runtime, KnobDomain.Diagnostics, "File path to append per-block free-VRAM readings to during the Wan video transformer forward.");

    /// <summary>Directory for a safetensors dump of every YuE decode-stage intermediate, for Python reference parity.</summary>
    public static readonly Knob<string?> YueDump =
        Str("diagnostics.yueDump", null, KnobScope.Runtime, KnobDomain.Diagnostics, "Directory for a safetensors dump of every YuE decode-stage intermediate, for Python reference parity.");

    /// <summary>Accumulates proj+D2H / host-argmax / feed-queue timings across YuE stage-2 residual decode frames.</summary>
    public static readonly Knob<bool> YueProfile =
        Bool("diagnostics.yueProfile", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Accumulates proj+D2H / host-argmax / feed-queue timings across YuE stage-2 residual decode frames.");

    /// <summary>Directory for raw F32 per-stage dumps of ZetaChroma's first (step-0 cond) forward, for the torch oracle.</summary>
    public static readonly Knob<string?> ZetaDump =
        Str("diagnostics.zetaDump", null, KnobScope.Runtime, KnobDomain.Diagnostics, "Directory for raw F32 per-stage dumps of ZetaChroma's first (step-0 cond) forward, for the torch oracle.");

    /// <summary>Logs min/max/NaN of every Z-Image block intermediate (first ~300 forwards) to locate F16 overflow.</summary>
    public static readonly Knob<bool> ZimageF16trace =
        Bool("diagnostics.zimageF16trace", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Logs min/max/NaN of every Z-Image block intermediate (first ~300 forwards) to locate F16 overflow.");

    /// <summary>Enables Z-Image's host-side prediction scans before the CFG combine to localize non-finite Base outputs.</summary>
    public static readonly Knob<bool> ZimagePredStats =
        Bool("diagnostics.zimagePredStats", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Enables Z-Image's host-side prediction scans before the CFG combine to localize non-finite Base outputs.");

    /// <summary>Enables per-channel latent/VAE min/max/mean host scans in the Z-Image pipeline.</summary>
    public static readonly Knob<bool> ZimageStats =
        Bool("diagnostics.zimageStats", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Enables per-channel latent/VAE min/max/mean host scans in the Z-Image pipeline.");

    /// <summary>Logs ZipVoice frame/token alignment counts and the phonemized reference text before decoding.</summary>
    public static readonly Knob<bool> ZipvoiceDebug =
        Bool("diagnostics.zipvoiceDebug", false, KnobScope.Runtime, KnobDomain.Diagnostics, "Logs ZipVoice frame/token alignment counts and the phonemized reference text before decoding.");

}
