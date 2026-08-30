namespace HartsyInference.Engine.Planning;

/// <summary>Hash-only identities and published contracts for known H3 artifacts.</summary>
internal static class VideoProfileManifest
{
    private const string LightXUrl = "https://github.com/ModelTC/Minimax-H3-Turbo#model-specs";
    private const string PddUrl = "https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs";

    private static readonly IReadOnlyDictionary<string, VideoKnownArtifact> _byHash = BuildByHash();
    private static readonly IReadOnlyDictionary<string, VideoKnownArtifact> _byId =
        _byHash.Values.GroupBy(artifact => artifact.Id, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(group => group.Key, group => group.First(), StringComparer.OrdinalIgnoreCase);

    /// <summary>Looks up a lowercase or uppercase full-file hash.</summary>
    public static bool TryGetByHash(string sha256, out VideoKnownArtifact? artifact) =>
        _byHash.TryGetValue(sha256.ToLowerInvariant(), out artifact);

    /// <summary>Looks up a stable artifact/profile id.</summary>
    public static bool TryGetById(string id, out VideoKnownArtifact? artifact) => _byId.TryGetValue(id, out artifact);

    private static IReadOnlyDictionary<string, VideoKnownArtifact> BuildByHash()
    {
        VideoKnownArtifact[] artifacts =
        [
            Main("e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a",
                "minimax-h3-fl2va-base-int8-convrot", "MiniMax-H3 FL2VA int8 ConvRot", VideoTaskFamily.Fl2Va),
            Main("9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779",
                "minimax-h3-ref2va-base-int8-convrot", "MiniMax-H3 Ref2VA int8 ConvRot", VideoTaskFamily.Ref2Va),
            Main("12944c1f7791637e7de12208aef04da82bd26b95271b1b47d817364315ade993",
                "minimax-h3-fl2va-base-fp8-scaled", "MiniMax-H3 FL2VA pruned fp8 scaled", VideoTaskFamily.Fl2Va,
                provenanceUrl: "https://huggingface.co/Comfy-Org/MiniMax-H3"),
            Main("f86f2f79ebd2d76eb8eeb46091e83982e6ff51d255747e7b16e92834b392b8e9",
                "minimax-h3-ref2va-base-fp8-scaled", "MiniMax-H3 Ref2VA pruned fp8 scaled", VideoTaskFamily.Ref2Va,
                provenanceUrl: "https://huggingface.co/Comfy-Org/MiniMax-H3"),
            Main("9ad5c98b533894c122050d32804a14f49fca8edc16c52564a281cdc5825ac934",
                "minimax-h3-fl2va-pulpcut-turbo8", "PulpCut MiniMax-H3 FL2VA baked Turbo", VideoTaskFamily.Fl2Va,
                VideoAccelerationKind.Turbo, steps: 8, flowShift: 12f, audioFlowShift: 3f,
                provenanceUrl: "https://huggingface.co/PulpCut/MiniMax-H3-Turbo-INT8-ConvRot"),
            Main("e64cef63bc2785bcd72e6103c52aa78c6cd2c4f9870a7ce79675083fd65cf2e7",
                "minimax-h3-ref2va-pulpcut-turbo8", "PulpCut MiniMax-H3 Ref2VA baked Turbo", VideoTaskFamily.Ref2Va,
                VideoAccelerationKind.Turbo, steps: 8, flowShift: 12f, audioFlowShift: 3f,
                provenanceUrl: "https://huggingface.co/PulpCut/MiniMax-H3-Ref2VA-Turbo-INT8-ConvRot"),
            Main("e0441d26414f6e0c28f43d580e6cc56fad424da0fa4d261b698ca73188aa6332",
                "minimax-h3-hybrid8-dasiwa", "DaSiWa MiniMax-H3 Hybrid 8 Turbo", VideoTaskFamily.Hybrid,
                VideoAccelerationKind.Turbo, steps: 8, flowShift: 12f, audioFlowShift: 3f,
                provenanceUrl: "https://civitai.red/models/2877206?modelVersionId=3275408"),
            Main("497c0ff6377eb239d8b446c991a52a69e0817447d92c4525bd85ff8b449fcbaa",
                "minimax-h3-ref2va-zs05-int8", "Joey MiniMax-H3 Ref2VA ZS05 int8", VideoTaskFamily.Ref2Va,
                provenanceUrl: "https://huggingface.co/joeygambino/MiniMax-H3-x-Z-Image-native"),
            Main("7221ae65d78780354d51e5048d29728d9f1f8fb9baf50b1dd3df85f5101413d3",
                "minimax-h3-fast-vsa-comfysol64-v1", "Kijai MiniMax-H3 FastH3 VSA", VideoTaskFamily.T2Va,
                VideoAccelerationKind.Vsa, steps: 4, flowShift: 12f, audioFlowShift: 3f,
                attention: VideoAttentionKind.ComfySol64V1),

            Rejected("1dfe28c517a937fb9876f0975f224fd6e7ecb8744219f89bb8ba954403e10dc3",
                "minimax-h3-fl2va-pulpcut-input-major", "PulpCut FL2VA input-major ConvRot",
                "This input-major ConvRot layout is incompatible with Hartsy's output-major int8 linear contract."),
            Rejected("5ca6696fe1cd9a8f254594ac67ee541f151b2377735dea3557364bd868270463",
                "minimax-h3-ref2va-pulpcut-input-major", "PulpCut Ref2VA input-major ConvRot",
                "This input-major ConvRot layout is incompatible with Hartsy's output-major int8 linear contract."),

            Adapter("5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c",
                "minimax-h3-lightx-ref4", "LightX Ref2V Turbo 4-step", VideoTaskFamily.Ref2Va, 4, 12f, 3f,
                VideoReferenceSizing.MatchTarget, LightXUrl),
            Adapter("9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99",
                "minimax-h3-lightx-ref4", "LightX Ref2V Turbo 4-step (Diffusers)", VideoTaskFamily.Ref2Va,
                4, 12f, 3f, VideoReferenceSizing.MatchTarget, LightXUrl),
            Adapter("08cfe946033af7d27719b964b6e0a0e50c32138daabbd6ce4137e23df6bf9980",
                "minimax-h3-lightx-fl8-768p", "LightX FL2V Turbo 8-step 768p", VideoTaskFamily.Fl2Va, 8, 6f, 3f,
                VideoReferenceSizing.MatchTarget, LightXUrl, 1344, 768),
            Adapter("9b0efe3613b43a84e30febaa43af27432ea9d0711eac7bba904b2556b175f6d4",
                "minimax-h3-lightx-fl8-768p", "LightX FL2V Turbo 8-step 768p (Diffusers)",
                VideoTaskFamily.Fl2Va, 8, 6f, 3f, VideoReferenceSizing.MatchTarget, LightXUrl, 1344, 768),
            Adapter("2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e",
                "minimax-h3-lightx-fl8-544p", "LightX FL2V Turbo 8-step 544p", VideoTaskFamily.Fl2Va, 8, 6f, 3f,
                VideoReferenceSizing.MatchTarget, LightXUrl),
            Adapter("e16ac20824d6e6649b193806f8fb095639bd9946c97b1bb84b4248eab1cc807f",
                "minimax-h3-lightx-fl8-544p", "LightX FL2V Turbo 8-step 544p (Diffusers)",
                VideoTaskFamily.Fl2Va, 8, 6f, 3f, VideoReferenceSizing.MatchTarget, LightXUrl),
            Adapter("c396a9a06f58399e9df9754b18299818d84a2ddd371724ba48fe4a41221437dc",
                "minimax-h3-lightx-fl4-768p-v1.0", "LightX FL2V Turbo 4-step 768p v1.0", VideoTaskFamily.Fl2Va,
                4, 6f, 3f, VideoReferenceSizing.MatchTarget, LightXUrl, 1344, 768),
            Adapter("1bdabc2e9fce20b1db563b96bcf6e46adcad4c1964f423676436bf266cc7416c",
                "minimax-h3-lightx-fl4-768p-v1.0", "LightX FL2V Turbo 4-step 768p v1.0 (Diffusers)",
                VideoTaskFamily.Fl2Va, 4, 6f, 3f, VideoReferenceSizing.MatchTarget, LightXUrl, 1344, 768),
            Adapter("449d80f301ac571622c72e28b8fd72a4b3681b7a8df8a92f17c8f6ec43f56558",
                "minimax-h3-lightx-fl4-768p-v1.1", "LightX FL2V Turbo 4-step 768p v1.1", VideoTaskFamily.Fl2Va,
                4, 6f, 3f, VideoReferenceSizing.MatchTarget, LightXUrl, 1344, 768),
            Adapter("b5e25a59292d51bca3fc02b9a0b2284e11b4eb20921a9c5adc2db785956b8966",
                "minimax-h3-lightx-fl4-768p-v1.1", "LightX FL2V Turbo 4-step 768p v1.1 (Diffusers)",
                VideoTaskFamily.Fl2Va, 4, 6f, 3f, VideoReferenceSizing.MatchTarget, LightXUrl, 1344, 768),
            Adapter("5ff4a12c8b4599fec716e1b15a45e504e0d1129111896bdcde5ac4a15e395b29",
                "minimax-h3-lightx-fl4-diffusers-v0.1", "LightX FL2V Turbo 4-step Diffusers v0.1",
                VideoTaskFamily.Fl2Va, 4, 6f, 3f, VideoReferenceSizing.MatchTarget, LightXUrl),
            Adapter("0b29be7042d883970eb0c20774a9ba03d95669ed80a721bb4d21be8ea0d0a196",
                "minimax-h3-pdd-fl", "Official MiniMax-H3 PDD FL adapter", VideoTaskFamily.Fl2Va, 8, 12f, 3f,
                VideoReferenceSizing.Native, PddUrl, acceleration: VideoAccelerationKind.Pdd),
            Adapter("111c82e669f6e20e628228172edf39395f1a9fc3ad049793895e542c0f55b18c",
                "minimax-h3-pdd-ref", "Official MiniMax-H3 PDD Ref adapter", VideoTaskFamily.Ref2Va, 8, 12f, 3f,
                VideoReferenceSizing.Native, PddUrl, acceleration: VideoAccelerationKind.Pdd),

            Component("919a48acb525dc8fc70287fcd94ec1f5e5e289a77f1df14d01099c6ce204eb02",
                "minimax-h3-fun-controlnet-union", "MiniMax-H3 Fun ControlNet-Union", VideoProfileArtifactRole.ControlNet,
                "https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union"),
            Component("9bb2d96f218c76babd85e0611b85ca8fb330a90546c01a0005e8a58a59593410",
                "minimax-h3-video-vae-int8-convrot", "MiniMax-H3 video VAE int8 ConvRot",
                VideoProfileArtifactRole.VideoVae, "https://huggingface.co/Comfy-Org/MiniMax-H3"),
        ];
        return artifacts.ToDictionary(artifact => artifact.Sha256, StringComparer.OrdinalIgnoreCase);
    }

    private static VideoKnownArtifact Main(string hash, string id, string name, VideoTaskFamily task,
        VideoAccelerationKind acceleration = VideoAccelerationKind.None, int? steps = null, float? flowShift = null,
        float? audioFlowShift = null, string? provenanceUrl = null,
        VideoAttentionKind attention = VideoAttentionKind.Dense) =>
        new VideoKnownArtifact
        {
            Sha256 = hash,
            Id = id,
            DisplayName = name,
            Role = VideoProfileArtifactRole.Main,
            Task = task,
            Acceleration = acceleration,
            Attention = attention,
            Steps = steps,
            FlowShift = flowShift,
            AudioFlowShift = audioFlowShift,
            ProvenanceUrl = provenanceUrl,
        };

    private static VideoKnownArtifact Adapter(string hash, string id, string name, VideoTaskFamily task, int steps,
        float flowShift, float audioFlowShift, VideoReferenceSizing sizing, string provenanceUrl, int? width = null,
        int? height = null, VideoAccelerationKind acceleration = VideoAccelerationKind.Turbo) =>
        new VideoKnownArtifact
        {
            Sha256 = hash,
            Id = id,
            DisplayName = name,
            Role = VideoProfileArtifactRole.Adapter,
            Task = task,
            Acceleration = acceleration,
            Steps = steps,
            FlowShift = flowShift,
            AudioFlowShift = audioFlowShift,
            Width = width,
            Height = height,
            ReferenceSizing = sizing,
            ProvenanceUrl = provenanceUrl,
        };

    private static VideoKnownArtifact Rejected(string hash, string id, string name, string reason) =>
        new VideoKnownArtifact
        {
            Sha256 = hash,
            Id = id,
            DisplayName = name,
            Role = VideoProfileArtifactRole.Rejected,
            Task = VideoTaskFamily.Unknown,
            Acceleration = VideoAccelerationKind.None,
            RejectionReason = reason,
        };

    private static VideoKnownArtifact Component(string hash, string id, string name, VideoProfileArtifactRole role,
        string provenanceUrl) =>
        new VideoKnownArtifact
        {
            Sha256 = hash,
            Id = id,
            DisplayName = name,
            Role = role,
            Task = VideoTaskFamily.Unknown,
            Acceleration = VideoAccelerationKind.None,
            ProvenanceUrl = provenanceUrl,
        };
}
