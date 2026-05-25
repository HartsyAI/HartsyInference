using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.ModelHandler.SafeTensors;
using SharpInference.Tokenizers;

namespace SharpInference.Vision.Clip;

/// <summary>Loads a CLIP checkpoint (single <c>model.safetensors</c> file in HuggingFace's
/// <c>CLIPModel</c> layout) and constructs ready-to-use <see cref="ClipImageEncoder"/> +
/// <see cref="ClipTextEmbedder"/> instances. Owns the safetensors mmap and the tokenizer; disposes
/// them on <see cref="Dispose"/>.
/// <para>Expected key layout (HuggingFace <c>openai/clip-vit-large-patch14</c> conventions):</para>
/// <list type="bullet">
///   <item><c>text_model.embeddings.{token,position}_embedding.weight</c></item>
///   <item><c>text_model.encoder.layers.{i}.{self_attn,layer_norm1,layer_norm2,mlp}.*</c></item>
///   <item><c>text_model.final_layer_norm.{weight,bias}</c></item>
///   <item><c>text_projection.weight</c> — shape <c>[projDim, hiddenSize]</c></item>
///   <item><c>vision_model.embeddings.{patch,class,position}_embedding.weight</c></item>
///   <item><c>vision_model.pre_layrnorm.{weight,bias}</c> (note the OpenAI typo; both spellings accepted by <see cref="ClipVisionEncoder"/>)</item>
///   <item><c>vision_model.encoder.layers.{i}.*</c></item>
///   <item><c>vision_model.post_layernorm.{weight,bias}</c></item>
///   <item><c>visual_projection.weight</c> — shape <c>[projDim, hiddenSize]</c></item>
/// </list></summary>
public sealed class ClipModelLoader : IDisposable
{
    private SafeTensorsLoader? _loader;
    private Dictionary<string, Tensor>? _weights;
    private ClipTokenizer? _tokenizer;
    private ClipTextEncoder? _textEncoder;
    private ClipVisionEncoder? _visionEncoder;
    private int _disposed;

    /// <summary>Preset used to build the encoders. Determines hidden sizes, layer counts, projection dim, GELU variant.</summary>
    public ClipPreset Preset { get; }

    /// <summary>Loaded image encoder. Available after <see cref="LoadFromSingleFile"/> succeeds.</summary>
    public ClipImageEncoder ImageEncoder
    {
        get
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            return _visionEncoder is null
                ? throw new InvalidOperationException("Vision encoder not loaded. Call LoadFromSingleFile first.")
                : new ClipImageEncoder(_visionEncoder);
        }
    }

    /// <summary>Loaded text embedder. Available after <see cref="LoadFromSingleFile"/> succeeds.</summary>
    public ClipTextEmbedder TextEmbedder
    {
        get
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            if (_textEncoder is null || _tokenizer is null)
                throw new InvalidOperationException("Text encoder not loaded. Call LoadFromSingleFile first.");
            return new ClipTextEmbedder(_textEncoder, _tokenizer);
        }
    }

    /// <summary>Creates a loader for the given preset. Does not perform I/O — call <see cref="LoadFromSingleFile"/> to actually load weights.</summary>
    public ClipModelLoader(ClipPreset preset)
    {
        ArgumentNullException.ThrowIfNull(preset);
        Preset = preset;
    }

    /// <summary>Loads weights from a single safetensors file (HuggingFace <c>model.safetensors</c>
    /// layout). The optional <paramref name="tokenizer"/> override lets callers reuse an existing
    /// tokenizer instance; if null, the embedded OpenAI CLIP vocab/merges are used.</summary>
    /// <param name="safetensorsPath">Path to the <c>model.safetensors</c> file.</param>
    /// <param name="tokenizer">Optional tokenizer override. Pass null to use the embedded OpenAI vocab/merges (correct for CLIP-L/H/G — they all share the 49,408-token BPE).</param>
    public void LoadFromSingleFile(string safetensorsPath, ClipTokenizer? tokenizer = null)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (!File.Exists(safetensorsPath))
            throw new FileNotFoundException($"CLIP safetensors file not found: {safetensorsPath}", safetensorsPath);

        SafeTensorsLoader loader = new();
        loader.Load(safetensorsPath);
        Dictionary<string, Tensor> weights = loader.GetAllTensors();

        ClipTextEncoder textEncoder = new(Preset.TextConfig);
        textEncoder.LoadWeights(weights, prefix: "text_model");

        ClipVisionEncoder visionEncoder = new(Preset.VisionConfig);
        visionEncoder.LoadWeights(weights, prefix: "vision_model");

        _loader = loader;
        _weights = weights;
        _tokenizer = tokenizer ?? new ClipTokenizer();
        _textEncoder = textEncoder;
        _visionEncoder = visionEncoder;
    }

    /// <summary>Yields every weight tensor for GPU preload via <c>backend.PreloadWeights(loader.EnumerateWeights())</c>. Includes both encoders. Throws if not yet loaded.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (_textEncoder is null || _visionEncoder is null)
            throw new InvalidOperationException("Cannot enumerate weights before LoadFromSingleFile.");
        foreach (Tensor t in _textEncoder.EnumerateWeights()) yield return t;
        foreach (Tensor t in _visionEncoder.EnumerateWeights()) yield return t;
    }

    /// <summary>Releases the safetensors mmap, the tokenizer, and any F32-promoted weight copies
    /// the encoders created during loading. The mmap-backed tensors themselves are non-owning
    /// (created via <see cref="SafeTensorsLoader.GetTensor"/>) so disposing them is a no-op; the
    /// mmap goes away when the loader is disposed.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _tokenizer?.Dispose();
        _loader?.Dispose();
        _tokenizer = null;
        _textEncoder = null;
        _visionEncoder = null;
        _weights = null;
        _loader = null;
    }
}
