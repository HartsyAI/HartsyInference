using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection;

/// <summary>Common surface for any YOLO-family detection model — YOLOv8, YOLO11, and future
/// variants. Lets <see cref="YoloPipeline"/> hold a single reference regardless of which
/// architecture was loaded. Implementations are responsible for the full backbone + neck + head
/// forward pass; the pipeline owns only the preprocessor + post-processor.
/// <para>Not <see cref="IDisposable"/> — weight tensors are owned by the safetensors mmap that
/// outlives the model object. Disposing them is the loader's job.</para></summary>
public interface IYoloDetectModel
{
    /// <summary>The variant + scaling parameters this model was constructed with.</summary>
    YoloConfig Config { get; }

    /// <summary>Number of classes the head was built for (typically 80 for COCO).</summary>
    int NumClasses { get; }

    /// <summary>Per-detect-scale strides — always <c>[8, 16, 32]</c> for the YOLOv8/11 families.</summary>
    IReadOnlyList<int> Strides { get; }

    /// <summary>Loads BN-folded safetensors weights into the model.</summary>
    void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "model");

    /// <summary>Yields every weight tensor for GPU preload via <c>backend.PreloadWeights</c>.</summary>
    IEnumerable<Tensor> EnumerateWeights();

    /// <summary>Runs the full forward pass. Input is <c>[1, 3, H, W]</c> with H/W a multiple of 32;
    /// output is <c>[1, 4 + numClasses, totalAnchors]</c> with xywh boxes in input-pixel coords
    /// and sigmoid'd class probs. Caller owns the returned tensor.</summary>
    Tensor Forward(IBackend backend, Tensor input);
}
