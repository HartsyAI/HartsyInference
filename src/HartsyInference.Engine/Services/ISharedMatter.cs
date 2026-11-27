using HartsyInference.ThreeD.Pipelines.Stages;

namespace HartsyInference.Engine.Services;

/// <summary>The one matting model the 3D side keeps resident: the engine's input preparation and the pipelines that matte their own input all use it.</summary>
internal interface ISharedMatter : IForegroundMatter, IDisposable
{
    /// <summary>The checkpoint the resident model was read from; null while nothing is loaded.</summary>
    string? LoadedFrom { get; }
}
