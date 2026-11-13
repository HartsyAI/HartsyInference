namespace HartsyInference.Engine.Requests;

/// <summary>One 3D pipeline the engine holds resident.</summary>
/// <param name="Kind">The family: <c>image</c> (TripoSR, Hunyuan3D, TRELLIS), <c>trellis2</c>, <c>pixal3d</c>, <c>hairdeck</c>, <c>scene</c>, <c>rig</c>, <c>paint</c> or <c>part</c>; <c>matting</c> is the BiRefNet checkpoint shared by the input preparation and the TRELLIS.2 and Pixal3D pipelines.</param>
/// <param name="Name">The checkpoint it was loaded from, relative to the models root when it lies under it, else its file or folder name; never an absolute path. Pass it to <see cref="Services.IMeshLifecycleService.Free"/>.</param>
public sealed record MeshLoadedModel(string Kind, string Name);
