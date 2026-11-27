namespace HartsyInference.Engine.Requests;

/// <summary>The coordinate frame a glb result is returned in.</summary>
public enum MeshFrame
{
    /// <summary>The frame the model produces, as today: its own scale and orientation.</summary>
    Native = 0,

    /// <summary>glTF convention: Y up, +Z to the front, the longest side of the bounding box one metre, centred on the origin. Applied as one root node of the scene, so the vertex data is untouched. A model whose native front direction is not verified keeps its front as it is and says so in <see cref="MeshResult.Warnings"/>.</summary>
    GltfYUpMetres = 1,
}
