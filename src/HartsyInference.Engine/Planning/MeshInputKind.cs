namespace HartsyInference.Engine.Planning;

/// <summary>What a 3D model takes as its input.</summary>
public enum MeshInputKind
{
    /// <summary>One image.</summary>
    Image,

    /// <summary>One image whose alpha plane is the mask of the object to lift.</summary>
    ImageWithMask,

    /// <summary>A triangle mesh or scene (obj or glb).</summary>
    SceneMesh,

    /// <summary>A triangle mesh (obj or glb) and a reference image.</summary>
    SceneMeshAndImage,

    /// <summary>One image and a control signal: a point cloud, voxels, a bounding box or a skeleton.</summary>
    ImageAndControl,
}
