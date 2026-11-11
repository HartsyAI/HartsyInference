namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>Object-to-camera pose in the PyTorch3D convention: unit quaternion (w, x, y, z), translation and per-axis scale.</summary>
public readonly record struct SamPose(
    float QuatW, float QuatX, float QuatY, float QuatZ, float TranslationX, float TranslationY, float TranslationZ,
    float ScaleX, float ScaleY, float ScaleZ);
