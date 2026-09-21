// chw_f32_to_hwc_u8: VAE output [B,3,H,W] F32 in [-1,1] -> [H,W,3] u8 in [0,255].
//   byte = uint(clamp((x + 1) * 0.5, 0, 1) * 255 + 0.5)
// The +0.5 before the truncating cast is round-half-up, matching IBackend.ChwF32ToHwcU8; one pixel off by
// one is a byte off by one in the PNG.
//
// One invocation per output WORD, not per pixel: GLSL has no byte-addressable storage here, and a pixel is
// three bytes, so pixels straddle word boundaries and no per-pixel scheme avoids a read-modify-write race.
// The caller rounds the allocation up to whole words; lanes past the image write zero.

#version 460

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0)          buffer Out_ { uint  out_data[]; };
layout(set = 0, binding = 1) readonly buffer In_  { float in_data[];  };

layout(push_constant) uniform Push {
    uint pixels;    // height * width
    uint words;     // ceil(pixels * 3 / 4)
} pc;

void main() {
    uint word = gl_GlobalInvocationID.x;
    if (word >= pc.words) return;

    uint totalBytes = pc.pixels * 3u;
    uint packed = 0u;
    for (uint lane = 0u; lane < 4u; lane++) {
        uint byteIndex = word * 4u + lane;
        if (byteIndex >= totalBytes) {
            break;
        }
        uint pixel = byteIndex / 3u;
        uint channel = byteIndex - pixel * 3u;
        float v = (in_data[channel * pc.pixels + pixel] + 1.0) * 0.5;
        v = clamp(v, 0.0, 1.0);
        packed |= uint(v * 255.0 + 0.5) << (lane * 8u);
    }
    out_data[word] = packed;
}
