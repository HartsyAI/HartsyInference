// chw_f32_to_hwc_u8: the last step of an image generation. Takes the VAE's [B,3,H,W] F32 output in
// [-1,1] and writes [H,W,3] u8 in [0,255], which is what an encoder wants.
//
//   v = clamp((x + 1) * 0.5, 0, 1)
//   byte = uint(v * 255 + 0.5)
//
// Matches IBackend.ChwF32ToHwcU8's reference exactly, including the +0.5 before the truncating cast —
// that is round-half-up, not round-to-even, and one pixel differing by one is a byte differing in the PNG.
//
// One invocation per output WORD, not per pixel. GLSL has no byte-addressable storage without an
// extension, so the destination is declared as uint and each invocation composes four bytes and stores
// them whole. Per-pixel invocations would instead have to merge into shared words, which means either
// atomics over a buffer somebody has to zero first or a read-modify-write race — and a pixel is three
// bytes, so pixels straddle word boundaries and no per-pixel scheme avoids that.
//
// The caller rounds the allocation up to a whole number of words for the same reason: the last word may
// cover bytes past the image, and those lanes write zero rather than reading out of bounds.
//
// Compile: glslc chw_f32_to_hwc_u8.comp.glsl -o chw_f32_to_hwc_u8_f32.spv

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
