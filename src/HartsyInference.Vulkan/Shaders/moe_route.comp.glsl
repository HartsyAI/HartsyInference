// moe_route: per-token router, one workgroup per token, matching MoeReference.Route.
//
// score (softmax | sigmoid | sqrt-softplus) of logits / divisor, selection score = score + bias (altBias when the
// token kind is non-zero), optional node-limited group masking, then k rounds of "largest selection score, lowest
// index wins a tie" over the not-yet-taken experts. The weights are the unbiased scores at the picks, renormalised
// by their sum plus epsilon and scaled. The reference's strict-greater scan never picks a -inf score, so when no
// candidate is left the lowest untaken index is taken; the reduction tracks that alongside the best.
//
// Shared memory: 1024 experts x 4 arrays + 512 groups x 2 + 3 reduction arrays, about 23 KB.
// WGSIZE must equal the dispatch's local_size_x.

#version 460

#define WGSIZE 256
#define MAX_EXPERTS 1024
#define MAX_GROUPS 512

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(set = 0, binding = 0) writeonly buffer TopkIdx_    { int topk_idx[]; };
layout(set = 0, binding = 1) writeonly buffer TopkWeight_ { float topk_weight[]; };
layout(set = 0, binding = 2) readonly buffer Logits_      { float logits[]; };
layout(set = 0, binding = 3) readonly buffer Bias_        { float bias_data[]; };
layout(set = 0, binding = 4) readonly buffer AltBias_     { float alt_bias_data[]; };
layout(set = 0, binding = 5) readonly buffer Kinds_       { int kinds_data[]; };

layout(push_constant) uniform Push {
    uint numExperts;
    uint topK;
    uint scoring;        // 0 softmax, 1 sigmoid, 2 sqrt-softplus
    uint groupCount;     // 0 or 1 = ungrouped
    uint groupsKept;
    float maskedValue;
    uint renormalize;
    float renormEps;
    float scale;
    float logitDivisor;
    uint hasBias;
    uint hasAltAndKinds;
} pc;

shared float sRaw[MAX_EXPERTS];
shared float sSel[MAX_EXPERTS];
shared uint sTaken[MAX_EXPERTS];
shared int sPick[MAX_EXPERTS];
shared float sGroupScore[MAX_GROUPS];
shared uint sKept[MAX_GROUPS];
shared float sRedVal[WGSIZE];
shared uint sRedIdx[WGSIZE];
shared uint sRedFirst[WGSIZE];
shared float sScalar;

const uint NONE = 0xFFFFFFFFu;

// exp accurate to about one ulp on every driver: the built-in is allowed a few ulp and on some hardware lowers to
// ex2(x * log2(e)), whose product error grows with |x|. Range-reduce by hand with a split ln2, then a degree-7 series.
float accExp(float x) {
    if (x > 88.0) return uintBitsToFloat(0x7F800000u);
    if (x < -87.0) return 0.0;
    float kf = floor(x * 1.44269504089 + 0.5);
    float r = fma(-kf, 0.693145751953125, x);
    r = fma(-kf, 1.42860677e-6, r);
    float p = 1.0 + r * (1.0 + r * 0.5 * (1.0 + r * (1.0 / 3.0) * (1.0 + r * 0.25 * (1.0 + r * 0.2
        * (1.0 + r * (1.0 / 6.0) * (1.0 + r * (1.0 / 7.0)))))));
    return ldexp(p, int(kf));
}

// softplus(x) = max(x, 0) + log1p(exp(-|x|)); log1p by the atanh series, which keeps its relative precision
// for tiny arguments where log(1 + t) cannot (the built-in log is only absolutely accurate near 1).
float softplusScalar(float x) {
    if (x > 20.0) return x;
    float t = accExp(-abs(x));
    float s = t / (2.0 + t);
    float q = s * s;
    float series = 1.0 + q * (1.0 / 3.0 + q * (1.0 / 5.0 + q * (1.0 / 7.0 + q * (1.0 / 9.0 + q * (1.0 / 11.0
        + q * (1.0 / 13.0 + q * (1.0 / 15.0 + q * (1.0 / 17.0))))))));
    return max(x, 0.0) + 2.0 * s * series;
}

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint token = gl_WorkGroupID.x;
    uint e = pc.numExperts;
    uint k = pc.topK;
    uint base = token * e;

    for (uint i = tid; i < e; i += WGSIZE) {
        float v = logits[base + i];
        sRaw[i] = (pc.logitDivisor == 1.0) ? v : v / pc.logitDivisor;
        sTaken[i] = 0u;
    }
    barrier();

    if (pc.scoring == 0u) {
        float m = -3.402823e38;
        for (uint i = tid; i < e; i += WGSIZE) m = max(m, sRaw[i]);
        sRedVal[tid] = m;
        barrier();
        for (uint s = WGSIZE / 2u; s > 0u; s >>= 1u) {
            if (tid < s) sRedVal[tid] = max(sRedVal[tid], sRedVal[tid + s]);
            barrier();
        }
        float mx = sRedVal[0];
        barrier();
        float sum = 0.0;
        for (uint i = tid; i < e; i += WGSIZE) {
            float v = accExp(sRaw[i] - mx);
            sRaw[i] = v;
            sum += v;
        }
        sRedVal[tid] = sum;
        barrier();
        for (uint s = WGSIZE / 2u; s > 0u; s >>= 1u) {
            if (tid < s) sRedVal[tid] += sRedVal[tid + s];
            barrier();
        }
        float total = sRedVal[0];
        barrier();
        for (uint i = tid; i < e; i += WGSIZE) sRaw[i] = sRaw[i] / total;
    } else if (pc.scoring == 1u) {
        for (uint i = tid; i < e; i += WGSIZE) sRaw[i] = 1.0 / (1.0 + accExp(-sRaw[i]));
    } else {
        for (uint i = tid; i < e; i += WGSIZE) sRaw[i] = sqrt(softplusScalar(sRaw[i]));
    }
    barrier();

    int kind = (pc.hasAltAndKinds != 0u) ? kinds_data[token] : 0;
    bool useAlt = kind != 0;
    bool useBias = useAlt || pc.hasBias != 0u;
    for (uint i = tid; i < e; i += WGSIZE) {
        float r = sRaw[i];
        sSel[i] = useBias ? (r + (useAlt ? alt_bias_data[i] : bias_data[i])) : r;
    }
    barrier();

    if (pc.groupCount > 1u) {
        uint groups = pc.groupCount;
        uint per = e / groups;
        for (uint g = tid; g < groups; g += WGSIZE) {
            float top1 = uintBitsToFloat(0xFF800000u);
            float top2 = top1;
            for (uint j = 0u; j < per; j++) {
                float v = sSel[g * per + j];
                if (v > top1) { top2 = top1; top1 = v; }
                else if (v > top2) top2 = v;
            }
            sGroupScore[g] = top1 + top2;
            sKept[g] = 0u;
        }
        barrier();
        if (tid == 0u) {
            for (uint r = 0u; r < pc.groupsKept; r++) {
                int bestG = -1;
                float bestV = uintBitsToFloat(0xFF800000u);
                for (uint g = 0u; g < groups; g++) {
                    if (sKept[g] == 0u && sGroupScore[g] > bestV) { bestV = sGroupScore[g]; bestG = int(g); }
                }
                if (bestG >= 0) sKept[bestG] = 1u;
            }
        }
        barrier();
        for (uint i = tid; i < e; i += WGSIZE) {
            if (sKept[i / per] == 0u) sSel[i] = pc.maskedValue;
        }
        barrier();
    }

    float negInf = uintBitsToFloat(0xFF800000u);
    for (uint j = 0u; j < k; j++) {
        float bestVal = negInf;
        uint bestIdx = NONE;
        uint first = NONE;
        for (uint i = tid; i < e; i += WGSIZE) {
            if (sTaken[i] != 0u) continue;
            if (first == NONE) first = i;
            float v = sSel[i];
            if (v > bestVal) { bestVal = v; bestIdx = i; }
        }
        sRedVal[tid] = bestVal;
        sRedIdx[tid] = bestIdx;
        sRedFirst[tid] = first;
        barrier();
        for (uint s = WGSIZE / 2u; s > 0u; s >>= 1u) {
            if (tid < s) {
                uint oi = sRedIdx[tid + s];
                if (oi != NONE) {
                    float ov = sRedVal[tid + s];
                    uint ci = sRedIdx[tid];
                    if (ci == NONE || ov > sRedVal[tid] || (ov == sRedVal[tid] && oi < ci)) {
                        sRedVal[tid] = ov;
                        sRedIdx[tid] = oi;
                    }
                }
                sRedFirst[tid] = min(sRedFirst[tid], sRedFirst[tid + s]);
            }
            barrier();
        }
        if (tid == 0u) {
            uint best = sRedIdx[0] != NONE ? sRedIdx[0] : sRedFirst[0];
            sTaken[best] = 1u;
            sPick[j] = int(best);
        }
        barrier();
    }

    if (tid == 0u) {
        float wsum = 0.0;
        for (uint j = 0u; j < k; j++) wsum += sRaw[sPick[j]];
        sScalar = wsum;
    }
    barrier();
    float wsum = sScalar;
    for (uint j = tid; j < k; j += WGSIZE) {
        float w = sRaw[sPick[j]];
        if (pc.renormalize != 0u) w /= wsum + pc.renormEps;
        w *= pc.scale;
        topk_idx[token * k + j] = sPick[j];
        topk_weight[token * k + j] = w;
    }
}
