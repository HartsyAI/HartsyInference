// Minimal nvrtc-based CUDA→PTX compiler. Used where nvcc is not installed but libnvrtc is
// present (the runtime CUDA compiler ships in the driver/toolkit runtime libs). Declares the
// nvrtc API inline and dlopens libnvrtc so it needs no headers and no link-time CUDA toolkit.
//
// Build:  cc -O2 -o nvrtc_compile nvrtc_compile.c -ldl
// Usage:  LD_LIBRARY_PATH=~/.local/lib/cuda13 ./nvrtc_compile in.cu out.ptx compute_80
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

typedef int nvrtcResult;
typedef void* nvrtcProgram;

typedef nvrtcResult (*fn_create)(nvrtcProgram*, const char*, const char*, int, const char* const*, const char* const*);
typedef nvrtcResult (*fn_compile)(nvrtcProgram, int, const char* const*);
typedef nvrtcResult (*fn_ptxsize)(nvrtcProgram, size_t*);
typedef nvrtcResult (*fn_getptx)(nvrtcProgram, char*);
typedef nvrtcResult (*fn_logsize)(nvrtcProgram, size_t*);
typedef nvrtcResult (*fn_getlog)(nvrtcProgram, char*);
typedef const char* (*fn_errstr)(nvrtcResult);

static char* read_file(const char* path, size_t* outLen)
{
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(n + 1);
    fread(buf, 1, n, f);
    buf[n] = 0;
    fclose(f);
    if (outLen) *outLen = (size_t)n;
    return buf;
}

int main(int argc, char** argv)
{
    if (argc < 4) { fprintf(stderr, "usage: %s in.cu out.ptx compute_XX\n", argv[0]); return 1; }
    const char* srcPath = argv[1];
    const char* outPath = argv[2];
    const char* arch = argv[3];

    void* lib = dlopen("libnvrtc.so", RTLD_NOW);
    if (!lib) lib = dlopen("libnvrtc.so.13", RTLD_NOW);
    if (!lib) lib = dlopen("libnvrtc.so.12", RTLD_NOW);
    if (!lib) { fprintf(stderr, "dlopen libnvrtc failed: %s\n", dlerror()); return 3; }

    fn_create  nvrtcCreateProgram   = (fn_create)  dlsym(lib, "nvrtcCreateProgram");
    fn_compile nvrtcCompileProgram  = (fn_compile) dlsym(lib, "nvrtcCompileProgram");
    fn_ptxsize nvrtcGetPTXSize      = (fn_ptxsize) dlsym(lib, "nvrtcGetPTXSize");
    fn_getptx  nvrtcGetPTX          = (fn_getptx)  dlsym(lib, "nvrtcGetPTX");
    fn_logsize nvrtcGetProgramLogSize = (fn_logsize) dlsym(lib, "nvrtcGetProgramLogSize");
    fn_getlog  nvrtcGetProgramLog   = (fn_getlog)  dlsym(lib, "nvrtcGetProgramLog");
    fn_errstr  nvrtcGetErrorString  = (fn_errstr)  dlsym(lib, "nvrtcGetErrorString");

    size_t srcLen = 0;
    char* src = read_file(srcPath, &srcLen);

    nvrtcProgram prog;
    nvrtcResult r = nvrtcCreateProgram(&prog, src, srcPath, 0, NULL, NULL);
    if (r != 0) { fprintf(stderr, "create failed: %s\n", nvrtcGetErrorString(r)); return 4; }

    char archOpt[64];
    snprintf(archOpt, sizeof(archOpt), "--gpu-architecture=%s", arch);
    // Extra argv (argv[4..]) are include directories → passed as --include-path=<dir> (for <mma.h>, <cuda_fp16.h>,
    // etc. when the WMMA/tensor-core kernels need the CUDA headers under bare nvrtc). Backward compatible: none → arch-only.
    int nInc = argc - 4;
    int nOpts = 1 + (nInc > 0 ? nInc : 0);
    const char** opts = (const char**)malloc(sizeof(char*) * nOpts);
    opts[0] = archOpt;
    for (int i = 0; i < nInc; i++)
    {
        char* incOpt = (char*)malloc(strlen(argv[4 + i]) + 32);
        sprintf(incOpt, "--include-path=%s", argv[4 + i]);
        opts[1 + i] = incOpt;
    }
    r = nvrtcCompileProgram(prog, nOpts, opts);

    size_t logSize = 0;
    nvrtcGetProgramLogSize(prog, &logSize);
    if (logSize > 1)
    {
        char* log = (char*)malloc(logSize);
        nvrtcGetProgramLog(prog, log);
        fprintf(stderr, "%s\n", log);
        free(log);
    }
    if (r != 0) { fprintf(stderr, "compile failed: %s\n", nvrtcGetErrorString(r)); return 5; }

    size_t ptxSize = 0;
    nvrtcGetPTXSize(prog, &ptxSize);
    char* ptx = (char*)malloc(ptxSize);
    nvrtcGetPTX(prog, ptx);

    FILE* out = fopen(outPath, "wb");
    if (!out) { fprintf(stderr, "cannot write %s\n", outPath); return 6; }
    fwrite(ptx, 1, ptxSize - 1, out);
    fclose(out);
    fprintf(stderr, "wrote %s (%zu bytes)\n", outPath, ptxSize - 1);
    return 0;
}
