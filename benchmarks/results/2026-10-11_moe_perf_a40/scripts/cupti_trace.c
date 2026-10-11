// LD_PRELOAD kernel tracer: records every kernel (graph nodes included) with its GPU start/end timestamps to $CUPTI_TRACE_OUT as
// "start_ns,end_ns,gridX,gridY,gridZ,blockX,blockY,blockZ,graphNodeId,name" lines. Activity tracing needs no profiling-counter permission.
#include <cupti.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define BUF_SIZE (16 * 1024 * 1024)
static FILE *g_out;

static void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
    *size = BUF_SIZE;
    *buffer = (uint8_t *)aligned_alloc(8, BUF_SIZE);
    *maxNumRecords = 0;
}

static void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
    CUpti_Activity *record = NULL;
    while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
        if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL || record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            CUpti_ActivityKernel9 *k = (CUpti_ActivityKernel9 *)record;
            fprintf(g_out, "%llu,%llu,%d,%d,%d,%d,%d,%d,%llu,%s\n", (unsigned long long)k->start, (unsigned long long)k->end,
                    k->gridX, k->gridY, k->gridZ, k->blockX, k->blockY, k->blockZ, (unsigned long long)k->graphNodeId, k->name);
        }
    }
    free(buffer);
}

__attribute__((constructor)) static void trace_init(void)
{
    const char *path = getenv("CUPTI_TRACE_OUT");
    g_out = fopen(path ? path : "/tmp/cupti_trace.csv", "w");
    cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
}

__attribute__((destructor)) static void trace_fini(void)
{
    cuptiActivityFlushAll(1);
    if (g_out) fclose(g_out);
}
