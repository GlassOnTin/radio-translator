// SPDX-License-Identifier: AGPL-3.0-or-later
/* Minimal receive-only SDRplay RSPdx IQ capture via the SDRplay API v3.
 * Writes raw interleaved int16 I/Q to a file. No transmit (RSPdx is RX-only).
 * Usage: sdrplay_cap <freq_hz> <fs_hz> <duration_s> <out.iq> [antenna A|B|C] [grdB]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "sdrplay_api.h"

static FILE *out = NULL;
static volatile int running = 1;
static unsigned long long total = 0, limit = 0;

static void stream_a(short *xi, short *xq, sdrplay_api_StreamCbParamsT *params,
                     unsigned int numSamples, unsigned int reset, void *ctx) {
    (void)params; (void)reset; (void)ctx;
    if (!running || !out) return;
    unsigned int n = numSamples;
    if (total + n > limit) n = (unsigned int)(limit - total);
    if (n == 0) { running = 0; return; }
    short *buf = (short *)malloc((size_t)n * 2 * sizeof(short));
    if (!buf) { running = 0; return; }
    for (unsigned int i = 0; i < n; i++) { buf[2*i] = xi[i]; buf[2*i+1] = xq[i]; }
    fwrite(buf, sizeof(short), (size_t)n * 2, out);
    free(buf);
    total += n;
    if (total >= limit) running = 0;
}

static void event_cb(sdrplay_api_EventT eventId, sdrplay_api_TunerSelectT tuner,
                     sdrplay_api_EventParamsT *params, void *ctx) {
    (void)tuner; (void)params; (void)ctx;
    if (eventId == sdrplay_api_DeviceRemoved || eventId == sdrplay_api_DeviceFailure) {
        fprintf(stderr, "device removed/failure event\n"); running = 0;
    }
}

#define CHK(call) do { sdrplay_api_ErrT e = (call); if (e != sdrplay_api_Success) { \
    fprintf(stderr, "%s failed: %s\n", #call, sdrplay_api_GetErrorString(e)); \
    sdrplay_api_Close(); return 1; } } while (0)

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s freq_hz fs_hz duration_s out.iq [A|B|C] [grdB]\n", argv[0]); return 1; }
    double freq = atof(argv[1]), fs = atof(argv[2]), dur = atof(argv[3]);
    const char *outfile = argv[4];
    int antenna = (argc > 5) ? (argv[5][0] - 'A') : 0;
    int grdB = (argc > 6) ? atoi(argv[6]) : 30;
    limit = (unsigned long long)(fs * dur);

    CHK(sdrplay_api_Open());
    float ver; sdrplay_api_ApiVersion(&ver);
    fprintf(stderr, "API version %.2f\n", ver);
    sdrplay_api_DeviceT devs[4]; unsigned int nd = 0;
    CHK(sdrplay_api_GetDevices(devs, &nd, 4));
    int idx = -1;
    for (unsigned int i = 0; i < nd; i++) {
        fprintf(stderr, "dev[%u] SerNo=%s hwVer=%d valid=%d\n", i, devs[i].SerNo, devs[i].hwVer, devs[i].valid);
        if (devs[i].valid && idx < 0) idx = (int)i;
    }
    if (idx < 0) { fprintf(stderr, "no valid device\n"); sdrplay_api_Close(); return 1; }
    CHK(sdrplay_api_SelectDevice(&devs[idx]));
    sdrplay_api_DebugEnable(devs[idx].dev, sdrplay_api_DbgLvl_Warning);
    sdrplay_api_DeviceParamsT *dp = NULL;
    CHK(sdrplay_api_GetDeviceParams(devs[idx].dev, &dp));

    dp->devParams->fsFreq.fsHz = fs;
    dp->devParams->mode = sdrplay_api_ISOCH;
    dp->devParams->rspDxParams.antennaSel = (sdrplay_api_RspDx_AntennaSelectT)antenna;
    dp->devParams->rspDxParams.biasTEnable = 0;
    dp->devParams->rspDxParams.hdrEnable = 0;
    dp->devParams->rspDxParams.rfNotchEnable = 0;
    dp->devParams->rspDxParams.rfDabNotchEnable = 0;
    dp->rxChannelA->tunerParams.rfFreq.rfHz = freq;
    dp->rxChannelA->tunerParams.bwType = sdrplay_api_BW_1_536;
    dp->rxChannelA->tunerParams.ifType = sdrplay_api_IF_Zero;
    dp->rxChannelA->tunerParams.gain.gRdB = grdB;
    dp->rxChannelA->tunerParams.gain.LNAstate = 0;
    dp->rxChannelA->ctrlParams.agc.enable = sdrplay_api_AGC_DISABLE;
    dp->rxChannelA->ctrlParams.dcOffset.DCenable = 1;
    dp->rxChannelA->ctrlParams.dcOffset.IQenable = 1;
    dp->rxChannelA->ctrlParams.decimation.enable = 0;

    out = fopen(outfile, "wb");
    if (!out) { fprintf(stderr, "cannot open %s\n", outfile); sdrplay_api_Close(); return 1; }
    sdrplay_api_CallbackFnsT cb; memset(&cb, 0, sizeof(cb));
    cb.StreamACbFn = stream_a; cb.EventCbFn = event_cb;
    CHK(sdrplay_api_Init(devs[idx].dev, &cb, NULL));
    fprintf(stderr, "capturing %.0f Hz fs=%.0f dur=%.1fs ant=%c grdB=%d -> %s\n",
            freq, fs, dur, 'A' + antenna, grdB, outfile);
    while (running) usleep(100000);
    fflush(out); fclose(out);
    sdrplay_api_Uninit(devs[idx].dev);
    sdrplay_api_ReleaseDevice(&devs[idx]);
    sdrplay_api_Close();
    fprintf(stderr, "captured %llu samples (%.2f s)\n", total, (double)total / fs);
    return 0;
}