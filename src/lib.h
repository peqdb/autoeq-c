/*
 * Copyright (C) 2026 PEQdB Inc.
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#pragma once

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <stdbool.h>

#define INFO(...) fprintf(stderr, __VA_ARGS__)

enum {
    MAX_N = 32,
	K     = 384,
};

typedef int32_t  i32;
typedef uint32_t u32;
typedef float   f32;
typedef double  f64;

#define FOR_K() for (i32 k = 0; k < K; ++k)
#define FOR_N() for (i32 n = 0; n < N; ++n)

typedef enum {
	PK,
	LSC,
	HSC,
    _TYPE_N,
	_TYPE_INVALID = -1,
} Type;

typedef struct {
	f32 b0, b1, b2, a0, a1, a2;

	f32 db0_dA, db0_dalpha, db0_dcos;
	f32 db1_dA, db1_dcos;
	f32 db2_dA, db2_dalpha, db2_dcos;
	f32 da0_dA, da0_dalpha, da0_dcos;
	f32 da1_dA, da1_dcos;
	f32 da2_dA, da2_dalpha, da2_dcos;
} Biquad;

typedef struct {
	f32 f0, gain, Q;
} Filter;

typedef struct {
	f32 lo, hi;
} Lim;

typedef Biquad (*BiquadFn)(f32 A, f32 cos_w, f32 alpha);
typedef Filter (*InitFn)(f32 *restrict y, const f32 *restrict f, f32 fs, Lim lim_f0, Lim lim_gain, Lim lim_q);

extern const char *const TYPE_NAMES[_TYPE_N];
extern const BiquadFn BIQUAD_FNS[_TYPE_N];
extern const InitFn INIT_FNS[_TYPE_N];

inline static f32 clip(f32 x, f32 lo, f32 hi)
{
	return fminf(fmaxf(x, lo), hi);
}

inline static f32 sq(f32 x)
{
	return x * x;
}

inline static f32 exp10(f32 x)
{
	return expf((f32)M_LN10 * x);
}

inline static bool limit(f32 *x, Lim lim)
{
	f32 orig = *x;
	*x = clip(*x, lim.lo, lim.hi);
	return *x != orig;
}

void spectrum(Type type, f32 f0, f32 gain, f32 Q, f32 fs, const f32 *restrict f, f32 *restrict y);
