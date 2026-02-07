/*
 * Copyright (C) 2026 PEQdB Inc.
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include "lib.h"

typedef struct {
	f32 width, height;
	i32 idx;
} Peak;

/* function adapted from scipy */
static Peak largest_peak(const f32 *restrict x, const f32 *restrict f, Lim lim)
{
	enum {H = K / 2};

	/* find_peaks */

	i32 peaks[H];

	i32 n = 0, i_max = K - 1;
	for (i32 i = 1; i < i_max; ++i) {
		if (f[i] < lim.lo || f[i] > lim.hi)
			continue;

		if (x[i - 1] >= x[i])
			continue;

		i32 i_ahead = i + 1;
		while (i_ahead < i_max && x[i_ahead] == x[i])
			++i_ahead;

		if (x[i_ahead] < x[i]) {
			i32 left_edge = i,
				right_edge = i_ahead - 1;

			peaks[n] = (left_edge + right_edge) / 2;

			++n;
			i = i_ahead;
		}
	}

	/* peak_prominences */

	f32 prominences[H];
	i32 left_bases[H],
		right_bases[H];

	for (i32 p = 0; p < n; ++p) {
		i32 peak = peaks[p],
			i_min = 0,
			i_max = K - 1;

		f32 x_peak = x[peak];

		left_bases[p] = peak;
		f32 left_min = x_peak;
		for (i32 i = peak; i_min <= i && x[i] <= x_peak; --i) {
			if (x[i] < left_min) {
				left_min = x[i];
				left_bases[p] = i;
			}
		}

		right_bases[p] = peak;
		f32 right_min = x_peak;
		for (i32 i = peak; i <= i_max && x[i] <= x_peak; ++i) {
			if (x[i] < right_min) {
				right_min = x[i];
				right_bases[p] = i;
			}
		}

		prominences[p] = x_peak - fmaxf(left_min, right_min);
	}

	/* peak_widths */

	Peak largest = {.idx = -1};
	f32 largest_size = 0.f;

	for (i32 p = 0; p < n; ++p) {
		i32 i_min = left_bases[p],
			i_max = right_bases[p],
			peak = peaks[p];

		f32 x_peak = x[peak],
			height = x_peak - .5f*prominences[p];

		i32 i = peak;
		while (i_min < i && height < x[i])
			--i;

		f32 left_ip = (f32)i;
		if (x[i] < height)
			left_ip += (height - x[i]) / (x[i + 1] - x[i]);

		i = peak;
		while (i < i_max && height < x[i])
			++i;

		f32 right_ip = (f32)i;
		if (x[i] < height)
			right_ip -= (height - x[i]) / (x[i - 1] - x[i]);

		f32 width = right_ip - left_ip,
			size  = width * x_peak;

		if (size > largest_size) {
			largest = (Peak){.idx = peak, .width = width, .height = x_peak};
			largest_size = size;
		}
	}

	return largest;
}

/* initializtation stategies adapted from jaakkopasanen/AutoEq */

static Filter init_pk(f32 *restrict y, const f32 *restrict f, f32 fs, Lim lim_f0, Lim lim_gain, Lim lim_q)
{
	(void)fs;

	f32 rect[K];

	FOR_K() rect[k] = fmaxf( y[k], 0.f);
	Peak peak = largest_peak(rect, f, lim_f0);

	FOR_K() rect[k] = fmaxf(-y[k], 0.f);
	Peak dip  = largest_peak(rect, f, lim_f0);

	Peak p = peak.width*peak.height > dip.width*dip.height ? peak : dip;

	f32 f0 = f[p.idx],
		gain = p.idx == peak.idx ? peak.height : -dip.height,
		bw = p.width * log2f(f[1] / f[0]),
	 	bw_exp2 = exp2f(bw),
		Q = sqrtf(bw_exp2) / (bw_exp2 - 1.f);

	limit(&gain, lim_gain);
	limit(&Q, lim_q);

	return (Filter){.f0 = f0, .gain = gain, .Q = Q};
}

static Filter init_lsc(f32 *restrict y, const f32 *restrict f, f32 fs, Lim lim_f0, Lim lim_gain, Lim lim_q)
{
	(void)lim_q;

	lim_f0.lo = fmaxf(lim_f0.lo, 40.);
	lim_f0.hi = fminf(lim_f0.hi, 10000.);

	f32 best = 0.f;
	i32 best_idx = -1;

	f32 a = 0.f;
	FOR_K() {
		a += y[k];

		f32 avg = fabsf(a / (f32)(k + 1));
		if (avg > best) {
			best = avg;
			best_idx = k;
		}
	}

	f32 f0 = f[best_idx],
		Q = M_SQRT1_2;

	limit(&f0, lim_f0);
	limit(&Q, lim_q);

	f32 w[K] = {0};
	spectrum(LSC, f0, 1.f, Q, fs, f, w);

	f32 p = 0.f, c = 0.f;
	FOR_K() {
		p += w[k] * y[k];
		c += w[k];
	}

	f32 gain = p / c;
	limit(&gain, lim_gain);

	return (Filter){.f0 = f0, .gain = gain, .Q = Q};
}

static Filter init_hsc(f32 *restrict y, const f32 *restrict f, f32 fs, Lim lim_f0, Lim lim_gain, Lim lim_q)
{
	(void)lim_q;

	lim_f0.lo = fmaxf(lim_f0.lo, 40.);
	lim_f0.hi = fminf(lim_f0.hi, 10000.);

	f32 best = 0.f;
	i32 best_idx = -1;

	f32 a = 0.f;
	FOR_K() {
		a += y[K - 1 - k];

		f32 avg = fabsf(a / (f32)(k + 1));
		if (avg > best) {
			best = avg;
			best_idx = K - 1 - k;
		}
	}

	f32 f0 = f[best_idx],
		Q = M_SQRT1_2;

	limit(&f0, lim_f0);
	limit(&Q, lim_q);

	f32 w[K] = {0};
	spectrum(HSC, f0, 1.f, Q, fs, f, w);

	f32 p = 0.f, c = 0.f;
	FOR_K() {
		p += w[k]*y[k];
		c += w[k];
	}

	f32 gain = p / c;
	limit(&gain, lim_gain);

	return (Filter){.f0 = f0, .gain = gain, .Q = Q};
}

const InitFn INIT_FNS[] = {
	[PK ] = init_pk ,
	[LSC] = init_lsc,
	[HSC] = init_hsc,
};
