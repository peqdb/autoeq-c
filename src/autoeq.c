/*
 * Copyright (C) 2026 PEQdB Inc.
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include "lib.h"

#ifdef WASM
#	include <emscripten/emscripten.h>
#	define EXPORT EMSCRIPTEN_KEEPALIVE
#else
#	define EXPORT
#endif

typedef struct {
	const Type *restrict types;
	const f32 *restrict phi,
			  *restrict r;
	f32 fs;
	i32 N;
	bool opt_amp;
} Consts;

/* 3n (lf0 gain bw) + 1 (amp) */
#define W_FROM_N(N) (3*(N) + 1)

enum {MAX_W = W_FROM_N(MAX_N)};

typedef struct {
	f32 v[MAX_W];
} Wrt;

#define FOR_W() for (i32 w = 0; w < W_FROM_N(N); ++w)

#define LF_AT(wrt, i)	(wrt)->v[0*N + (i)]
#define GAIN_AT(wrt, i) (wrt)->v[1*N + (i)]
#define BW_AT(wrt, i)	(wrt)->v[2*N + (i)]
#define AMP_AT(wrt)		(wrt)->v[3*N]

static f32 q_to_bw(f32 Q)
{
	return 2.f/(f32)M_LN2 * asinhf(.5f/Q);
}

static f32 bw_to_q(f32 bw)
{
	return .5f / sinhf(.5f*(f32)M_LN2 * bw);
}

static f32 grad(const Consts *restrict c, Wrt *restrict x, Wrt *restrict g)
{
	i32 N = c->N;
	f32 rK = 1.f / (f32)K;

	bool opt_amp = __builtin_expect(c->opt_amp, true);

	f32 dy_dw0[MAX_N][K],
		dy_dgain[MAX_N][K],
		dy_dbw[MAX_N][K];

	f32 w0_v[MAX_N];

	f32 pred[K],
		pred_init = opt_amp ? exp10(AMP_AT(x) / 10.f) : 1.f;

	FOR_K() pred[k] = pred_init;

	FOR_N() {
		/* forward */

		f32 f0   = expf(LF_AT(x, n)),
			gain = GAIN_AT(x, n),
			bw   = BW_AT(x, n);

		f32 A     = exp10(gain / 40.f),
			w0    = 2.f*(f32)M_PI/c->fs * f0,
			cos_w = cosf(w0),
			sin_w = sinf(w0),
			kQ    = sinhf(.5f*(f32)M_LN2 * bw),
			alpha = sin_w * kQ;

		w0_v[n] = w0;

		Biquad s = BIQUAD_FNS[c->types[n]](A, cos_w, alpha);

		f32 dA_dgain   =  A * (f32)M_LN10/40.f,
			dalpha_dw0 =  cos_w * kQ,
			dalpha_dbw =  sin_w * coshf(.5f*(f32)M_LN2 * bw) * .5f*(f32)M_LN2,
			dcos_dw0   = -sin_w;

		f32 b_x0 = sq(s.b0 + s.b1 + s.b2),
			b_x1 = -4.f*(s.b0*s.b1 + 4.f*s.b0*s.b2 + s.b1*s.b2),
			b_x2 = 16.f*s.b0*s.b2,
			a_x0 = sq(s.a0 + s.a1 + s.a2),
			a_x1 = -4.f*(s.a0*s.a1 + 4.f*s.a0*s.a2 + s.a1*s.a2),
			a_x2 = 16.f*s.a0*s.a2;

		f32 ba = s.b0 + s.b1 + s.b2,
			aa = s.a0 + s.a1 + s.a2;

		FOR_K() {
			f32 phi_k = c->phi[k];

			f32 b_poly = b_x0 + phi_k*(b_x1 + phi_k*b_x2),
				a_poly = a_x0 + phi_k*(a_x1 + phi_k*a_x2);

			pred[k] *= b_poly / a_poly;

			/* backward */

			f32 _8phi2 = 8.f*sq(phi_k), _2phi = 2.f*phi_k;

			f32 bm =  20.f/(f32)M_LN10 / b_poly,
				am = -20.f/(f32)M_LN10 / a_poly;

			f32 dy_db0 = bm * (ba - _2phi*(s.b1 + 4.f*s.b2) + _8phi2*s.b2),
				dy_db1 = bm * (ba - _2phi*(s.b0 + s.b2)),
				dy_db2 = bm * (ba - _2phi*(4.f*s.b0 + s.b1) + _8phi2*s.b0);

			f32 dy_da0 = am * (aa - _2phi*(s.a1 + 4.f*s.a2) + _8phi2*s.a2),
				dy_da1 = am * (aa - _2phi*(s.a0 + s.a2)),
				dy_da2 = am * (aa - _2phi*(4.f*s.a0 + s.a1) + _8phi2*s.a0);

			f32 dy_dA
				  = dy_db0*s.db0_dA
				  + dy_db1*s.db1_dA
				  + dy_db2*s.db2_dA
				  + dy_da0*s.da0_dA
				  + dy_da1*s.da1_dA
				  + dy_da2*s.da2_dA,
				dy_dalpha
				  = dy_db0*s.db0_dalpha
				  + dy_db2*s.db2_dalpha
				  + dy_da0*s.da0_dalpha
				  + dy_da2*s.da2_dalpha,
				dy_dcos
				  = dy_db0*s.db0_dcos
				  + dy_db1*s.db1_dcos
				  + dy_db2*s.db2_dcos
				  + dy_da0*s.da0_dcos
				  + dy_da1*s.da1_dcos
				  + dy_da2*s.da2_dcos;

			dy_dw0[n][k]   = dy_dalpha*dalpha_dw0 + dy_dcos*dcos_dw0;
			dy_dgain[n][k] = dy_dA*dA_dgain;
			dy_dbw[n][k]   = dy_dalpha*dalpha_dbw;
		}
	}

	f32 L = 0.f,
		dL_dy[K],
		dL_dy_sum = 0.f;

	FOR_K() {
		f32 d = 10.f*log10f(pred[k]) - c->r[k];
		L += sq(d);
		dL_dy_sum += dL_dy[k] = 2.f*d;
	}

	L *= rK;
	AMP_AT(g) = opt_amp ? dL_dy_sum * rK : 0.f;

	FOR_N() {
		f32 glf   = 0.f,
			ggain = 0.f,
			gbw   = 0.f;

		FOR_K() {
			glf   += dL_dy[k]*dy_dw0[n][k];
			ggain += dL_dy[k]*dy_dgain[n][k];
			gbw   += dL_dy[k]*dy_dbw[n][k];
		}

		LF_AT(g, n)   = glf * rK * w0_v[n];
		GAIN_AT(g, n) = ggain * rK;
		BW_AT(g, n)   = gbw * rK;
	}

	return L;
}

static f64 time_s(void)
{
	struct timespec ts;
	timespec_get(&ts, TIME_UTC);
	return (f64)ts.tv_sec + 1e-9*(f64)ts.tv_nsec;
}

/* AdaBelief optimizer
 * https://arxiv.org/abs/2010.07468 */

typedef struct {
	Wrt m, s;
	f32 b1, b2,
		b1t, b2t,
		eps, eps_root,
		lr;
	i32 N, step;
} AdaBelief;

static void make_adabelief(AdaBelief *s, i32 N)
{
	s->b1t = s->b1 = .9f;
	s->b2t = s->b2 = .99f;
	s->eps = 1e-12f;
	s->eps_root = 1e-12f;
	s->lr = 3e-2f;
	s->N = N;
	s->step = 0;

	FOR_W() s->m.v[w] = s->s.v[w] = 0.f;
}

static void adabelief_step(AdaBelief *restrict s, Wrt *restrict x, Wrt *restrict g)
{
	i32 N = s->N;

	FOR_W() {
		s->m.v[w] = s->b1*s->m.v[w] + (1.f - s->b1)*g->v[w];
		s->s.v[w] = s->b2*s->s.v[w] + (1.f - s->b2)*sq(g->v[w] - s->m.v[w]);

		f32 m_hat = s->m.v[w] / (1.f - s->b1t),
			s_hat = s->s.v[w] / (1.f - s->b2t);

		f32 den = sqrtf(s_hat + s->eps_root) + s->eps;

		x->v[w] -= s->lr * m_hat / den;
	}

	s->b1t *= s->b1;
	s->b2t *= s->b2;

	++s->step;
}

static f32 fit(
	i32 steps,
	const Type *restrict types,
	f32 *restrict f0,
	f32 *restrict gain,
	f32 *restrict Q,
	f32 *restrict amp,
	const Lim *restrict f0_lim,
	const Lim *restrict gain_lim,
	const Lim *Q_lim,
	i32 N,
	const f32 *restrict f,
	const f32 *restrict r,
	f32 fs)
{
	enum {LOG = 250};

	/* easy reparameterization that improves conditioning
	 * f -> log f
	 * q -> bw */

	Lim lf_lim[MAX_N],
		bw_lim[MAX_N];

	FOR_N() {
		lf_lim[n] = (Lim){
			.lo = logf(f0_lim[n].lo),
			.hi = logf(f0_lim[n].hi),
		};
		bw_lim[n] = (Lim){
			.lo = q_to_bw(Q_lim[n].hi),
			.hi = q_to_bw(Q_lim[n].lo),
		};
	}

	f32 phi[K];
	FOR_K() phi[k] = sq(sinf((f32)M_PI / fs * f[k]));

	Wrt x;

	FOR_N() {
		LF_AT(&x, n)   = logf(f0[n]);
		GAIN_AT(&x, n) = gain[n];
		BW_AT(&x, n)   = q_to_bw(Q[n]);
	}
	AMP_AT(&x) = amp ? *amp : 0.f;

	INFO("\n");
	if (amp)
		INFO("%12s: %34.2f dB\n", "P", *amp);
	FOR_N() {
		INFO("%12i: %8s %12.1f Hz %9.2f dB %10.3f Q\n",
			n + 1,
			TYPE_NAMES[types[n]],
			f0[n],
			gain[n],
			Q[n]);
	}
	INFO("\n");

	Wrt g, best;
	f32 L, best_L = 1e9f;

	f64 t0 = time_s();

	Consts c = {
		.types = types,
		.phi = phi,
		.r = r,
		.fs = fs,
		.N = N,
		.opt_amp = !!amp,
	};

	AdaBelief opt;
	make_adabelief(&opt, N);

	for (i32 step = 0; step < steps; ++step) {
		L = grad(&c, &x, &g);

		adabelief_step(&opt, &x, &g);

		/* box constraints via projection
		 * maybe worth experimenting around with barrier methods */
		FOR_N() {
			if (limit(&LF_AT(&x, n), lf_lim[n]))
				LF_AT(&opt.m, n) = 0.f;

			if (limit(&GAIN_AT(&x, n), gain_lim[n]))
				GAIN_AT(&opt.m, n) = 0.f;

			if (limit(&BW_AT(&x, n), bw_lim[n]))
				BW_AT(&opt.m, n) = 0.f;
		}

		if (L < best_L) {
			best_L = L;
			FOR_W() best.v[w] = x.v[w];
		}

		if (step % LOG == 0 && step) {
			f64 t1 = time_s();
			INFO("L: %.5f | %6i/%i [%6.0f it/s]\n", L, step, steps, (f32)LOG / (t1 - t0));
			t0 = t1;
		}
	}

	FOR_N() {
		f0[n]   = expf(LF_AT(&best, n));
		gain[n] = GAIN_AT(&best, n);
		Q[n]    = bw_to_q(BW_AT(&best, n));
	}

	if (amp)
		*amp = AMP_AT(&best);

	INFO("L: %.5f\n\n", best_L);
	if (amp)
		INFO("%12s: %34.2f dB\n", "P", *amp);
	FOR_N() {
		INFO("%12i: %8s %12.1f Hz %9.2f dB %10.3f Q\n",
			n + 1, TYPE_NAMES[types[n]], f0[n], gain[n], Q[n]);
	}
	INFO("\n");

	return L;
}

static i32 search(const f32 *x, i32 n, f32 v)
{
	i32 idx = -1;
	f32 best = 1e9f;
	for (i32 i = 0; i < n; ++i) {
		f32 d = fabsf(x[i] - v);
		if (d < best) {
			best = d;
			idx = i;
		}
	}
	return idx;
}

/* sigmoid-based step function */
static f32 sgm(f32 x, f32 x0, f32 x1)
{
	f32 SMOOTH = 4.f;

	f32 k = SMOOTH / (x1 - x0),
		m = .5f * (x0 + x1),
		y = k * (x - m);

	return .5f*tanhf(.5f*y) + .5f;
}

typedef struct {
	f32 smooth_f0, smooth_f1,

		smooth_lo,
		smooth_hi,

		bias_f0, bias_f1,
		bias_f2, bias_f3,

		bias_lo,
		bias_md,
		bias_hi,

		clip_f;
} Smooth;

/* progressively smooth data from 1kHz onwards using bilateral-ish filtering.
 * idea is that this prevents the optimizer from killing the 8 kHz resonant
 *  peak found in IEM measurements, and prevents drastic changes to the treble.
 * frequencies below 1 kHz are still smoothed a small amount which may perhaps
 *  help the optimizer out by reducing noise.
 * the method from jaakkopasanen/AutoEq averages the loss of the treble,
 *  however this method is essentially an extreme smoothing effect, which isn't
 *  really what you want; this tuned adaptive smoothing seems to perform
 *  much better */
static void adaptive_smooth(const Smooth *s, const f32 *restrict f, f32 *restrict r)
{
	/* currently designed around K=384 */
	_Static_assert(K == 384, "");

	enum {H = 48};

	f32 smooth_l0 = logf(s->smooth_f0),
		smooth_l1 = logf(s->smooth_f1),

		bias_l0   = logf(s->bias_f0),
		bias_l1   = logf(s->bias_f1),
		bias_l2   = logf(s->bias_f2),
		bias_l3   = logf(s->bias_f3);

	f32 x[K];
	memcpy(x, r, sizeof(*r) * K);

	i32 clip_idx = search(f, K, s->clip_f);

	FOR_K() {
		f32 f_k = f[k],
			l = logf(f_k),
			x_k = x[k],

			/* smoothing */
			sigma = s->smooth_lo + (s->smooth_hi - s->smooth_lo)*sgm(l, smooth_l0, smooth_l1),

			/* positive / negative bias */
			bias = s->bias_lo + (s->bias_md - s->bias_lo)*sgm(l, bias_l0, bias_l1)
				              + (s->bias_hi - s->bias_md)*sgm(l, bias_l2, bias_l3),

			a = 0.f,
			c = 0.f;

		for (i32 j = -H; j <= H; ++j) {
			i32 s = k + j;
			s = s < 0 ? 0
			  : s > clip_idx ? clip_idx
			  : s;

			f32 x_s = x[s],
				d_spatial = sq((f32)j * sigma),
				d_range   = bias * (x_s - x_k);

			f32 w = expf(-.5f*d_spatial + d_range);

			a += w * x[s];
			c += w;
		}

		r[k] = a / c;
	}
}

static f32 center_mean(f32 *x, i32 n)
{
	f32 sum = 0.f;
	for (i32 i = 0; i < n; ++i)
		sum += x[i];

	f32 mean = sum / (f32)n;
	for (i32 i = 0; i < n; ++i)
		x[i] -= mean;

	return mean;
}

/* roll-off the treble in the residual to stop the optimizer from wasting effort
 *  trying to compensate it (especially an issue due to using MSE) */
static void treble_rolloff(const f32 *restrict f, f32 *restrict r, f32 f_treble)
{
	i32 treble_idx = search(f, K, f_treble),
		n_treble = K - treble_idx;

	f32 inv = 1.f / (f32)(n_treble - 1);

	for (i32 i = 0; i < n_treble; ++i) {
		f32 t = (f32)i * inv,
			w = cosf(.5f * (f32)M_PI * t);
		r[treble_idx + i] *= w;
	}
}

EXPORT f32 preprocess(
	const f32 *restrict f, const f32 *restrict dst,
	const f32 *restrict src, f32 *restrict r, const Smooth *smooth, bool demean)
{
	f32 F_TREBLE_SMOOTH   = 16000.f,
		F_TREBLE_UNSMOOTH = 18500.f;

	f32 b[K];
	memcpy(b, src, sizeof(*src) * K);

	if (smooth)
		adaptive_smooth(smooth, f, b);

	FOR_K() r[k] = dst[k] - b[k];

	f32 mean = 0.f;
	if (demean)
		mean = center_mean(r, K);

	/* currently depends on dst, which isn't the best */
	treble_rolloff(f, r, smooth ? F_TREBLE_SMOOTH : F_TREBLE_UNSMOOTH);

	return mean;
}

EXPORT f32 autoeq(
	i32 steps,
	const Type *restrict types,
	f32 *restrict f0,
	f32 *restrict gain,
	f32 *restrict Q,
	f32 *restrict amp,
	const Lim *restrict f0_lim,
	const Lim *restrict gain_lim,
	const Lim *restrict Q_lim,
	i32 N,
	const f32 *restrict f,
	const f32 *restrict r,
	f32 fs)
{
	f32 r_init[K];
	memcpy(r_init, r, sizeof(*r) * K);

	FOR_N() {
		Type type = types[n];

		Filter p = INIT_FNS[type](r_init, f, fs, f0_lim[n], gain_lim[n], Q_lim[n]);
		spectrum(type, p.f0, -p.gain, p.Q, fs, f, r_init);

		f0[n]   = p.f0;
		gain[n] = p.gain;
		Q[n]    = p.Q;
	}

	if (amp)
		*amp = 0.f;

	return fit(steps, types, f0, gain, Q, amp, f0_lim, gain_lim, Q_lim, N, f, r, fs);
}

const Smooth *get_ie_smooth(void)
{
	static const Smooth IE_SMOOTH = {
		.smooth_lo = .3f,
		.smooth_hi = .03f,

		.smooth_f0 = 3000.f, .smooth_f1 = 12000.f,

		.bias_lo = .0f,
		.bias_md = .15f,
		.bias_hi = .03f,

		.bias_f0 = 10000.f, .bias_f1 = 13000.f,
		.bias_f2 = 14000.f, .bias_f3 = 20000.f,

		.clip_f = 18500.f,
	};

	return &IE_SMOOTH;
}

const Smooth *get_oe_smooth(void)
{
	static const Smooth OE_SMOOTH = {
		.smooth_lo = .3f,
		.smooth_hi = .03f,

		.smooth_f0 = 5000.f, .smooth_f1 = 15000.f,

		.bias_lo = .0f,
		.bias_md = .3f,
		.bias_hi = .2f,

		.bias_f0 = 6000.f, .bias_f1 =  9000.f,
		.bias_f2 = 9000.f, .bias_f3 = 20000.f,

		.clip_f = 17000.f,
	};

	return &OE_SMOOTH;
}

#ifndef WASM

typedef struct {
	Type type;
	Lim f0, gain, Q;
} Spec;

#define A(...) {__VA_ARGS__}
#define FILTERS(X) \
	X(LSC, A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 3.f)) \
	X(HSC, A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 3.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f)) \
	X(PK , A(20.f, 16000.f), A(-16.f, 16.f), A(.4f, 4.f))

#define X_TYPE(t, f, g, q) t,
#define   X_F0(t, f, g, q) f,
#define X_GAIN(t, f, g, q) g,
#define    X_Q(t, f, g, q) q,

const Type TYPES[]   = { FILTERS(X_TYPE) };
const Lim F0_LIM[]   = { FILTERS(X_F0) },
		  GAIN_LIM[] = { FILTERS(X_GAIN) },
		  Q_LIM[]    = { FILTERS(X_Q) };

_Static_assert(sizeof(TYPES)/sizeof(*TYPES) == MAX_N, "");

int main(int argc, char *argv[])
{
	f64 t0 = time_s();

	if (argc < 3) {
		INFO("usage: autoeq [iox] <N> <?steps>\n");
		return -1;
	}

	const Smooth *smooth;
	if (argv[1][0] == 'i') {
		smooth = get_ie_smooth();
	} else if (argv[1][0] == 'o') {
		smooth = get_oe_smooth();
	} else if (argv[1][0] == 'x') {
		smooth = NULL;
	} else {
		INFO("invalid type\n");
		return -1;
	}

	char *end;
	i32 N = (i32)strtol(argv[2], &end, 10);
	if (N <= 0 || N > MAX_N) {
		INFO("invalid count\n");
		return -1;
	}

	i32 steps = 3000;
	if (argc == 4) {
		steps = (i32)strtol(argv[3], &end, 10);
		if (steps <= 0) {
			INFO("invalid steps\n");
			return -1;
		}
	}

	const f32
		FS = 48000.f,

		F0 = 20.f,
		F1 = 20000.f,

		L0 = logf(F0),
		L1 = logf(F1),
		LR = L1 - L0;

	f32 f[K];
	FOR_K() f[k] = expf(L0 + LR/(f32)(K - 1)*(f32)k);

	f32 dst[K], src[K];

	u32 n_read = 0;
	while (n_read != K)
		n_read += fread(dst, sizeof(*dst), K - n_read, stdin);

	n_read = 0;
	while (n_read != K)
		n_read += fread(src, sizeof(*src), K - n_read, stdin);

	f32 gain[MAX_N], f0[MAX_N], Q[MAX_N], amp = 0.;

	f32 r[K];
	preprocess(f, dst, src, r, smooth, true);

	autoeq(steps, TYPES, f0, gain, Q, &amp, F0_LIM, GAIN_LIM, Q_LIM, N, f, r, FS);

	f32 y[K] = {0};
	FOR_N()
		spectrum(TYPES[n], f0[n], gain[n], Q[n], FS, f, y);

	f32 max = -1e9f;
	FOR_K() max = fmaxf(max, y[k]);

	printf("%.7e %.7e\n", amp, max);
	FOR_N() {
		printf("%s %.7e %.7e %.7e\n",
			TYPE_NAMES[TYPES[n]], f0[n], gain[n], Q[n]);
	}

	f64 t1 = time_s();

	INFO("\ntime: %.0f ms\n\n", 1000.*(t1 - t0));
}

#undef X_Q
#undef X_GAIN
#undef X_F0
#undef X_TYPE

#undef X_LIST
#undef A

#endif
