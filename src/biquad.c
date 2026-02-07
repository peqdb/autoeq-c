/*
 * Copyright (C) 2026 PEQdB Inc.
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include "lib.h"

static Biquad pk(f32 A, f32 cos_w, f32 alpha)
{
	f32 rA = 1.f / A;

	return (Biquad){
		.b0 = A*alpha + 1.f,
		.db0_dA = alpha,
		.db0_dalpha = A,
		.db0_dcos = 0.f,

		.b1 = -2.f*cos_w,
		.db1_dA = 0.f,
		.db1_dcos = -2.,

		.b2 = -A*alpha + 1.f,
		.db2_dA = -alpha,
		.db2_dalpha = -A,
		.db2_dcos = 0.f,

		.a0 = (A + alpha)*rA,
		.da0_dA = -alpha*sq(rA),
		.da0_dalpha = rA,
		.da0_dcos = 0.f,

		.a1 = -2.f*cos_w,
		.da1_dA = 0.f,
		.da1_dcos = -2.f,

		.a2 = (A - alpha)*rA,
		.da2_dA = alpha*sq(rA),
		.da2_dalpha = -rA,
		.da2_dcos = 0.f,
	};
}

static Biquad lsc(f32 A, f32 cos_w, f32 alpha)
{
	f32 p1 = A + 1.f,
		m1 = A - 1.f,
		sqrt_A = sqrtf(A),
		k = 2*sqrt_A*alpha,
		dk_dA = alpha / sqrt_A,
		dk_dalpha = 2.f*sqrt_A;

	return (Biquad){
		.b0 = A*(-cos_w*m1 + k + p1),
		.db0_dA = A*dk_dA - A*cos_w + A - cos_w*m1 + k + p1,
		.db0_dalpha = A*dk_dalpha,
		.db0_dcos = -A*m1,

		.b1 = 2.f*A*(-cos_w*p1 + m1),
		.db1_dA = -2.f*A*cos_w + 2.f*A - 2.f*cos_w*p1 + 2.f*m1,
		.db1_dcos = -2.f*A*p1,

		.b2 = A*(-cos_w*m1 - k + p1),
		.db2_dA = -A*dk_dA - A*cos_w + A - cos_w*m1 - k + p1,
		.db2_dalpha = -A*dk_dalpha,
		.db2_dcos = -A*m1,

		.a0 = cos_w*m1 + k + p1,
		.da0_dA = dk_dA + cos_w + 1.f,
		.da0_dalpha = dk_dalpha,
		.da0_dcos = m1,

		.a1 = -2.f*cos_w*p1 - 2.f*m1,
		.da1_dA = -2.f*cos_w - 2.f,
		.da1_dcos = -2.f*p1,

		.a2 = cos_w*m1 - k + p1,
		.da2_dA = -dk_dA + cos_w + 1.f,
		.da2_dalpha = -dk_dalpha,
		.da2_dcos = m1,
	};
}

static Biquad hsc(f32 A, f32 cos_w, f32 alpha)
{
	f32 p1 = A + 1.f,
		m1 = A - 1.f,
		sqrt_A = sqrtf(A),
		k = 2*sqrt_A*alpha,
		dk_dA = alpha / sqrt_A,
		dk_dalpha = 2.f*sqrt_A;

	return (Biquad){
		.b0 = A*(cos_w*m1 + k + p1),
		.db0_dA = A*dk_dA + A*cos_w + A + cos_w*m1 + k + p1,
		.db0_dalpha = A*dk_dalpha,
		.db0_dcos = A*m1,

		.b1 = -2.f*A*(cos_w*p1 + m1),
		.db1_dA = -2.f*A*cos_w - 2.f*A - 2.f*cos_w*p1 - 2.f*m1,
		.db1_dcos = -2.f*A*p1,

		.b2 = A*(cos_w*m1 - k + p1),
		.db2_dA = -A*dk_dA + A*cos_w + A + cos_w*m1 - k + p1,
		.db2_dalpha = -A*dk_dalpha,
		.db2_dcos = A*m1,

		.a0 = -cos_w*m1 + k + p1,
		.da0_dA = dk_dA - cos_w + 1.f,
		.da0_dalpha = dk_dalpha,
		.da0_dcos = -m1,

		.a1 = -2.f*cos_w*p1 + 2.f*m1,
		.da1_dA = 2.f - 2.f*cos_w,
		.da1_dcos = -2.f*p1,

		.a2 = -cos_w*m1 - k + p1,
		.da2_dA = -dk_dA - cos_w + 1.f,
		.da2_dalpha = -dk_dalpha,
		.da2_dcos = -m1,
	};
}

const char *const TYPE_NAMES[] = {
	[PK ] = "PK",
	[LSC] = "LSC",
	[HSC] = "HSC",
};

const BiquadFn BIQUAD_FNS[] = {
	[PK ] = pk,
	[LSC] = lsc,
	[HSC] = hsc,
};

void spectrum(Type type, f32 f0, f32 gain, f32 Q, f32 fs, const f32 *restrict f, f32 *restrict y)
{
	f32 A = exp10(gain / 40.f),
		w0 = 2.f*(f32)M_PI/fs * f0,
		cos_w = cosf(w0),
		sin_w = sinf(w0),
		alpha = sin_w * .5f / Q;

	Biquad s = BIQUAD_FNS[type](A, cos_w, alpha);

	f32 b_x0 = sq(s.b0 + s.b1 + s.b2),
		b_x1 = -4.f*(s.b0*s.b1 + 4.f*s.b0*s.b2 + s.b1*s.b2),
		b_x2 = 16.f*s.b0*s.b2,
		a_x0 = sq(s.a0 + s.a1 + s.a2),
		a_x1 = -4.f*(s.a0*s.a1 + 4.f*s.a0*s.a2 + s.a1*s.a2),
		a_x2 = 16.f*s.a0*s.a2;

	FOR_K() {
		f32 phi = sq(sinf((f32)M_PI/fs * f[k]));

		f32 b_poly = b_x0 + phi*(b_x1 + phi*b_x2),
			a_poly = a_x0 + phi*(a_x1 + phi*a_x2);

		y[k] += 10.f*log10f(b_poly / a_poly);
	}
}

