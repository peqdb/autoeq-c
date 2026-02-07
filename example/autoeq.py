from typing import Literal
import numpy as np
import subprocess

K = 384
X0 = 20
X1 = 20_000

LX0 = np.log(X0)
LX1 = np.log(X1)

LX = np.linspace(LX0, LX1, K)
X = np.exp(LX)

def interp(x: np.ndarray, y: np.ndarray):
	return np.interp(LX, np.log(x), y)

def run_autoeq(dst: np.ndarray, src: np.ndarray, n: int, smooth: Literal['i', 'o', 'x'], steps = 3000):
	dst = np.asarray(dst, dtype=np.float32)
	src = np.asarray(src, dtype=np.float32)

	payload = dst.tobytes() + src.tobytes()

	p = subprocess.run(
		['./autoeq', smooth, str(n), str(steps)],
		input=payload,
		stdout=subprocess.PIPE,
		check=True,
	)

	lines = p.stdout.decode().strip().splitlines()
	amp, max_db = map(float, lines[0].split())

	filters = []
	for line in lines[1:]:
		filt_type, f0, gain, q = line.split()
		filters.append((filt_type, float(f0), float(gain), float(q)))

	return amp, max_db, filters


def main():
	x = np.array([20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000])

	dst_raw = 6*np.sin(x)
	src_raw = 6*np.cos(x)

	dst = interp(x, dst_raw)
	src = interp(x, src_raw)

	amp, max_db, filters = run_autoeq(dst, src, n=8, smooth='i')

	for filt_type, f0, gain, q in filters:
		print(f'type={filt_type}, {f0=}, {gain=}, {q=}')

	print(f'{amp=}, {max_db=}')

if __name__ == '__main__':
	main()
