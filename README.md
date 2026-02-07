# autoeq-c

Fast parametric EQ fitter in C.

This repo contains:

- A WebAssembly build (Emscripten) intended to be called from TypeScript/JavaScript
- A native C implementation (for CLI / embedding)

## WASM build/usage

Requirements:

- Emscripten (`emcc`) available in `PATH` (via `emsdk`)

Build:

- `./build`

This produces a single-file Emscripten module `autoeq.js` in the current folder.

The intended consumer is the TypeScript wrapper in [example/autoeq.ts](example/autoeq.ts).

Three functions are provided:

- `make()`
- `run(inst, dst, src, config, smooth, steps, fs)`
- `interp(x, y)`

Notes:

- The frequency axis is fixed to 384 log-spaced points `X` from 20 Hz → 20 kHz. If your data isn’t already on the internal `X` grid, use the provide linear interpolation function `interp(x, y)` from `example/autoeq.ts`.
- `interp()` is linear interpolation. If you resample very dense data (e.g. raw FFT / measurement output) directly down to 384 points you will introduce aliasing artifacts. Prefer to bin/average onto a log-spaced grid and/or apply some smoothing before interpolating.
- The number of filters must be <= 32. Default configs are provided as `autoeq.CONFIGS.STANDARD(N)` and `autoeq.CONFIGS.PRECISE(N)`.

### TypeScript example ([example/test.ts](example/test.ts))

```ts
import * as autoeq from './autoeq';

const inst = await autoeq.make();

// suppose you have arbitrary measurement points
// resample them onto the internal 384-point grid `autoeq.X` before running

const x = [20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000];

// these are just example curves
const dstRaw = x.map(x => 6*Math.sin(x)); // target curve in dB at x
const srcRaw = x.map(x => 6*Math.cos(x)); // source curve in dB at x

const dst = autoeq.interp(x, dstRaw);
const src = autoeq.interp(x, srcRaw);

// autoeq src to dst with 8 filters and IE smoothing

const config = autoeq.CONFIGS.STANDARD(8);
const res = autoeq.run(inst, dst, src, config, autoeq.Smooth.IE);

if (res) {
    console.log(res.filters);
    console.log({ amp: res.amp, loss: res.loss, time: res.time });
}
```

Example output:

```
[
  {
    type: "LSC",
    f0: 23.711910247802734,
    gain: 7.024210453033447,
    q: 0.565746009349823,
  },
  /* ... */
  {
    type: "PK",
    f0: 9249.556640625,
    gain: 3.049382209777832,
    q: 1.3290164470672607,
  }
]
{
  amp: 1.4850640296936035,
  loss: 0.027285713702440262,
  time: 30.329625,
}
```

## Native build/usage

Build:

- `cc src/*.c -lm -o autoeq`

The native binary is a low-level tool meant to be called from a script or part of a larger program.

- `./autoeq [iox] <N> <?steps>`

Where:

- `iox`
    - `i` = In-ear tuned smoothing preset
    - `o` = Over-ear smoothing preset
    - `x` = No smoothing
- `N` = Number of filters to fit (<= 32)
- `steps` (optional) = Number of optimizer steps

Input (stdin):

- 384 `float32` values for `dst` followed by 384 `float32` values for `src`

Output (stdout):

- First line: `<amp> <max>`
  - `amp` is the fitted overall gain offset (includes mean removal)
  - `max` is the max dB of the fitted filter stack response
- Then `N` lines, one per filter:
  - `<type> <f0> <gain> <Q>`

There is a minimal Python example in [example/autoeq.py](example/autoeq.py) with `NumPy` and `subprocess` that demonstrates how to call the native binary.

## License

LGPL-3.0-or-later. See [LICENSE](LICENSE).
