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
