import numpy as np, wave
from scipy.signal import fftconvolve, butter, sosfilt

SR = 48000; DUR = 15.0; N = int(SR * DUR)
L = np.zeros(N); R = np.zeros(N); VERB = np.zeros(N)
rs = np.random.default_rng(3)
BEAT = 0.5

def add(sig, t0, gain=1.0, pan=0.0, verb=0.0):
    i = int(t0 * SR); n = min(len(sig), N - i)
    if n <= 0: return
    s = sig[:n] * gain
    L[i:i+n] += s * np.sqrt(0.5 * (1 - pan)); R[i:i+n] += s * np.sqrt(0.5 * (1 + pan))
    VERB[i:i+n] += s * verb

def tt(d): return np.arange(int(d * SR)) / SR
def bp(x, lo, hi, order=2): return sosfilt(butter(order, [lo, hi], 'band', fs=SR, output='sos'), x)
def hp(x, f): return sosfilt(butter(2, f, 'high', fs=SR, output='sos'), x)
def lp(x, f): return sosfilt(butter(2, f, 'low', fs=SR, output='sos'), x)

def kick(d=0.5, f0=48, f1=170, k=32, dec=7):
    t = tt(d); f = f0 + f1 * np.exp(-t * k); ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * np.exp(-t * dec); s[:120] += rs.uniform(-1, 1, 120) * np.linspace(.6, 0, 120)
    return np.tanh(s * 1.6)
def hat(d=0.05): t = tt(d); return hp(rs.uniform(-1, 1, len(t)), 7500) * np.exp(-t * 90)
def clap(d=0.25):
    t = tt(d); n = bp(rs.uniform(-1, 1, len(t)), 900, 5000); e = np.exp(-t * 18)
    for o in (0.0, 0.011, 0.022): e += np.where(t >= o, np.exp(-(t - o) * 160), 0) * .6
    return n * e * .6
def boom(d=2.5):
    t = tt(d); s = kick(d, 36, 120, 14, 1.6) * 1.1
    s += lp(rs.uniform(-1, 1, len(t)), 900) * np.exp(-t * 5) * .8
    s += np.sin(2 * np.pi * 36 * t) * np.exp(-t * 1.4) * .7
    return np.tanh(s)
def blip(f, d=0.12, dec=40):
    t = tt(d); return (np.sin(2 * np.pi * f * t) + .3 * np.sin(4 * np.pi * f * t)) * np.exp(-t * dec) * np.minimum(1, t * 3000)
def pluck(f, d=0.6, dec=7, bright=6):
    t = tt(d); s = sum(np.sin(2 * np.pi * f * h * t) * np.exp(-t * dec * (1 + h * .6)) / h for h in range(1, bright + 1))
    return s * np.minimum(1, t * 2000)
def sweep_noise(d, f_from, f_to, q=1.8, curve=2.0, env='rise'):
    t = tt(d); x = rs.uniform(-1, 1, len(t)); u = t / d
    fc = f_from * (f_to / f_from) ** (u ** curve)
    # state variable filter (bandpass)
    y = np.zeros_like(x); low = band = 0.0
    fcoef = 2 * np.sin(np.pi * np.minimum(fc, SR / 6) / SR); damp = 1 / q
    for i in range(len(x)):
        high = x[i] - low - damp * band; band += fcoef[i] * high; low += fcoef[i] * band; y[i] = band
    e = u ** 2 if env == 'rise' else (np.sin(np.pi * u) ** 1.5 if env == 'swell' else (1 - u) ** 2)
    return y * e / (np.max(np.abs(y)) + 1e-9)
def glide(f0, f1, d, curve=None, dec=0.0):
    t = tt(d); u = t / d; c = curve(u) if curve else u
    f = f0 + (f1 - f0) * c; ph = 2 * np.pi * np.cumsum(f) / SR
    return (np.sin(ph) + .25 * np.sin(2 * ph)) * np.exp(-t * dec) * np.minimum(1, t * 800) * np.minimum(1, (d - t) * 400)

A = 55.0
def note(semi, base=A): return base * 2 ** (semi / 12)

# ---------- scene 1 ----------
add(blip(1760, .2, 30), 0.04, .35, 0, .5)
add(sweep_noise(.44, 300, 5000, env='swell'), 0.3, .35, 0, .3)
add(glide(220, 880, .42, lambda u: u * u), 0.3, .08, 0, .3)
add(kick(.6), 0.95, .9); add(clap(), 0.95, .5, 0, .4)
add(sweep_noise(.35, 2000, 7000, env='fall'), 0.95, .3, -.5)
for i in range(0, 46, 2): add(blip(3200 + (i % 5) * 180, .03, 180), 1.15 + i * .45 / 46, .12, (i / 46 - .5) * .6)
add(sweep_noise(.22, 800, 6000, env='rise'), 1.78, .5, .4)
add(glide(110, 220, .7, lambda u: np.sqrt(u)), 1.3, .06)

# ---------- scene 2: each principle has its sound ----------
for b in range(4): add(kick(), 2 + b * BEAT, .95)
add(boom(1.2), 2.0, .55, 0, .3)
add(kick(.4, 40, 90, 20, 9), 2.16, .7); add(clap(.2), 2.16, .35)
add(glide(300, 900, .34, lambda u: 1 + 2.4 * (u - 1) ** 3 + 1.4 * (u - 1) ** 2, 4), 2.5, .12, 0, .4)
for i in range(6): add(blip(note(12 * 3 + [0, 3, 7, 10, 12, 15][i]), .09, 50), 3.0 + i * .025, .18, -.6 + i * .24, .4)
add(glide(160, 40, .45, lambda u: 1 - (1 - u) ** 4, 5), 3.5, .5); add(kick(.5, 36, 60, 10, 6), 3.5, .5)
add(sweep_noise(.25, 500, 8000), 3.75, .35, -.4)

# ---------- scene 3: groove + arpeggio following the ripple ----------
for b in range(5): add(kick(), 4 + b * BEAT, .95); add(clap(), 4.5 + b * BEAT * 2, .35, .1, .3) if b < 2 else None
add(boom(1.2), 4.0, .45, 0, .3)
for i in range(20): add(hat(), 4.0 + i * .125 + .0625 * (i % 2 == 0), .22 if i % 2 else .12, .35 if i % 2 else -.35)
arp = [0, 7, 12, 15, 19, 15, 12, 7]
for i in range(16): add(pluck(note(24 + arp[i % 8]), .35, 10, 5), 4.0 + i * .125, .1 * (1 - i / 22), [-.5, .5][i % 2], .5)
for i in range(4): add(pluck(note(12), .6, 5, 3), 4.25 + i * .5, .35)
add(sweep_noise(.42, 6000, 300, curve=.6, env='rise'), 6.05, .5, 0, .3)   # suck-in
add(glide(900, 120, .4, lambda u: u ** 2), 6.08, .08)

# ---------- scene 4: particles ----------
add(boom(2.2), 6.5, 1.0, 0, .4); add(sweep_noise(.9, 7000, 400, curve=.5, env='fall'), 6.5, .45, 0, .5)
for b in range(5): add(kick(.4), 7.0 + b * BEAT, .8)
for i in range(38):
    add(blip(note(48 + [0, 3, 7, 10, 14][i % 5]) * (1 + .003 * i), .25, 14), 6.6 + i * .04 + rs.uniform(0, .02), .045, rs.uniform(-.9, .9), .8)
for i in range(16): add(hat(.03), 7.0 + i * .125, .14, .5 if i % 2 else -.5)
add(sweep_noise(.6, 400, 9000, curve=1.4, env='rise'), 8.4, .6, -.2)
add(sweep_noise(.35, 9000, 1500, env='fall'), 8.8, .3, .8)

# ---------- scene 5: sonified easing curve ----------
add(kick(.6), 9.0, .9); add(boom(1.2), 9.0, .35, 0, .3)
add(pluck(note(36), .8, 5), 9.0, .35, -.4, .5); add(pluck(note(43), .8, 5), 9.12, .35, .4, .5)
add(blip(1320, .15, 25), 9.45, .2, -.3, .6); add(blip(1760, .15, 25), 9.47, .2, .3, .6)
bz = lambda u: 1 - (1 - u) ** 4.2
add(glide(note(36), note(60), .8, bz, .6), 9.75, .12, 0, .5)
for i in range(16):
    v = bz(i / 16); add(blip(note(36 + round(v * 24)), .06, 60), 9.75 + i * .05, .1, -.6 + v * 1.2, .4)
for b in range(4): add(kick(.4, 50, 100, 30, 12), 9.5 + b * BEAT, .45)
add(sweep_noise(.28, 800, 9000), 10.72, .45, .7)

# ---------- scene 6: loops ----------
add(boom(1.0), 11.0, .45, 0, .2)
for i in range(6): add(sweep_noise(.22, 3000, 700, env='fall'), 11.0 + i * .05, .18, [-.8, 0, .8, -.8, 0, .8][i])
for b in range(3): add(kick(), 11 + b * BEAT, .95)
for b in (1,): add(clap(), 11 + b * BEAT * 2 - .5, .35)
for i in range(12): add(hat(), 11.0 + i * .125, .2 if i % 2 else .1, .3 if i % 2 else -.3)
for i in range(12): add(pluck(note(24 + [0, 12, 7, 15][i % 4]), .3, 12, 4), 11 + i * .125, .09, .2, .4)
add(sweep_noise(.5, 250, 9000, curve=2.4, env='rise'), 12.5, .75, 0, .3)
add(glide(110, 880, .5, lambda u: u ** 3), 12.5, .1)
for i in range(8): add(kick(.2, 60, 80, 40, 20), 12.5 + i * .0625 * (1 - i * .06), .25 + i * .06)

# ---------- scene 7: end card ----------
add(boom(2.5), 13.0, 1.1, 0, .6); add(clap(.4), 13.0, .5, 0, .8)
for f, p in [(0, -.4), (7, .3), (12, -.1), (15, .5), (19, 0)]: add(pluck(note(24 + f), 1.8, 1.6, 6), 13.02 + p * .01, .09, p, .9)
add(pluck(note(48 + 7), .9, 4), 13.55, .22, .2, .8)  # the dot
add(blip(2637, .3, 12), 13.57, .1, .3, .9)

# pad (Am9) with sidechain pump on the groove sections
t = np.arange(N) / SR; pad = np.zeros(N)
for s in [0, 7, 10, 14, 15, 19]:
    f = note(12 + s)
    for dt in (-.07, .07): pad += np.sin(2 * np.pi * f * (1 + dt / 100) * t + rs.uniform(0, 6))
pad = lp(pad, 1400) * .05
pe = np.clip((t - .9) / .6, 0, 1) * np.clip((15 - t) / 1.9, 0, 1)
pump = np.ones(N)
for b in np.arange(2, 13, BEAT):
    i = int(b * SR); m = min(int(.4 * SR), N - i); pump[i:i+m] = np.minimum(pump[i:i+m], 1 - .8 * np.exp(-np.arange(m) / SR * 9))
pad *= pe * pump
L += pad; R += np.roll(pad, 300)

# reverb send
ir_t = np.arange(int(2.2 * SR)) / SR
irL = rs.normal(0, 1, len(ir_t)) * np.exp(-ir_t * 3.2); irR = rs.normal(0, 1, len(ir_t)) * np.exp(-ir_t * 3.2)
v = hp(VERB, 250); L += fftconvolve(v, irL)[:N] * .012; R += fftconvolve(v, irR)[:N] * .012

# master: glue, fade, normalize
mix = np.stack([L, R], 1); mix = np.tanh(mix * 1.1)
fade = np.clip((DUR - t) / .35, 0, 1)[:, None]; mix *= fade
mix *= 0.89 / np.max(np.abs(mix))
pcm = (mix * 32767).astype('<i2')
with wave.open('audio.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print('ok', np.max(np.abs(mix)), np.sqrt(np.mean(mix ** 2)))
