#!/usr/bin/env python3
"""
Generate the Kairos StarVector icon.

The motif is the program itself: a planet point on a degree scale with Gann
rays radiating from it in all four directions, and a price curve crossing
through. Drawn at 8x and downsampled, because Pillow's shape primitives are
not anti-aliased and a 1:1 render produces visibly stepped diagonals - and
this icon is almost entirely diagonals.

Sizing decisions are driven by the small end. At 16 pixels a ray thinner than
one pixel disappears entirely and a fan of nine ratios turns into a grey
smudge, so the ray count and thickness are set for legibility at 16 and 32,
not for detail at 1024.
"""
from __future__ import annotations

import math
import os

from PIL import Image, ImageDraw, ImageFilter

SS = 8                      # supersample factor
S = 1024                    # final master size
W = S * SS

# Palette taken from the app's dark theme.
BG_TOP = (18, 24, 33)
BG_BOTTOM = (11, 15, 21)
RING = (58, 71, 88)
RAY = (79, 168, 224)        # the blue used for planetary traces
RAY_WARM = (242, 193, 78)   # the amber used for the primary marker
PLANET = (242, 193, 78)
PLANET_EDGE = (11, 16, 21)
CURVE = (61, 190, 134)      # the green price curve
NOW = (224, 82, 75)         # the red "now" line
TICK = (120, 134, 153)


def rounded_rect_mask(size: int, radius_frac: float = 0.22) -> Image.Image:
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    d.rounded_rectangle([0, 0, size - 1, size - 1],
                        radius=int(size * radius_frac), fill=255)
    return m


def vertical_gradient(size: int, top, bottom) -> Image.Image:
    g = Image.new("RGB", (1, size))
    px = g.load()
    for y in range(size):
        t = y / max(size - 1, 1)
        px[0, y] = (
            int(top[0] + (bottom[0] - top[0]) * t),
            int(top[1] + (bottom[1] - top[1]) * t),
            int(top[2] + (bottom[2] - top[2]) * t),
        )
    return g.resize((size, size), Image.Resampling.NEAREST)


def clip_ray(x0, y0, dx, dy, box):
    """Trim a ray to the drawing box, mirroring how the app clips to the band."""
    lo_x, lo_y, hi_x, hi_y = box
    t = float("inf")
    if dx > 0:
        t = min(t, (hi_x - x0) / dx)
    elif dx < 0:
        t = min(t, (lo_x - x0) / dx)
    if dy > 0:
        t = min(t, (hi_y - y0) / dy)
    elif dy < 0:
        t = min(t, (lo_y - y0) / dy)
    if not math.isfinite(t) or t <= 0:
        return None
    return (x0, y0, x0 + dx * t, y0 + dy * t)


def build_master() -> Image.Image:
    base = vertical_gradient(W, BG_TOP, BG_BOTTOM).convert("RGBA")

    pad = int(W * 0.085)
    box = (pad, pad, W - pad, W - pad)

    # --- degree ticks down the left edge -------------------------------
    ticks = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    td = ImageDraw.Draw(ticks)
    n_ticks = 7                       # 0, 30, 60 ... 180
    for i in range(n_ticks):
        y = box[1] + (box[3] - box[1]) * i / (n_ticks - 1)
        long_tick = (i % 2 == 0)
        length = W * (0.055 if long_tick else 0.032)
        td.line([pad, y, pad + length, y],
                fill=TICK + ((150,) if long_tick else (90,)),
                width=int(W * 0.006))
    base = Image.alpha_composite(base, ticks)

    # --- Gann rays, four quadrants from one origin ----------------------
    rays = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    rd = ImageDraw.Draw(rays)

    ox = box[0] + (box[2] - box[0]) * 0.36
    oy = box[1] + (box[3] - box[1]) * 0.44

    # Second interval point, ahead of the now line.
    ox2 = box[0] + (box[2] - box[0]) * 0.80
    oy2 = box[1] + (box[3] - box[1]) * 0.66

    ray_w = int(W * 0.010)

    # Secondary ratios first so the 1x1 draws over them.
    for r in (0.5, 2.0):
        for sx in (1, -1):
            for sy in (1, -1):
                seg = clip_ray(ox, oy, sx * 1.0, sy * r, box)
                if seg:
                    rd.line(seg, fill=RAY + (105,), width=int(ray_w * 0.85))

    # The 1x1 in gold, in all four quadrants. This is the reference line the
    # whole method is built on, so it is the one that has to survive at 16px.
    for sx in (1, -1):
        for sy in (1, -1):
            seg = clip_ray(ox, oy, sx * 1.0, sy * 1.0, box)
            if seg:
                rd.line(seg, fill=RAY_WARM + (240,), width=int(ray_w * 1.5))

    # A single crossing pair from the future point, which is the thing the
    # program is actually for. Kept to two rays so it reads rather than adds
    # to the lattice.
    for sy in (1, -1):
        seg = clip_ray(ox2, oy2, -1.0, sy * 1.0, box)
        if seg:
            rd.line(seg, fill=RAY + (185,), width=int(ray_w * 1.1))
    base = Image.alpha_composite(base, rays)

    # --- the "now" vertical ---------------------------------------------
    nowl = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    nd = ImageDraw.Draw(nowl)
    nx = box[0] + (box[2] - box[0]) * 0.68
    nd.line([nx, box[1], nx, box[3]], fill=NOW + (190,), width=int(W * 0.013))
    base = Image.alpha_composite(base, nowl)

    # --- price curve ----------------------------------------------------
    curve = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    cd = ImageDraw.Draw(curve)
    pts = []
    span = box[2] - box[0]
    height = box[3] - box[1]
    for i in range(240):
        t = i / 239.0
        x = box[0] + span * t
        # Rising left to right with a mid dip, and held in the lower band so
        # it does not run through the planet marker and muddle both.
        y = (box[1] + height * 0.88
             - height * 0.34 * t
             + height * 0.09 * math.sin(t * math.pi * 2.2)
             + height * 0.04 * math.sin(t * math.pi * 5.0) * (1 - t))
        pts.append((x, max(box[1], min(box[3], y))))
    cd.line(pts, fill=CURVE + (255,), width=int(W * 0.019), joint="curve")
    base = Image.alpha_composite(base, curve)

    # --- planet marker at the ray origin --------------------------------
    dot = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dot)
    rr = W * 0.072
    dd.ellipse([ox - rr, oy - rr, ox + rr, oy + rr],
               fill=PLANET + (255,),
               outline=PLANET_EDGE + (255,), width=int(W * 0.012))
    # the future interval point, origin of the backward rays
    rr2 = W * 0.048
    dd.ellipse([ox2 - rr2, oy2 - rr2, ox2 + rr2, oy2 + rr2],
               fill=RAY + (255,),
               outline=PLANET_EDGE + (255,), width=int(W * 0.010))
    base = Image.alpha_composite(base, dot)

    # --- border ring ----------------------------------------------------
    ring = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    gd = ImageDraw.Draw(ring)
    gd.rounded_rectangle([0, 0, W - 1, W - 1],
                         radius=int(W * 0.22),
                         outline=RING + (255,), width=int(W * 0.016))
    base = Image.alpha_composite(base, ring)

    # --- mask to the rounded square and downsample ----------------------
    base.putalpha(rounded_rect_mask(W))
    out = base.resize((S, S), Image.Resampling.LANCZOS)
    return out


def build_simple(size: int) -> Image.Image:
    """
    A purpose-drawn variant for 48 pixels and below.

    Downsampling the detailed master to 16 pixels produces a smudge, because
    ten thin diagonals and a set of tick marks cannot survive in 256 pixels
    of canvas no matter how good the filter is. So the small sizes are drawn
    from scratch with only what still carries meaning at that scale: the gold
    1x1 cross, the planet point at its centre, and the price curve. The tick
    marks, the secondary ratios, the now line and the second point are all
    dropped rather than blurred.
    """
    ss = 16
    w = size * ss
    base = vertical_gradient(w, BG_TOP, BG_BOTTOM).convert("RGBA")

    pad = int(w * 0.06)
    box = (pad, pad, w - pad, w - pad)
    cx = box[0] + (box[2] - box[0]) * 0.42
    cy = box[1] + (box[3] - box[1]) * 0.42

    rays = Image.new("RGBA", (w, w), (0, 0, 0, 0))
    rd = ImageDraw.Draw(rays)
    ray_w = max(int(w * 0.055), ss)
    for sx in (1, -1):
        for sy in (1, -1):
            seg = clip_ray(cx, cy, sx * 1.0, sy * 1.0, box)
            if seg:
                rd.line(seg, fill=RAY_WARM + (255,), width=ray_w)
    base = Image.alpha_composite(base, rays)

    curve = Image.new("RGBA", (w, w), (0, 0, 0, 0))
    cd = ImageDraw.Draw(curve)
    span = box[2] - box[0]
    height = box[3] - box[1]
    pts = []
    for i in range(120):
        tt = i / 119.0
        x = box[0] + span * tt
        y = (box[1] + height * 0.90
             - height * 0.30 * tt
             + height * 0.10 * math.sin(tt * math.pi * 2.0))
        pts.append((x, max(box[1], min(box[3], y))))
    cd.line(pts, fill=CURVE + (255,), width=max(int(w * 0.075), ss),
            joint="curve")
    base = Image.alpha_composite(base, curve)

    dot = Image.new("RGBA", (w, w), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dot)
    rr = w * 0.135
    dd.ellipse([cx - rr, cy - rr, cx + rr, cy + rr],
               fill=PLANET + (255,),
               outline=PLANET_EDGE + (255,), width=max(int(w * 0.030), ss))
    base = Image.alpha_composite(base, dot)

    base.putalpha(rounded_rect_mask(w, radius_frac=0.20))
    img = base.resize((size, size), Image.Resampling.LANCZOS)
    return img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=70,
                                              threshold=2))


def build_small(master: Image.Image, size: int) -> Image.Image:
    """
    Downsample with a light sharpen at the small end.

    Straight LANCZOS from 1024 to 16 leaves the diagonals mushy; a small
    unsharp pass restores enough edge for the rays to still read as lines.
    """
    img = master.resize((size, size), Image.Resampling.LANCZOS)
    if size <= 64:
        img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=90,
                                                threshold=2))
    return img


def main() -> int:
    out_dir = "/mnt/user-data/outputs"
    os.makedirs(out_dir, exist_ok=True)

    master = build_master()

    sizes = [1024, 512, 256, 128, 64, 48, 32, 24, 16]
    made = {}
    for s in sizes:
        if s == S:
            img = master
        elif s <= 48:
            img = build_simple(s)          # drawn, not downsampled
        else:
            img = build_small(master, s)
        path = os.path.join(out_dir, f"kairos_icon_{s}.png")
        img.save(path, "PNG")
        made[s] = img
        print(f"  wrote {path}")

    # Windows .ico with every size embedded, which is what the build needs.
    # Pillow's ICO writer downsamples one image for every entry, which would
    # throw away the purpose-drawn 16 and 32. append_images keeps each frame
    # exactly as generated.
    ico = os.path.join(out_dir, "icon.ico")
    frames = [made[s] for s in (128, 64, 48, 32, 24, 16)]
    made[256].save(ico, "ICO", append_images=frames,
                   sizes=[(s, s) for s in (256, 128, 64, 48, 32, 24, 16)])
    print(f"  wrote {ico}")

    # BMP has no alpha, so flatten onto the theme background rather than
    # letting the transparent corners come out white.
    flat = Image.new("RGB", (256, 256), BG_BOTTOM)
    flat.paste(made[256], (0, 0), made[256])
    bmp = os.path.join(out_dir, "kairos_icon_256.bmp")
    flat.save(bmp, "BMP")
    print(f"  wrote {bmp}")

    # A wide preview strip so the small sizes can be eyeballed together.
    strip_sizes = [256, 128, 64, 48, 32, 24, 16]
    pad = 16
    width = sum(strip_sizes) + pad * (len(strip_sizes) + 1)
    strip = Image.new("RGB", (width, 256 + pad * 2), (30, 34, 40))
    x = pad
    for s in strip_sizes:
        strip.paste(made[s], (x, pad + (256 - s) // 2), made[s])
        x += s + pad
    preview = os.path.join(out_dir, "kairos_icon_preview.png")
    strip.save(preview, "PNG")
    print(f"  wrote {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
