#!/usr/bin/env python3
"""make_balloon.py — Generate print-ready gore patterns and visual previews
for a balloon of a given size, color, and logo.

Outputs (SVG, Affinity-Designer friendly):
  <prefix>_flat.svg            all gores laid out side by side (print layout)
  <prefix>_with_logo.svg       single-gore view demonstrating logo slicing
  <prefix>_preview.svg         per-gore slicing preview (logo cut across gores)
  <prefix>_assembled.svg       side-on preview of the assembled balloon

Usage:
  python3 make_balloon.py --size 200 --color yellow --logo love_you_logo.svg
  python3 make_balloon.py --size 150 --color red --gores 8 --logo love_you_logo.svg
"""
from __future__ import annotations

import argparse
import base64
import math
import os
import re
import sys
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Color name -> CMYK (process %) + display RGB (for on-screen previews).
# CMYK values are what you'd send to a print shop; RGB is just an
# approximation used when rendering the preview SVGs.
# ---------------------------------------------------------------------------
COLORS: dict[str, dict] = {
    "white":  {"cmyk": (0,   0,   0,   0),  "rgb": "#ffffff"},
    "black":  {"cmyk": (0,   0,   0,   100), "rgb": "#111111"},
    "yellow": {"cmyk": (0,   0,   100, 0),  "rgb": "#ffe600"},
    "gold":   {"cmyk": (0,   15,  100, 10), "rgb": "#e3b505"},
    "orange": {"cmyk": (0,   50,  100, 0),  "rgb": "#ff8c1a"},
    "red":    {"cmyk": (0,   100, 100, 0),  "rgb": "#e6001f"},
    "pink":   {"cmyk": (0,   60,  10,  0),  "rgb": "#ff7fb3"},
    "magenta":{"cmyk": (0,   100, 0,   0),  "rgb": "#d6008c"},
    "purple": {"cmyk": (60,  90,  0,   0),  "rgb": "#7a28c8"},
    "blue":   {"cmyk": (100, 70,  0,   0),  "rgb": "#1a46c8"},
    "sky":    {"cmyk": (70,  15,  0,   0),  "rgb": "#3fb8ff"},
    "teal":   {"cmyk": (90,  0,   40,  0),  "rgb": "#00a79d"},
    "green":  {"cmyk": (80,  0,   100, 0),  "rgb": "#23a03a"},
    "lime":   {"cmyk": (40,  0,   100, 0),  "rgb": "#a5d82a"},
    "brown":  {"cmyk": (20,  60,  80,  40), "rgb": "#6f4a1a"},
    "silver": {"cmyk": (0,   0,   0,   25), "rgb": "#c0c0c0"},
}


def resolve_color(name: str) -> dict:
    key = name.strip().lower()
    if key not in COLORS:
        raise ValueError(
            f"Unknown color '{name}'. Known: {', '.join(sorted(COLORS))}"
        )
    c = COLORS[key]
    return {"name": key, "cmyk": c["cmyk"], "rgb": c["rgb"]}


def cmyk_string(cmyk: tuple[int, int, int, int]) -> str:
    c, m, y, k = cmyk
    return f"C{c} M{m} Y{y} K{k}"


# ---------------------------------------------------------------------------
# Balloon / gore geometry.
#
# Model: a sphere of diameter D (cm) built from N identical lens-shaped
# "gores". When unrolled flat, a single gore has:
#     length  = pi * R                (pole to pole)
#     width(s)= (2*pi*R / N) * cos(s/R)
# where s is the arc-length from the equator (-pi*R/2 .. pi*R/2).
# ---------------------------------------------------------------------------


@dataclass
class Balloon:
    diameter_cm: float          # full balloon diameter
    n_gores: int                # number of gores
    seam_allowance_mm: float = 10.0  # extra mm added around each gore

    @property
    def radius_cm(self) -> float:
        return self.diameter_cm / 2.0

    @property
    def gore_length_cm(self) -> float:
        return math.pi * self.radius_cm

    @property
    def gore_max_half_width_cm(self) -> float:
        return math.pi * self.radius_cm / self.n_gores

    def half_width_at(self, s_cm: float) -> float:
        """Half-width of a gore at arc-length s from the equator (cm)."""
        R = self.radius_cm
        return (math.pi * R / self.n_gores) * math.cos(s_cm / R)


def gore_outline_points(bln: Balloon, n_samples: int = 101) -> list[tuple[float, float]]:
    """Return a closed polygon (in cm) for a single gore centered on (0,0)
    with the long axis along y."""
    half_len = bln.gore_length_cm / 2.0
    xs_right = []
    xs_left = []
    for i in range(n_samples):
        s = -half_len + (2 * half_len) * i / (n_samples - 1)
        hw = bln.half_width_at(s)
        # Clamp tiny negatives from float error at the poles.
        hw = max(hw, 0.0)
        xs_right.append((hw, s))
        xs_left.append((-hw, s))
    return xs_right + list(reversed(xs_left))


def svg_path_from_points(pts: list[tuple[float, float]]) -> str:
    if not pts:
        return ""
    out = [f"M {pts[0][0]:.3f} {pts[0][1]:.3f}"]
    for x, y in pts[1:]:
        out.append(f"L {x:.3f} {y:.3f}")
    out.append("Z")
    return " ".join(out)


# ---------------------------------------------------------------------------
# Logo handling.
#
# We treat the supplied logo SVG as a texture that is "painted" on the
# balloon surface. For a logo of (logo_w x logo_h) cm centered at the
# equator and longitude 0:
#
#   longitude(u) = u / R            (u is horizontal cm from logo center)
#   latitude (v) = v / R            (v is vertical cm from logo center)
#
# A gore k spans longitude in [2*pi*k/N - pi/N, 2*pi*k/N + pi/N].
# On the flat gore pattern (y along the arc length axis, x across) the
# horizontal coordinate of a logo pixel is:
#   gore_x = (longitude - 2*pi*k/N) * R * cos(latitude)   [arc-length width]
#          ~ u - k * (2*pi*R/N)  for small latitudes
# We use the simplified mapping (no cos(theta) compression) for the preview
# -- adequate for logos confined near the equator.
# ---------------------------------------------------------------------------


def read_logo_svg(path: str) -> tuple[str, float, float]:
    """Return (inner svg content minus outer <svg> tags, viewbox_w, viewbox_h)."""
    with open(path, "r") as f:
        text = f.read()
    m = re.search(r"<svg[^>]*viewBox=\"([^\"]+)\"", text)
    if m:
        vb = [float(x) for x in m.group(1).split()]
        vw, vh = vb[2], vb[3]
    else:
        wm = re.search(r"<svg[^>]*\bwidth=\"([0-9.]+)", text)
        hm = re.search(r"<svg[^>]*\bheight=\"([0-9.]+)", text)
        vw = float(wm.group(1)) if wm else 400.0
        vh = float(hm.group(1)) if hm else 400.0
    body = re.sub(r"^.*?<svg[^>]*>", "", text, count=1, flags=re.DOTALL)
    body = re.sub(r"</svg>\s*$", "", body)
    return body, vw, vh


def logo_as_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def _draw_warped_logo_on_gore(
    svg_lines: list,
    gore_index: int,
    n_gores: int,
    panel_cx: float,
    panel_cy: float,
    radius_mm: float,
    logo_w_mm: float,
    logo_h_mm: float,
    logo_uri: str,
    gore_clip_id: str,
    n_strips: int = 24,
) -> None:
    """Project (bake) the logo onto a spherical gore and render it inside
    the given gore clip shape.

    The logo is assumed to be painted on the sphere centered on gore 0 at
    the equator (longitude 0, latitude 0) using equirectangular mapping:
    logo pixel (u, v) <-> sphere (lon=u/R, lat=v/R). On the flat gore k:

        x_flat = (u - lon_k * R) * cos(v/R)
        y_flat = v

    so each horizontal row of the logo is horizontally compressed by
    cos(latitude) and shifted by -effective_offset * W * cos(latitude).
    We approximate the continuous warp by slicing the logo into
    ``n_strips`` horizontal bands and applying a constant scale per band.

    The gore index is remapped to a signed offset in [-N/2, +N/2] so that
    the logo correctly wraps across the seam between gore N-1 and gore 0
    instead of the far side of the balloon.
    """
    # Signed offset with wraparound: gores on the "left" side of gore 0
    # get negative offsets so the left edge of the logo lands on them.
    if gore_index <= n_gores // 2:
        effective = gore_index
    else:
        effective = gore_index - n_gores

    W_mm = 2.0 * math.pi * radius_mm / n_gores  # gore width at equator

    svg_lines.append(f'    <g clip-path="url(#{gore_clip_id})">\n')
    for j in range(n_strips):
        v_start = -logo_h_mm / 2.0 + j * logo_h_mm / n_strips
        v_end = v_start + logo_h_mm / n_strips
        v_mid = (v_start + v_end) / 2.0
        theta_mid = v_mid / radius_mm
        scale_x = math.cos(theta_mid)
        if scale_x <= 0:  # Past the pole -- nothing to draw for this strip.
            continue

        strip_id = f"strip_{gore_index}_{j}"
        svg_lines.append(
            f'      <clipPath id="{strip_id}" clipPathUnits="userSpaceOnUse">'
            f'<rect x="-50000" y="{panel_cy + v_start:.3f}" '
            f'width="100000" height="{(v_end - v_start):.3f}"/></clipPath>\n'
        )

        tx = panel_cx - effective * W_mm * scale_x
        ty = panel_cy
        svg_lines.append(
            f'      <g clip-path="url(#{strip_id})">\n'
            f'        <g transform="translate({tx:.3f},{ty:.3f}) scale({scale_x:.6f},1)">\n'
            f'          <image xlink:href="{logo_uri}" href="{logo_uri}" '
            f'x="{-logo_w_mm/2.0:.3f}" y="{-logo_h_mm/2.0:.3f}" '
            f'width="{logo_w_mm:.3f}" height="{logo_h_mm:.3f}" '
            f'preserveAspectRatio="none"/>\n'
            f'        </g>\n'
            f'      </g>\n'
        )
    svg_lines.append('    </g>\n')


# ---------------------------------------------------------------------------
# SVG generation helpers.
# ---------------------------------------------------------------------------

MM_PER_CM = 10.0


def svg_header(width_mm: float, height_mm: float, title: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"\n'
        f'     width="{width_mm:.2f}mm" height="{height_mm:.2f}mm"\n'
        f'     viewBox="0 0 {width_mm:.2f} {height_mm:.2f}">\n'
        f'  <title>{title}</title>\n'
    )


def svg_footer() -> str:
    return "</svg>\n"


def generate_flat_layout(
    bln: Balloon, color: dict, out_path: str
) -> None:
    """All N gores laid out side by side for printing (no logo)."""
    gore_pts_cm = gore_outline_points(bln)
    # Convert to mm.
    gore_pts_mm = [(x * MM_PER_CM, y * MM_PER_CM) for x, y in gore_pts_cm]
    # Bounding box of a single gore.
    xs = [p[0] for p in gore_pts_mm]
    ys = [p[1] for p in gore_pts_mm]
    gore_w = max(xs) - min(xs)
    gore_h = max(ys) - min(ys)
    margin = 20.0
    gap = 15.0
    total_w = bln.n_gores * gore_w + (bln.n_gores - 1) * gap + 2 * margin
    total_h = gore_h + 2 * margin + 30
    svg = [svg_header(total_w, total_h, f"Flat gores {bln.diameter_cm:g}cm {color['name']}")]
    svg.append(
        f'  <text x="{margin}" y="18" font-family="Helvetica,Arial,sans-serif"'
        f' font-size="10" fill="#222">Balloon {bln.diameter_cm:g}cm — {bln.n_gores} gores'
        f' — {color["name"]} ({cmyk_string(color["cmyk"])}) — gore length'
        f' {bln.gore_length_cm:.1f}cm, max width {2*bln.gore_max_half_width_cm:.1f}cm</text>\n'
    )
    for i in range(bln.n_gores):
        cx = margin + i * (gore_w + gap) + gore_w / 2.0
        cy = margin + 20 + gore_h / 2.0
        offset_x = cx
        offset_y = cy
        pts_shifted = [
            (x + offset_x, y + offset_y) for x, y in gore_pts_mm
        ]
        path_d = svg_path_from_points(pts_shifted)
        svg.append(
            f'  <path d="{path_d}" fill="{color["rgb"]}" fill-opacity="0.35"'
            f' stroke="#222" stroke-width="0.6"/>\n'
        )
        svg.append(
            f'  <text x="{cx}" y="{cy}" font-family="Helvetica,Arial,sans-serif"'
            f' font-size="12" fill="#222" text-anchor="middle">gore {i+1}/{bln.n_gores}</text>\n'
        )
    svg.append(svg_footer())
    with open(out_path, "w") as f:
        f.write("".join(svg))


def generate_slicing_preview(
    bln: Balloon,
    color: dict,
    logo_path: str,
    logo_width_cm: float,
    out_path: str,
) -> None:
    """All N gores side by side with the logo sliced across them.

    The logo is conceptually painted around the equator starting centered
    on gore 0. For each gore k we render the full logo, offset so that
    only the portion belonging to that gore is visible, and clip to the
    gore outline. The result is what the printer would actually output.
    """
    gore_pts_cm = gore_outline_points(bln)
    gore_pts_mm = [(x * MM_PER_CM, y * MM_PER_CM) for x, y in gore_pts_cm]
    xs = [p[0] for p in gore_pts_mm]
    ys = [p[1] for p in gore_pts_mm]
    gore_w = max(xs) - min(xs)
    gore_h = max(ys) - min(ys)
    equator_gap_mm = 2 * bln.gore_max_half_width_cm * MM_PER_CM  # spacing between gore centers ~ arc-length

    # Logo physical size (mm)
    logo_w_mm = logo_width_cm * MM_PER_CM
    # Preserve aspect ratio from the source SVG.
    _, vw, vh = read_logo_svg(logo_path)
    logo_h_mm = logo_w_mm * (vh / vw)
    logo_uri = logo_as_data_uri(logo_path)

    margin = 20.0
    gap = 15.0
    title_h = 40
    total_w = bln.n_gores * gore_w + (bln.n_gores - 1) * gap + 2 * margin
    total_h = gore_h + 2 * margin + title_h

    svg = [svg_header(total_w, total_h, f"Logo slicing preview — {bln.diameter_cm:g}cm {color['name']}")]
    svg.append('  <defs>\n')
    for i in range(bln.n_gores):
        path_d = svg_path_from_points(gore_pts_mm)
        svg.append(f'    <clipPath id="gore{i}"><path d="{path_d}"/></clipPath>\n')
    svg.append('  </defs>\n')

    svg.append(
        f'  <text x="{margin}" y="18" font-family="Helvetica,Arial,sans-serif"'
        f' font-size="12" fill="#222"><tspan font-weight="700">Logo slicing preview</tspan> '
        f'— {bln.diameter_cm:g}cm balloon, {bln.n_gores} gores, {color["name"]} '
        f'({cmyk_string(color["cmyk"])}), logo width {logo_width_cm:g}cm</text>\n'
    )
    svg.append(
        f'  <text x="{margin}" y="34" font-family="Helvetica,Arial,sans-serif"'
        f' font-size="9" fill="#555">Each panel shows one gore as it will be printed. '
        f'The logo is painted around the equator starting on gore 1 and wraps left/right '
        f'onto neighbors as needed.</text>\n'
    )

    for i in range(bln.n_gores):
        cx = margin + i * (gore_w + gap) + gore_w / 2.0
        cy = margin + title_h + gore_h / 2.0
        # Draw gore background.
        pts_shifted = [(x + cx, y + cy) for x, y in gore_pts_mm]
        path_d = svg_path_from_points(pts_shifted)
        svg.append(
            f'  <g>\n'
            f'    <path d="{path_d}" fill="{color["rgb"]}" fill-opacity="0.30"'
            f' stroke="#222" stroke-width="0.6"/>\n'
        )
        # Clip-path definition relative to this gore's local origin.
        clip_id = f"gclip{i}"
        svg.append(
            f'    <clipPath id="{clip_id}">\n'
            f'      <path d="{path_d}"/>\n'
            f'    </clipPath>\n'
        )
        # Project (bake) the logo onto the spherical surface of this gore.
        # Uses wraparound so gore N-1 shows the left edge of the logo and
        # gore 1 shows the right edge, and applies cos(latitude) horizontal
        # compression so the reassembled balloon matches the source design.
        _draw_warped_logo_on_gore(
            svg_lines=svg,
            gore_index=i,
            n_gores=bln.n_gores,
            panel_cx=cx,
            panel_cy=cy,
            radius_mm=bln.radius_cm * MM_PER_CM,
            logo_w_mm=logo_w_mm,
            logo_h_mm=logo_h_mm,
            logo_uri=logo_uri,
            gore_clip_id=clip_id,
            n_strips=24,
        )
        svg.append(
            f'    <text x="{cx}" y="{margin + title_h + gore_h + 12}"'
            f' font-family="Helvetica,Arial,sans-serif" font-size="10"'
            f' text-anchor="middle" fill="#222">gore {i+1}</text>\n'
        )
        svg.append('  </g>\n')

    svg.append(svg_footer())
    with open(out_path, "w") as f:
        f.write("".join(svg))


def generate_with_logo_detail(
    bln: Balloon,
    color: dict,
    logo_path: str,
    logo_width_cm: float,
    out_path: str,
) -> None:
    """Single large gore drawing showing the logo placement and slice lines."""
    gore_pts_cm = gore_outline_points(bln)
    gore_pts_mm = [(x * MM_PER_CM, y * MM_PER_CM) for x, y in gore_pts_cm]
    xs = [p[0] for p in gore_pts_mm]
    ys = [p[1] for p in gore_pts_mm]
    gore_w = max(xs) - min(xs)
    gore_h = max(ys) - min(ys)

    logo_w_mm = logo_width_cm * MM_PER_CM
    _, vw, vh = read_logo_svg(logo_path)
    logo_h_mm = logo_w_mm * (vh / vw)
    logo_uri = logo_as_data_uri(logo_path)

    margin = 30.0
    title_h = 40
    total_w = gore_w + 2 * margin + 120  # room for caption
    total_h = gore_h + 2 * margin + title_h

    cx = margin + gore_w / 2.0
    cy = margin + title_h + gore_h / 2.0

    svg = [svg_header(total_w, total_h, f"Gore with logo — {bln.diameter_cm:g}cm {color['name']}")]
    svg.append(
        f'  <text x="{margin}" y="18" font-family="Helvetica,Arial,sans-serif"'
        f' font-size="12" fill="#222" font-weight="700">Gore 1 with logo placement</text>\n'
    )
    svg.append(
        f'  <text x="{margin}" y="34" font-family="Helvetica,Arial,sans-serif"'
        f' font-size="9" fill="#555">Balloon {bln.diameter_cm:g}cm, '
        f'{bln.n_gores} gores, {color["name"]} ({cmyk_string(color["cmyk"])}). '
        f'Logo {logo_width_cm:g}cm wide, centered on equator.</text>\n'
    )
    pts_shifted = [(x + cx, y + cy) for x, y in gore_pts_mm]
    path_d = svg_path_from_points(pts_shifted)
    svg.append(
        f'  <clipPath id="gore_main"><path d="{path_d}"/></clipPath>\n'
    )
    svg.append(
        f'  <path d="{path_d}" fill="{color["rgb"]}" fill-opacity="0.35"'
        f' stroke="#222" stroke-width="0.6"/>\n'
    )
    # Project the logo onto gore 0 with cos(latitude) warping so the
    # shown single gore matches what the print shop will actually print.
    _draw_warped_logo_on_gore(
        svg_lines=svg,
        gore_index=0,
        n_gores=bln.n_gores,
        panel_cx=cx,
        panel_cy=cy,
        radius_mm=bln.radius_cm * MM_PER_CM,
        logo_w_mm=logo_w_mm,
        logo_h_mm=logo_h_mm,
        logo_uri=logo_uri,
        gore_clip_id="gore_main",
        n_strips=24,
    )
    # Equator line + pole markers.
    svg.append(
        f'  <line x1="{cx - gore_w/2.0 - 5}" y1="{cy}"'
        f' x2="{cx + gore_w/2.0 + 5}" y2="{cy}"'
        f' stroke="#c33" stroke-width="0.6" stroke-dasharray="4 3"/>\n'
    )
    svg.append(
        f'  <text x="{cx + gore_w/2.0 + 8}" y="{cy + 3}"'
        f' font-family="Helvetica,Arial,sans-serif" font-size="8" fill="#c33">equator</text>\n'
    )
    svg.append(svg_footer())
    with open(out_path, "w") as f:
        f.write("".join(svg))


def generate_assembled_preview(
    bln: Balloon,
    color: dict,
    logo_path: str,
    logo_width_cm: float,
    out_path: str,
) -> None:
    """Side-on preview of the balloon with the logo wrapped on the front.

    We draw a circle (to scale in mm) shaded to look spherical, overlay
    longitude lines for the gore seams, and place the logo centered with a
    slight horizontal squash to mimic wrapping.
    """
    D_mm = bln.diameter_cm * MM_PER_CM
    pad = 40.0
    total_w = D_mm + 2 * pad
    total_h = D_mm + 2 * pad + 40

    cx = pad + D_mm / 2.0
    cy = pad + 20 + D_mm / 2.0
    R = D_mm / 2.0

    logo_w_mm = logo_width_cm * MM_PER_CM
    _, vw, vh = read_logo_svg(logo_path)
    logo_h_mm = logo_w_mm * (vh / vw)
    # Foreshortening: logo near the front appears ~unchanged horizontally
    # but is visually squashed at the edges. We approximate with 85%.
    logo_w_vis = logo_w_mm * 0.92

    logo_uri = logo_as_data_uri(logo_path)

    svg = [svg_header(total_w, total_h, f"Assembled balloon — {bln.diameter_cm:g}cm {color['name']}")]
    svg.append(
        f'  <text x="{pad}" y="20" font-family="Helvetica,Arial,sans-serif"'
        f' font-size="12" fill="#222" font-weight="700">Assembled preview — '
        f'{bln.diameter_cm:g}cm {color["name"]} balloon ({bln.n_gores} gores)</text>\n'
    )
    # Spherical gradient.
    svg.append('  <defs>\n')
    svg.append(
        f'    <radialGradient id="sphere" cx="38%" cy="32%" r="70%">\n'
        f'      <stop offset="0%" stop-color="#ffffff" stop-opacity="0.85"/>\n'
        f'      <stop offset="35%" stop-color="{color["rgb"]}" stop-opacity="1"/>\n'
        f'      <stop offset="100%" stop-color="#000000" stop-opacity="0.55"/>\n'
        f'    </radialGradient>\n'
    )
    svg.append(f'    <clipPath id="ball"><circle cx="{cx}" cy="{cy}" r="{R}"/></clipPath>\n')
    svg.append('  </defs>\n')
    svg.append(
        f'  <circle cx="{cx}" cy="{cy}" r="{R}" fill="{color["rgb"]}"/>\n'
    )
    svg.append(
        f'  <circle cx="{cx}" cy="{cy}" r="{R}" fill="url(#sphere)"/>\n'
    )
    # Gore seams as ellipses (longitude lines viewed side-on).
    for k in range(bln.n_gores):
        # Longitude of this seam relative to center.
        lon = (2 * math.pi * k / bln.n_gores) - math.pi / bln.n_gores
        # Ellipse x-radius = R * sin(lon) -> can be negative/zero
        rx = abs(R * math.sin(lon))
        # If rx ~ 0 the seam is the vertical pole line.
        if rx < 0.5:
            svg.append(
                f'  <line x1="{cx}" y1="{cy - R}" x2="{cx}" y2="{cy + R}"'
                f' stroke="#000" stroke-width="0.4" stroke-opacity="0.35"'
                f' clip-path="url(#ball)"/>\n'
            )
        else:
            svg.append(
                f'  <ellipse cx="{cx}" cy="{cy}" rx="{rx:.2f}" ry="{R:.2f}"'
                f' fill="none" stroke="#000" stroke-width="0.4"'
                f' stroke-opacity="0.35" clip-path="url(#ball)"/>\n'
            )
    # Logo centered on the front.
    lx = cx - logo_w_vis / 2.0
    ly = cy - logo_h_mm / 2.0
    svg.append(
        f'  <image xlink:href="{logo_uri}" href="{logo_uri}"'
        f' x="{lx:.2f}" y="{ly:.2f}"'
        f' width="{logo_w_vis:.2f}" height="{logo_h_mm:.2f}"'
        f' clip-path="url(#ball)" preserveAspectRatio="xMidYMid meet"/>\n'
    )
    # Equator line.
    svg.append(
        f'  <ellipse cx="{cx}" cy="{cy}" rx="{R}" ry="{R * 0.15}"'
        f' fill="none" stroke="#000" stroke-width="0.4"'
        f' stroke-opacity="0.25" stroke-dasharray="3 3"/>\n'
    )
    # Outline.
    svg.append(
        f'  <circle cx="{cx}" cy="{cy}" r="{R}" fill="none"'
        f' stroke="#222" stroke-width="0.8"/>\n'
    )
    # Caption.
    svg.append(
        f'  <text x="{pad}" y="{pad + 20 + D_mm + 18}"'
        f' font-family="Helvetica,Arial,sans-serif" font-size="9" fill="#555">'
        f'Color CMYK: {cmyk_string(color["cmyk"])}. Logo {logo_width_cm:g}cm wide '
        f'on the equator (approx. {logo_width_cm / (math.pi * bln.diameter_cm) * 100:.0f}% of circumference).'
        f'</text>\n'
    )
    svg.append(svg_footer())
    with open(out_path, "w") as f:
        f.write("".join(svg))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate balloon gore patterns with a logo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--size", type=float, default=200.0,
                        help="Balloon diameter in cm (e.g. 200 for 2m).")
    parser.add_argument("--color", type=str, default="yellow",
                        help="Balloon color name (see --list-colors).")
    parser.add_argument("--logo", type=str, default="love_you_logo.svg",
                        help="Path to logo SVG file.")
    parser.add_argument("--logo-width", type=float, default=None,
                        help="Logo width on balloon in cm (default = 30%% of circumference).")
    parser.add_argument("--gores", type=int, default=6,
                        help="Number of gores.")
    parser.add_argument("--prefix", type=str, default=None,
                        help="Output filename prefix (default auto).")
    parser.add_argument("--outdir", type=str, default=".",
                        help="Output directory.")
    parser.add_argument("--list-colors", action="store_true",
                        help="List available color names and exit.")
    args = parser.parse_args(argv)

    if args.list_colors:
        print("Available colors:")
        for name in sorted(COLORS):
            c = COLORS[name]
            print(f"  {name:<10s} CMYK {cmyk_string(c['cmyk']):<22s} RGB {c['rgb']}")
        return 0

    try:
        color = resolve_color(args.color)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    bln = Balloon(diameter_cm=args.size, n_gores=args.gores)

    if args.logo_width is None:
        # 30% of equator circumference by default -> a bold but not absurd logo.
        logo_width_cm = 0.30 * math.pi * bln.diameter_cm
    else:
        logo_width_cm = args.logo_width

    if not os.path.isfile(args.logo):
        print(f"error: logo file not found: {args.logo}", file=sys.stderr)
        return 2

    os.makedirs(args.outdir, exist_ok=True)

    logo_stem = os.path.splitext(os.path.basename(args.logo))[0]
    prefix = args.prefix or f"balloon_{int(args.size)}cm_{color['name']}_{logo_stem}"

    flat_path = os.path.join(args.outdir, f"{prefix}_flat.svg")
    detail_path = os.path.join(args.outdir, f"{prefix}_with_logo.svg")
    preview_path = os.path.join(args.outdir, f"{prefix}_preview.svg")
    assembled_path = os.path.join(args.outdir, f"{prefix}_assembled.svg")

    generate_flat_layout(bln, color, flat_path)
    generate_with_logo_detail(bln, color, args.logo, logo_width_cm, detail_path)
    generate_slicing_preview(bln, color, args.logo, logo_width_cm, preview_path)
    generate_assembled_preview(bln, color, args.logo, logo_width_cm, assembled_path)

    print(f"Balloon: {args.size:g}cm {color['name']} ({cmyk_string(color['cmyk'])}), "
          f"{args.gores} gores.")
    print(f"Gore length: {bln.gore_length_cm:.1f}cm, "
          f"max half-width: {bln.gore_max_half_width_cm:.1f}cm.")
    print(f"Logo: {args.logo} at {logo_width_cm:.1f}cm wide "
          f"({logo_width_cm / (math.pi * args.size) * 100:.0f}% of circumference).")
    print("Wrote:")
    for p in (flat_path, detail_path, preview_path, assembled_path):
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
