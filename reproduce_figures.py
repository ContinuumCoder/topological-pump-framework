import time
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.collections import LineCollection

# Importar la biblioteca de lógica central
from symplectic_tools import (
    symplectic_J, pump_S_from_w, is_symplectic, labels_XZ,
    pretty_mapping_from_S, changed_columns, verify_transvection_equivalence
)

# -------------------------
# Utilidades de Trazado (Optimizadas)
# -------------------------
def add_grid_fast(ax, Lx, Ly, color="#e9e9e9", lw=0.8, draw_outer=True, outer_color="#cfcfcf"):
    """Dibuja una cuadrícula de manera eficiente usando LineCollection."""
    segs = []
    for x in range(Lx + 1):
        segs.append([(x, 0), (x, Ly)])
    for y in range(Ly + 1):
        segs.append([(0, y), (Lx, y)])
    lc = LineCollection(segs, colors=color, linewidths=lw, zorder=0)
    ax.add_collection(lc)
    if draw_outer:
        ax.plot([0, Lx], [0, 0], color=outer_color, lw=1.2, zorder=1)
        ax.plot([0, Lx], [Ly, Ly], color=outer_color, lw=1.2, zorder=1)
        ax.plot([0, 0], [0, Ly], color=outer_color, lw=1.2, zorder=1)
        ax.plot([Lx, Lx], [0, Ly], color=outer_color, lw=1.2, zorder=1)

def draw_S_heatmap(ax, S: np.ndarray, lbls: list[str], title: str | None = None, bottom_title: str | None = None):
    """Dibuja el mapa de calor de la matriz S."""
    ax.imshow(S, cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks(range(len(lbls))); ax.set_yticks(range(len(lbls)))
    ax.set_xticklabels(lbls, rotation=45, ha='left', fontsize=10)
    ax.set_yticklabels(lbls, fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True, length=0)
    if title:
        ax.set_title(title, fontsize=12, pad=6)
    if bottom_title:
        ax.set_xlabel(bottom_title, fontsize=12, labelpad=8)

def draw_w_bar(ax, w: np.ndarray, lbls: list[str], title="w"):
    """Dibuja el vector w como una barra vertical."""
    ax.imshow(w.reshape(-1, 1), cmap="Reds", interpolation="nearest", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(lbls))); ax.set_yticklabels(lbls, fontsize=10)
    ax.set_xticks([0]); ax.set_xticklabels([title], fontsize=10)
    for spine in ax.spines.values(): spine.set_visible(False)

# -------------------------
# Reporte de Datos
# -------------------------
def report_pump(k: int, w: np.ndarray, S: np.ndarray, name: str):
    """Imprime un informe detallado sobre la transvección S y su vector w."""
    lbls = labels_XZ(k)
    J = symplectic_J(k)
    w = w.reshape(-1, 1).astype(np.uint8)
    Jw = (J @ w) % 2
    H = (w @ Jw.T) % 2
    maps = pretty_mapping_from_S(S, lbls)
    changed = changed_columns(S, lbls)

    print(f"=== {name} ===")
    print(f"k = {k}, 2k = {2*k}")
    print(f"w (len={len(w)}): {w.ravel().tolist()}")
    print(f"Jw: {Jw.ravel().tolist()}")
    nz = np.where(H == 1)
    nz_pairs = list(zip(nz[0].tolist(), nz[1].tolist()))
    print(f"H = w (Jw)^T nonzeros (row,col): {nz_pairs[:16]}{' ...' if len(nz_pairs) > 16 else ''}")
    sp_ok = is_symplectic(S)
    tv_ok = verify_transvection_equivalence(S, w.ravel())
    print(f"Symplectic check (S^T J S == J mod 2): {sp_ok}")
    print(f"Equivalence check (S = I + w (Jw)^T): {tv_ok}")
    print(f"Changed columns ({len(changed)}): {changed}")
    print("Mapping summary:")
    for line in maps:
        print("  " + line)
    print()

# -------------------------
# Figura 3: Torus ε-loop (k=2)
# -------------------------
def figure_torus_swap(out_png="fig_torus_pump_swap.png", out_svg=None, dpi=220, fast=False):
    t0 = time.perf_counter()
    Lx, Ly = 12, 8

    fig = plt.figure(figsize=(11.5, 5.6), dpi=dpi, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.08, hspace=0.02, wspace=0.02)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 0.24])

    axL = fig.add_subplot(gs[0, 0])
    axH = fig.add_subplot(gs[0, 1])
    axW = fig.add_subplot(gs[0, 2])

    grid_color = "#efefef" if not fast else "#f6f6f6"
    add_grid_fast(axL, Lx, Ly, color=grid_color, lw=0.7 if not fast else 0.5, draw_outer=True)

    arr_kw = dict(arrowstyle='-|>', mutation_scale=14, color="#606060", lw=1.2)
    axL.add_patch(FancyArrowPatch((-0.4, -0.6), (Lx+0.4, -0.6), **arr_kw))
    axL.text(Lx/2, -0.95, "x ≡ x + Lx", ha="center", va="top", fontsize=9, color="#606060")
    axL.add_patch(FancyArrowPatch((Lx+0.6, -0.3), (Lx+0.6, Ly+0.3), **arr_kw))
    axL.text(Lx+0.95, Ly/2, "y ≡ y + Ly", rotation=90, ha="left", va="center", fontsize=9, color="#606060")

    y = Ly // 2
    axL.plot([-0.1, Lx+0.1], [y + 0.5, y + 0.5], color="#e53935", lw=3.0, solid_capstyle="round", zorder=5)
    axL.text(0.6, y + 0.9, "ε-loop W", color="#e53935", fontsize=11, weight="bold")

    k = 2
    lbls = labels_XZ(k)
    w = np.array([1, 0, 1, 0], dtype=np.uint8)  # w = X1 + Z1
    S = pump_S_from_w(w)

    axL.text(0.02, 0.82, "Choose w = X1 + Z1", transform=axL.transAxes,
             ha="left", va="bottom", fontsize=12, color="#1a73e8", zorder=10)
    maps = pretty_mapping_from_S(S, lbls)
    axL.text(0.02, 0.14, "Effect:\n" + "\n".join(maps),
             transform=axL.transAxes, ha="left", va="bottom",
             fontsize=10, family="monospace",
             bbox=dict(boxstyle="round,pad=0.35", fc="#f7f7f7", ec="#d0d0d0", alpha=0.95),
             zorder=10)

    axL.set_xlim(-1.2, Lx + 1.8)
    axL.set_ylim(-1.6, Ly + 3.5)
    axL.set_aspect("equal")
    axL.axis("off")

    draw_S_heatmap(axH, S, lbls, title="Transvection S = I + w (J w)^T",
                   bottom_title="S for w = X1 + Z1 (k=2)")
    draw_w_bar(axW, w, lbls, title="w")
    fig.suptitle("Logical pump on a torus", fontsize=14, x= 0.3, y=0.915)

    t1 = time.perf_counter()
    saved = []
    if out_png:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        saved.append(out_png)
    if out_svg:
        fig.savefig(out_svg, bbox_inches="tight")
        saved.append(out_svg)
    t2 = time.perf_counter()
    plt.close(fig)

    report_pump(k, w, S, "Torus ε-loop (Fig 3)")
    if saved:
        print(f"[Saved] {' and '.join(saved)}")
    print(f"[Timing] compute/draw: {(t1-t0)*1e3:.1f} ms, save: {(t2-t1)*1e3:.1f} ms\n")

# -------------------------
# Figura 4: Plana Perforada (k=3)
# -------------------------
def figure_punctured_3holes(out_png="fig_punctured3_pump_12.png", out_svg=None,
                            dpi=220, fast=False):
    t0 = time.perf_counter()
    Lx, Ly = 14, 10
    holes = [(4, 7), (8, 7), (6, 3)]

    fig = plt.figure(figsize=(12.7, 5.9), dpi=dpi, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.08, hspace=0.02, wspace=0.02)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 0.24])
    axL = fig.add_subplot(gs[0, 0])
    axH = fig.add_subplot(gs[0, 1])
    axW = fig.add_subplot(gs[0, 2])

    grid_color = "#eeeeee" if not fast else "#f6f6f6"
    add_grid_fast(axL, Lx, Ly, color=grid_color, lw=0.8 if not fast else 0.5, draw_outer=True)

    for i, (cx, cy) in enumerate(holes, start=1):
        rect = Rectangle((cx, cy), 1, 1, facecolor="#fff2b2", edgecolor="#d4a514", lw=1.1, zorder=2)
        axL.add_patch(rect)
        axL.text(cx + 0.5, cy + 0.5, f"{i}", ha="center", va="center", fontsize=12, weight="bold", color="#7a5d00", zorder=3)

    x0, x1 = min(holes[0][0], holes[1][0]) - 0.2, max(holes[0][0], holes[1][0]) + 1.2
    y0, y1 = min(holes[0][1], holes[1][1]) - 0.2, max(holes[0][1], holes[1][1]) + 1.2
    path = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    segs = np.array([[path[i], path[i+1]] for i in range(len(path)-1)])
    lc = LineCollection(segs, colors="#d81b60", linewidths=3.0, zorder=4, capstyle="round", joinstyle="round")
    axL.add_collection(lc)
    axL.text((x0+x1)/2, y1 + 0.35, "W encircling holes 1 & 2", ha="center", va="bottom", color="#d81b60", fontsize=11)

    k = 3
    lbls = labels_XZ(k)
    w = np.array([0, 0, 0, 1, 1, 0], dtype=np.uint8)  # w = Z1 + Z2
    S = pump_S_from_w(w)

    axL.text(0.02, 0.02, "k = 3, choose w = Z1 + Z2", transform=axL.transAxes,
             ha="left", va="bottom", fontsize=12, color="#1a73e8", zorder=10)
    key_lines = ["Effect:", "  X1 -> X1 Z1 Z2", "  X2 -> X2 Z1 Z2", "  others unchanged"]
    axL.text(0.02, 0.12, "\n".join(key_lines),
             transform=axL.transAxes, ha="left", va="bottom",
             fontsize=10, family="monospace",
             bbox=dict(boxstyle="round,pad=0.35", fc="#f7f7f7", ec="#d0d0d0", alpha=0.95),
             zorder=10)

    axL.set_xlim(-0.8, Lx + 0.8)
    axL.set_ylim(-1.2, Ly + 3.2)
    axL.set_aspect("equal")
    axL.axis("off")

    draw_S_heatmap(axH, S, lbls, title="W around holes 1&2 → coupled shear on X1, X2",
                   bottom_title="S for w = Z1 + Z2 (k=3)")
    draw_w_bar(axW, w, lbls, title="w")
    fig.suptitle("Punctured planar code", fontsize=14, x=0.3, y=0.915)

    t1 = time.perf_counter()
    saved = []
    if out_png:
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        saved.append(out_png)
    if out_svg:
        fig.savefig(out_svg, bbox_inches="tight")
        saved.append(out_svg)
    t2 = time.perf_counter()
    plt.close(fig)
    
    report_pump(k, w, S, "Planar code (3 holes) (Fig 4)")
    if saved:
        print(f"[Saved] {' and '.join(saved)}")
    print(f"[Timing] compute/draw: {(t1-t0)*1e3:.1f} ms, save: {(t2-t1)*1e3:.1f} ms\n")

# -------------------------
# Punto de Entrada Principal
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Reproduce figures for the topological ε-pump paper.")
    p.add_argument("--fast", action="store_true", help="Speed up drawing (lower DPI, simpler grid).")
    p.add_argument("--dpi", type=int, default=220, help="Figure DPI (PNG).")
    gfmt = p.add_mutually_exclusive_group()
    gfmt.add_argument("--png-only", action="store_true", help="Export PNG only.")
    gfmt.add_argument("--svg-only", action="store_true", help="Export SVG only.")
    return p.parse_args()

def main():
    args = parse_args()
    dpi = min(args.dpi, 160) if args.fast else args.dpi
    
    png1, svg1 = "fig_torus_pump_swap.png", "fig_torus_pump_swap.svg"
    png2, svg2 = "fig_punctured3_pump_12.png", "fig_punctured3_pump_12.svg"
    
    if args.png_only:
        svg1, svg2 = None, None
    if args.svg_only:
        png1, png2 = None, None
        
    matplotlib.rcParams["savefig.pad_inches"] = 0.02

    figure_torus_swap(out_png=png1, out_svg=svg1, dpi=dpi, fast=args.fast)
    figure_punctured_3holes(out_png=png2, out_svg=svg2, dpi=dpi, fast=args.fast)

if __name__ == "__main__":
    main()