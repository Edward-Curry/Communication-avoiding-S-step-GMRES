#!/usr/bin/env python3
"""
@file scripts/generate_scaling_matrix.py
@brief Generates three-dimensional convection-diffusion test matrices.
@author Edward Curry
@date 2026-08-23

The diagonal boost controls conditioning. The scatter block applies a
symmetric permutation that changes the halo topology without changing the
underlying linear system.
"""


import argparse
import math
import random
from pathlib import Path

FORWARD_3D = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]


def counts(n1, dim):
    """Return the row count and stored entries for the requested stencil."""
    rows = n1 ** dim
    edges = sum(
        (n1 - abs(dx)) * (n1 - abs(dy)) * (n1 - abs(dz))
        for dx, dy, dz in FORWARD_3D[:dim]
    )
    return rows, rows + 2 * edges          # diagonal + both triangles


def build_permutation(n, block, seed):
    """Return a deterministic blockwise symmetric-permutation index map."""
    perm = list(range(n))
    if block <= 1:
        return perm
    rng = random.Random(seed)
    step = min(block, n)
    for lo in range(0, n, step):
        hi = min(n, lo + step)
        chunk = perm[lo:hi]
        rng.shuffle(chunk)
        perm[lo:hi] = chunk
    return perm


def generate(n1, dim, diag_boost, advection,
             scatter_block, seed, out):
    """Write one general Matrix Market convection-diffusion matrix."""
    if dim != 3:
        raise SystemExit("only dim = 3 is implemented")
    if abs(advection) >= 1.0:
        raise SystemExit("advection must satisfy |gamma| < 1 to keep the spectrum real")

    rows, stored = counts(n1, dim)
    perm = build_permutation(rows, scatter_block, seed)

    diag = 2.0 * dim + diag_boost
    up = -1.0 + advection          # neighbour at +stride
    lo = -1.0 - advection          # neighbour at -stride
    strides = [1, n1, n1 * n1]

    kappa = (4.0 * dim + diag_boost) / diag_boost if diag_boost > 0 else float("inf")
    lam_min_laplace = dim * math.pi ** 2 / (n1 + 1) ** 2

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"  n1={n1} dim={dim} rows={rows} stored={stored}")
    print(f"  diag={diag:.6g} boost={diag_boost:g} -> kappa ~ {kappa:.4g}")
    print(f"  Laplacian lambda_min = {lam_min_laplace:.4g} "
          f"({'boost dominates, grid-independent' if diag_boost > lam_min_laplace else 'GRID-DEPENDENT: boost is below lambda_min'})")
    if advection:
        r = math.sqrt((1 + advection) / (1 - advection))
        print(f"  advection={advection:g} -> kappa(X) ~ {r ** (dim * (n1 - 1)):.4g}")

    with out.open("w", newline="\n") as fh:
        fh.write("%%MatrixMarket matrix coordinate real general\n")
        fh.write(f"% 3D convection-diffusion, n1={n1} dim={dim} "
                 f"diag_boost={diag_boost:g} advection={advection:g} "
                 f"scatter_block={scatter_block}\n")
        fh.write(f"{rows} {rows} {stored}\n")

        buf = []
        for gid in range(rows):
            r0 = perm[gid] + 1
            buf.append(f"{r0} {r0} {diag:.17g}\n")

            rest = gid
            for d in range(dim):
                k = rest % n1
                rest //= n1
                st = strides[d]
                if k > 0:
                    buf.append(f"{r0} {perm[gid - st] + 1} {lo:.17g}\n")
                if k < n1 - 1:
                    buf.append(f"{r0} {perm[gid + st] + 1} {up:.17g}\n")

            if len(buf) >= 2_000_000:
                fh.writelines(buf)
                buf.clear()
        fh.writelines(buf)

    print(f"  wrote {out}  ({out.stat().st_size / 2**30:.2f} GB)")


def main():
    """Parse command-line arguments and generate one scaling matrix."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, required=True, help="n1; rows = n1^dim")
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--diag-boost", type=float, default=0.01,
                    help="added to the diagonal; sets kappa ~ (4*dim+d)/d (default 0.01)")
    ap.add_argument("--advection", type=float, default=0.0,
                    help="gamma; non-normality. Keep <= 0.01 at large n1")
    ap.add_argument("--scatter-block", type=int, default=1,
                    help="1 = stencil ordering; >= rows = global shuffle")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    if a.output.exists():
        print(f"  {a.output} already exists, nothing to do")
        return
    generate(a.size, a.dim, a.diag_boost, a.advection,
             a.scatter_block, a.seed, a.output)


if __name__ == "__main__":
    main()
