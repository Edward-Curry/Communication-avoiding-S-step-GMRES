#!/usr/bin/env python3

import argparse
from pathlib import Path


FORWARD_NEIGHBORS = [
    (dx, dy, dz)
    for dz in range(-1, 2)
    for dy in range(-1, 2)
    for dx in range(-1, 2)
    if (dz > 0 or (dz == 0 and dy > 0) or (dz == 0 and dy == 0 and dx > 0))
]


def stored_entries(size: int) -> int:
    points = size**3
    edges = sum(
        (size - abs(dx)) * (size - abs(dy)) * (size - abs(dz))
        for dx, dy, dz in FORWARD_NEIGHBORS
    )
    return points + edges


def diagonal_value(index: int) -> float:
    # Deterministic variation keeps the all-ones solution from being an eigenvector.
    bucket = (index * 2654435761) % 1000
    return 0.75 + 0.5 * bucket / 999.0


def generate(size: int, output_path: Path) -> None:
    points = size**3
    entries = stored_entries(size)
    plane = size * size
    buffer = []
    flush_size = 65536

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="ascii", newline="\n") as output:
        output.write("%%MatrixMarket matrix coordinate real symmetric\n")
        output.write("% Easy 3D 27-point SPD matrix for distributed GMRES tests.\n")
        output.write(f"{points} {points} {entries}\n")

        for z in range(size):
            for y in range(size):
                for x in range(size):
                    row_zero = z * plane + y * size + x
                    row = row_zero + 1
                    buffer.append(f"{row} {row} {diagonal_value(row_zero):.17g}\n")

                    for dx, dy, dz in FORWARD_NEIGHBORS:
                        nx = x + dx
                        ny = y + dy
                        nz = z + dz

                        if 0 <= nx < size and 0 <= ny < size and 0 <= nz < size:
                            col = nz * plane + ny * size + nx + 1
                            buffer.append(f"{row} {col} -0.005\n")

                    if len(buffer) >= flush_size:
                        output.write("".join(buffer))
                        buffer.clear()

            print(f"Generated layer {z + 1}/{size}", flush=True)

        if buffer:
            output.write("".join(buffer))

    expanded_nonzeros = 2 * entries - points
    print(f"Wrote {output_path}")
    print(f"Rows: {points}")
    print(f"Expanded nonzeros: {expanded_nonzeros}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an easy, large SPD Matrix Market test matrix."
    )
    parser.add_argument(
        "--size",
        type=int,
        default=100,
        help="Grid length; total rows are size^3 (default: 100).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/matrices/easy_3d_100.mtx"),
        help="Output Matrix Market path.",
    )
    args = parser.parse_args()

    if args.size < 2:
        parser.error("--size must be at least 2")

    generate(args.size, args.output)


if __name__ == "__main__":
    main()
