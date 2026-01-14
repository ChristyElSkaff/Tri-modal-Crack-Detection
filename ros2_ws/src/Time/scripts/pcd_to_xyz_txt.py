#!/usr/bin/env python3
import argparse
import numpy as np
import struct

def parse_header(lines):
    header = {}
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        key, *rest = ln.split()
        key = key.upper()
        header[key] = rest
        if key == "DATA":
            break
    # normalize
    fields = header.get("FIELDS", header.get("FIELD", []))
    sizes  = list(map(int, header.get("SIZE", [])))
    types  = header.get("TYPE", [])
    counts = list(map(int, header.get("COUNT", ["1"]*len(fields))))
    width  = int(header.get("WIDTH", ["0"])[0])
    height = int(header.get("HEIGHT", ["1"])[0])
    points = int(header.get("POINTS", [str(width*height)])[0])
    data   = header.get("DATA", ["ascii"])[0].lower()
    return fields, sizes, types, counts, width, height, points, data

def dtype_from_pcd(fields, sizes, types, counts):
    # Build numpy dtype for one point
    dt = []
    for f, sz, ty, ct in zip(fields, sizes, types, counts):
        if ty == "F":
            base = {4: np.float32, 8: np.float64}[sz]
        elif ty == "U":
            base = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[sz]
        elif ty == "I":
            base = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}[sz]
        else:
            raise ValueError(f"Unsupported TYPE {ty}")
        if ct == 1:
            dt.append((f, base))
        else:
            dt.append((f, base, (ct,)))
    return np.dtype(dt)

def read_pcd(path):
    with open(path, "rb") as f:
        # read header lines until DATA
        header_lines = []
        while True:
            ln = f.readline()
            if not ln:
                raise ValueError("Unexpected EOF while reading PCD header")
            header_lines.append(ln.decode("utf-8", errors="ignore"))
            if header_lines[-1].strip().upper().startswith("DATA"):
                break

        fields, sizes, types, counts, width, height, points, data = parse_header(header_lines)
        dt = dtype_from_pcd(fields, sizes, types, counts)

        if data == "ascii":
            # remaining file is text
            txt = f.read().decode("utf-8", errors="ignore").strip().splitlines()
            rows = []
            for ln in txt:
                if not ln.strip():
                    continue
                rows.append(list(map(float, ln.split())))
            arr = np.array(rows, dtype=np.float64)
            # Map columns to fields (assumes count=1 for x,y,z in ascii; typical)
            # We'll just return a dict-like view for x,y,z if present.
            return fields, arr

        elif data in ("binary", "binary_compressed"):
            if data == "binary_compressed":
                raise ValueError("binary_compressed PCD not supported by this simple script.")
            # binary: read points * point_step bytes
            raw = f.read(points * dt.itemsize)
            if len(raw) < points * dt.itemsize:
                raise ValueError("PCD binary data shorter than expected.")
            cloud = np.frombuffer(raw, dtype=dt, count=points)
            return fields, cloud

        else:
            raise ValueError(f"Unsupported DATA mode: {data}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcd", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fields, cloud = read_pcd(args.pcd)

    if "x" not in [f.lower() for f in fields] or "y" not in [f.lower() for f in fields] or "z" not in [f.lower() for f in fields]:
        # Some PCDs store fields as X Y Z uppercase
        # We'll handle case-insensitively
        pass

    # Extract x,y,z case-insensitively
    if isinstance(cloud, np.ndarray) and cloud.dtype.names is None:
        # ASCII numeric array: assume x,y,z are first 3 columns
        xyz = cloud[:, :3]
    else:
        # binary structured array
        names = {n.lower(): n for n in cloud.dtype.names}
        xyz = np.vstack([cloud[names["x"]], cloud[names["y"]], cloud[names["z"]]]).T.astype(np.float64)

    # Remove NaNs / inf
    m = np.isfinite(xyz).all(axis=1)
    xyz = xyz[m]

    np.savetxt(args.out, xyz, fmt="%.6f")
    print("Wrote:", args.out, "points:", xyz.shape[0])

if __name__ == "__main__":
    main()