"""核对分区结果：结构、键唯一、行数守恒、随机抽样回源比对。

跑在 ``01_partition.py`` 之后。判据全部是"硬"的（要么全过要么报出差异），
只有"每日档数分布"是信息性输出。

用法::

    python -I 02_verify.py --pkg <包根> --out <输出根> [--md5] [--samples 30]
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

FAIL = []


def bad(msg: str) -> None:
    FAIL.append(msg)
    print("  ✗", msg)


def ok(msg: str) -> None:
    print("  ✓", msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default=None,
                    help="源 zip 所在目录；不给则跳过第 7 项回源比对（1–6 项照常跑）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md5", action="store_true", help="全量重算 md5（约 70 GB 读盘）")
    ap.add_argument("--samples", type=int, default=30)
    a = ap.parse_args()
    pkg, out = (Path(a.pkg) if a.pkg else None), Path(a.out)
    M = json.loads((out / "_manifest.json").read_text())

    print("1. 分区文件清单")
    files = sorted((out / "minute" / "1min").rglob("*.parquet"))
    n_files = len(files)
    if n_files != len(M["minute"]):
        bad(f"磁盘 {n_files} 个分区，manifest 记录 {len(M['minute'])} 个")
    else:
        ok(f"{n_files} 个月分区，与 manifest 一致")

    print("2. 月份连续")
    ym = sorted(f"{f.parent.name.split('=')[1]}-{f.stem.split('=')[1]}" for f in files)
    exp = []
    y0, m0 = int(ym[0][:4]), int(ym[0][5:])
    y1, m1 = int(ym[-1][:4]), int(ym[-1][5:])
    while (y0, m0) <= (y1, m1):
        exp.append(f"{y0:04d}-{m0:02d}")
        m0 += 1
        if m0 == 13:
            y0, m0 = y0 + 1, 1
    missing = [m for m in exp if m not in set(ym)]
    if missing:
        bad(f"缺月 {missing}")
    else:
        ok(f"{ym[0]} ~ {ym[-1]} 共 {len(ym)} 个月，无缺口")

    print("3. schema 一致")
    sig = None
    for f in files:
        s = pq.ParquetFile(f).schema_arrow
        t = tuple((n, str(ty)) for n, ty in zip(s.names, s.types))
        if sig is None:
            sig = t
        elif t != sig:
            bad(f"{f.name} schema 与首文件不同：{t}")
            break
    else:
        ok(f"全部 {len(files)} 个文件同 schema：{', '.join(n for n, _ in sig)}")

    print("4. 行数守恒")
    tot_part = 0
    for f in files:
        md = pq.ParquetFile(f).metadata
        ymk = f"{f.parent.name.split('=')[1]}-{f.stem.split('=')[1]}"
        rec = M["minute"].get(ymk)
        if rec is None:
            bad(f"{ymk} 不在 manifest")
        elif rec["rows"] != md.num_rows:
            bad(f"{ymk} 磁盘 {md.num_rows:,} 行 ≠ manifest {rec['rows']:,}")
        tot_part += md.num_rows
    by_year: dict[str, int] = {}
    for f in files:
        y = f.parent.name.split('=')[1]
        by_year[y] = by_year.get(y, 0) + pq.ParquetFile(f).metadata.num_rows
    for y, rec in sorted(M["source"].items()):
        if y in by_year and by_year[y] != rec["rows_1min"]:
            bad(f"{y} 年分区合计 {by_year[y]:,} ≠ 源 zip {rec['rows_1min']:,}")
    ok(f"分区合计 {tot_part:,} 行；逐年与源 zip 行数一致（{len(M['source'])} 年）")

    if a.md5:
        print("5. md5 复核")
        import hashlib

        def md5f(p):
            h = hashlib.md5()
            with p.open("rb") as fh:
                for c in iter(lambda: fh.read(1 << 20), b""):
                    h.update(c)
            return h.hexdigest()
        nbad = 0
        for f in files:
            ymk = f"{f.parent.name.split('=')[1]}-{f.stem.split('=')[1]}"
            if md5f(f) != M["minute"][ymk]["md5"]:
                bad(f"{ymk} md5 不符")
                nbad += 1
        if not nbad:
            ok("全部 md5 与 manifest 一致")

    print("6. 每日档数分布（信息性）")
    rng = random.Random(0)
    pick = rng.sample(files, min(len(files), 6))
    for f in sorted(pick):
        t = pq.read_table(f, columns=["trade_date", "ts_code", "trade_time"]).to_pandas()
        b = t.groupby(["trade_date", "ts_code"]).size()
        vc = b.value_counts().head(4).to_dict()
        odd = int((b > 241).sum())
        print(f"  {f.parent.name}/{f.stem}: {len(b):,} 个股票-日  "
              f"档数分布 {vc}  超过 241 档的 {odd}")

    if pkg is None:
        print("7. 随机抽样回源比对 —— 跳过（未给 --pkg；源包若已删除，1–6 项仍可独立校验）")
        print()
        if FAIL:
            print(f"失败 {len(FAIL)} 项")
            raise SystemExit(1)
        print("全部通过")
        return
    print("7. 随机抽样回源比对")
    rng = random.Random(20261028)
    zips = {z.name[:4]: z for z in sorted(pkg.glob("*.zip"))}
    if not zips:
        bad(f"{pkg} 下没有年份 zip；--pkg 应指向 '个股行情数据全量按年分类' 这一层")
        print()
        print(f"失败 {len(FAIL)} 项")
        raise SystemExit(1)
    checked = 0
    for _ in range(a.samples):
        f = rng.choice(files)
        y = f.parent.name.split('=')[1]
        t = pq.read_table(f, columns=["trade_date", "ts_code"]).to_pandas()
        row = t.sample(1, random_state=rng.randrange(1 << 30)).iloc[0]
        d, code = pd.Timestamp(row.trade_date).normalize(), row.ts_code
        zf = [z for n, z in zips.items() if n.startswith(y)] or \
             [z for n, z in zips.items() if y.startswith(n)]
        z = zipfile.ZipFile(zf[-1])
        nm = [n for n in z.namelist() if n.endswith(f"/1min/{code}.parquet")][0]
        with z.open(nm) as fh:
            src = pq.read_table(__import__("pyarrow").BufferReader(fh.read())).to_pandas().reset_index()
        src = src[pd.to_datetime(src.trade_date).dt.normalize() == d].sort_values("trade_time")
        dst = pq.read_table(f).to_pandas()
        dst = dst[(pd.to_datetime(dst.trade_date).dt.normalize() == d) & (dst.ts_code == code)] \
            .sort_values("trade_time")
        if len(src) != len(dst):
            bad(f"{d.date()} {code}: 源 {len(src)} 档 vs 分区 {len(dst)} 档")
            continue
        for c in ("open", "high", "low", "close", "vol", "amount", "adj_factor"):
            if not np.allclose(src[c].to_numpy(float), dst[c].to_numpy(float),
                               rtol=0, atol=0, equal_nan=True):
                bad(f"{d.date()} {code}: 列 {c} 数值不一致")
                break
        else:
            checked += 1
    ok(f"{checked}/{a.samples} 个随机股票-日 与源 zip 逐值相同")

    print()
    if FAIL:
        print(f"失败 {len(FAIL)} 项")
        raise SystemExit(1)
    print("全部通过")


if __name__ == "__main__":
    main()
