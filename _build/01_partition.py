"""把按年打包的行情 zip 重排成 Hive 分区（year=/month=）。

源：``个股行情数据全量按年分类/{年份}.zip``，内部结构
``<year>/1min/<code>.parquet``（每股一个文件，10 列）+ ``<year>/day/stock_daily.parquet``。

目标（``--out``）：:

    minute/1min/year=YYYY/month=MM.parquet   按 (trade_date, ts_code, trade_time) 排序
    daily/year=YYYY.parquet                  33 列，按 (trade_date, ts_code) 排序
    _manifest.json                           每文件行数 / 字节 / md5 / 来源

只做**搬运和重排**，不做任何数值变换：价格保持未复权，``adj_factor`` 原样保留，
``09:30`` 竞价档原样保留（不与其它档合并）。任何复权/对齐都是下游的事。

用法::

    python -I 01_partition.py --pkg <包根> --out <输出根> [--years 2009-2026]
                             [--months 2025-06] [--flush-rows 3000000]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

#: 统一的列顺序。源文件里 ``trade_date``/``trade_time`` 是 pandas 索引列，
#: 读出来位置不固定，一律按这里重排；``trade_date`` 放第一列供行组统计剪枝。
MIN_COLS = ["trade_date", "ts_code", "trade_time", "open", "high", "low", "close",
            "vol", "amount", "adj_factor"]
DAY_COLS = ["trade_date", "ts_code", "open", "high", "low", "close", "pre_close", "change",
            "pct_chg", "vol", "amount", "up_limit", "down_limit", "turnover_rate",
            "turnover_rate_f", "volume_ratio", "pe", "pe_ttm", "pb", "ps", "ps_ttm",
            "dv_ratio", "dv_ttm", "total_share", "float_share", "free_share",
            "total_mv", "circ_mv", "adj_factor", "suspend_timing", "suspend_type",
            "is_st", "listed_days"]

PQ_KW = dict(compression="zstd", compression_level=3, use_dictionary=True,
             write_statistics=True, version="2.6", row_group_size=1_000_000)


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(tbl: pa.Table, cols: list[str]) -> pa.Table:
    """按 cols 重排、统一类型、去掉 pandas schema metadata。

    必须显式 cast：2026 那批源文件是新版 pandas 写的，``ts_code`` 是 ``large_string``，
    其余年份是 ``string``；不统一的话 214 个分区会长出两种 schema。
    """
    have = tbl.column_names
    missing = [c for c in cols if c not in have]
    if missing:
        raise SystemExit(f"源文件缺列 {missing}；实际列 {have}")
    arrs = []
    for c in cols:
        a = tbl.column(c)
        if c == "ts_code":
            a = a.cast(pa.string())
        elif c in ("trade_date", "trade_time"):
            a = a.cast(pa.timestamp("ns"))
        arrs.append(a)
    return pa.Table.from_arrays(arrs, names=cols)


def month_col(tbl: pa.Table) -> pa.ChunkedArray:
    return pc.strftime(tbl.column("trade_date").cast(pa.timestamp("s")), format="%Y-%m")


def flush(buf: dict[str, list[pa.Table]], stage: Path, seq: list[int]) -> int:
    n = 0
    for ym, tables in sorted(buf.items()):
        if not tables:
            continue
        d = stage / ym
        d.mkdir(parents=True, exist_ok=True)
        tbl = pa.concat_tables(tables)
        pq.write_table(tbl, d / f"part_{seq[0]:04d}.parquet", **PQ_KW)
        seq[0] += 1
        n += tbl.num_rows
        tables.clear()
    return n


def read_min(f) -> pa.Table:
    """整块读进内存再解析。对 zip 流来说，让 parquet 自己 seek 会把同一段解压两遍。"""
    t = pq.read_table(pa.BufferReader(f.read()))
    return norm(t, MIN_COLS)


def finalize_month(stage_dir: Path, dst: Path, ym: str, root: Path) -> dict:
    parts = sorted(stage_dir.glob("part_*.parquet"))
    if not parts:
        raise SystemExit(f"{ym} 无 part")
    tbl = pa.concat_tables([pq.ParquetFile(p).read() for p in parts])
    tbl = tbl.sort_by([("trade_date", "ascending"), ("ts_code", "ascending"),
                       ("trade_time", "ascending")])
    # 键唯一：排序后相邻三列同时相等即为重复
    n = tbl.num_rows
    if n > 1:
        same = None
        for c in ("trade_date", "ts_code", "trade_time"):
            col = tbl.column(c)
            eq = pc.equal(col[1:], col[:-1])
            same = eq if same is None else pc.and_(same, eq)
        ndup = pc.sum(pc.cast(same, pa.int64())).as_py()
        if ndup:
            raise SystemExit(f"{ym}: 主键 (trade_date, ts_code, trade_time) 重复 {ndup} 行")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    pq.write_table(tbl, tmp, **PQ_KW)
    os.replace(tmp, dst)
    shutil.rmtree(stage_dir, ignore_errors=True)
    # path 存相对根目录，数据集整体挪位置时 manifest 不会失效
    return {"path": dst.relative_to(root).as_posix(), "rows": n,
            "size_bytes": dst.stat().st_size, "md5": md5_file(dst), "parts": len(parts)}


def do_year_min(zf: Path, year: int, stage: Path, out: Path,
                flush_rows: int, manifest: dict, only_months: set[str] | None) -> None:
    z = zipfile.ZipFile(zf)
    names = sorted(n for n in z.namelist() if "/1min/" in n and n.endswith(".parquet"))
    if not names:
        raise SystemExit(f"{zf.name}: 没有 1min 文件")
    buf: dict[str, list[pa.Table]] = {}
    seq = [0]
    bufrows = 0
    src_rows = 0
    t0 = time.time()
    for i, nm in enumerate(names, 1):
        try:
            with z.open(nm) as f:
                tbl = read_min(f)
        except Exception as e:                     # noqa: BLE001
            # 坏成员（CRC 不符 / 头损坏）不中断整年，记下来由 02_verify 报出
            manifest.setdefault("corrupt", {}).setdefault(str(year), []).append(
                {"file": nm, "error": f"{type(e).__name__}: {e}"})
            print(f"  ! {nm} 损坏，跳过：{type(e).__name__}", flush=True)
            continue
        src_rows += tbl.num_rows
        ymcol = month_col(tbl)
        for ym in pc.unique(ymcol).to_pylist():
            if only_months and ym not in only_months:
                continue
            buf.setdefault(ym, []).append(tbl.filter(pc.equal(ymcol, ym)))
        bufrows += tbl.num_rows
        if bufrows >= flush_rows:
            flush(buf, stage, seq)
            bufrows = 0
        if i % 500 == 0:
            print(f"  {year}: {i}/{len(names)} 只  {time.time()-t0:6.1f}s", flush=True)
    if buf:
        flush(buf, stage, seq)

    months = sorted(d.name for d in stage.iterdir() if d.is_dir())
    for ym in months:
        rec = finalize_month(stage / ym, out / f"year={ym[:4]}" / f"month={ym[5:]}.parquet",
                             ym, out)
        manifest["minute"][ym] = rec
        print(f"  {year}: {ym} -> {rec['rows']:>12,} 行  {rec['size_bytes']/2**20:7.1f} MiB", flush=True)
    manifest["source"][str(year)] = {"zip": zf.name, "zip_bytes": zf.stat().st_size,
                                     "files_1min": len(names), "rows_1min": src_rows}


def do_year_day(zf: Path, out: Path, manifest: dict) -> None:
    z = zipfile.ZipFile(zf)
    names = [n for n in z.namelist() if "/day/" in n and n.endswith(".parquet")]
    if len(names) != 1:
        raise SystemExit(f"{zf.name}: day 文件数 {len(names)}，预期 1")
    with z.open(names[0]) as f:
        tbl = norm(pq.read_table(pa.BufferReader(f.read())), DAY_COLS)
    tbl = tbl.sort_by([("trade_date", "ascending"), ("ts_code", "ascending")])
    dst = out / f"year={tbl.column('trade_date')[0].as_py().year}.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, dst, **PQ_KW)
    manifest["daily"][dst.stem] = {"path": dst.relative_to(out).as_posix(),
                                   "rows": tbl.num_rows,
                                   "size_bytes": dst.stat().st_size, "md5": md5_file(dst)}
    print(f"  day: {dst.name}  {tbl.num_rows:,} 行", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--years", default="2009-2026")
    ap.add_argument("--months", default=None, help="只做这些月份，逗号分隔，如 2025-06")
    ap.add_argument("--flush-rows", type=int, default=3_000_000)
    a = ap.parse_args()

    pkg, out = Path(a.pkg), Path(a.out)
    lo, hi = (int(x) for x in a.years.split("-"))
    only_months = set(a.months.split(",")) if a.months else None
    out.mkdir(parents=True, exist_ok=True)
    stage_root = out / "_stage"
    stage_root.mkdir(exist_ok=True)

    manifest_path = out / "_manifest.json"
    manifest = {"minute": {}, "daily": {}, "source": {},
                "columns": {"minute": MIN_COLS, "daily": DAY_COLS},
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    zips = sorted(pkg.glob("*.zip"))
    for y in range(lo, hi + 1):
        cand = [z for z in zips if z.name.startswith(str(y))]
        if not cand:
            print(f"{y}: 跳过（无 zip）")
            continue
        zf = sorted(cand)[-1]          # 同一年多个 zip 时取名字最大的（更新到更晚的）
        print(f"{y}: {zf.name}")
        do_year_min(zf, y, stage_root / str(y), out / "minute" / "1min",
                    a.flush_rows, manifest, only_months)
        do_year_day(zf, out / "daily", manifest)
        shutil.rmtree(stage_root / str(y), ignore_errors=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))

    print("完成：", manifest_path)


if __name__ == "__main__":
    main()
