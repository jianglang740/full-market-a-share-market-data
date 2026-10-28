"""全量审计：逐年的股票数/交易日数/行数，逐月明细，以及"每日档数"分布。

产出 ``_build/_audit.json``，README 里的所有统计数字都从这里取。

关注点：常规股票每交易日 241 档（09:30 竞价 + 09:31–11:30 + 13:01–15:00）；
北交所（``.BJ``）另有 15:01–15:30 盘后固定价格交易，因此是 271 档。

用法::

    python -I 04_audit.py --out <输出根>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)

    years: dict[str, dict] = {}
    monthly = []
    for f in sorted((out / "minute" / "1min").rglob("*.parquet")):
        y = f.parent.name.split("=")[1]
        m = f.stem.split("=")[1]
        t = pq.read_table(f, columns=["trade_date", "ts_code", "trade_time"])
        g = t.group_by(["trade_date", "ts_code"]).aggregate([("trade_time", "count")])
        cnt = g.column("trade_time_count").to_numpy()
        codes = g.column("ts_code").to_numpy(zero_copy_only=False)
        bars = Counter(cnt.tolist())
        over = cnt > 241
        over_codes = set(codes[over].tolist()) if over.any() else set()
        d = years.setdefault(y, {"months": 0, "rows": 0, "codes": set(), "days": set(),
                                 "over241": 0, "over_codes": set(), "bar_hist": Counter()})
        d["months"] += 1
        d["rows"] += t.num_rows
        d["codes"] |= set(t.column("ts_code").unique().to_pylist())
        d["days"] |= set(t.column("trade_date").unique().to_pylist())
        d["over241"] += int(over.sum())
        d["over_codes"] |= over_codes
        d["bar_hist"].update(bars)
        monthly.append({"ym": f"{y}-{m}", "rows": t.num_rows,
                        "codes": len(set(t.column("ts_code").unique().to_pylist())),
                        "days": len(set(t.column("trade_date").unique().to_pylist())),
                        "over241": int(over.sum()),
                        "bar_hist": {str(k): v for k, v in sorted(bars.items())}})

    res = {"years": {}, "monthly": monthly}
    for y in sorted(years):
        d = years[y]
        days = sorted(d["days"])
        res["years"][y] = {
            "months": d["months"],
            "rows": d["rows"],
            "codes": len(d["codes"]),
            "days": len(days),
            "first_day": str(days[0])[:10],
            "last_day": str(days[-1])[:10],
            "over241_stock_days": d["over241"],
            "over241_codes": sorted(d["over_codes"]),
            "bar_hist": {str(k): v for k, v in sorted(d["bar_hist"].items())},
        }
        r = res["years"][y]
        print(f"{y}: {r['days']:4d} 日  {r['codes']:5d} 只  {r['rows']:14,} 行  "
              f"超241档 {r['over241_stock_days']:8,} 个股票-日  "
              f"{r['first_day']} ~ {r['last_day']}", flush=True)

    (out / "_build" / "_audit.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1))
    print("写出", out / "_build" / "_audit.json")


if __name__ == "__main__":
    main()
