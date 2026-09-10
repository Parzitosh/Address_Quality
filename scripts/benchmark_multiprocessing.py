"""
P1 multiprocessing benchmark.

Measures the geography-resolution workload using the same masters as the
production pipeline. It does NOT change production routing or enable
multiprocessing automatically.
"""
from pathlib import Path
from multiprocessing import get_context
import argparse, time, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(SCRIPT_DIR))
from master_loader import load_masters, prepare_indexes
from entity_resolution import GeographicResolver, normalize_pin

_WORKER = None

def _init_worker(masters_dir):
    global _WORKER
    masters = load_masters(masters_dir)
    indexes = prepare_indexes(masters)
    _WORKER = GeographicResolver(masters, indexes)

def _resolve(row):
    address, city, state, pin = row
    return _WORKER.resolve_address(address=address, city=city, state=state, pin=pin)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input", default=str(ROOT/"Cleaned_Address_Data.csv"))
    p.add_argument("--sample", type=int, default=2000)
    p.add_argument("--workers", type=int, default=2)
    args=p.parse_args()

    df=pd.read_csv(args.input)
    def pick(names):
        lookup={str(c).strip().lower():c for c in df.columns}
        for n in names:
            if n.lower() in lookup:return lookup[n.lower()]
        raise ValueError(f"Missing column; expected one of {names}")
    ac=pick(["Clean Full Address","Full Address"])
    cc=pick(["Clean City","City"])
    sc=pick(["Clean State","State"])
    pc=pick(["Clean Pincode","Pincode","PIN"])
    rows=df[[ac,cc,sc,pc]].fillna("").astype(str).head(args.sample).itertuples(index=False,name=None)
    rows=list(rows)
    masters_dir=ROOT/"masters"

    masters=load_masters(masters_dir); indexes=prepare_indexes(masters)
    resolver=GeographicResolver(masters,indexes)
    t=time.perf_counter()
    for row in rows: _resolve_local(resolver,row)
    single=time.perf_counter()-t

    ctx=get_context("spawn")
    t=time.perf_counter()
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(str(masters_dir),)) as pool:
        list(pool.imap(_worker_resolve, rows, chunksize=32))
    multi=time.perf_counter()-t

    print(f"Sample: {len(rows):,}")
    print(f"Single-process: {single:.2f}s ({len(rows)/single:.1f} rows/sec)")
    print(f"{args.workers}-process: {multi:.2f}s ({len(rows)/multi:.1f} rows/sec)")
    print(f"Speedup: {single/multi:.2f}x")
    print("Note: worker startup/master-loading is included in multiprocessing time.")

def _resolve_local(resolver,row):
    address,city,state,pin=row
    return resolver.resolve_address(address=address,city=city,state=state,pin=normalize_pin(pin))

def _worker_resolve(row):
    address,city,state,pin=row
    return _WORKER.resolve_address(address=address,city=city,state=state,pin=normalize_pin(pin))

if __name__=="__main__": main()
