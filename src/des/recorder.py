# src/des/recorder.py
from __future__ import annotations
import queue, threading
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from des.phenotype_cache import StrainTable

_SCHEMA = pa.schema([
    ("tick", pa.int32()), ("cell_x", pa.int32()), ("cell_y", pa.int32()),
    ("strain", pa.string()), ("faction", pa.int8()), ("count", pa.int32()),
])

class Recorder:
    def __init__(self, path: str, table: StrainTable, queue_size: int = 4) -> None:
        self._table = table
        self._writer = pq.ParquetWriter(path, _SCHEMA)
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._exc: BaseException | None = None   # set if the writer thread dies
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while True:
                job = self._q.get()
                if job is None:
                    self._q.task_done()
                    break
                tick, ys, xs, sids, facs, cnts = job
                strains = [".".join(self._table.sequence_of(int(s))) for s in sids]
                batch = pa.record_batch([
                    pa.array([tick] * len(sids), pa.int32()),
                    pa.array(xs, pa.int32()),
                    pa.array(ys, pa.int32()),
                    pa.array(strains, pa.string()),
                    pa.array(facs, pa.int8()),
                    pa.array(cnts, pa.int32()),
                ], schema=_SCHEMA)
                self._writer.write_batch(batch)
                self._q.task_done()
        except BaseException as e:   # surface, never swallow — a crashed writer means data loss
            self._exc = e
            # drain so a bounded queue can't deadlock dump() on a dead thread
            while True:
                try:
                    self._q.get_nowait()
                    self._q.task_done()
                except queue.Empty:
                    break

    def _check_thread(self) -> None:
        if self._exc is not None:
            raise RuntimeError("Recorder writer thread died") from self._exc

    def dump(self, tick: int, world) -> None:
        self._check_thread()
        cnt = world.count.to("cpu")
        sid = world.strain_id.to("cpu")
        fac = world.faction.to("cpu")
        nz = torch.nonzero(cnt > 0, as_tuple=False)   # [M,3] = (y,x,k)
        ys = nz[:, 0].tolist(); xs = nz[:, 1].tolist()
        ks = nz[:, 2]
        sids = sid[nz[:, 0], nz[:, 1], ks].tolist()
        facs = fac[nz[:, 0], nz[:, 1], ks].tolist()
        cnts = cnt[nz[:, 0], nz[:, 1], ks].tolist()
        self._q.put((tick, ys, xs, sids, facs, cnts))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._q.put(None)
        self._thread.join()
        self._writer.close()
        self._check_thread()   # re-raise any error the writer hit before exit
