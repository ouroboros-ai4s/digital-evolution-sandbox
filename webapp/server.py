"""aiohttp viz server for the DES engine. This module's pure helpers
(layout_from_slots / build_engine_from_config) are unit-tested without the
event loop; the aiohttp app + WebSocket live loop are added on top."""
from __future__ import annotations
import os
import datetime
import torch
from aiohttp import web
from des.recorder import Recorder
from des.run import (
    PALETTE, _DEFAULTS, layout_from_slots, build_engine_from_config, pick_device,
)
from webapp.frame import encode_frame, cell_detail
from webapp.drilldown import frame_at_tick, cell_at_tick, strain_trajectory

PLAYGROUND_DIR = os.path.join("data", "playground")
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
_DEVICE_KEY = web.AppKey("device", torch.device)
_LIVE_KEY = web.AppKey("live_runs", dict)   # parquet path -> running Engine


async def _config(request):
    return web.json_response({
        "palette": PALETTE, "slots": sorted(_SLOTS),
        "locked": {str(k): v for k, v in _LOCKED.items()},
        "defaults": _DEFAULTS,
    })


async def _frame_at_tick(request):
    path = request.query["path"]; tick = int(request.query["tick"])
    return web.json_response(frame_at_tick(path, tick))


async def _cell(request):
    q = request.query
    path = q["path"]; y = int(q["y"]); x = int(q["x"])
    live = request.app[_LIVE_KEY].get(path)
    if live is not None:
        # live run: the parquet footer isn't written until close(); read the
        # in-memory world instead (spec §6: live=memory, replay=parquet).
        return web.json_response(cell_detail(live.world, live.table, y, x))
    return web.json_response(cell_at_tick(path, int(q["tick"]), y, x))


async def _trajectory(request):
    q = request.query
    return web.json_response(strain_trajectory(q["path"], q["strain"]))


async def _index(request):
    index = os.path.join(_STATIC_DIR, "index.html")
    if not os.path.exists(index):
        raise web.HTTPServiceUnavailable(
            text="前端未构建:先 cd webapp/frontend && npm install && npm run build")
    return web.FileResponse(index)


async def _ws(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    device = request.app[_DEVICE_KEY]
    async for msg in ws:
        if msg.type != web.WSMsgType.TEXT:
            continue
        data = msg.json()
        if data.get("cmd") != "start":
            continue
        try:
            eng, cfg = build_engine_from_config(data.get("config") or {}, device)
        except ValueError as e:
            await ws.send_json({"event": "error", "msg": str(e)})
            continue
        os.makedirs(PLAYGROUND_DIR, exist_ok=True)
        tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = os.path.join(PLAYGROUND_DIR, f"{tag}-live.parquet")
        rec = Recorder(path, eng.table)
        request.app[_LIVE_KEY][path] = eng
        await ws.send_json({"event": "started", "config": _jsonable(cfg), "path": path})
        try:
            for _ in range(int(cfg["T"])):
                if ws.closed:
                    break
                eng.step()
                rec.dump(eng.T, eng.world)
                frame = encode_frame(eng.world, eng.table, eng.T, eng.H, eng.W)
                await ws.send_json(frame)   # engine-speed: no sleep, no buffer
        finally:
            rec.close()
            request.app[_LIVE_KEY].pop(path, None)
        if not ws.closed:
            await ws.send_json({"event": "done", "path": path})
    return ws


def _jsonable(cfg: dict) -> dict:
    out = dict(cfg)
    out["layouts"] = [list(lay) for lay in cfg["layouts"]]
    out["players"] = [{"slots": {str(k): v for k, v in (p.get("slots") or {}).items()}}
                      for p in cfg["players"]]
    return out


def make_app(device=None) -> web.Application:
    app = web.Application()
    app[_DEVICE_KEY] = pick_device(device)
    app[_LIVE_KEY] = {}
    app.router.add_get("/config", _config)
    app.router.add_get("/api/frame_at_tick", _frame_at_tick)
    app.router.add_get("/api/cell", _cell)
    app.router.add_get("/api/trajectory", _trajectory)
    app.router.add_get("/ws", _ws)
    app.router.add_get("/", _index)
    if os.path.isdir(os.path.join(_STATIC_DIR, "_astro")):
        app.router.add_static("/_astro", os.path.join(_STATIC_DIR, "_astro"))
    return app


def main() -> None:
    if not os.path.exists(os.path.join(_STATIC_DIR, "index.html")):
        print("[warn] 前端未构建:先 cd webapp/frontend && npm install && npm run build")
    web.run_app(make_app(), port=8000)


if __name__ == "__main__":
    main()
