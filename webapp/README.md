# DES 可视化 Web App

实时游戏界面(验收之眼):选 BB0 对称局起始基因型 → 四阵营从四象限扩张 →
肉眼看红皇后共存动力学。后端 aiohttp 跑现有 Engine 流帧,前端纯 HTML/CSS/JS。

## 启动

从 repo 根(PowerShell):

```powershell
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m webapp.server
```

> 必须用 `-m webapp.server`(模块形式,从 repo 根跑)。`webapp/server.py` 用了绝对导入
> `from webapp.frame import ...`;直接 `python webapp/server.py` 会把 `webapp/` 放进 sys.path
> 而非 repo 根,导致 `ModuleNotFoundError: No module named 'webapp'`。

浏览器开 http://localhost:8000。

> **注意:** 服务仅绑定 localhost:8000,无鉴权,为单用户本地演示专用,请勿暴露于不受信任的网络。

## 数据

每局边跑边写 parquet 到 `data/playground/<timestamp>-live.parquet`(隔离目录,
gitignore;绝不与 `data/runs/` 正式采集混池)。schema 同正式 run:
`(tick, cell_x, cell_y, strain, faction, count)`。

## 红线

对称局四阵营同条 layout;locked 位只读、只 6 槽可改;live 帧忠实张量(无平滑);
读数与 `scripts/analyze_batch.py` 共用 `webapp/readouts.py` 单一来源。
