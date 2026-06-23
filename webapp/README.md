# DES 可视化 Web App

实时游戏界面(验收之眼):选 BB0 对称局起始基因型 → 四阵营从四象限扩张 →
肉眼看红皇后共存动力学。后端 aiohttp 跑现有 Engine 流帧,前端用 Astro 构建静态产物,由 aiohttp 单端口托管。

## 启动

从 repo 根(PowerShell):

**1. 构建前端(首次或前端改动后):**

```powershell
cd webapp/frontend
npm install
npm run build
```

**2. 起服务(务必模块形式):**

```powershell
$env:PYTHONPATH='src'; D:/anaconda3/envs/basic/python.exe -m webapp.server
```

> 必须用 `-m webapp.server`(模块形式,从 repo 根跑)。`webapp/server.py` 用了绝对导入
> `from webapp.frame import ...`;直接 `python webapp/server.py` 会把 `webapp/` 放进 sys.path
> 而非 repo 根,导致 `ModuleNotFoundError: No module named 'webapp'`。

**3. 浏览器开 http://localhost:8000。**

> **注意:** 服务仅绑定 localhost:8000,无鉴权,为单用户本地演示专用,请勿暴露于不受信任的网络。

## 数据

每局边跑边写 parquet 到 `data/playground/<timestamp>-live.parquet`(隔离目录,
gitignore;绝不与 `data/runs/` 正式采集混池)。schema 同正式 run:
`(tick, cell_x, cell_y, strain, faction, count)`。

## 红线

四阵营共用同一 BB0 模板结构,各自插槽取值可不同;locked 位只读、只 6 槽可改;live 帧忠实张量(无平滑、无暂停/调速);
读数与 `scripts/analyze_batch.py` 共用 `webapp/readouts.py` 单一来源。
