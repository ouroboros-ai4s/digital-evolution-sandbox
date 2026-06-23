// ui-variants-carousel.js — 4 UI variants over shared hardcoded "running" data.
// ←/→ switches version. Center canvas + right readouts are identical across versions;
// LEFT rail (player config), LEGEND (horz/vert), PLAY control differ. ponytail: all fake.

const FC=[[255,77,77],[77,155,255],[63,208,127],[255,204,77]];
const FNAME=["阵营0","阵营1","阵营2","阵营3"];
const LOCKED={1:'F4Nr4',5:'BroadSweep',7:'P_base'};
const SLOTS=new Set([0,2,3,9,10,13]);
const PALETTE=['N0','F4Nr1','F4Nr4','P_base','P_hotspot','BroadSweep'];
const N=128,K=64,T=450,cx=[32,96,32,96],cy=[32,32,96,96];
const fmt=(n)=>n>=1000?(n/1000).toFixed(n>=10000?0:1)+"k":String(n);
const fmtPct=(v)=>(v*100).toFixed(1)+'%';

// per-player slot picks (all start N0 = symmetric); kept so accordions are independent
const playerSlots=[0,1,2,3].map(()=>({0:'N0',2:'N0',3:'N0',9:'N0',10:'N0',13:'N0'}));

// ---- procedural frame: 4 factions grow from quadrants, meet, contest ----
function blob(x,y,fx,fy,reach){const d=Math.hypot(x-fx,y-fy);return Math.max(0,1-d/reach);}
function makeFrame(tick){
  const reach=8+58*Math.min(1,tick/345);
  const cells=[]; const share=[0,0,0,0]; let total=0,occ=0;
  for(let y=0;y<N;y++)for(let x=0;x<N;x++){
    let c=[0,0,0,0],tot=0;
    for(let f=0;f<4;f++){const b=blob(x,y,cx[f],cy[f],reach);const v=Math.round(b*b*K);c[f]=v;tot+=v;}
    if(tot<=0) continue;
    cells.push([y,x,c[0],c[1],c[2],c[3]]);
    for(let f=0;f<4;f++) share[f]+=c[f];
    total+=tot; occ++;
  }
  for(let f=0;f<4;f++) share[f]=total?share[f]/total:0.25;
  const grown=Math.min(1,tick/345);
  const distinct=Math.round(200+18800*grown*(0.9+0.1*Math.sin(tick/9)));
  const n2=22+4*grown+1.5*Math.sin(tick/20);
  const dmax=0.04+0.06*grown;
  return {tick,H:N,W:N,cells,total,occ,share,distinct,n2,dmax};
}

// ---- compositing (verbatim from app.js) ----
let img=null,ctx=null;
function setupCanvas(){const cv=document.getElementById("grid");cv.width=N;cv.height=N;
  ctx=cv.getContext("2d");img=ctx.createImageData(N,N);}
function drawFrame(fr){
  img.data.fill(0);
  for(let i=3;i<img.data.length;i+=4) img.data[i]=255;
  for(const c of fr.cells){
    const y=c[0],x=c[1],cf=[c[2],c[3],c[4],c[5]];
    const tot=cf[0]+cf[1]+cf[2]+cf[3]; if(tot<=0) continue;
    let r=0,gg=0,bb=0;
    for(let f=0;f<4;f++){const w=cf[f]/tot;r+=FC[f][0]*w;gg+=FC[f][1]*w;bb+=FC[f][2]*w;}
    const dens=Math.min(1,tot/K),a=0.15+0.85*dens,o=(y*N+x)*4;
    img.data[o]=r*a;img.data[o+1]=gg*a;img.data[o+2]=bb*a;img.data[o+3]=255;
  }
  ctx.putImageData(img,0,0);
}
function line(id,series,colors,ymax){
  const cv=document.getElementById(id);if(!cv)return;const c=cv.getContext("2d");
  cv.width=cv.clientWidth;cv.height=cv.clientHeight;
  const W=cv.width,H=cv.height;c.clearRect(0,0,W,H);
  series.forEach((s,si)=>{if(!s.length)return;
    c.strokeStyle=colors[si];c.lineWidth=1.5;c.beginPath();
    s.forEach((v,i)=>{const px=s.length>1?i/(s.length-1)*W:0;const py=H-(v/ymax)*H*0.92-2;
      i?c.lineTo(px,py):c.moveTo(px,py);});c.stroke();});
}

// ===== shared HTML fragments =====
function genomeHTML(slots, compact){
  let h=`<div class="genome${compact?' compact':''}">`;
  for(let i=0;i<16;i++){
    if(LOCKED[i]!==undefined) h+=`<div class="pos locked">${LOCKED[i]}</div>`;
    else if(SLOTS.has(i)){
      const opts=PALETTE.map(p=>`<option ${p===(slots[i]||'N0')?'selected':''}>${p}</option>`).join('');
      h+=`<div class="pos slot"><select>${opts}</select></div>`;
    } else h+=`<div class="pos locked">N0</div>`;
  }
  return h+`</div>`;
}
function globalParamsHTML(){
  return `<h2 style="margin:8px 0 6px"><span>全局参数</span></h2>
    <div class="row"><span>网格</span><input value="128"></div>
    <div class="row"><span>K</span><input value="64"></div>
    <div class="row"><span>fill/格</span><input value="20"></div>
    <div class="row"><span>T (ticks)</span><input value="450"></div>
    <div class="row"><span>seed</span><input value="0"></div>
    <div class="row"><span>z_max</span><input value="8.0"></div>`;
}
// player accordion entry (V2/V3/V4). showGlobal: include per-player global? no — global is shared.
function playerAccordion(f, opts){
  const o=opts||{};
  const slots=playerSlots[f];
  const pct=o.pct!==undefined?`<span class="pct">${fmtPct(o.pct)}</span>`:'';
  const open=o.open?'open':'';
  return `<details class="player" ${open}>
     <summary><span class="dot" style="background:rgb(${FC[f]})"></span>
       <span>玩家 ${f} · ${FNAME[f]}</span>${pct}
       <span class="chev">▸</span></summary>
     <div class="pbody">
       <div style="font-size:10px;color:var(--dim);margin-bottom:6px">BB0 基因型(16 位)· 灰=锁死 蓝=插槽</div>
       ${genomeHTML(slots, o.compact)}
     </div></details>`;
}
function centerHTML(legendMode){
  let legend;
  if(legendMode==='vert'){
    legend=`<div class="legend vert" id="legend">`+
      [0,1,2,3].map(f=>`<div class="lrow"><i style="background:rgb(${FC[f]})"></i>${FNAME[f]}<span class="lpct" data-f="${f}">25.0%</span></div>`).join('')+
      `<div class="lrow" style="color:var(--dim);min-width:0">亮度=密度 · 混色=争夺</div></div>`;
  } else {
    legend=`<div class="legend horz"><span><i style="background:var(--f0)"></i>阵营0</span>
      <span><i style="background:var(--f1)"></i>阵营1</span><span><i style="background:var(--f2)"></i>阵营2</span>
      <span><i style="background:var(--f3)"></i>阵营3</span><span>亮度=密度 · 混色=争夺格</span></div>`;
  }
  return `<main class="stage">
    <div class="stagehead"><span class="title">验收之眼 — 128×128 世界</span>${legend}</div>
    <div class="canvaswrap"><canvas id="grid" width="128" height="128"></canvas></div>
    <div class="transport" id="transport"></div>
  </main>`;
}
function rightHTML(){
  return `<aside class="read">
    <div class="cards">
      <div class="card"><div class="k">tick</div><div class="v" id="m-tick">0</div></div>
      <div class="card"><div class="k">总个体</div><div class="v" id="m-total">0</div></div>
      <div class="card"><div class="k">占用格</div><div class="v" id="m-occ">0</div></div>
      <div class="card"><div class="k">活株</div><div class="v" id="m-distinct">0</div></div>
      <div class="card"><div class="k">N2</div><div class="v" id="m-n2">0</div></div>
      <div class="card"><div class="k">d_max</div><div class="v" id="m-dmax">0</div></div>
    </div>
    <h3>阵营占比(主判决曲线 → 全贴 0.25=共存)</h3>
    <canvas class="chart" id="shareChart"></canvas>
    <h3>多样性(distinct / N2)</h3>
    <canvas class="chart" id="divChart"></canvas>
    <h3>主导株排行(top 5)</h3><ul class="lead" id="lead"></ul>
    <h3>该格明细(点格子)</h3>
    <ul class="lead" id="cellDetail"><li style="color:var(--dim)">点中心 canvas 的格子查看</li></ul>
  </aside>`;
}

// ===== VERSION BUILDERS (left rail + center + right; transport differs) =====
const VERSIONS=[
  {name:'基线',desc:'当前 UI(单对称配置+横图例)',
   build:()=>`<aside class="rail" id="rail">
       <button class="railbtn" onclick="document.getElementById('rail').classList.toggle('collapsed')">⚙</button>
       <h2><span>配置(对称局)</span></h2>
       <div class="body">
         <div class="note">这是<b>对称起始基因型</b>(四阵营全同)<br>不是角色/阵营差异系统</div>
         <h2 style="margin:0 0 6px"><span>BB0 基因型(16 位)</span></h2>
         ${genomeHTML(playerSlots[0])}
         <div style="font-size:11px;color:var(--dim);margin-bottom:10px">灰=锁死功能位 · 蓝=可选插槽</div>
         ${globalParamsHTML()}
         <button class="go" id="startBtn">▶ 开始</button>
       </div></aside>${centerHTML('horz')}${rightHTML()}`,
   transport:()=>`<span class="ticklabel" id="tickLabel">tick 0 / 450</span>
     <span id="status" style="color:var(--dim)">运行中</span>
     <span style="color:var(--dim);margin-left:auto">点格子 → {strain:count}</span>`},

  {name:'核心',desc:'四玩家折叠+竖图例+▶⏸切换',
   build:()=>`<aside class="rail">
       <h2><span>玩家配置</span></h2>
       <div class="body">
         ${[0,1,2,3].map(f=>playerAccordion(f,{open:f===0})).join('')}
         ${globalParamsHTML()}
         <button class="go" id="startBtn">▶ 开始</button>
       </div></aside>${centerHTML('vert')}${rightHTML()}`,
   transport:()=>`<button class="pbtn run" id="playBtn">▶ 播放中</button>
     <span class="ticklabel" id="tickLabel">tick 0 / 450</span>
     <span style="color:var(--dim);margin-left:auto">点格子 → {strain:count}</span>`},

  {name:'增强',desc:'占比数字+速度滑块',
   build:()=>`<aside class="rail">
       <h2><span>玩家配置</span></h2>
       <div class="body">
         ${[0,1,2,3].map(f=>playerAccordion(f,{pct:0.25})).join('')}
         ${globalParamsHTML()}
         <div class="row"><span>播放速度</span><input type="range" min="1" max="5" value="3" id="speedSlider" style="width:90px"></div>
         <button class="go" id="startBtn">▶ 开始</button>
       </div></aside>${centerHTML('vert')}${rightHTML()}`,
   transport:()=>`<button class="pbtn run" id="playBtn">▶ 播放中</button>
     <span class="ticklabel" id="tickLabel">tick 0 / 450</span>
     <span style="margin-left:auto;color:var(--dim)">速度 <b id="speedInd">×3</b></span>`},

  {name:'极简',desc:'默认收起,只显 leading 玩家',
   build:()=>`<aside class="rail">
       <h2><span>玩家(展开配置)</span></h2>
       <div class="body">
         ${[0,1,2,3].map(f=>playerAccordion(f,{pct:0.25,compact:true})).join('')}
         ${globalParamsHTML()}
         <button class="go" id="startBtn">▶ 开始</button>
       </div></aside>${centerHTML('vert')}${rightHTML()}`,
   transport:()=>`<button class="pbtn run" id="playBtn">▶</button>
     <span class="ticklabel" id="tickLabel">tick 0 / 450</span>
     <span style="color:var(--dim);margin-left:auto">点格子钻取</span>`}
];

// ===== playback state (shared across versions) =====
let tick=0,timer=null,running=false,started=false;
const setStatus=(t)=>{const e=document.getElementById("status");if(e)e.textContent=t;};
const shareSeries=[[],[],[],[]],distinctSeries=[],n2Series=[];
function step(){
  const fr=makeFrame(tick);
  drawFrame(fr); updateReadouts(fr);
  tick+=3; if(tick>T){pause();setStatus("完成");}
}
function play(){
  if(running)return; running=true; started=true;
  clearInterval(timer); const speed=document.getElementById("speedSlider")?.value||3;
  timer=setInterval(step,Math.max(20,120-speed*15));
  const pb=document.getElementById("playBtn");
  if(pb){pb.textContent="⏸ 暂停";pb.classList.add("paused");}
  const sb=document.getElementById("startBtn");
  if(sb)sb.textContent="⟳ 重置";
  setStatus("运行中");
}
function pause(){
  running=false; clearInterval(timer);
  const pb=document.getElementById("playBtn");
  if(pb){pb.textContent="▶ 继续";pb.classList.remove("paused");}
  setStatus("已暂停");
}
function reset(){
  tick=0; for(const s of shareSeries)s.length=0; distinctSeries.length=0; n2Series.length=0;
  running=false; started=false; clearInterval(timer);
  const pb=document.getElementById("playBtn");
  if(pb){pb.textContent="▶ 播放中";pb.classList.remove("paused");}
  const sb=document.getElementById("startBtn");
  if(sb)sb.textContent="▶ 开始";
  setStatus("未开始");
}

function updateReadouts(fr){
  document.getElementById("m-tick").textContent=fr.tick;
  document.getElementById("m-total").textContent=fmt(fr.total);
  document.getElementById("m-occ").textContent=fmt(fr.occ);
  document.getElementById("m-distinct").textContent=fmt(fr.distinct);
  document.getElementById("m-n2").textContent=fr.n2.toFixed(1);
  document.getElementById("m-dmax").textContent=fr.dmax.toFixed(3);
  document.getElementById("tickLabel").textContent=`tick ${fr.tick} / ${T}`;
  for(let f=0;f<4;f++){shareSeries[f].push(fr.share[f]);
    const el=document.querySelector(`.lpct[data-f="${f}"]`);
    if(el)el.textContent=fmtPct(fr.share[f]);}
  distinctSeries.push(fr.distinct); n2Series.push(fr.n2);
  line("shareChart",shareSeries,["#ff4d4d","#4d9bff","#3fd07f","#ffcc4d"],1.0);
  const dmax=Math.max(1,...distinctSeries),nmax=Math.max(1,...n2Series);
  line("divChart",[distinctSeries.map(v=>v/dmax),n2Series.map(v=>v/nmax)],["#a371f7","#39c5cf"],1.0);
  renderLeaderboard();
}
function renderLeaderboard(){
  const lb=[['…F4Nr4.N0.BroadSweep…',0.018,0],['…F4Nr4.P_hotspot.N0…',0.014,1],
            ['…F4Nr4.F4Nr1.N0…',0.011,2],['…BroadSweep.N0.N0…',0.009,3],['…P_base.N0.N0…',0.007,1]];
  const ul=document.getElementById("lead");ul.innerHTML="";
  lb.forEach(([seq,fr,f])=>{const li=document.createElement("li");
    li.innerHTML=`<span class="seq" title="${seq}">${seq}</span>
      <span style="color:rgb(${FC[f]})">f${f}</span>
      <span style="margin-left:auto;font-variant-numeric:tabular-nums">${fmtPct(fr)}</span>`;
    ul.appendChild(li);});
}

// ===== version switcher =====
let curV=0;
function showVersion(i){
  curV=i; reset();
  const v=VERSIONS[i];
  document.getElementById("app").innerHTML=v.build();
  document.getElementById("transport").innerHTML=v.transport();
  document.getElementById("vNum").textContent=i+1;
  document.getElementById("vName").textContent=v.name;
  const dots=document.getElementById("vDots");
  dots.innerHTML=VERSIONS.map((_,j)=>`<i class="${j===i?'on':''}"></i>`).join('');
  setupCanvas(); reset(); drawFrame(makeFrame(0));   // static tick-0 frame, not running
  // rebind: startBtn = 开始(首次)/ 重置(已开始); playBtn = 暂停/继续(仅已开始时)
  const sb=document.getElementById("startBtn");if(sb)sb.onclick=()=>{
    if(started){reset();drawFrame(makeFrame(0));}else{play();}};
  const pb=document.getElementById("playBtn");if(pb)pb.onclick=()=>{
    if(!started)return; running?pause():play();};
  const sp=document.getElementById("speedSlider");if(sp)sp.oninput=()=>{
    document.getElementById("speedInd").textContent="×"+sp.value; if(running){pause();play();}};
  document.getElementById("grid").addEventListener("click",onCanvasClick);
}
function onCanvasClick(ev){
  const cv=document.getElementById("grid"),rect=cv.getBoundingClientRect();
  const x=Math.floor((ev.clientX-rect.left)/rect.width*N);
  const y=Math.floor((ev.clientY-rect.top)/rect.height*N);
  const fr=makeFrame(tick),hit=fr.cells.find(c=>c[0]===y&&c[1]===x);
  const ul=document.getElementById("cellDetail");ul.innerHTML="";
  if(!hit){ul.innerHTML=`<li style="color:var(--dim)">格 (${y},${x}) 空</li>`;return;}
  const cf=[hit[2],hit[3],hit[4],hit[5]],fakeStrains=['F4Nr4.N0.BroadSweep.N0','F4Nr4.P_hotspot.N0','F4Nr4.F4Nr1.N0'];
  cf.forEach((cnt,f)=>{if(cnt<=0)return;const li=document.createElement("li");
    li.innerHTML=`<span class="seq" title="${fakeStrains[f%3]}">${fakeStrains[f%3]}</span>
      <span style="color:rgb(${FC[f]})">f${f}</span>
      <span style="margin-left:auto;font-variant-numeric:tabular-nums">${cnt}</span>`;
    ul.appendChild(li);});
}

document.addEventListener("keydown",e=>{
  if(e.key==="ArrowLeft")showVersion(Math.max(0,curV-1));
  if(e.key==="ArrowRight")showVersion(Math.min(VERSIONS.length-1,curV+1));
});
showVersion(0);
