// v3-enhanced.js — V3 standalone: 4-player accordion, vertical legend with live %,
// speed slider, start/reset button, pause/resume. No auto-start. ponytail: hardcoded data.

const FC=[[255,77,77],[77,155,255],[63,208,127],[255,204,77]];
const FNAME=["阵营0","阵营1","阵营2","阵营3"];
const LOCKED={1:'F4Nr4',5:'BroadSweep',7:'P_base'};
const SLOTS=new Set([0,2,3,9,10,13]);
const PALETTE=['N0','F4Nr1','F4Nr4','P_base','P_hotspot','BroadSweep'];
const N=128,K=64,T=450,cx=[32,96,32,96],cy=[32,32,96,96];
const fmt=(n)=>n>=1000?(n/1000).toFixed(n>=10000?0:1)+"k":String(n);
const fmtPct=(v)=>(v*100).toFixed(1)+'%';
const playerSlots=[0,1,2,3].map(()=>({0:'N0',2:'N0',3:'N0',9:'N0',10:'N0',13:'N0'}));

// ---- procedural frame ----
function blob(x,y,fx,fy,reach){const d=Math.hypot(x-fx,y-fy);return Math.max(0,1-d/reach);}
function makeFrame(tick){
  const reach=8+58*Math.min(1,tick/345);
  const cells=[],share=[0,0,0,0]; let total=0,occ=0;
  for(let y=0;y<N;y++)for(let x=0;x<N;x++){
    let c=[0,0,0,0],tot=0;
    for(let f=0;f<4;f++){const b=blob(x,y,cx[f],cy[f],reach);const v=Math.round(b*b*K);c[f]=v;tot+=v;}
    if(tot<=0) continue;
    cells.push([y,x,c[0],c[1],c[2],c[3]]);
    for(let f=0;f<4;f++)share[f]+=c[f];
    total+=tot; occ++;
  }
  for(let f=0;f<4;f++)share[f]=total?share[f]/total:0.25;
  const grown=Math.min(1,tick/345);
  const distinct=Math.round(200+18800*grown*(0.9+0.1*Math.sin(tick/9)));
  const n2=22+4*grown+1.5*Math.sin(tick/20);
  const dmax=0.04+0.06*grown;
  return {tick,H:N,W:N,cells,total,occ,share,distinct,n2,dmax};
}

// ---- compositing ----
let img=null,ctx=null;
function setupCanvas(){const cv=document.getElementById("grid");cv.width=N;cv.height=N;
  ctx=cv.getContext("2d");img=ctx.createImageData(N,N);}
function drawFrame(fr){
  img.data.fill(0);
  for(let i=3;i<img.data.length;i+=4)img.data[i]=255;
  for(const c of fr.cells){
    const y=c[0],x=c[1],cf=[c[2],c[3],c[4],c[5]];
    const tot=cf[0]+cf[1]+cf[2]+cf[3]; if(tot<=0)continue;
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

// ---- build player accordions + legend ----
function buildPlayers(){
  const h=[];
  for(let f=0;f<4;f++){
    const slots=playerSlots[f];
    h.push(`<details class="player" ${f===0?'open':''}>
       <summary><span class="dot" style="background:rgb(${FC[f]})"></span>
         <span>玩家 ${f} · ${FNAME[f]}</span><span class="pct">25.0%</span>
         <span class="chev">▸</span></summary>
       <div class="pbody">
         <div style="font-size:10px;color:var(--dim);margin-bottom:6px">BB0 基因型(16 位)· 灰=骨架噪声 · 深=功能锁死 · 蓝=可选插槽</div>
         <div class="genome-list">`);
    for(let i=0;i<16;i++){
      const idx=`<span class="gi">#${i}</span>`;
      if(LOCKED[i]!==undefined){
        h.push(`<div class="grow fn">${idx}<span class="gtype">功能·锁死</span><span class="gtag">${LOCKED[i]}</span></div>`);
      }else if(SLOTS.has(i)){
        const opts=PALETTE.map(p=>`<option ${p===(slots[i]||'N0')?'selected':''}>${p}</option>`).join('');
        h.push(`<div class="grow slot">${idx}<span class="gtype">插槽</span><select>${opts}</select></div>`);
      }else{
        h.push(`<div class="grow bb">${idx}<span class="gtype">骨架</span><span class="gtag">N0</span></div>`);
      }
    }
    h.push(`</div></div></details>`);
  }
  document.getElementById("players").innerHTML=h.join('');
}
function buildLegend(){
  const h=[0,1,2,3].map(f=>`<div class="lrow"><i style="background:rgb(${FC[f]})"></i>${FNAME[f]}<span class="lpct" data-f="${f}">25.0%</span></div>`);
  h.push(`<div class="lrow" style="color:var(--dim);min-width:0">亮度=密度 · 混色=争夺</div>`);
  document.getElementById("legend").innerHTML=h.join('');
}

// ---- playback ----
let tick=0,timer=null,running=false,started=false;
const shareSeries=[[],[],[],[]],distinctSeries=[],n2Series=[];
function step(){
  const fr=makeFrame(tick);
  drawFrame(fr); updateReadouts(fr);
  tick+=3; if(tick>T){pause();document.getElementById("startBtn").textContent="▶ 开始";}
}
function play(){
  if(running)return; running=true; started=true;
  clearInterval(timer); const speed=document.getElementById("speedSlider").value||3;
  timer=setInterval(step,Math.max(20,120-speed*15));
  document.getElementById("playBtn").textContent="⏸ 暂停";
  document.getElementById("startBtn").textContent="⟳ 重置";
}
function pause(){
  running=false; clearInterval(timer);
  document.getElementById("playBtn").textContent="▶ 继续";
}
function reset(){
  tick=0; for(const s of shareSeries)s.length=0; distinctSeries.length=0; n2Series.length=0;
  running=false; started=false; clearInterval(timer);
  document.getElementById("playBtn").textContent="▶ 播放中";
  document.getElementById("startBtn").textContent="▶ 开始";
  drawFrame(makeFrame(0));
}

function updateReadouts(fr){
  document.getElementById("m-tick").textContent=fr.tick;
  document.getElementById("m-total").textContent=fmt(fr.total);
  document.getElementById("m-occ").textContent=fmt(fr.occ);
  document.getElementById("m-distinct").textContent=fmt(fr.distinct);
  document.getElementById("m-n2").textContent=fr.n2.toFixed(1);
  document.getElementById("m-dmax").textContent=fr.dmax.toFixed(3);
  document.getElementById("tickLabel").textContent=`tick ${fr.tick} / ${T}`;
  for(let f=0;f<4;f++){
    shareSeries[f].push(fr.share[f]);
    const pctEl=document.querySelector(`.player[open] .pct`);  // only update open accordion
    const lpctEl=document.querySelector(`.lpct[data-f="${f}"]`);
    if(lpctEl)lpctEl.textContent=fmtPct(fr.share[f]);
  }
  // update open player accordion pct
  document.querySelectorAll('.player').forEach((p,f)=>{
    const pct=p.querySelector('.pct');
    if(pct)pct.textContent=fmtPct(fr.share[f]);
  });
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

// ---- draggable column resizers (left rail / right readout) ----
function setupResizers(){
  const app=document.querySelector(".app");
  function drag(el,varName,col){
    el.addEventListener("mousedown",(e)=>{
      e.preventDefault(); el.classList.add("drag");
      const startX=e.clientX, startW=app.children[col].offsetWidth;  // read actual rendered width
      const move=(ev)=>{
        const dx=ev.clientX-startX;
        const w=Math.max(160,Math.min(560,startW+(col===0?dx:-dx)));
        app.style.setProperty(varName,w+"px");
      };
      const up=()=>{el.classList.remove("drag");
        document.removeEventListener("mousemove",move);document.removeEventListener("mouseup",up);
        line("shareChart",shareSeries,["#ff4d4d","#4d9bff","#3fd07f","#ffcc4d"],1.0);
        line("divChart",[distinctSeries.map(v=>v/Math.max(1,...distinctSeries)),n2Series.map(v=>v/Math.max(1,...n2Series))],["#a371f7","#39c5cf"],1.0);
      };
      document.addEventListener("mousemove",move);document.addEventListener("mouseup",up);
    });
  }
  drag(document.getElementById("lresizer"),"--lw",0);   // col 0 = .rail
  drag(document.getElementById("rresizer"),"--rw",4);   // col 4 = .read
}

// ---- init ----
window.addEventListener("DOMContentLoaded",()=>{
  buildPlayers(); buildLegend(); setupCanvas(); setupResizers(); reset();
  document.getElementById("startBtn").onclick=()=>{started?reset():play();};
  document.getElementById("playBtn").onclick=()=>{if(!started)return; running?pause():play();};
  document.getElementById("speedSlider").oninput=()=>{
    document.getElementById("speedInd").textContent="×"+document.getElementById("speedSlider").value;
    if(running){pause();play();}};
  document.getElementById("grid").addEventListener("click",(ev)=>{
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
  });
});
