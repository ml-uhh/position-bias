"use strict";
(() => {
  const opening=document.querySelector('.opening');
  const openingControl=document.getElementById('opening-control');
  const reducedMotion=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let openingTimer;
  function finishOpening(){
    document.documentElement.classList.add('intro-done');
    opening.classList.remove('replaying');
    openingControl.textContent='Replay illustration';
  }
  if(reducedMotion){finishOpening();openingControl.hidden=true;}
  else openingTimer=setTimeout(finishOpening,6600);
  openingControl.addEventListener('click',()=>{
    clearTimeout(openingTimer);
    if(!document.documentElement.classList.contains('intro-done')){finishOpening();return;}
    if(opening.classList.contains('replaying')){finishOpening();return;}
    if(reducedMotion)return;
    opening.classList.remove('replaying');
    // Restart only the illustration; keep the explanation visible on replay.
    void opening.offsetWidth;
    opening.classList.add('replaying');
    openingControl.textContent='Finish illustration';
    openingTimer=setTimeout(finishOpening,6100);
  });
  const data = window.POSITION_BIAS_DATA;
  if (!data) return;
  const variants = ["attention", "fixed", "adaptive"];
  const names = {attention:"Attention only",fixed:"Fixed residual",adaptive:"Adaptive residual"};
  const rules = {attention:"λ = 1",fixed:"λ = ½",adaptive:"Measured layerwise λ"};
  const ns = "http://www.w3.org/2000/svg";
  const slider = document.getElementById("depth");
  const button = document.getElementById("play");
  const note = document.getElementById("frame-note");
  const select = document.getElementById("rope-model");
  let layer = data.depth, playing = false, lastTime = 0, handle = null;
  const panels = new Map();

  function el(tag, attrs = {}, text = null) {
    const node = document.createElementNS(ns, tag);
    for (const [key,value] of Object.entries(attrs)) node.setAttribute(key,value);
    if (text !== null) node.textContent = text;
    return node;
  }
  function createPanels(id) {
    const parent = document.getElementById(id);
    const list = variants.map(variant => {
      const panel = document.createElement("div");
      panel.className = "chart-panel"; panel.dataset.variant = variant;
      const heading = document.createElement("div"); heading.className = "chart-header";
      const headingGroup = document.createElement("div"); headingGroup.className = "chart-heading";
      const approach = document.createElement("div"); approach.className = "approach-label";
      approach.textContent = variant === "adaptive" ? "Our approach" : "Baseline";
      const h3 = document.createElement("h3"); h3.textContent = names[variant];
      const rule = document.createElement("div"); rule.className = "chart-rule"; rule.textContent = rules[variant];
      headingGroup.append(approach,h3);
      heading.append(headingGroup,rule);
      const svg = el("svg",{role:"img"});
      const metric = document.createElement("p"); metric.className = "distance";
      const metricName = document.createElement("span"); metricName.textContent = "Profile distance";
      const value = document.createElement("strong"); value.textContent = "—";
      metric.append(metricName,value);
      const tooltip = document.createElement("div"); tooltip.className = "tooltip"; tooltip.hidden = true;
      panel.append(heading,svg,metric,tooltip); parent.append(panel);
      return {panel,svg,value,variant,tooltip};
    });
    panels.set(id,list);
  }
  function axisCeiling(value) {
    if (value <= 0) return 1;
    const p = 10 ** Math.floor(Math.log10(value));
    return Math.ceil(value / p * 2) / 2 * p;
  }
  function draw(target,prediction,measurement,metric,description) {
    const {panel,svg,value,tooltip} = target;
    const width = Math.max(270,panel.getBoundingClientRect().width);
    const height = width < 410 ? 230 : 250;
    const margin = {left:48,right:10,top:14,bottom:44};
    const w = width - margin.left - margin.right, h = height - margin.top - margin.bottom;
    const peak = Math.max(...prediction,...(measurement || []));
    const ceiling = Math.min(1,axisCeiling(peak * 1.06));
    const x = i => margin.left + i / (data.n-1) * w;
    const y = p => margin.top + h * (1 - p / ceiling);
    const fmt = p => ceiling >= .1 ? p.toFixed(1) : p.toFixed(3);
    svg.replaceChildren(); svg.setAttribute("viewBox",`0 0 ${width} ${height}`);
    svg.append(el("title",{},`${names[target.variant]}: ${description}`));
    svg.append(el("desc",{},"Normalized influence by input token position. A separate linear y axis is used for each panel. Dashed black curve, when present, is measured full-model influence."));
    for (let i=0;i<=3;i++) {
      const p = ceiling * i/3, yy = y(p);
      svg.append(el("line",{x1:margin.left,x2:width-margin.right,y1:yy,y2:yy,class:"grid-line"}));
      svg.append(el("text",{x:margin.left-7,y:yy+4,"text-anchor":"end",class:"tick-label"},fmt(p)));
    }
    svg.append(el("line",{x1:margin.left,x2:width-margin.right,y1:y(0),y2:y(0),class:"axis-line"}));
    svg.append(el("line",{x1:margin.left,x2:margin.left,y1:margin.top,y2:y(0),class:"axis-line"}));
    for (const i of [0,127,255]) svg.append(el("text",{x:x(i),y:height-25,"text-anchor":i===0?"start":i===255?"end":"middle",class:"tick-label"},String(i+1)));
    svg.append(el("text",{x:margin.left+w/2,y:height-6,"text-anchor":"middle",class:"axis-label"},"Input token position"));
    svg.append(el("text",{transform:`translate(12 ${margin.top+h/2}) rotate(-90)`,"text-anchor":"middle",class:"axis-label"},"Normalized influence"));
    const path = p => p.map((v,i)=>`${i?"L":"M"}${x(i).toFixed(2)},${y(v).toFixed(2)}`).join(" ");
    svg.append(el("path",{d:path(prediction),class:"prediction"}));
    if(measurement) svg.append(el("path",{d:path(measurement),class:"observation"}));
    value.textContent = metric === null ? "At final layer" : metric.toFixed(2);
    const guide = el("line",{y1:margin.top,y2:y(0),class:"hover-guide",visibility:"hidden"});svg.append(guide);
    const hit = el("rect",{x:margin.left,y:margin.top,width:w,height:h,fill:"transparent"});svg.append(hit);
    hit.addEventListener("pointermove",event=>{
      const box=svg.getBoundingClientRect();
      const xx=(event.clientX-box.left)*width/box.width;
      const i=Math.max(0,Math.min(data.n-1,Math.round((xx-margin.left)/w*(data.n-1))));
      guide.setAttribute("x1",x(i));guide.setAttribute("x2",x(i));guide.setAttribute("visibility","visible");
      tooltip.textContent=`Position ${i+1}: ${(prediction[i]*100).toFixed(2)}%`;
      if(measurement) tooltip.textContent+=` · measured ${(measurement[i]*100).toFixed(2)}%`;
      tooltip.hidden=false;
    });
    hit.addEventListener("pointerleave",()=>{tooltip.hidden=true;guide.setAttribute("visibility","hidden");});
  }
  function renderAnimation() {
    slider.value=layer;
    document.getElementById("depth-value").value=`${layer} / ${data.depth}`;
    const atEnd=layer===data.depth;
    for(const panel of panels.get("animation-charts")) draw(panel,data.frames[panel.variant][layer],atEnd?data.measured:null,atEnd?data.w1[panel.variant]:null,`${data.model}, ${layer} composed layers`);
    if(layer===0) note.textContent="Before any transport, all rollout mass starts at the final token. Attention can move it backward; the residual path preserves its current position.";
    else if(atEnd) note.textContent="After all 36 layers, adaptive mixing retains both primacy and recency. Its profile is much closer to measured influence than either baseline.";
    else {
      const attention=data.frames.attention[layer].slice(0,64).reduce((a,b)=>a+b,0);
      const adaptive=data.frames.adaptive[layer].slice(192).reduce((a,b)=>a+b,0);
      note.textContent=`After ${layer} layer${layer===1?"":"s"}, attention-only transport assigns ${(attention*100).toFixed(0)}% of its mass to the first quarter. Adaptive mixing retains ${(adaptive*100).toFixed(0)}% in the last quarter.`;
    }
  }
  function renderRoPE() {
    const model=data.rope[Number(select.value)];
    for(const panel of panels.get("rope-charts")) draw(panel,model.profiles[panel.variant],model.profiles.measured,model.w1[panel.variant],`${model.name}, final rollout`);
  }
  function pause() {playing=false;cancelAnimationFrame(handle);button.textContent=layer===data.depth?"Play through layers":"Continue";button.setAttribute("aria-pressed","false");}
  function tick(now) {
    if(!playing)return;
    if(now-lastTime>=260){lastTime=now;layer++;renderAnimation();if(layer>=data.depth){pause();return;}}
    handle=requestAnimationFrame(tick);
  }
  button.addEventListener("click",()=>{
    if(playing){pause();return;}
    if(layer===data.depth)layer=0;
    playing=true;lastTime=performance.now();button.textContent="Pause";button.setAttribute("aria-pressed","true");renderAnimation();handle=requestAnimationFrame(tick);
  });
  slider.addEventListener("input",()=>{pause();layer=Number(slider.value);renderAnimation();button.textContent=layer===data.depth?"Play through layers":"Continue";});
  document.getElementById("final").addEventListener("click",()=>{layer=data.depth;pause();renderAnimation();});
  document.addEventListener("visibilitychange",()=>{if(document.hidden)pause();});
  data.rope.forEach((model,i)=>{const option=document.createElement("option");option.value=i;option.textContent=model.name;select.append(option);});
  select.addEventListener("change",renderRoPE);
  createPanels("animation-charts");createPanels("rope-charts");renderAnimation();renderRoPE();
  let resizeTimer;
  new ResizeObserver(()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{renderAnimation();renderRoPE();},100);}).observe(document.querySelector("main"));
})();
