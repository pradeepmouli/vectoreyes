"use strict";
const bundle=JSON.parse(document.getElementById('bundle-data').textContent);
const $=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg';
let track='shape',region=null,selection=null,viewport=null,trial=null;
const layers=new Set(['regions']);
const versions=bundle.versions;
function text(tag,value,cls){const e=document.createElement(tag);e.textContent=value;if(cls)e.className=cls;return e}
let svgSerial=0;
function svgNode(markup){const root=document.importNode(new DOMParser().parseFromString(markup,'image/svg+xml').documentElement,true),prefix='view'+(++svgSerial)+'-',ids=new Map();for(const e of [root,...root.querySelectorAll('[id]')])if(e.id){ids.set(e.id,prefix+e.id);e.id=prefix+e.id}for(const e of [root,...root.querySelectorAll('*')])for(const attr of [...e.attributes]){let value=attr.value;if(attr.localName==='href'&&value.startsWith('#')&&ids.has(value.slice(1)))value='#'+ids.get(value.slice(1));value=value.replace(/url\(\s*(['"]?)#([^)'"\s]+)\1\s*\)/g,(whole,quote,id)=>ids.has(id)?'url(#'+ids.get(id)+')':whole);if(value!==attr.value)e.setAttributeNS(attr.namespaceURI,attr.name,value)}return root}
function download(name,content,mime){const url=URL.createObjectURL(new Blob([content],{type:mime}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
function showReview(on){$('workspace').hidden=!on;$('handoff').hidden=on;$('art-tab').setAttribute('aria-pressed',String(!on));$('review-tab').setAttribute('aria-pressed',String(on));if(on)render()}
function current(){return trial?.candidate_version&&versions[trial.candidate_version]?trial.candidate_version:trial?.base_version&&versions[trial.base_version]?trial.base_version:Object.keys(versions)[0]}
function fullViewport(){const v=versions[current()];return [0,0,v?.inspection.width||1254,v?.inspection.height||1254]}
function applyViewport(){for(const s of document.querySelectorAll('.viewport>svg'))s.setAttribute('viewBox',(viewport||fullViewport()).join(' '))}
function labelForRegion(id){return id}
function inspect(id){selection=id;const snapshot=versions[current()]?.inspection;const r=snapshot?.regions.find(r=>r.id===region);let item=r;if(id&&r){for(const p of r.paths){if(p.id===id)item=p;for(const c of p.commands||[])if(c.id===id)item=c;for(const a of p.anchors||[])if(a.id===id||id.endsWith('/'+a.id))item=a}}$('selected-name').textContent=id||'Whole drawing';$('address').textContent=id||'No selection';$('metadata').textContent=item?JSON.stringify(item===r?{geometry_type:r.geometry?.type,path_editability:r.path_editability}:item,null,2):'No retained metadata';$('diagnostics').replaceChildren();if(r)for(const [k,v]of Object.entries(r.counts)){if(typeof v==='object')continue;const row=text('div','', 'metric');row.append(text('span',k.replaceAll('_',' ')),text('span',String(v)));$('diagnostics').appendChild(row)}for(const element of document.querySelectorAll('.viewport [data-command-id],.viewport [data-anchor-id]'))element.classList.toggle('selected',element.getAttribute('data-command-id')===id||element.getAttribute('data-anchor-id')===id);$('mapping-note').textContent=correspondingSource(id).length?'Retained source span available; source overlay shows the recorded supporting samples.':'Unmapped selection across versions: no counterpart is guessed from command indices.';renderSourceSelection();highlightCorrespondence()}
function correspondingSource(id,version=current()){const snapshot=versions[version]?.inspection;if(!id||!snapshot)return [];const feature=id.split('/').pop();let ids=[];for(const record of snapshot.correspondence||[]){for(const span of record.correspondence?.spans||[])if((span.output_command_ids||[]).includes(id)||span.feature_id===feature)ids.push(...span.source_command_ids||[]);for(const end of record.correspondence?.endpoints||[])if(end.output_command_id===id)ids.push(end.source_command_id)}return [...new Set(ids)]}
function highlightCorrespondence(){const source=new Set(correspondingSource(selection));for(const svg of document.querySelectorAll('.viewport>svg[data-version]'))for(const element of svg.querySelectorAll('[data-command-id],[data-anchor-id]')){const id=element.getAttribute('data-command-id')||element.getAttribute('data-anchor-id'),version=svg.dataset.version;const matched=version===current()?id===selection:source.size>0&&correspondingSource(id,version).some(id=>source.has(id));element.classList.toggle('selected',matched)}}
function renderSourceSelection(){const svg=$('source-svg');if(!svg)return;svg.querySelector('[data-source-selection]')?.remove();const ids=new Set(correspondingSource(selection));if(!ids.size)return;const group=document.createElementNS(NS,'g');group.setAttribute('data-source-selection','true');for(const path of versions[current()].inspection.source_paths){let run=[];const flush=()=>{if(run.length<2){run=[];return}const p=document.createElementNS(NS,'polyline');p.setAttribute('points',run.map(c=>c.end.join(',')).join(' '));p.setAttribute('fill','none');p.setAttribute('stroke','#864767');p.setAttribute('stroke-width','4');group.appendChild(p);run=[]};for(const c of path.commands){if(ids.has(c.id))run.push(c);else flush()}flush()}svg.appendChild(group)}
function sourcePane(){const s=document.createElementNS(NS,'svg');s.id='source-svg';s.setAttribute('viewBox',fullViewport().join(' '));const img=document.createElementNS(NS,'image');img.setAttribute('href',track==='fill'?(versions[current()]?.fill_reference||''):(bundle.source||''));const b=fullViewport();img.setAttribute('width',b[2]);img.setAttribute('height',b[3]);s.appendChild(img);return s}
function pane(title,svg){const p=text('div','', 'pane');p.appendChild(text('h2',title));const v=text('div','', 'viewport');v.appendChild(svg);let start=null;v.onpointerdown=e=>{if(e.target.closest('[data-command-id],[data-anchor-id],[data-region-id]'))return;start={x:e.clientX,y:e.clientY,box:[...(viewport||fullViewport())]};v.setPointerCapture(e.pointerId)};v.onpointermove=e=>{if(!start)return;const scale=Math.min(v.clientWidth/start.box[2],v.clientHeight/start.box[3]);viewport=[start.box[0]-(e.clientX-start.x)/scale,start.box[1]-(e.clientY-start.y)/scale,start.box[2],start.box[3]];applyViewport()};v.onpointerup=()=>start=null;v.onpointercancel=()=>start=null;p.appendChild(v);return p}
function versionSvg(version){const v=versions[version],s=svgNode(track==='shape'?v.inspector_svg:v.svg);s.dataset.version=version;if(track==='shape'){for(const e of s.querySelectorAll('[data-command-id]'))e.style.display=layers.has('commands')&&(!region||e.getAttribute('data-command-id').startsWith(region+'.'))?'':'none';for(const e of s.querySelectorAll('[data-anchor-id],[data-anchor-ref]')){const id=e.getAttribute('data-anchor-id')||e.getAttribute('data-anchor-ref');e.style.display=layers.has('features')&&(!region||id.startsWith(region+'.'))?'':'none'}for(const [attr,layer]of [['data-path-id','paths'],['data-region-id','regions'],['data-source-path-id','source']])for(const e of s.querySelectorAll('['+attr+']')){const selected=attr==='data-region-id'?e.getAttribute(attr)===region:!region||(attr==='data-source-path-id'?e.getAttribute('data-source-region-id')===region:e.getAttribute(attr).startsWith(region+'.'));e.style.display=layers.has(layer)&&(attr==='data-region-id'||selected)?'':'none'}}s.onclick=e=>{const target=e.target.closest('[data-command-id],[data-anchor-id],[data-region-id],[data-path-id]');if(!target)return;const id=target.getAttribute('data-command-id')||target.getAttribute('data-anchor-id')||target.getAttribute('data-region-id')||target.getAttribute('data-path-id');if(version!==current()){$('mapping-note').textContent='Parent selection: '+id+' in '+version+'. Select the candidate to author its plan; version IDs are not interchangeable.';return}if(target.hasAttribute('data-region-id'))selectRegion(id);else inspect(id)};return s}
function renderParameters(){
  renderPassHistory();
  if(!$('parameters')){
    const box=text('details','','card');box.id='parameters';
    box.append(text('summary','Parameters used'),text('p','','muted'));
    box.lastChild.id='parameter-summary';
    for(const [id,title] of [['requested-parameters','Submitted trial request'],['effective-parameters','Retained parameters for displayed artwork']]){
      box.append(text('h2',title));const pre=text('pre','');pre.id=id;box.append(pre);
    }
    const exact=text('a','Download exact trial record');exact.id='exact-request';box.append(exact);
    $('workspace').append(box);
  }
  const version=current(),parameters=versions[version]?.parameters;
  const options=parameters?.trace_options?.value||parameters?.observed_trace_options;
  const traceSummary=options?['fit_strategy','trace_level','max_sampling_colors'].filter(key=>options[key]!=null).map(key=>key.replaceAll('_',' ')+': '+options[key]).join(' · '):'';
  $('parameter-summary').textContent=(version?'Displayed artwork '+version+'. ':'')+(traceSummary?'Trace: '+traceSummary+'. ':'')+
    (trial?.execution==='failed'?'Rejected request: no candidate parameters were produced. ':'')+
    'Null fields are unavailable, not defaults. Per-action overrides are listed below.';
  $('requested-parameters').textContent=JSON.stringify(trial?.request||null,null,2);
  $('effective-parameters').textContent=JSON.stringify(parameters||{availability:'No captured parameters'},null,2);
  $('exact-request').hidden=!trial;
  if(trial)$('exact-request').href='trials/'+encodeURIComponent(trial.id)+'.json';
}

function renderPassHistory(){
  if(!bundle.passes?.length)return;
  if(!$('pass-history'))$('workspace').insertBefore($('pass-history-template').content.cloneNode(true),$('workspace').querySelector('.inspector'));
  $('pass-list').replaceChildren();
  for(const stage of bundle.passes){
    const button=text('option',(stage.status==='disabled'?'☐ ':'☑ ')+stage.name.replace(/_pass$/,'').replaceAll('_',' ')+
      ' · '+stage.status+(stage.version?' · '+stage.version:''),'pass-row');
    button.value=stage.version||'';button.selected=stage.version===current();
    button.disabled=!stage.version||!versions[stage.version];
    $('pass-list').append(button);
  }
  $('pass-list').onchange=()=>{trial=bundle.trials.find(t=>t.candidate_version===$('pass-list').value);if(trial)$('trial').value=trial.id;region=selection=viewport=null;render()};
}
let comparisonReference='parent';
let detailScale=4,detailOverlay=false,detailMetric='unsupported';
function renderDetailOverlay(){
  document.querySelectorAll('[data-detail-overlay]').forEach(e=>e.remove());
  if(!detailOverlay||track!=='shape')return;
  const value=versions[current()],key=detailMetric+'_curvature_per_px';
  for(const row of value?.boundary_detail?.regions||[]){
    if(region&&row.id!==region)continue;
    for(const boundary of row.boundaries){
      const scale=boundary.scales.find(s=>s.scale_px===detailScale);if(!scale)continue;
      const samples=[...scale.hotspots].filter(s=>s[key]>0).sort((a,b)=>b[key]-a[key]).slice(0,6);
      for(const svg of document.querySelectorAll('.viewport>svg')){
        const source=svg.id==='source-svg';if(!source&&svg.dataset.version!==current())continue;
        const group=document.createElementNS(NS,'g');group.setAttribute('data-detail-overlay','true');group.setAttribute('pointer-events','none');
        for(const sample of samples){
          const point=source?sample.source_point:sample.candidate_point,circle=document.createElementNS(NS,'circle');
          circle.setAttribute('cx',point[0]*boundary.policy.units_per_pixel);circle.setAttribute('cy',point[1]*boundary.policy.units_per_pixel);
          circle.setAttribute('r','4');circle.setAttribute('fill',detailMetric==='unsupported'?'#a23d70':'#24759a');circle.setAttribute('stroke','white');circle.setAttribute('stroke-width','1');
          const title=document.createElementNS(NS,'title');title.textContent=row.id+' / '+boundary.candidate_path_id+' · '+sample[key].toFixed(4)+' rad/px';circle.append(title);group.append(circle);
        }
        svg.append(group);
      }
    }
  }
}
function renderDetail(){
  let box=$('boundary-detail');if(!box){box=text('section','','card');box.id='boundary-detail';$('workspace').insertBefore(box,$('workspace').querySelector('.inspector'))}
  const value=versions[current()];box.replaceChildren();box.hidden=!value||track!=='shape';if(box.hidden)return;
  box.append(text('h2','Source-supported boundary detail'));
  box.append(text('p','Turning differences from retained material polylines, not a quality score. Fine-scale differences can be anti-aliasing or raster noise. Feature displacement can contribute to both directions.','muted'));
  const controls=text('div','','toolbar'),label=text('label','Scale '),select=document.createElement('select');select.setAttribute('aria-label','Boundary detail scale');
  const scales=[...new Set((value.boundary_detail?.regions||[]).flatMap(r=>r.boundaries.flatMap(b=>b.scales.map(s=>s.scale_px))))].sort((a,b)=>a-b);
  if(!scales.includes(detailScale)&&scales.length)detailScale=scales[0];
  for(const scale of scales){const option=text('option',scale+' px');option.value=scale;select.append(option)}
  select.value=detailScale;select.disabled=!scales.length;select.onchange=()=>{detailScale=Number(select.value);renderDetail();renderDetailOverlay()};label.append(select);controls.append(label);
  const locations=text('label','Show locations '),check=document.createElement('input');check.type='checkbox';check.checked=detailOverlay;check.onchange=()=>{detailOverlay=check.checked;renderDetailOverlay()};locations.append(check);controls.append(locations);
  const metric=document.createElement('select');metric.setAttribute('aria-label','Boundary detail locations');for(const kind of ['unsupported','missing']){const option=text('option',kind==='unsupported'?'Unsupported turning':'Missing turning');option.value=kind;metric.append(option)}metric.value=detailMetric;metric.onchange=()=>{detailMetric=metric.value;renderDetailOverlay()};controls.append(metric);
  if(value.boundary_detail_file){const link=text('a','Download full detail JSON','button');link.href=value.boundary_detail_file;link.download='';controls.append(link)}box.append(controls);
  const table=text('table'),head=text('tr');for(const title of ['Region / path','Unsupported (rad)','Missing (rad)','Signed mismatch (rad)','Alignment RMS (px)','Status']){const th=text('th',title);th.scope='col';head.append(th)}const thead=text('thead');thead.append(head);table.append(thead);const body=text('tbody');
  const rows=value.boundary_detail?.regions||[];
  for(const row of rows){if(region&&row.id!==region)continue;
    const boundaries=row.boundaries.length?row.boundaries:[null];for(const boundary of boundaries){
      const scale=boundary?.scales.find(s=>s.scale_px===detailScale),tr=text('tr');tr.dataset.detailRegionId=row.id;
      const th=text('th',row.id+(boundary?' / '+boundary.candidate_path_id:''));th.scope='row';tr.append(th);
      for(const item of [scale?.unsupported_turn_radians,scale?.missing_turn_radians,scale?.signed_turn_difference_radians,boundary?.alignment_rms_px])tr.append(text('td',item==null?'—':item.toFixed(3)));
      tr.append(text('td',(row.reason||boundary?.reason||boundary?.status||'not_assessed').replaceAll('_',' ')));body.append(tr);
    }
  }
  table.append(body);const scroll=text('div','','count-table');scroll.append(table);box.append(scroll);
  if(!rows.length)box.append(text('p','Not assessed: no captured source-boundary evidence.','muted'));
  box.append(text('p','Alignment uses cyclic spatial matching and normalized arc length—not semantic feature correspondence. Locations mark up to six sampled differences per boundary, not detected corners. Full JSON includes provenance, settings, and every sample.','muted'));
}
function renderCounts(){
  renderDetail();
  queueMicrotask(renderDetailOverlay);
  let box=$('region-counts');
  if(!box){
    box=text('section','','card');box.id='region-counts';
      $('workspace').append(box);
  }
  box.replaceChildren();
  const version=current(),value=versions[version];box.hidden=!value;if(!value)return;
  box.append(text('h2','Per-region complexity'));
  const controls=text('div','','toolbar'),label=text('label','Compare with '),select=document.createElement('select');
  select.setAttribute('aria-label','Count comparison reference');
  for(const [key,title] of [['parent','Parent'],['baseline','Baseline']]){
    const option=text('option',title);option.value=key;option.disabled=!value.comparisons?.[key];select.append(option);
  }
  if(!value.comparisons?.[comparisonReference])comparisonReference='baseline';
  select.value=comparisonReference;select.onchange=()=>{comparisonReference=select.value;renderCounts()};
  label.append(select);controls.append(label);
  const exportButton=text('button','Download counts JSON');
  exportButton.onclick=()=>download(version+'-counts.json',JSON.stringify({drawing_id:value.inspection.drawing_id,version,
    regions:value.inspection.regions.map(r=>({id:r.id,counts:r.counts})),comparisons:value.comparisons},null,2),'application/json');
  controls.append(exportButton);box.append(controls);
  const comparison=value.comparisons?.[comparisonReference];
  const comparable=comparison?.status==='compared';
  box.append(text('p',(comparable?'Comparing '+comparison.reference_version+' → '+version+'. ':'Comparison unavailable; showing '+version+' only. ')+
    'Local retained paths and commands, excluding moves. Primitive and clone geometry has no local path commands. Counts are not a quality verdict.','muted'));
  const table=text('table'),head=text('tr');
  for(const title of ['Region','Paths','L / Q / C / A','Free','Corner-owned','Corners (point / rounded)','Δ segments','Δ corners','Match']){
    const th=text('th',title);th.scope='col';head.append(th);
  }
  const thead=text('thead');thead.append(head);table.append(thead);const body=text('tbody');
  const rows=comparable?comparison.regions:value.inspection.regions.map(r=>({id:r.id,after:r.counts,status:'comparison_unavailable'}));
  const number=v=>v==null?'Unavailable':String(v);
  const types=c=>c?['line','quadratic','cubic','arc'].map(k=>c.by_type?.[k]??0).join(' / '):'—';
  const corners=c=>c?c.corners==null?'Unavailable':c.corners+' ('+c.point_corners+' / '+c.rounded_corners+')':'—';
  const pair=(row,format)=>row.before?format(row.before)+' → '+(row.after?format(row.after):'—'):row.after?format(row.after):'—';
  const delta=v=>v==null?'—':v>0?'+'+v:String(v);
  for(const row of rows){
    if(region&&row.id!==region)continue;
    const tr=text('tr');tr.dataset.regionCountId=row.id;
    const th=text('th',row.id);th.scope='row';tr.append(th);
    for(const cell of [pair(row,c=>number(c.paths)),pair(row,types),pair(row,c=>number(c.free_commands)),
      pair(row,c=>number(c.corner_commands)),pair(row,corners),delta(row.delta?.commands),delta(row.delta?.corners),row.status.replaceAll('_',' ')])tr.append(text('td',cell));
    body.append(tr);
  }
  table.append(body);const scroll=text('div','','count-table');scroll.append(table);box.append(scroll);
  box.append(text('p','Unavailable means missing retained evidence, not zero detected features. Deltas require the same drawing, region ID and source-region lineage; merges and splits are not guessed.','muted'));
}
function render(){const version=current();renderParameters();renderCounts();$('panes').replaceChildren();if(!version){$('panes').appendChild(text('p','No published artwork. Inspect the trial error below.'));return}const r=versions[version].inspection;$('regions').replaceChildren();for(const item of r.regions){const b=text('button',labelForRegion(item.id),'region');b.setAttribute('aria-pressed',String(item.id===region));b.onclick=()=>selectRegion(item.id);$('regions').appendChild(b)}$('panes').appendChild(pane('Original source',sourcePane()));if(track==='shape'&&trial?.base_version&&versions[trial.base_version]&&trial.base_version!==version)$('panes').appendChild(pane('Parent · '+trial.base_version,versionSvg(trial.base_version)));$('panes').appendChild(pane((trial?.candidate_version===version?'Candidate':'Baseline / parent')+' · '+version,versionSvg(version)));applyViewport();inspect(selection||region);$('export-svg').disabled=false;$('export-png').disabled=false;$('mapping-note').textContent=track==='fill'?'Fill track: geometry remains fixed for each exported version. Cross-version paint verdicts require verified common geometry/support; no fill score is inferred from these pictures.':$('mapping-note').textContent}
function selectRegion(id){region=id;selection=id;layers.add('commands');layers.add('features');for(const check of document.querySelectorAll('#layers input'))check.checked=layers.has(check.value);render()}
function focus(){if(!region)return;const svg=versionSvg(current());svg.style.position='absolute';svg.style.visibility='hidden';document.body.appendChild(svg);const paths=svg.querySelectorAll('[data-path-id]');let boxes=[];for(const p of paths)if(p.getAttribute('data-path-id').startsWith(region+'.')){p.style.display='';const box=p.getBBox();if(box.width||box.height)boxes.push(box)}if(!boxes.length){svg.remove();$('mapping-note').textContent='No exact local path bounds. Keep whole drawing or pan/zoom manually.';return}const minX=Math.min(...boxes.map(b=>b.x))-18,minY=Math.min(...boxes.map(b=>b.y))-18,maxX=Math.max(...boxes.map(b=>b.x+b.width))+18,maxY=Math.max(...boxes.map(b=>b.y+b.height))+18;svg.remove();viewport=[minX,minY,maxX-minX,maxY-minY];applyViewport()}
function zoom(factor){const b=viewport||fullViewport(),w=b[2]*factor,h=b[3]*factor;viewport=[b[0]+(b[2]-w)/2,b[1]+(b[3]-h)/2,w,h];applyViewport()}
function selectedView(){const s=document.querySelector('.viewport svg[data-version="'+CSS.escape(current())+'"]');if(!s)throw Error('No view');const clone=s.cloneNode(true);clone.setAttribute('xmlns',NS);clone.setAttribute('viewBox',(viewport||fullViewport()).join(' '));clone.setAttribute('width','1200');clone.setAttribute('height',String(Math.round(1200*(viewport||fullViewport())[3]/(viewport||fullViewport())[2])));return new XMLSerializer().serializeToString(clone)}
function renderHistory(){for(const t of bundle.trials){const box=text('div','', 'trial');box.append(text('strong',t.id+' · '+t.disposition),text('p',t.hypothesis));const states=text('div','', 'verdicts');for(const [k,v]of [['Execution',t.execution],['Shape',t.shape.status],['Fill',t.fill.status]])states.appendChild(text('span',k+': '+v,'verdict'));box.appendChild(states);if(t.error)box.appendChild(text('pre',t.error));if(t.artifact_errors&&Object.keys(t.artifact_errors).length)box.appendChild(text('pre',JSON.stringify(t.artifact_errors,null,2)));for(const [track,review]of [['Shape',t.shape],['Fill',t.fill]])if(review.reason)box.appendChild(text('p',track+' · '+review.reason));box.appendChild(text('pre',JSON.stringify(t.request,null,2)));$('history').appendChild(box)}}
$('title').textContent=bundle.run.title;document.title=bundle.run.title+' · VectorMark';const accepted=bundle.run.accepted_version,display=accepted||bundle.run.baseline_version,v=versions[display];$('summary').textContent=bundle.run.title;$('status').textContent=accepted?'Accepted':'Provisional · review incomplete';$('delivery-note').textContent=accepted?'Exact accepted SVG, ready to download.':'No accepted version yet. This download is the provisional baseline.';$('limitations').textContent=bundle.trials.some(t=>t.execution==='failed'||t.execution==='unknown')?'Some trials are blocked. Their errors remain in review history; no unpublished result is shown.':accepted?'Shape and fill reviews are recorded in the trial history.':'Shape and fill still need explicit independent review.';if(v){$('artwork').appendChild(svgNode(v.svg));$('download-svg').href=v.file;$('download-svg').download=bundle.run.title.replace(/[^a-z0-9_-]/gi,'-')+(accepted?'':'-provisional')+'.svg';$('download-svg').textContent=accepted?'Download SVG':'Download provisional SVG'}else{$('artwork').appendChild(text('p','No captured SVG is available.'));$('download-svg').hidden=true}
for(const t of bundle.trials){const o=text('option',t.id+' · '+t.disposition);o.value=t.id;$('trial').appendChild(o)}trial=bundle.trials.find(t=>t.candidate_version===display)||bundle.trials[0];if(trial)$('trial').value=trial.id;$('trial').onchange=()=>{trial=bundle.trials.find(t=>t.id===$('trial').value);region=selection=viewport=null;render()};for(const name of ['regions','paths','commands','features','source']){const label=text('label','', 'layer'),c=document.createElement('input');c.type='checkbox';c.value=name;c.checked=layers.has(name);c.onchange=()=>{if(c.checked)layers.add(name);else layers.delete(name);render()};label.append(c,document.createTextNode(name==='source'?'Source polylines':name[0].toUpperCase()+name.slice(1)));$('layers').appendChild(label)}
for(const id of ['review-tab','compare-original','enter-review'])$(id).onclick=()=>showReview(true);for(const id of ['art-tab','back','brand'])$(id).onclick=e=>{e.preventDefault();showReview(false)};for(const name of ['shape','fill'])$(name+'-tab').onclick=()=>{track=name;$('shape-tab').setAttribute('aria-pressed',String(name==='shape'));$('fill-tab').setAttribute('aria-pressed',String(name==='fill'));render()};$('focus-region').onclick=focus;$('whole').onclick=()=>{viewport=null;applyViewport()};$('zoom-in').onclick=()=>zoom(.8);$('zoom-out').onclick=()=>zoom(1.25);$('copy').onclick=async()=>{try{await navigator.clipboard.writeText(selection||region||'');$('copy-note').textContent='Copied exact version-scoped ID.'}catch{$('copy-note').textContent='Select and copy the displayed ID.'}};
$('export-svg').onclick=()=>download(track+'-view.svg',selectedView(),'image/svg+xml');$('export-json').onclick=()=>download('view.json',JSON.stringify({version:current(),track,region_ids:region?[region]:[],layers:[...layers],viewport:viewport||fullViewport(),selected_id:selection,inspection:versions[current()]?.inspection},null,2),'application/json');$('export-png').onclick=()=>{const svg=selectedView(),url=URL.createObjectURL(new Blob([svg],{type:'image/svg+xml'})),image=new Image();image.onload=()=>{const canvas=document.createElement('canvas');canvas.width=image.width;canvas.height=image.height;canvas.getContext('2d').drawImage(image,0,0);URL.revokeObjectURL(url);canvas.toBlob(blob=>{if(blob)download(track+'-view.png',blob,'image/png')})};image.onerror=()=>{URL.revokeObjectURL(url);$('copy-note').textContent='PNG unavailable. Export the SVG and view JSON instead.'};image.src=url};renderHistory();
