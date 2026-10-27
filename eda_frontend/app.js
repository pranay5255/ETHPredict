const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(v,n=2)=>v==null?'—':Number(v).toLocaleString(undefined,{maximumFractionDigits:n});
const colors={'completed':'#58d6c2','no trade':'#f3bf63','failed':'#ec7d86','skipped':'#a98deb','unknown':'#a0afbf'};
const plotLayout=(y)=>({paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{color:'#a0afbf',size:12},margin:{l:65,r:25,t:15,b:70},xaxis:{gridcolor:'#263445',zerolinecolor:'#40556c',automargin:true},yaxis:{title:y,gridcolor:'#263445',zerolinecolor:'#40556c',automargin:true},legend:{orientation:'h',y:-.25},hoverlabel:{bgcolor:'#1b2636',font:{color:'#e9f1f8'}}});
const plotConfig={responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']};
let all=[],report={},filtered=[];
const fields=[['family','Family'],['cohort','Cohort'],['horizon','Horizon'],['split','Split'],['status','Status']];
function options(field){let vals=new Set();all.forEach(r=>{if(field==='split'){Object.entries(r.backtest||{}).forEach(([k,v])=>{if(v?.trades!=null)vals.add(k)});(r.forecast||[]).forEach(x=>vals.add(x.split));}else vals.add(r[field]??'unknown')});return [...vals].sort()}
function initFilters(){ $('filters').innerHTML=fields.map(([key,title])=>`<label>${title}<select id="filter-${key}"><option value="">All ${title.toLowerCase()}s</option>${options(key).map(v=>`<option value="${esc(v)}">${esc(v)}</option>`).join('')}</select></label>`).join('');fields.forEach(([key])=>{$(`filter-${key}`).addEventListener('change',render)}) }
function selection(){return Object.fromEntries(fields.map(([k])=>[k,$(`filter-${k}`).value]))}
function rowsForSplit(rows,split){return rows.filter(r=>!split||r.backtest?.[split]?.trades!=null||(r.forecast||[]).some(f=>f.split===split))}
function render(){let f=selection();filtered=all.filter(r=>['family','cohort','horizon','status'].every(k=>!f[k]||String(r[k]??'unknown')===f[k]));filtered=rowsForSplit(filtered,f.split);renderEvidence();renderForecast();renderComparison();renderTrading();renderStability();renderLedger()}
function renderEvidence(){let research=filtered.filter(r=>r.cohort!=='Legacy staged'),counts={};research.forEach(r=>counts[r.status]=(counts[r.status]||0)+1);let cards=[['Runs',research.length,''],['Completed',counts.completed||0,''],['No trade',counts['no trade']||0,'warn'],['Failed',counts.failed||0,'bad'],['Skipped',counts.skipped||0,'warn'],['Missing manifest',research.filter(r=>!r.artifact_available?.manifest).length,'warn'],['Missing artifact files',research.filter(r=>r.missing_artifacts?.length).length,'warn'],['Trackio aggregate only',research.filter(r=>r.evidence==='Trackio aggregate only').length,'warn']];$('cards').innerHTML=cards.map(([label,value,cls])=>`<div class="card ${cls}"><strong>${value}</strong><span>${label}</span></div>`).join('');$('evidence-note').textContent=`${research.filter(r=>r.discrepancies?.length).length} runs have local/Trackio metric conflicts. ${filtered.filter(r=>r.cohort==='Legacy staged').length} legacy staged rows stay in their own cohort. ${research.filter(r=>r.test_reuse_flag).length} runs flag possible final-test reuse because the guard was disabled.`}
function draw(id,traces,y,opts={}){if(!traces.length||traces.every(t=>!t.x?.length)){Plotly.purge(id);$(id).innerHTML='<div class="empty">No compatible evidence under these filters.</div>';return}Plotly.react(id,traces,{...plotLayout(y),...opts},plotConfig)}
function hover(r,extra=''){return `${r.name}<br>Cohort: ${r.cohort}<br>Role: ${r.selection_role}<br>Trial count: ${r.trial_count}<br>Validation trades: ${fmt(r.backtest?.validation?.trades,0)}<br>Evidence: ${r.evidence}${extra}`}
function renderForecast(){
  let split=selection().split||'validation';
  $('mae-title').textContent=`${split==='test'?'Archived test':'Validation'} MAE by horizon and fold`;
  $('logloss-title').textContent=`${split==='test'?'Archived test':'Validation'} direction log loss`;
  let points=filtered.flatMap(r=>(r.forecast||[]).filter(f=>f.split===split).map(f=>({...f,row:r})));
  for(let [id,key,label] of [['mae','mae','MAE'],['logloss','log_loss','Log loss']]){
    let traces=[];let groups=Object.groupBy(points.filter(p=>p[key]!=null),p=>`${p.row.cohort} · ${p.row.stage}`);
    for(let [name,pts] of Object.entries(groups))traces.push({type:'scatter',mode:'markers',name,
      x:pts.map(p=>`${p.horizon} · fold ${p.fold??'aggregate'}`),y:pts.map(p=>p[key]),
      text:pts.map(p=>hover(p.row,`<br>Source: ${p.source}<br>Split: ${split}`)),
      hovertemplate:'%{text}<br>'+label+': %{y:.5f}<extra></extra>',marker:{size:10,opacity:.78}});
    draw(id,traces,label);
  }
  let reliable=filtered.flatMap(r=>(r.reliability||[]).map(p=>({...p,row:r})))
    .filter(p=>!selection().horizon||p.horizon===selection().horizon);
  let grouped=Object.groupBy(reliable,p=>`${p.row.name} / ${p.horizon} / fold ${p.fold??'all'}`);
  let traces=Object.entries(grouped).slice(0,20).map(([name,pts])=>({type:'scatter',mode:'lines+markers',name,
    x:pts.map(p=>p.mean_probability),y:pts.map(p=>p.realized_frequency),
    text:pts.map(p=>`n=${p.count}, fold=${p.fold??'all'}, split=validation`),
    hovertemplate:'%{text}<br>Predicted %{x:.3f}<br>Observed %{y:.3f}<extra></extra>'}));
  if(traces.length)traces.unshift({type:'scatter',mode:'lines',name:'ideal',x:[0,1],y:[0,1],line:{dash:'dash',color:'#a0afbf'}});
  draw('reliability',traces,'Observed direction frequency');
  let bucket=Object.entries(grouped).slice(0,20).flatMap(([name,pts])=>[
    {type:'scatter',mode:'lines+markers',name:name+' predicted',x:pts.map(p=>p.mean_probability),y:pts.map(p=>p.mean_pred_return)},
    {type:'scatter',mode:'lines+markers',name:name+' realized',x:pts.map(p=>p.mean_probability),y:pts.map(p=>p.mean_realized_return),line:{dash:'dot'}}]);
  draw('returns',bucket,'Mean return');
}
function compatibleKey(r){if([r.cohort,r.data_id,r.split_settings_id,r.horizon,r.cost_id].some(v=>v==null||v===''))return null;return [r.cohort,r.data_id,r.split_settings_id,r.horizon,r.cost_id].join('|')}
function variant(r){return r.family==='bars'?r.bar_type:r.family==='features'?(r.feature_set||[]).join('+'):r.family==='sampling'?r.sampling_mode:null}
function renderComparison(){
  let split=selection().split||'validation';
  $('pairs-title').textContent=`Paired ${split} candidate coverage`;
  $('candidates-title').textContent=`${split==='test'?'Archived test':'Validation'} candidate coverage and trade count`;
  let groups=Object.groupBy(filtered.filter(r=>['bars','features','sampling'].includes(r.family)
    &&r.stage==='meta_label_mvp_trial'&&r.backtest?.[split]?.candidate_coverage!=null
    &&compatibleKey(r)&&variant(r)),r=>`${r.family}|${compatibleKey(r)}`);
  let traces=[],paired=0;
  for(let [key,group] of Object.entries(groups)){
    let variants=Object.groupBy(group,variant);if(Object.keys(variants).length<2)continue;
    let names=Object.keys(variants).sort(),base=variants[names[0]][0];
    for(let name of names.slice(1)){
      let other=variants[name][0];traces.push({type:'scatter',mode:'lines+markers',name:key.split('|')[0],
        x:[`${key.split('|')[0]} · ${names[0]}`,`${key.split('|')[0]} · ${name}`],
        y:[base.backtest[split].candidate_coverage,other.backtest[split].candidate_coverage],
        text:[hover(base),hover(other)],hovertemplate:'%{text}<br>Candidate coverage %{y:.3f}<extra></extra>',showlegend:paired===0});paired++;
    }
  }
  draw('pairs',traces,'Candidate coverage');
  $('pair-note').textContent=`${paired} compatible variant pair(s). Trial count and selection role appear on hover. Matching split settings do not guarantee identical sample indices. Zero-trade variants are compared on candidate coverage only.`;
  let points=filtered.filter(r=>r.backtest?.[split]?.candidate_coverage!=null);
  draw('candidates',[{type:'scatter',mode:'markers',x:points.map(r=>r.backtest[split].candidate_coverage),
    y:points.map(r=>r.backtest[split].trades),text:points.map(r=>hover(r,`<br>Split: ${split}<br>Trades/day: ${fmt(r.backtest[split].trades_per_day)}`)),
    hovertemplate:'%{text}<br>Candidate coverage: %{x:.3f}<br>Trades: %{y}<extra></extra>',
    marker:{size:10,color:points.map(r=>colors[r.status]||colors.unknown),opacity:.8}}],`${split} trades`,
    {xaxis:{...plotLayout('').xaxis,title:'Candidate coverage'}});
}
function renderTrading(){
  let split=selection().split||'validation',label=split==='test'?'Archived test':'Validation';
  $('pnl-title').textContent=`${label} PnL and cost components`;
  $('risk-title').textContent=`${label} turnover and drawdown`;
  $('trades-title').textContent=`${label} equity paths`;
  $('trade-dist-title').textContent=`${label} trade net PnL distributions`;
  let rows=filtered.filter(r=>r.backtest?.[split]?.trades>0)
    .sort((a,b)=>(b.backtest[split].net_pnl??-Infinity)-(a.backtest[split].net_pnl??-Infinity)).slice(0,24);
  let keys=[['gross_pnl','Gross','#86b8f7'],['fees','Fees','#f3bf63'],['net_pnl','Net','#58d6c2']];
  draw('pnl',keys.map(([key,name,color])=>({type:'bar',name,x:rows.map(r=>r.name),
    y:rows.map(r=>r.backtest[split][key]),text:rows.map(r=>hover(r,`<br>Split: ${split}`)),
    hovertemplate:'%{text}<br>'+name+': %{y:.2f}<extra></extra>',marker:{color}})),'PnL / fees',
    {barmode:'group',xaxis:{tickangle:-35,automargin:true}});
  draw('risk',[{type:'scatter',mode:'markers',x:rows.map(r=>r.backtest[split].turnover),
    y:rows.map(r=>r.backtest[split].max_drawdown),text:rows.map(r=>hover(r,`<br>Split: ${split}`)),
    hovertemplate:'%{text}<br>Turnover %{x:.0f}<br>Drawdown %{y:.2f}<extra></extra>',
    marker:{size:rows.map(r=>Math.max(8,Math.min(23,5+Math.sqrt(r.backtest[split].trades||0)))),color:'#86b8f7',opacity:.8}}],
    'Max drawdown',{xaxis:{...plotLayout('').xaxis,title:'Turnover'}});
  let traces=filtered.filter(r=>r.trades?.[split]?.equity?.length).slice(0,15).map(r=>({type:'scatter',mode:'lines',name:r.name,
    x:r.trades[split].equity.map((_,i)=>i),y:r.trades[split].equity,
    text:hover(r,`<br>Parquet trades: ${r.trades[split].trade_count}<br>Split: ${split}`),
    hovertemplate:'%{text}<br>Cumulative net PnL: %{y:.2f}<extra></extra>'}));
  draw('trades',traces,'Cumulative net PnL');
  let distributions=filtered.filter(r=>r.trades?.[split]?.net_pnl?.length).slice(0,8).map(r=>({
    type:'histogram',name:r.name,x:r.trades[split].net_pnl,opacity:.55,
    hovertemplate:'Trade net PnL %{x:.2f}<br>Count %{y}<extra>'+r.name+'</extra>'}));
  draw('trade-dist',distributions,'Trade count',{barmode:'overlay',xaxis:{...plotLayout('').xaxis,title:'Net PnL per trade'}});
}
function renderStability(){let points=filtered.filter(r=>r.backtest?.validation?.trades>0&&r.backtest?.test?.trades>0&&r.backtest.validation.net_pnl!=null&&r.backtest.test.net_pnl!=null);draw('stability-plot',[{type:'scatter',mode:'markers',x:points.map(r=>r.backtest.validation.net_pnl),y:points.map(r=>r.backtest.test.net_pnl),text:points.map(r=>hover(r,`<br>Test trades: ${fmt(r.backtest.test.trades,0)}<br>Final-test reuse flag: ${r.test_reuse_flag?'yes':'no'}`)),hovertemplate:'%{text}<br>Validation %{x:.2f}<br>Test %{y:.2f}<extra></extra>',marker:{size:11,color:points.map(r=>r.test_reuse_flag?'#f3bf63':'#58d6c2')}}],'Archived test net PnL',{xaxis:{...plotLayout('').xaxis,title:'Validation net PnL'}})}
function renderLedger(){let sorted=[...filtered].sort((a,b)=>a.cohort.localeCompare(b.cohort)||a.name.localeCompare(b.name));$('ledger').innerHTML=sorted.map(r=>{let v=r.backtest?.validation||{},t=r.backtest?.test||{};let link=r.source_url?`<a href="${esc(r.source_url)}" target="_blank" rel="noopener">Open source run ↗</a>`:'Historical local artifact';return `<article><details><summary><b class="mono">${esc(r.name)}</b><span class="pill ${r.status==='no trade'?'amber':r.status==='failed'?'red':''}">${esc(r.status)}</span><span>${esc(r.cohort)}</span><span>${esc(r.conclusion)}</span></summary><div class="detail"><p>${link} · <span class="mono">${esc(r.id)}</span></p><p><b>Selection:</b> ${esc(r.selection_role)} · <b>Evidence:</b> ${esc(r.evidence)} · <b>Horizon:</b> ${esc(r.horizon)} · <b>Trial count:</b> ${r.trial_count}</p><p><b>Validation:</b> ${fmt(v.trades,0)} trades (${fmt(v.trades_per_day)}/day) · gross ${fmt(v.gross_pnl)} · fees ${fmt(v.fees)} · net ${fmt(v.net_pnl)} · turnover ${fmt(v.turnover)} · drawdown ${fmt(v.max_drawdown)}</p><p><b>Archived test:</b> ${fmt(t.trades,0)} trades (${fmt(t.trades_per_day)}/day) · gross ${fmt(t.gross_pnl)} · fees ${fmt(t.fees)} · net ${fmt(t.net_pnl)} ${r.test_reuse_flag?'<span class="pill amber">Possible reuse</span>':''}</p><p><b>Insufficient evidence:</b> ${esc((r.insufficient_reasons||[]).join('; ')||'No specific missing-evidence flag')}</p><p><b>Supporting plots:</b> <a href="#forecast">forecast</a> · <a href="#comparison">candidates</a> · <a href="#trading">trading</a> · <a href="#stability">stability</a></p><p class="small mono">${esc(r.manifest_path||'No local manifest')}</p></div></details></article>`}).join('')||'<div class="empty">No runs match these filters.</div>'}
fetch('./data.json').then(r=>{if(!r.ok)throw Error(`HTTP ${r.status}`);return r.json()}).then(data=>{all=data.rows;report=data.report;$('updated').textContent=`Built ${new Date(report.generated_utc).toLocaleString()} · ${report.source_runs} Trackio research runs · ${report.historical_rows} historical rows`;initFilters();render()}).catch(error=>{$('cards').innerHTML=`<div class="note">Could not load EDA data: ${esc(error.message)}. Run the build command.</div>`});
