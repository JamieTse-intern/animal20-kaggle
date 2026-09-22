"""Build run_log.html — the training-history artifact, with all data inlined from the
local run signatures so the page needs no network and no backing service."""
import csv, json, io

runs = list(csv.DictReader(open('experiment_history_runs.csv', encoding='utf-8')))
lb = list(csv.DictReader(open('experiment_history_leaderboard.csv', encoding='utf-8')))
for r in runs:
    r['run_dir'] = r['run_dir'].replace('\\', '/')

phases = [
 dict(span="09-07", name="Domain matching", lead="0.740 -> 0.859", flag="",
      body="The first model validated at 98.9% and scored 0.740. Cause: training images were clean, test images are 64x64 JPEG q75 4:2:0. Replicating the exact degradation pipeline &mdash; squash + LANCZOS to 64px, re-encoded with the test set's own quantisation tables &mdash; was worth more than every architecture change that followed."),
 dict(span="09-08", name="CLIP ensemble + external renditions", lead="0.922 -> 0.947", flag="",
      body="About 20% of the test set is non-photographic: paintings, sketches, 3D renders. Training was 100% photos. DomainNet painting data, degraded through the same pipeline and filtered on radial-spectrum distance to the test set, closed the gap. Sketch and clipart failed the spectrum check and cost points."),
 dict(span="09-09", name="ImageNet-R + self-distillation", lead="0.962 -> 0.96499", flag="",
      body="ImageNet-R covered the two DomainNet blind spots (hen&rarr;chickens, sea_lion&rarr;seals). Pseudo-labels were seeded from a saved stronger ensemble rather than each model's own weaker predictions."),
 dict(span="09-09 - 09-11", name="Manual correction", lead="retracted", flag="void",
      body="Hand-reviewing low-confidence test predictions and overriding labels reached 0.97417. It also violates the assignment's rule against manually or AI-labelling the test set, and was retracted in full &mdash; including a teacher probability file that had corrections baked into it and was used to seed pseudo-labels for retraining. Those scores are struck through below and are not the project's result."),
 dict(span="09-10 - 09-11", name="ConvNeXt-V2 FCMAE pivot", lead="0.97065 -> 0.97342", flag="peak",
      body="Masked-autoencoder pretraining beat CLIP contrastive pretraining on this rendition-heavy problem. A single ConvNeXt-V2-Large then beat every ensemble it was a member of, including the mixed ensemble at 0.97267."),
 dict(span="09-11", name="Scaling fails three ways", lead="0.95869 / tie / tie", flag="",
      body="ConvNeXt-V2-Huge (658M) alone: 0.95869, severe overfitting on ~8k real images. A same-architecture different-seed ensemble disagreed on only 0.51% of rows and scored below the single model. Widening TTA from 6 to 18 views changed 20 rows and tied exactly."),
 dict(span="09-12", name="Data integrity, then the noise floor", lead="0.97257", flag="fault",
      body="An audit found DomainNet's bottlecap class is bottle-cap mosaic art: 76% of the bottles external pool were fish, logos, a portrait. Fixed. Class-balanced sampling tripled the starved classes. Local held-out rendition accuracy rose 1.97pt; the leaderboard moved &minus;0.085pt, one third of a standard error. Every data-side knob now lands inside the instrument's resolution."),
 dict(span="09-12", name="Resolution: one finding, one failure", lead="pending", flag="open",
      body="Renditions lose accuracy 3-4x faster than photos under downsampling &mdash; the one diagnosed mechanism nothing had touched. At 256px the base stage gained 1.04pt on held-out renditions. The distillation stage then collapsed: best epoch 1 of 16, &minus;4.4pt, p&lt;0.001. The base-stage probe is built and unsubmitted; the collapsed model must not be submitted."),
]

HTML = r'''<title>Animal-20 Run Log</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>
:root{
  --paper:#f5f7f8; --panel:#ffffff; --ink:#111820; --ink-2:#414e5a; --ink-3:#6b7885;
  --rule:#d9dfe4; --rule-2:#eaeef1;
  --signal:#0c7a70; --signal-soft:#0c7a7018;
  --void:#8a6412; --fault:#a6382c; --open:#3f5d8f;
  --band:#0c7a7014;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --paper:#0c1116; --panel:#121a21; --ink:#e6ecf1; --ink-2:#a3b0bb; --ink-3:#77848f;
    --rule:#243039; --rule-2:#1a242c;
    --signal:#3fbaad; --signal-soft:#3fbaad1f;
    --void:#c9971f; --fault:#e0705f; --open:#7ba0d8;
    --band:#3fbaad14;
  }
}
:root[data-theme="dark"]{
  --paper:#0c1116; --panel:#121a21; --ink:#e6ecf1; --ink-2:#a3b0bb; --ink-3:#77848f;
  --rule:#243039; --rule-2:#1a242c;
  --signal:#3fbaad; --signal-soft:#3fbaad1f;
  --void:#c9971f; --fault:#e0705f; --open:#7ba0d8;
  --band:#3fbaad14;
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);
  font-family:"Source Serif 4",Georgia,serif;font-size:16px;line-height:1.6;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:40px 22px 80px}
h1,h2,h3{font-family:Archivo,"Helvetica Neue",Arial,sans-serif}
.eyebrow{font-family:Archivo,sans-serif;font-size:11px;font-weight:600;letter-spacing:.13em;
  text-transform:uppercase;color:var(--ink-3)}
h1{font-size:clamp(30px,5vw,44px);font-weight:700;letter-spacing:-.02em;margin:6px 0 10px;
  text-wrap:balance;line-height:1.08}
.sub{color:var(--ink-2);max-width:62ch;margin:0}
header{border-bottom:1px solid var(--rule);padding-bottom:28px}
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(162px,1fr));
  margin:26px 0 0;border-top:1px solid var(--rule-2)}
.fig{padding:16px 18px 14px;border-bottom:1px solid var(--rule-2)}
.fig+.fig{border-left:1px solid var(--rule-2)}
.fig .v{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums;
  font-size:25px;font-weight:600;letter-spacing:-.01em;display:block;line-height:1.2}
.fig .k{font-family:Archivo,sans-serif;font-size:11px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink-3);margin-top:5px;display:block}
.fig .n{font-size:13px;color:var(--ink-2);margin-top:6px;display:block;line-height:1.45}
.fig.best .v{color:var(--signal)}
.u{font-size:15px}
section{margin-top:52px}
h2{font-size:21px;font-weight:600;letter-spacing:-.01em;margin:0 0 6px}
.lede{color:var(--ink-2);margin:0 0 22px;max-width:66ch;font-size:15px}
.chartbox{background:var(--panel);border:1px solid var(--rule);padding:18px 14px 10px;overflow-x:auto}
svg{display:block;min-width:660px;width:100%;height:auto}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin-top:12px;padding:0 4px}
.lg{display:flex;align-items:center;gap:7px;font-size:12.5px;color:var(--ink-2)}
.lg i{width:13px;height:13px;display:inline-block;flex:none}
ol.phases{list-style:none;padding:0;margin:0;counter-reset:p}
ol.phases li{counter-increment:p;display:grid;grid-template-columns:66px 1fr;gap:20px;
  padding:22px 0;border-top:1px solid var(--rule-2)}
ol.phases li:first-child{border-top:1px solid var(--rule)}
.pn{font-family:"IBM Plex Mono",monospace;font-size:11.5px;color:var(--ink-3);padding-top:3px}
.pn b{display:block;font-size:19px;font-weight:500;letter-spacing:-.02em;margin-bottom:2px}
.pn b::before{content:counter(p,decimal-leading-zero)}
.ph h3{font-size:17px;font-weight:600;margin:0 0 2px;letter-spacing:-.01em}
.ph .lead{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;font-size:13px;
  color:var(--signal);margin:0 0 9px;font-weight:500}
.ph[data-flag="void"] .lead{color:var(--void)}
.ph[data-flag="fault"] .lead{color:var(--fault)}
.ph[data-flag="open"] .lead{color:var(--open)}
.ph p{margin:0;color:var(--ink-2);font-size:15.2px;max-width:68ch}
.tablebox{overflow-x:auto;border:1px solid var(--rule);background:var(--panel)}
table{border-collapse:collapse;width:100%;min-width:620px;
  font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums;font-size:12.7px}
th{font-family:Archivo,sans-serif;font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--ink-3);text-align:left;font-weight:600;padding:11px 12px;
  border-bottom:1px solid var(--rule);white-space:nowrap;position:sticky;top:0;background:var(--panel)}
td{padding:8px 12px;border-bottom:1px solid var(--rule-2);white-space:nowrap;color:var(--ink-2)}
tr:last-child td{border-bottom:none}
td.n{text-align:right}
td.s{color:var(--ink);font-weight:500}
.chip{font-family:Archivo,sans-serif;font-size:9.5px;letter-spacing:.07em;text-transform:uppercase;
  font-weight:600;padding:2.5px 6px;border:1px solid currentColor;white-space:nowrap}
.c-void{color:var(--void)} .c-fault{color:var(--fault)} .c-best{color:var(--signal)}
.strike{text-decoration:line-through;text-decoration-thickness:1px;opacity:.62}
.ctrl{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 12px}
button{font-family:Archivo,sans-serif;font-size:11.5px;font-weight:600;letter-spacing:.06em;
  text-transform:uppercase;padding:7px 13px;background:transparent;color:var(--ink-2);
  border:1px solid var(--rule);cursor:pointer}
button[aria-pressed="true"]{background:var(--signal-soft);color:var(--signal);border-color:var(--signal)}
button:focus-visible{outline:2px solid var(--signal);outline-offset:2px}
.note{font-size:13.5px;color:var(--ink-3);margin-top:11px;max-width:72ch}
code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12.3px;color:var(--ink-2)}
footer{margin-top:60px;padding-top:22px;border-top:1px solid var(--rule);
  font-size:13px;color:var(--ink-3)}
@media (max-width:560px){
  ol.phases li{grid-template-columns:1fr;gap:8px}
  .pn{display:flex;gap:10px;align-items:baseline;padding-top:0}
  .pn b{font-size:15px;margin-bottom:0}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">FIT5215 &middot; 20-class animal classification &middot; 64&times;64</div>
  <h1>Animal-20 Run Log</h1>
  <p class="sub">Six days, 94 training runs, 21 scored submissions. The record includes what was
  retracted and what landed inside the measurement noise &mdash; both are part of the result.</p>
  <div class="figs">
    <div class="fig best"><span class="v" id="f-best">&mdash;</span><span class="k">Best public</span>
      <span class="n">ConvNeXt-V2-Large FCMAE, single model, no ensemble</span></div>
    <div class="fig"><span class="v" id="f-runs">&mdash;</span><span class="k">Training runs</span>
      <span class="n">9 backbones, 09-07 to 09-12</span></div>
    <div class="fig"><span class="v">0.25<span class="u">pt</span></span><span class="k">Leaderboard SE</span>
      <span class="n">11,681 test images; 1pt is about 117 images</span></div>
    <div class="fig"><span class="v" id="f-gap">&mdash;</span><span class="k">Gap to first</span>
      <span class="n">0.97726, held by another entrant</span></div>
  </div>
</header>

<section>
  <h2>Every submission against the noise floor</h2>
  <p class="lede">The shaded band is &plusmn;1 standard error around the best score. Once a result
  lands inside it, one submission cannot tell it apart from the best &mdash; which is where every
  attempt since 09-11 has landed.</p>
  <div class="chartbox"><svg id="chart" viewBox="0 0 940 400" role="img"
    aria-label="Kaggle public score by submission, 0.74 to 0.98, with a plus or minus one standard error band around the best score"></svg></div>
  <div class="legend">
    <span class="lg"><i style="background:var(--signal)"></i>Counted submission</span>
    <span class="lg"><i style="background:var(--void)"></i>Retracted &mdash; manual test labelling</span>
    <span class="lg"><i style="background:var(--band);border:1px solid var(--signal)"></i>&plusmn;1 SE of best</span>
  </div>
</section>

<section>
  <h2>How it moved</h2>
  <p class="lede">Numbered because the order carries the argument: each phase is what the previous
  phase's failure made obvious.</p>
  <ol class="phases" id="phases"></ol>
</section>

<section>
  <h2>Submissions</h2>
  <p class="lede">Retracted rows are struck through. They were real scores; they are not results.</p>
  <div class="tablebox"><table id="t-lb"><thead><tr>
    <th>Date</th><th>File</th><th style="text-align:right">Public</th><th>Produced by</th><th>Note</th>
  </tr></thead><tbody></tbody></table></div>
</section>

<section>
  <h2>Training runs</h2>
  <p class="lede">Config read back from each run's cached signature, so what is shown is what
  actually trained &mdash; not what the notebook says today.</p>
  <div class="ctrl">
    <button id="b-all" aria-pressed="true">All 94</button>
    <button id="b-cnv2" aria-pressed="false">ConvNeXt-V2 line</button>
    <button id="b-flag" aria-pressed="false">Flagged only</button>
  </div>
  <div class="tablebox"><table id="t-runs"><thead><tr>
    <th>Finished</th><th>Backbone</th><th>Stage</th><th style="text-align:right">Best ep</th>
    <th style="text-align:right">Val %</th><th style="text-align:right">px</th>
    <th style="text-align:right">Ratio</th><th>External source</th><th>Flag</th>
  </tr></thead><tbody></tbody></table></div>
  <p class="note"><b>Reading the Val column.</b> 20 runs trained with
  <code>train_on_all_data=True</code>, which puts the validation images inside the training set
  &mdash; their 99.9% is leaked and means nothing. From 09-10 onward validation saturates anyway:
  the 0.97342 model and the 0.97257 model both validate at 98.94%. A pseudo stage that picks
  <code>best_epoch &le; 2</code> is flagged as collapse &mdash; the distillation loop reinforcing
  its own errors.</p>
</section>

<footer>Built from 94 <code>*_run.npz</code> signatures and the submission record.
Scores are Kaggle public leaderboard.</footer>
</div>

<script>
const RUNS=__RUNS__;
const LB=__LB__;
const PHASES=__PHASES__;
const BEST=0.97342, LEADER=0.97726, SE=0.0025, N_TEST=11681;

document.getElementById('f-best').textContent=BEST.toFixed(5);
document.getElementById('f-runs').textContent=RUNS.length;
document.getElementById('f-gap').innerHTML=Math.round((LEADER-BEST)*N_TEST)+'<span class="u"> img</span>';

(function(){
  const rows=LB.filter(r=>r.kaggle_public).map((r,i)=>({
    i:i, date:r.date, y:parseFloat(r.kaggle_public), file:r.submission,
    dead:/RETRACTED/.test(r.note), best:/BEST/.test(r.note)}));
  const W=940,H=400,L=62,R=136,T=26,B=44;
  const xs=i=>L+i*(W-L-R)/(rows.length-1);
  const lo=0.732, hi=0.986;
  const ys=v=>T+(hi-v)/(hi-lo)*(H-T-B);
  let s='';
  s+='<rect x="'+L+'" y="'+ys(BEST+SE)+'" width="'+(W-L-R)+'" height="'+(ys(BEST-SE)-ys(BEST+SE))+'" fill="var(--band)"/>';
  [0.75,0.80,0.85,0.90,0.95].forEach(function(v){
    s+='<line x1="'+L+'" y1="'+ys(v)+'" x2="'+(W-R)+'" y2="'+ys(v)+'" stroke="var(--rule-2)" stroke-width="1"/>';
    s+='<text x="'+(L-10)+'" y="'+(ys(v)+4)+'" text-anchor="end" fill="var(--ink-3)" font-size="11" font-family="IBM Plex Mono, monospace">'+v.toFixed(2)+'</text>';});
  [[BEST,'var(--signal)','best 0.97342'],[LEADER,'var(--ink-3)','leader 0.97726']].forEach(function(d){
    s+='<line x1="'+L+'" y1="'+ys(d[0])+'" x2="'+(W-R)+'" y2="'+ys(d[0])+'" stroke="'+d[1]+'" stroke-width="1" stroke-dasharray="4 3"/>';
    s+='<text x="'+(W-R+9)+'" y="'+(ys(d[0])+4)+'" fill="'+d[1]+'" font-size="11.5" font-weight="600" font-family="IBM Plex Mono, monospace">'+d[2]+'</text>';});
  s+='<text x="'+(W-R+9)+'" y="'+(ys(BEST-SE)+14)+'" fill="var(--ink-3)" font-size="10.5" font-family="Archivo, sans-serif">&#177;1 SE band</text>';
  const solid=rows.filter(function(r){return !r.dead;});
  s+='<polyline fill="none" stroke="var(--signal)" stroke-width="1.75" points="'+
     solid.map(function(r){return xs(r.i)+','+ys(r.y);}).join(' ')+'"/>';
  rows.forEach(function(r){
    const x=xs(r.i),y=ys(r.y);
    s+='<g><title>'+r.file+' &#183; '+r.y.toFixed(5)+' &#183; '+r.date+'</title>';
    if(r.dead){s+='<path d="M'+(x-4.4)+' '+(y-4.4)+'L'+(x+4.4)+' '+(y+4.4)+'M'+(x+4.4)+' '+(y-4.4)+'L'+(x-4.4)+' '+(y+4.4)+'" stroke="var(--void)" stroke-width="1.9" fill="none"/>';}
    else{s+='<circle cx="'+x+'" cy="'+y+'" r="'+(r.best?5.2:3.4)+'" fill="'+(r.best?'var(--signal)':'var(--panel)')+'" stroke="var(--signal)" stroke-width="1.75"/>';}
    s+='</g>';});
  const seen={};
  rows.forEach(function(r){if(seen[r.date])return;seen[r.date]=1;
    s+='<text x="'+xs(r.i)+'" y="'+(H-B+20)+'" text-anchor="middle" fill="var(--ink-3)" font-size="10.5" font-family="IBM Plex Mono, monospace">'+r.date+'</text>';});
  const f=rows[0], p=rows.filter(function(r){return r.best;})[0];
  s+='<text x="'+(xs(f.i)+9)+'" y="'+(ys(f.y)+5)+'" fill="var(--ink-3)" font-size="11" font-family="Archivo, sans-serif">v1 &#8212; clean training images, degraded test set</text>';
  s+='<text x="'+(xs(p.i)-9)+'" y="'+(ys(p.y)-12)+'" text-anchor="end" fill="var(--signal)" font-size="11" font-weight="600" font-family="Archivo, sans-serif">single model beats every ensemble</text>';
  document.getElementById('chart').innerHTML=s;
})();

document.getElementById('phases').innerHTML=PHASES.map(function(p){
 return '<li><div class="pn"><b></b>'+p.span+'</div><div class="ph" data-flag="'+p.flag+
  '"><h3>'+p.name+'</h3><p class="lead">'+p.lead+'</p><p>'+p.body+'</p></div></li>';}).join('');

document.querySelector('#t-lb tbody').innerHTML=LB.map(function(r){
  const dead=/RETRACTED/.test(r.note), best=/BEST/.test(r.note);
  const chip=dead?'<span class="chip c-void">retracted</span>':
    best?'<span class="chip c-best">best</span>':
    /DO NOT SUBMIT/.test(r.note)?'<span class="chip c-fault">do not submit</span>':'';
  const note=r.note.replace(/\*\*/g,'').replace(/RETRACTED\s*-?\s*/,'').replace(/BEST\s*-\s*/,'').trim();
  return '<tr class="'+(dead?'strike':'')+'"><td>'+r.date+'</td><td class="s">'+r.submission+
   '</td><td class="n s">'+(r.kaggle_public||'&mdash;')+'</td><td>'+(r.local_run||'&mdash;')+
   '</td><td>'+chip+' '+note+'</td></tr>';}).join('');

const tb=document.querySelector('#t-runs tbody');
function flagOf(r){
  if(r.val_leaked) return ['c-fault','val leaked'];
  if(r.stage.indexOf('pseudo')===0 && +r.best_epoch<=2) return ['c-fault','collapse'];
  if(r.image_size==='256') return ['c-best','256px'];
  return null;}
function draw(f){
  tb.innerHTML=RUNS.filter(f).map(function(r){
    const fl=flagOf(r);
    return '<tr><td>'+r.saved+'</td><td class="s">'+r.backbone+'</td><td>'+r.stage+
     '</td><td class="n">'+r.best_epoch+'</td><td class="n'+(r.val_leaked?'':' s')+'">'+r.val_acc+
     '</td><td class="n">'+(r.image_size||'&mdash;')+'</td><td class="n">'+(r.ext_ratio||'&mdash;')+
     '</td><td>'+(r.ext_source||'&mdash;')+'</td><td>'+
     (fl?'<span class="chip '+fl[0]+'">'+fl[1]+'</span>':'')+'</td></tr>';}).join('');}
const F={'b-all':function(){return true;},
         'b-cnv2':function(r){return /convnextv2/.test(r.backbone);},
         'b-flag':function(r){return !!flagOf(r);}};
Object.keys(F).forEach(function(id){
  document.getElementById(id).addEventListener('click',function(){
    Object.keys(F).forEach(function(o){
      document.getElementById(o).setAttribute('aria-pressed',String(o===id));});
    draw(F[id]);});});
draw(F['b-all']);
</script>
'''

html = (HTML.replace('__RUNS__', json.dumps(runs, ensure_ascii=False))
            .replace('__LB__', json.dumps(lb, ensure_ascii=False))
            .replace('__PHASES__', json.dumps(phases, ensure_ascii=False)))
io.open('run_log.html', 'w', encoding='utf-8').write(html)
print('run_log.html written: {:,} bytes, {} runs + {} submissions inlined'.format(
    len(html), len(runs), len(lb)))
