"""Export model-blind native RGB/IR train images and independent review workspaces."""

# ruff: noqa: E501
# The self-contained HTML/JavaScript below retains long literal lines.

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from aero_ir.data.antiuav import load_split_manifest
from aero_ir.registration.physical_review import review_template
from aero_ir.utils.manifest import file_sha256
from scripts.audit_antiuav300_dense_registration import _candidate_pairs, _read_at

HTML = r"""<!doctype html><meta charset="utf-8"><title>Independent physical review</title>
<style>body{font:16px sans-serif;margin:20px}canvas{border:1px solid #777;cursor:crosshair}
.view{overflow:auto;max-height:65vh;border:1px solid #aaa}button,input,select{margin:5px}
pre{white-space:pre-wrap}label{display:inline-block}</style>
<h1>Model-blind physical correspondence annotation</h1>
<p>Work independently. Do not inspect another review or model predictions. Native pixel
centres: upper-left pixel is (0,0). Annotate only the same identifiable physical feature.
Box corners, image text and crosshairs are NOT scene landmarks. No automatic matches shown.</p>
<label>Reviewer identity <input id="identity"></label>
<label><input type="checkbox" id="attest">I annotated independently without model output</label>
<button onclick="save()">Export review JSON</button>
<label>Resume YOUR JSON <input type="file" id="resume" accept=".json"></label>
<p>Save frequently. This page does not upload data or autosave.</p>
<button onclick="navigate(-1)">Previous</button><select id="pair"></select>
<button onclick="navigate(1)">Next</button>
<label>Display zoom <select id="zoom"><option>.5</option><option selected>1</option>
<option>2</option><option>4</option></select></label>
<p id="pairlabel"></p>
<label>Semantic landmark ID <input id="name" placeholder="e.g. left-front rotor hub"></label>
<label>Region <select id="region"><option>target</option><option>background</option></select></label>
<div id="views"></div>
<button onclick="add()">Add landmark</button><button onclick="resetPoints()">Clear pending points</button>
<p id="pending"></p><pre id="list"></pre>
<label>Delete landmark index (1-based)<input id="deleteindex" type="number" min="1"></label>
<button onclick="removeLandmark()">Delete</button><br>
<label>Frame notes / explanation if no landmarks <input id="notes" size="90"></label>
<label><input id="reviewed" type="checkbox">Frame review complete</label>
<script>
const manifest=__MANIFEST__, initial=__TEMPLATE__;
let review=initial, idx=0, pending={}, images={};
const modalities=['visible','infrared'];
const $=id=>document.getElementById(id);
manifest.pairs.forEach((p,i)=>{let o=document.createElement('option');o.value=i;o.textContent=p.pair_id;$('pair').append(o)});
modalities.forEach(m=>{
 let d=document.createElement('div');d.innerHTML=`<h3>${m}</h3>
 <label>Visibility <select id="${m}_visibility"><option>visible</option><option>unobservable</option></select></label>
 <label>Uncertainty radius (native px) <input type="number" min="0.01" step="0.25" id="${m}_uncertainty"></label>
 <label>Reason if unobservable <input id="${m}_reason" size="50"></label>
 <div class="view"><canvas id="${m}"></canvas></div>`;$('views').append(d);
 let c=$(m);c.onclick=e=>{let r=c.getBoundingClientRect();pending[m]=[
 Math.max(0,Math.min(c.width-1,(e.clientX-r.left-c.clientLeft)*c.width/c.clientWidth-.5)),
 Math.max(0,Math.min(c.height-1,(e.clientY-r.top-c.clientTop)*c.height/c.clientHeight-.5))];draw();};
});
function draw(){modalities.forEach(m=>{let c=$(m),ctx=c.getContext('2d');if(!images[m])return;
 ctx.clearRect(0,0,c.width,c.height);ctx.drawImage(images[m],0,0);
 c.style.width=(c.width*Number($('zoom').value))+'px';
 c.style.height=(c.height*Number($('zoom').value))+'px';
 if(pending[m]){let [x,y]=pending[m];ctx.strokeStyle='red';ctx.lineWidth=2;ctx.beginPath();
 ctx.arc(x+.5,y+.5,5,0,2*Math.PI);ctx.stroke();}});
 $('pending').textContent='Pending native points: '+JSON.stringify(pending);
 $('list').textContent=review.pairs[idx].landmarks.map((p,i)=>`${i+1}: ${JSON.stringify(p)}`).join('\n');}
function store(){review.reviewer_identity=$('identity').value;
review.independent_model_blind_attestation=$('attest').checked;
review.pairs[idx].notes=$('notes').value;review.pairs[idx].reviewed=$('reviewed').checked;}
function load(){pending={};images={};$('pair').value=idx;let p=manifest.pairs[idx];
 modalities.forEach(m=>{$(m+'_uncertainty').value='';$(m+'_reason').value='';$(m+'_visibility').value='visible'});
 $('pairlabel').textContent=p.pair_id+' — raw frame index '+p.frame_index;
 $('notes').value=review.pairs[idx].notes;$('reviewed').checked=review.pairs[idx].reviewed;
 modalities.forEach(m=>{let im=new Image(), c=$(m), entry=p.images[m];
 c.width=entry.shape[1];c.height=entry.shape[0];c.getContext('2d').clearRect(0,0,c.width,c.height);
 let expectedIndex=idx; im.onload=()=>{if(expectedIndex!==idx)return;images[m]=im;draw()};
 im.onerror=()=>alert('Cannot load '+entry.path+'. Keep images/ next to this HTML.');im.src=entry.path;});draw();}
function navigate(delta){store();idx=Math.max(0,Math.min(manifest.pairs.length-1,idx+delta));load();}
$('pair').onchange=()=>{store();idx=Number($('pair').value);load()};$('zoom').onchange=draw;
function resetPoints(){pending={};modalities.forEach(m=>{$(m+'_uncertainty').value='';$(m+'_reason').value='';$(m+'_visibility').value='visible'});draw()}
function add(){let name=$('name').value.trim();if(!name)return alert('Enter a physical landmark ID');
 if(review.pairs[idx].landmarks.some(p=>p.landmark_id===name))return alert('Duplicate landmark ID');
 let point={landmark_id:name,region:$('region').value};
 for(let m of modalities){let visibility=$(m+'_visibility').value;
 if(visibility==='unobservable'){let reason=$(m+'_reason').value.trim();if(!reason)return alert('Provide reason');
 point[m]={visibility,xy:null,uncertainty_px:null,reason};}
 else{let uncertainty=Number($(m+'_uncertainty').value);if(!pending[m]||!(uncertainty>0))return alert('Click '+m+' and enter positive native uncertainty');
 point[m]={visibility,xy:pending[m],uncertainty_px:uncertainty};}}
 review.pairs[idx].landmarks.push(point);resetPoints();}
function removeLandmark(){let n=Number($('deleteindex').value)-1;
 if(Number.isInteger(n)&&n>=0&&n<review.pairs[idx].landmarks.length){review.pairs[idx].landmarks.splice(n,1);draw()}}
function save(){store();let a=document.createElement('a'),url=URL.createObjectURL(new Blob([JSON.stringify(review,null,2)],{type:'application/json'}));
 a.href=url;a.download='review_'+review.reviewer_slot+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('resume').onchange=async e=>{try{let data=JSON.parse(await e.target.files[0].text());
 if(data.manifest_sha256!==initial.manifest_sha256||data.reviewer_slot!==initial.reviewer_slot||
 data.pairs.length!==manifest.pairs.length||data.pairs.some((p,i)=>p.pair_id!==manifest.pairs[i].pair_id))throw Error('Wrong panel/slot/order');
 review=data;$('identity').value=review.reviewer_identity;$('attest').checked=review.independent_model_blind_attestation;load();
 }catch(err){alert('Cannot resume: '+err.message)}};
load();</script>"""


def dump(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("/lustre/winston1214/dataset/Anti-UAV300")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite panel: {args.output_dir}")
    split_path = args.root / "label_new/train.json"
    manifest_split = load_split_manifest(args.root, "train")
    sequences = sorted(manifest_split)
    chosen = [sequences[i] for i in np.linspace(0, len(sequences) - 1, 16, dtype=int)]
    args.output_dir.mkdir(parents=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir()
    manifest = {
        "schema": "aero_physical_panel_v1",
        "split": "train",
        "purpose": "model-blind development observability and annotation; not final qualification",
        "coordinate_convention": "native pixel centres, top-left centre (0,0), no resizing",
        "sequence_selection": "sorted official train; linspace endpoints 16 dtype=int",
        "frame_selection": "usable indices at floor(q*(N-1)), q=0.25,0.50,0.75",
        "train_manifest_sha256": file_sha256(split_path),
        "preparation_source_sha256": file_sha256(__file__),
        "pairs": [],
    }
    for sequence in chosen:
        folder = args.root / "train" / sequence
        indices, _, _ = _candidate_pairs(folder)
        if len(indices) < 4:
            raise ValueError(f"insufficient usable frames in {sequence}")
        captures = {m: cv2.VideoCapture(str(folder / f"{m}.mp4")) for m in ("visible", "infrared")}
        try:
            source_hashes = {
                m: {
                    "video_sha256": file_sha256(folder / f"{m}.mp4"),
                    "annotation_sha256": file_sha256(folder / f"{m}.json"),
                }
                for m in captures
            }
            for q in (0.25, 0.50, 0.75):
                position = int(q * (len(indices) - 1))
                frame = indices[position]
                pair_id = f"{sequence}__{frame:06d}"
                pair = {
                    "pair_id": pair_id,
                    "sequence_id": sequence,
                    "frame_index": frame,
                    "usable_position": position,
                    "quantile": q,
                    "images": {},
                }
                for m, capture in captures.items():
                    image = _read_at(capture, frame, folder / f"{m}.mp4")
                    expected = (1080, 1920) if m == "visible" else (512, 640)
                    if image.shape[:2] != expected:
                        raise ValueError(f"unexpected native shape {sequence}/{m}")
                    path = image_dir / f"{pair_id}_{m}.png"
                    if not cv2.imwrite(str(path), image):
                        raise OSError(f"could not save {path}")
                    pair["images"][m] = {
                        "path": str(path.relative_to(args.output_dir)),
                        "shape": list(image.shape),
                        "sha256": file_sha256(path),
                        "decoded_bgr_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                        **source_hashes[m],
                    }
                manifest["pairs"].append(pair)
        finally:
            for capture in captures.values():
                capture.release()
        print(f"exported {sequence}", flush=True)
    manifest_path = args.output_dir / "manifest.json"
    dump(manifest_path, manifest)
    digest = file_sha256(manifest_path)
    for slot in ("A", "B"):
        template = review_template(manifest, digest, slot)
        dump(args.output_dir / f"review_{slot}_template.json", template)
        html = HTML.replace("__MANIFEST__", json.dumps(manifest).replace("<", "\\u003c"))
        html = html.replace("__TEMPLATE__", json.dumps(template).replace("<", "\\u003c"))
        with (args.output_dir / f"review_{slot}.html").open("x") as handle:
            handle.write(html)
    dump(
        args.output_dir / "physical_gate_draft.json",
        {
            "status": "unapproved",
            "manifest_sha256": digest,
            "thresholds": None,
            "reviewed_protocol_sha256": None,
            "note": "Freeze reviewed acceptance protocol before checkpoint evaluation. No automatic PASS.",
        },
    )
    print(f"Prepared {len(manifest['pairs'])} pairs: {args.output_dir}")


if __name__ == "__main__":
    main()
