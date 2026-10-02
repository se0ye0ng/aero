import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.prepare_registration_practice import PANEL_INDICES, crop_box, render


def test_selection_fixed_and_small():
    assert PANEL_INDICES == (1, 10, 19, 28, 37, 46)
    assert all(i % 3 == 1 for i in PANEL_INDICES)


def test_context_crop_bounds_and_contains_original_box():
    assert crop_box([0, 0, 10, 10], 100, 100) == [0, 0, 34, 34]
    x0, y0, x1, y1 = crop_box([499, 935, 163, 126], 1920, 1080)
    assert x0 <= 499 and y0 <= 935 and x1 >= 662 and y1 >= 1061
    assert x1 <= 1920 and y1 <= 1080


@pytest.mark.parametrize(
    "rect", [[0, 0, 0, 2], [-1, 0, 1, 2], [95, 0, 10, 1], [0, 0, float("nan"), 1]]
)
def test_reject_invalid_box(rect):
    with pytest.raises(ValueError):
        crop_box(rect, 100, 100)


def test_render_no_gt_schema_no_radius_and_safe_json(tmp_path):
    template = Path("assets/registration_practice.html").read_text()
    manifest = {"pairs": [], "note": "</script><script>bad</script>"}
    html = render(template, manifest, "abc", "A")
    assert "__MANIFEST__" not in html and "__SLOT__" not in html
    assert "</script><script>bad" not in html
    assert "aero_part_familiarization_v1" in html
    assert "aero_physical_review_v1" not in html
    assert 'id="visible_uncertainty"' not in html
    assert "practice_only_not_registration_gt" in html
    js = html.split("<script>", 1)[1].split("</script>", 1)[0]
    if shutil.which("node"):
        source = tmp_path / "practice.js"
        source.write_text(js)
        subprocess.run(["node", "--check", str(source)], check=True, capture_output=True)
    assert json.loads(json.dumps(manifest)) == manifest


def test_practice_interaction_export_and_native_clicks(tmp_path):
    """Exercise event handlers with a minimal DOM; not a browser layout test."""
    if not shutil.which("node"):
        pytest.skip("node unavailable")
    entry = {"shape": [100, 100, 3], "practice_crop_xyxy": [0, 0, 100, 100], "path": "x.png"}
    manifest = {
        "pairs": [{"pair_id": "practice1", "images": {"visible": entry, "infrared": entry}}]
    }
    html = render(Path("assets/registration_practice.html").read_text(), manifest, "hash", "A")
    js = html.split("<script>", 1)[1].split("</script>", 1)[0]
    harness = r"""
const assert=require('assert');const nodes={};const saved=[];
class Element{
 constructor(){this.value='';this.checked=false;this.width=520;this.height=390;this.children=[]}
 set id(v){this._id=v;nodes[v]=this}get id(){return this._id}
 set innerHTML(v){for(const match of v.matchAll(/id="([^"]+)"/g)){
 let n=new Element();n.id=match[1]}}
 append(n){this.children.push(n)}click(){}
 getContext(){return {clearRect(){},drawImage(){},beginPath(){},arc(){},stroke(){}}}
 getBoundingClientRect(){return {left:0,top:0,width:520,height:390}}
}
const document={getElementById(id){return nodes[id]||(nodes[id]=new Element())},
 createElement(){return new Element()}};
class Image{set src(v){this.onload()}}
class Blob{constructor(parts){this.parts=parts}}
const URL={createObjectURL(blob){saved.push(JSON.parse(blob.parts.join('')));return 'blob:test'},
 revokeObjectURL(){}};
function setTimeout(fn){fn()}
"""
    assertions = r"""
$('reviewer').value='tester';$('quiz1').value='overlay';$('quiz2').value='no';
$('quizcheck').onclick();assert($('quizfeedback').textContent.startsWith('맞습니다'));
for(const m of mods)for(const [k] of fields)$(m+'_'+k).value='clear';
$('decision').value='yes';$('decision').onchange();$('part').value='motor_axis';
$('complete').checked=true;$('export').onclick();assert.strictEqual(saved.length,0);
for(const m of mods)$(m).onclick({clientX:65+20.5*3.9,clientY:30.5*3.9});
assert(Math.abs(review.pairs[0].points.visible[0]-20)<1e-8);
assert(Math.abs(review.pairs[0].points.infrared[1]-30)<1e-8);
$('export').onclick();assert.strictEqual(saved.length,1);
assert.strictEqual(saved[0].purpose,'practice_only_not_registration_gt');
assert.strictEqual(saved[0].qualification,'not_assessed_practice_only');
assert.strictEqual(saved[0].all_frames_answered,true);
$('decision').value='unsure';$('decision').onchange();
assert.deepStrictEqual(review.pairs[0].points,{});
$('notes').value='같은 축인지 구별할 수 없음';$('export').onclick();
assert.strictEqual(saved.length,2);assert.deepStrictEqual(saved[1].pairs[0].points,{});
"""
    source = tmp_path / "interaction.js"
    source.write_text(harness + js + assertions)
    subprocess.run(["node", str(source)], check=True, capture_output=True)
