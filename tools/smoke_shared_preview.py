"""Real-browser checks for one source preview engine across algorithm presets."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

import httpx
import numpy as np
import tifffile
from playwright.sync_api import sync_playwright, expect


def main():
    root=Path(__file__).resolve().parents[1]
    output=root/'.test-output'/f'shared-preview-{int(time.time())}'
    output.mkdir(parents=True)
    yy,xx=np.mgrid[:72,:168]
    tifffile.imwrite(output/'sample.tif',np.stack([(xx+yy+i*3).astype(np.uint16) for i in range(80)]),photometric='minisblack',metadata={'axes':'TYX'})
    url='http://127.0.0.1:8887'
    with (output/'server.log').open('w',encoding='utf-8') as log:
        server=subprocess.Popen([sys.executable,str(root/'run_web.py'),'--port','8887','--no-browser','--workspace',str(output)],cwd=root,stdout=log,stderr=log,env=dict(os.environ,PYTHONIOENCODING='utf-8'),creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        try:
            for _ in range(100):
                try:
                    if httpx.get(url+'/api/state',timeout=.5).status_code==200:break
                except httpx.HTTPError:time.sleep(.1)
            with httpx.Client(base_url=url,timeout=30) as api,sync_playwright() as pw:
                api.headers['X-EIP-Token']=api.get('/api/state').json()['session_token']
                api.post('/api/sources/open',json={'path':str(output/'sample.tif'),'interval':.01}).raise_for_status()
                browser=pw.chromium.launch(channel='msedge' if sys.platform=='win32' else 'chromium',headless=True)
                page=browser.new_page(viewport={'width':1440,'height':960})
                errors=[];reads=[];geometry=[];inflight=set();maximum=[0]
                page.on('pageerror',lambda e:errors.append(str(e)))
                def requested(request):
                    if '/sources/' in request.url and request.url.endswith('/frame'):
                        inflight.add(request);maximum[0]=max(maximum[0],len(inflight))
                    if request.url.endswith('/split/geometry'):geometry.append(request.url)
                def finished(request):
                    if request in inflight:
                        inflight.remove(request)
                        reads.append(request.post_data_json)
                page.on('request',requested);page.on('requestfinished',finished);page.on('requestfailed',lambda r:inflight.discard(r))
                page.goto(url)
                expect(page.locator('#frame-info')).to_contain_text('第 1 帧 / 80')
                page.evaluate('window.sharedCanvas=document.getElementById("input-canvas");window.sharedResult=document.getElementById("result-canvas")')
                assert page.locator('.topbar #analysis-mode').count()==0
                assert page.locator('#general-flow-body #analysis-mode').count()==1
                def seek(index):
                    page.locator('#seek-value').fill(str(index));page.locator('#seek').click()
                    expect(page.locator('#frame-info')).to_contain_text(f'第 {index} 帧 / 80')
                def full(selector):
                    pane=page.locator(selector).bounding_box();host=page.locator('#viewers').bounding_box()
                    assert abs(pane['width']-host['width'])<1 and abs(pane['x']-host['x'])<1,(pane,host)
                seek(10)
                page.locator('[data-view="compare"]').click()
                handle=page.locator('[data-resize="compare"]');handle.focus();handle.press('ArrowRight')
                page.locator('[data-view="input"]').click();full('#input-pane')
                page.locator('[data-view="result"]').click();full('#result-pane')
                page.locator('[data-view="input"]').click()
                page.locator('#roi-full').click()
                for key,value in [('width',30),('height',20),('x',10),('y',10)]:
                    page.locator('#roi-'+key).fill(str(value));page.locator('#roi-'+key).press('Tab')
                page.locator('#tool-roi').click()
                page.locator('#actual').click()
                expect(page.locator('#zoom-value')).to_have_text('100%')
                before=len(reads)
                page.locator('#analysis-mode').select_option('split_correlation')
                expect(page.locator('#split-view-caption')).to_contain_text('原始帧 10')
                expect(page.locator('#tool-pan')).to_have_class('icon-button selected')
                assert len(reads)==before and not geometry
                assert page.evaluate('sharedCanvas===document.getElementById("input-canvas")&&sharedResult===document.getElementById("result-canvas")')
                assert page.locator('#split-canvas').count()==0
                assert page.locator('#split-flow-body #analysis-mode').count()==1
                center=page.locator('#split-center').input_value()
                performances={}
                for mode in ('general','split_correlation'):
                    page.locator('#analysis-mode').select_option(mode)
                    seek(10);before=len(reads)
                    page.locator('#play').click();page.wait_for_timeout(450);page.locator('#play').click()
                    expect(page.locator('#play')).to_have_attribute('aria-label','播放')
                    page.wait_for_timeout(100)
                    count=len(reads)-before;assert count>=3,(mode,count)
                    performances[mode]={'frames_in_450ms':count,'last_position':page.locator('#seek-value').input_value()}
                    assert not geometry and maximum[0]==1
                assert page.locator('#split-center').input_value()==center
                # Changing the algorithm preset during playback keeps the same clock.
                seek(10);page.locator('#play').click()
                page.locator('#analysis-mode').select_option('general')
                expect(page.locator('#play')).to_have_attribute('aria-label','暂停')
                page.wait_for_timeout(100)
                page.locator('#analysis-mode').select_option('split_correlation')
                expect(page.locator('#play')).to_have_attribute('aria-label','暂停')
                page.locator('#play').click()
                assert int(page.locator('#seek-value').input_value())>10
                assert not geometry and maximum[0]==1
                preview_geometry=len(geometry)
                # Dragging the shared scrubber coalesces intermediate positions.
                page.locator('#scrubber').evaluate("el=>{for(const value of [3,20,40,60]){el.value=value;el.dispatchEvent(new Event('input',{bubbles:true}));}}")
                expect(page.locator('#frame-info')).to_contain_text('第 61 帧 / 80')
                page.locator('#split-open-context').click();page.locator('#split-use-current').click()
                expect(page.locator('#split-center')).to_have_value('61')
                # An explicit derived view does not replace the source's saved transform.
                page.locator('#actual').click()
                page.locator('#split-view').select_option('left')
                expect(page.locator('#input-label')).to_have_text('左幅参考')
                page.locator('#analysis-mode').select_option('general')
                expect(page.locator('#frame-info')).to_contain_text('第 61 帧 / 80')
                expect(page.locator('#roi-x')).to_have_value('10')
                expect(page.locator('#roi-width')).to_have_value('30')
                expect(page.locator('#zoom-value')).to_have_text('100%')
                full('#input-pane')
                page.wait_for_timeout(450);page.reload()
                expect(page.locator('#frame-info')).to_contain_text('第 61 帧 / 80')
                page.locator('#analysis-mode').select_option('split_correlation')
                expect(page.locator('#split-center')).to_have_value('61')
                seek(80);page.locator('#play').click()
                expect(page.locator('#play')).to_have_attribute('aria-label','播放')
                expect(page.locator('#frame-info')).to_contain_text('第 80 帧 / 80')
                for width,height in ((1440,960),(1100,760),(960,760)):
                    page.set_viewport_size({'width':width,'height':height});page.wait_for_timeout(100)
                    full('#input-pane')
                    timeline=page.locator('.timeline').bounding_box()
                    assert timeline['y']+timeline['height']<=height
                    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                    page.screenshot(path=str(output/f'preview-{width}.png'))
                assert not errors,errors
                report={'one_canvas_pair':True,'single_pane_full_width':True,'mode_preserves_position':True,'switch_preset_during_playback':True,'roi_and_raw_transform_preserved':True,'one_frame_request_inflight':maximum[0],'geometry_requests_during_preview':preview_geometry,'explicit_geometry_requests':len(geometry),'independent_correlation_center':True,'latest_seek_and_eof':True,'performances':performances,'javascript_errors':errors}
                (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
                print(json.dumps({'output':str(output),**report},ensure_ascii=False));browser.close()
        finally:
            server.terminate()
            try:server.wait(timeout=10)
            except subprocess.TimeoutExpired:server.kill()


if __name__=='__main__':main()
