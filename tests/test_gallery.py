import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


class GalleryTests(unittest.TestCase):
    def test_http_catalogue_state_and_mutation_validation(self):
        from fly_brain_rc.dashboard.lab import LabServer,JobManager
        from threading import Thread
        from urllib.request import urlopen,Request
        from urllib.error import HTTPError
        with tempfile.TemporaryDirectory() as root:
            manager=JobManager(root);server=LabServer(manager,port=0)
            thread=Thread(target=server.http.serve_forever,daemon=True);thread.start()
            base=f'http://127.0.0.1:{server.address[1]}'
            try:
                with urlopen(base+'/api/gallery/maps') as response:data=json.load(response)
                self.assertEqual(len(data['maps']),231)
                with urlopen(base+'/api/gallery/state') as response:self.assertEqual(json.load(response)['status'],'idle')
                for origin,status in [(base,400),('http://evil.example',403)]:
                    req=Request(base+'/api/gallery/start',data=b'{"map_id":"missing"}',headers={'Content-Type':'application/json','Origin':origin})
                    with self.assertRaises(HTTPError) as raised:urlopen(req)
                    self.assertEqual(raised.exception.code,status)
            finally:
                server.http.shutdown();server.http.server_close();thread.join(2);manager.close()
                if hasattr(server,'gallery'):server.gallery.close()

    def test_catalogue_unique_reproducible_and_excludes_sealed_seeds(self):
        from fly_brain_rc.dashboard.gallery import catalogue
        maps=catalogue(Path('/missing/project'))['maps']
        self.assertEqual(len(maps),231);self.assertEqual(len({m['id'] for m in maps}),231)
        self.assertEqual(sum(m['group']=='old' for m in maps),7)
        self.assertEqual(sum(m['group']=='complex32' for m in maps),32)
        self.assertEqual(sum(m['group']=='independent' for m in maps),192)
        self.assertTrue(all('104000' not in m['name'] and '104001' not in m['name'] for m in maps))
        self.assertTrue(all(m['known_result'] is None for m in maps))
        self.assertEqual(maps,catalogue(Path('/missing/project'))['maps'])

    def test_unknown_ids_fields_and_invalid_speeds_rejected(self):
        from fly_brain_rc.dashboard.gallery import validate_request,catalogue
        maps={m['id']:m for m in catalogue(Path('/missing/project'))['maps']};key=next(iter(maps))
        for r in [dict(map_id='../../etc/passwd'),dict(map_id=key,speed=True),dict(map_id=key,speed=0),dict(map_id=key,speed=3),dict(map_id=key,path='x')]:
            with self.assertRaises(ValueError):validate_request(r,maps)
        self.assertEqual(validate_request(dict(map_id=key,speed=2),maps),(key,2))

    def test_serial_completion_stop_restart_preserves_results_and_ignores_stale_run(self):
        from fly_brain_rc.dashboard.gallery import GalleryManager
        with tempfile.TemporaryDirectory() as root:
            manager=GalleryManager(root);key=next(iter(manager.maps))
            script='import json; print(json.dumps({"type":"frame","frame":{"pose":[0,0,0]}}),flush=True); print(json.dumps({"type":"result","result":{"success":True,"contacts":0,"cancelled":False}}),flush=True)'
            with patch.object(manager,'_command',return_value=[sys.executable,'-c',script]):
                first=manager.start(dict(map_id=key));run=first['run_id']
                deadline=time.monotonic()+3
                while manager.state()['status'] not in ('finished','error') and time.monotonic()<deadline:time.sleep(.02)
                self.assertEqual(manager.state()['status'],'finished');self.assertIn(key,manager.state()['history'])
            with patch.object(manager,'_command',return_value=[sys.executable,'-c','import time;time.sleep(30)']):
                manager.start(dict(map_id=key))
                with self.assertRaises(RuntimeError):manager.start(dict(map_id=key))
                manager._receive(run,dict(type='result',result=dict(success=False,contacts=1,cancelled=False)))
                self.assertEqual(manager.state()['status'],'starting')
                manager.stop();deadline=time.monotonic()+5
                while manager.state()['status'] in ('starting','running','stopping') and time.monotonic()<deadline:time.sleep(.02)
                self.assertEqual(manager.state()['status'],'stopped')
                self.assertTrue(manager.state()['history'][key]['success'])
            manager.close()
