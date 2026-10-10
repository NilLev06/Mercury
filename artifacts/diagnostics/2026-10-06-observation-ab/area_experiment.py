import json, shutil, hashlib
from unittest.mock import patch
import experiment as ex

def area_state(controller, sensor, target, obstacle):
    state = ex.original_state(controller, sensor, target, obstacle)
    if controller.agent.task != 'bypass-v1':
        return state
    effective = target if target is not None else controller.navigation.recalled_target(sensor)
    detail = ('goal_area_none' if effective is None else
              'goal_area_far' if effective.apparent_area < .025 else
              'goal_area_middle' if effective.apparent_area < .1 else 'goal_area_near')
    return state + (detail,)

if __name__ == '__main__':
    before = hashlib.sha256(ex.source.read_bytes()).hexdigest()
    output = ex.root/'area.json'
    if output.exists():
        raise RuntimeError('refusing overwrite')
    shutil.copy2(ex.source, output)
    ex.detailed_state = area_state
    with ex.DetailContext():
        with patch('fly_brain_rc.bypass_learning.bypass_cases', return_value=ex.bypass_cases(73,24)):
            result = ex.train_bypass(episodes=24, steps=2400, output=output, seed=42)
            print('TRAIN_SUMMARY area', result, flush=True)
    ex.evaluate('area', output, ex.cases()+ex.bypass_cases(73,6,held_out=True), detailed=True)
    assert hashlib.sha256(ex.source.read_bytes()).hexdigest() == before
    print('SOURCE_UNCHANGED', before, flush=True)
