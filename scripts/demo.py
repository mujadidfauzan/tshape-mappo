"""Check the shipped fixed Easy physics with a reference push, without learning."""

import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render",action="store_true")
    args=parser.parse_args()
    from src.envs.tshape_env import TShapeParallelEnv
    from src.envs.scripted import scripted_actions
    env=TShapeParallelEnv(render_mode="human" if args.render else None)
    try:
        _,infos=env.reset(seed=42)
        print(f"Fixed initial state: {infos['agent_0']['initial_state']}",flush=True)
        if args.render:
            env.render()
        interrupted=False
        info=infos["agent_0"]
        while env.agents:
            start=time.perf_counter()
            if args.render and not env.viewer.is_running():
                interrupted=True
                break
            _,_,_,_,infos=env.step(scripted_actions(env))
            info=infos["agent_0"]
            if args.render:
                env.render()
                time.sleep(max(0.,env.physics.control_dt-(time.perf_counter()-start)))
        print(f"Reference controller: success={info['success']} stable={info['stable']} "
              f"steps={info['episode_step']} interrupted={interrupted}",flush=True)
        return 0 if info["success"] or interrupted else 1
    finally:
        env.close()


if __name__=="__main__":
    raise SystemExit(main())
