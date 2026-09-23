"""Open a scene XML in the MuJoCo simulator viewer (blocking).

Same viewer as the MuJoCo.app simulate window; the model is loaded
in-process so macOS file-type quirks do not matter.
"""
import argparse
import sys

import mujoco
import mujoco.viewer  # noqa: F401  (submodule not auto-imported)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?", default="scenes/scene_seed0.xml")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    print(f"scene: {args.scene}")
    print("keys: space=pause, F1/F2/F3=UI, F6=open new model (drag&drop also works), ESC=quit")
    try:
        mujoco.viewer.launch(model, data)
    except Exception as e:  # noqa: BLE001
        print(f"viewer error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()