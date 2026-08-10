from __future__ import annotations

import lerobot_unoarm  # noqa: F401


def main() -> None:
    import lerobot.envs  # noqa: F401
    from lerobot.scripts.lerobot_train import main as train_main

    train_main()


def eval_main() -> None:
    import lerobot.envs  # noqa: F401
    from lerobot.scripts.lerobot_eval import main as lerobot_eval_main

    lerobot_eval_main()
