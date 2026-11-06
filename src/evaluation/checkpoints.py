"""Load self-contained Easy/Medium checkpoints with saved geometry and spawns."""

from contextlib import contextmanager
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile

from src.algorithms.mappo import MAPPO
from src.envs.config import load_config


@contextmanager
def loaded_policy(checkpoint, device="cpu"):
    agent, payload = MAPPO.from_checkpoint(checkpoint, device=device)
    training = payload["training"]
    if training.get("task") not in ("fixed_easy_v1", "easy_v2", "medium_v1"):
        raise ValueError("this checkpoint was not created by the Easy/Medium trainer")
    xml = training["model_xml"]
    if hashlib.sha256(xml.encode()).hexdigest() != training["model_sha256"]:
        raise ValueError("checkpoint model hash mismatch")
    with tempfile.TemporaryDirectory(prefix="tshape-easy-") as directory:
        model = Path(directory)/"arena.xml"
        model.write_text(xml)
        config = replace(load_config(training["environment_config"]), model_path=str(model))
        if (training["task"] == "medium_v1") != (config.task == "medium"):
            raise ValueError("checkpoint task does not match its environment configuration")
        yield agent, config, payload
