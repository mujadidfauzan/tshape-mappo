"""Load saved policies with a local, verified and self-contained XML snapshot."""

from contextlib import contextmanager
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile

from src.algorithms.mappo import MAPPO
from src.envs.config import load_config, load_checkpoint_config
from src.training.trainer import feature_schema


@contextmanager
def loaded_policy(checkpoint, device="cpu", environment_override=None):
    agent, payload = MAPPO.from_checkpoint(checkpoint, device=device)
    training = payload["training"]
    if training.get("feature_schema") != feature_schema(agent.dimensions["num_agents"]):
        raise ValueError("checkpoint feature schema mismatch")
    with tempfile.TemporaryDirectory(prefix="tshape-benchmark-") as directory:
        if environment_override is None:
            xml = training["model_xml"]
            if hashlib.sha256(xml.encode()).hexdigest() != training["model_sha256"]:
                raise ValueError("checkpoint XML hash mismatch")
            path = Path(directory) / "arena.xml"
            path.write_text(xml)
            config = replace(load_checkpoint_config(training["environment_config"]), model_path=str(path))
        else:
            config = load_config(environment_override)
        yield agent, config, payload
