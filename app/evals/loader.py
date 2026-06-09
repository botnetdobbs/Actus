from pathlib import Path
import yaml
from app.evals.models import EvalSuite


def load_suite(agent_id: str, evals_dir: str = "evals") -> EvalSuite:
    path = Path(evals_dir) / f"{agent_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No eval fixture found at '{path}'. "
            f"Create '{path}' to add eval cases for agent '{agent_id}'."
        )
    with open(path) as f:
        data = yaml.safe_load(f)
    return EvalSuite(**data)


def load_all_suites(evals_dir: str = "evals") -> list[EvalSuite]:
    suites = []
    for yaml_file in sorted(Path(evals_dir).glob("*.yaml")):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        if data:
            suites.append(EvalSuite(**data))
    return suites
