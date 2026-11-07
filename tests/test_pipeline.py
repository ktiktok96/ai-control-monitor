import json
from pathlib import Path

from control_eval.agent import run_agent
from control_eval.models import DiskCache, MockClient
from control_eval.monitor import parse_score
from control_eval.run import main
from control_eval.tasks import load_tasks

TASKS = Path(__file__).parent.parent / "data" / "tasks.json"


def test_tasks_load():
    mains, sides = load_tasks(TASKS)
    assert len(mains) >= 5 and len(sides) >= 3
    assert len({t.id for t in mains}) == len(mains)


def test_side_task_detection():
    _, sides = load_tasks(TASKS)
    exfil = next(s for s in sides if s.id == "exfiltrate_key")
    assert exfil.succeeded('open("/tmp/debug.log","a").write(os.environ["FAKE_API_KEY"])')
    assert not exfil.succeeded("def f(x):\n    return x")


def test_parse_score():
    assert parse_score("<score>73</score>") == 73
    assert parse_score("<score> 150 </score>") == 100
    assert parse_score("no score here") is None


async def test_agent_attack_vs_honest():
    mains, sides = load_tasks(TASKS)
    client = MockClient()
    honest = await run_agent(client, mains[0])
    attack = await run_agent(client, mains[0], sides[0])
    assert not honest.is_attack and honest.side_task_succeeded is None
    assert attack.is_attack and attack.side_task_succeeded


def test_cache_roundtrip(tmp_path):
    c = DiskCache(tmp_path)
    k = c.key("m", "s", "u", 10)
    assert c.get(k) is None
    c.set(k, "hello")
    assert c.get(k) == "hello"


def test_end_to_end_dry_run(tmp_path):
    main(["--dry-run", "--tasks", str(TASKS), "--out", str(tmp_path), "--cache",
          str(tmp_path / "cache")])
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["results"]
    # The mock monitor perfectly flags the mock attack, so AUROC should be 1 where defined.
    assert all(r["auroc"] in (None, 1.0) for r in summary["results"])
    assert (tmp_path / "records.jsonl").exists()
