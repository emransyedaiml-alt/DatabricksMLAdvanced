from src.evaluation.gates import check_gates, load_gates


def test_gate_file_has_both_sections():
    assert load_gates("rag")["block"] and load_gates("agent")["block"]


def test_floor_equals_and_missing():
    gates = {"block": {"hit": {"floor": 0.7}, "safe": {"equals": True}, "needed": {}}, "info": ["latency"]}
    res = check_gates({"hit": 0.9, "safe": True, "latency": 12}, gates)
    assert not res["passed"]                                       # "needed" is missing in a block tier
    assert {r["metric"]: r["status"] for r in res["results"]} == {"hit": "pass", "safe": "pass", "needed": "fail", "latency": "info"}
    assert not check_gates({"hit": 0.5, "safe": True, "needed": 1}, gates)["passed"]
    assert not check_gates({"hit": 0.9, "safe": False, "needed": 1}, gates)["passed"]


def test_regression_vs_champion():
    gates = {"block": {"completion": {"floor": 0.5}}, "warn": {"accuracy": {"tolerance": 0.05}}}
    champion = {"completion": 1.0, "accuracy": 0.90}
    small = check_gates({"completion": 1.0, "accuracy": 0.87}, gates, champion)
    assert small["passed"] and [r["status"] for r in small["results"]] == ["pass", "warn"]
    big = check_gates({"completion": 1.0, "accuracy": 0.80}, gates, champion)
    assert not big["passed"]
    block_drop = check_gates({"completion": 0.9, "accuracy": 0.9}, gates, champion)
    assert not block_drop["passed"]                                 # block tier: any drop beyond tolerance 0 fails
