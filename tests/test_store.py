import pytest

from agent_runtime import events as ev
from agent_runtime.store import EventStore


def test_append_assigns_sequential_seq(store):
    rid = store.new_run_id()
    e0 = store.append(rid, ev.RUN_CREATED, {"agent": "a", "request": "r"})
    e1 = store.append(rid, ev.CHECKPOINT, {"n": 1})
    assert (e0.seq, e1.seq) == (0, 1)
    got = store.events(rid)
    assert [e.type for e in got] == [ev.RUN_CREATED, ev.CHECKPOINT]
    assert got[1].payload == {"n": 1}


def test_runs_are_isolated(store):
    a, b = store.new_run_id(), store.new_run_id()
    store.append(a, ev.RUN_CREATED, {})
    store.append(b, ev.RUN_CREATED, {})
    store.append(a, ev.CHECKPOINT, {})
    assert len(store.events(a)) == 2
    assert len(store.events(b)) == 1
    assert set(store.run_ids()) == {a, b}


def test_unknown_event_type_rejected(store):
    with pytest.raises(ValueError, match="unknown event type"):
        store.append("r1", "made_up_event", {})


def test_until_filters_by_seq(store):
    rid = "r-until"
    for i in range(5):
        store.append(rid, ev.CHECKPOINT, {"i": i})
    assert len(store.events(rid, until=2)) == 3
    assert store.events(rid, until=0)[0].payload == {"i": 0}


def test_payload_roundtrip_is_exact(store):
    payload = {"nested": {"a": [1, 2.5, "x"], "b": None}, "flag": True}
    store.append("rp", ev.CHECKPOINT, payload)
    assert store.events("rp")[0].payload == payload


def test_idempotency_first_write_wins(store):
    store.claim_execution("key1", "r1", {"result": "first"})
    store.claim_execution("key1", "r1", {"result": "second"})
    assert store.get_execution("key1") == {"result": "first"}
    assert store.get_execution("missing") is None


def test_deterministic_clock_controls_timestamps(tmp_path):
    ticks = iter([10.0, 11.5])
    st = EventStore(tmp_path / "c.db", clock=lambda: next(ticks))
    st.append("r", ev.RUN_CREATED, {})
    st.append("r", ev.CHECKPOINT, {})
    assert [e.ts for e in st.events("r")] == [10.0, 11.5]
    st.close()
