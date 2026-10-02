import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from coldledger.warehouse.relocation import (
    RelocationItem, Destination,
    plan_relocation, validate_plan, summarize, explain,
    RelocationDesk, RelocationInfeasible, RelocationCommitFailed,
    execute,
)
from coldledger.storage.event_store import EventStore

UTC = timezone.utc


def H(h):
    return timedelta(hours=h)


def item(iid, zone="-25C", qty="10", circuits=frozenset({"M", "G"}),
         max_transit=H(6), exposure=10.0, priority=0, redundant=False):
    return RelocationItem(iid, zone, Decimal(qty), circuits,
                          max_transit, exposure, priority, redundant)


def dest(did, zones=frozenset({"-25C"}), circuits=frozenset({"M"}),
         cap="100", transit=H(2), exposure=1.0, reserve="0", enabled=True):
    return Destination(did, zones, circuits, Decimal(cap), transit,
                       exposure, Decimal(reserve), enabled)


class PlanningTests(unittest.TestCase):
    def test_plan_respects_reserve_not_just_volume(self):
        # 100 capacity but 95 reserved -> only 5 usable; the move needs 10.
        d = dest("D1", cap="100", reserve="95")
        d2 = dest("D2", cap="10")
        plan = plan_relocation([item("I1")], [d, d2])
        totals = summarize(plan)
        self.assertEqual(totals, {"D1": Decimal("5"), "D2": Decimal("5")})
        validate_plan(plan, [item("I1")], [d, d2])

    def test_volume_alone_is_not_enough(self):
        # Big free volume, but wrong zone, dead circuit, late transit,
        # exhausted exposure budget - all four must be reported.
        d = dest("D1", zones=frozenset({"+5C"}), circuits=frozenset({"X"}),
                 cap="1000", transit=H(20), exposure=50.0)
        with self.assertRaises(RelocationInfeasible) as caught:
            plan_relocation([item("I1")], [d])
        codes = {r.code for r in caught.exception.rejections}
        self.assertEqual(codes, {"zone", "circuit", "transit", "exposure"})

    def test_exposure_budget_accumulates_across_splits(self):
        a = dest("A", cap="5", exposure=6.0)
        b = dest("B", cap="5", exposure=6.0)
        # qty 10 forces a 5/5 split; budget 10 cannot cover 6+6
        with self.assertRaises(RelocationInfeasible) as caught:
            plan_relocation([item("I1", exposure=10.0)], [a, b])
        self.assertIn("exposure",
                      {r.code for r in caught.exception.rejections})
        plan = plan_relocation([item("I1", exposure=12.0)], [a, b])
        self.assertEqual(summarize(plan),
                         {"A": Decimal("5"), "B": Decimal("5")})

    def test_redundant_load_needs_distinct_circuits(self):
        a = dest("A", cap="5", circuits=frozenset({"M"}))
        b = dest("B", cap="5", circuits=frozenset({"G"}))
        with self.assertRaises(RelocationInfeasible) as caught:
            plan_relocation([item("I1", circuits=frozenset({"M", "G"}),
                                  redundant=True, max_transit=H(10))], [a])
        self.assertIn("circuit", {r.code for r in caught.exception.rejections})
        plan = plan_relocation(
            [item("I1", circuits=frozenset({"M", "G"}),
                  redundant=True, max_transit=H(10))], [a, b])
        used = {x.circuit for x in plan.assignments}
        self.assertEqual(used, {"M", "G"})

    def test_priority_ordering_and_explain(self):
        plan = plan_relocation(
            [item("LOW", qty="5", priority=0), item("HOT", qty="5", priority=9)],
            [dest("D1", cap="6"), dest("D2", cap="10")])
        # HOT placed first into the tighter-fit destination by slack.
        self.assertTrue(explain(plan)[0].startswith("HOT"))

    def test_no_destinations(self):
        with self.assertRaises(RelocationInfeasible):
            plan_relocation([item("I1")], [])

    def test_disabled_destination_rejected(self):
        with self.assertRaises(RelocationInfeasible) as caught:
            plan_relocation([item("I1")], [dest("D1", enabled=False)])
        self.assertEqual(caught.exception.rejections[0].code, "disabled")


class AtomicCommitTests(unittest.TestCase):
    def test_commit_locks_inventory_and_capacity_once(self):
        items = [item("I1"), item("I2", qty="7")]
        dsts = [dest("D1", cap="20")]
        desk = RelocationDesk(dsts, items)
        prepared = desk.prepare(plan_relocation(items, dsts), "R1")
        locks = desk.commit(prepared)
        self.assertEqual(len(locks), 2)
        self.assertEqual(desk.available("I1"), Decimal("0"))
        self.assertEqual(desk.available("I2"), Decimal("0"))
        self.assertEqual(desk.held("D1"), Decimal("17"))
        # idempotent re-commit returns the same locks
        self.assertEqual(desk.commit(prepared), locks)

    def test_destination_failing_after_prepare_aborts_whole_move(self):
        items = [item("I1"), item("I2", qty="4")]
        dsts = [dest("D1"), dest("D2", cap="20")]
        desk = RelocationDesk(dsts, items)
        prepared = desk.prepare(plan_relocation(items, dsts), "R1")
        desk.disable_destination(prepared.plan.assignments[0].destination_id)
        with self.assertRaises(RelocationCommitFailed) as caught:
            desk.commit(prepared)
        self.assertIn("disabled", {r.code for r in caught.exception.rejections})
        # no half-transfer: nothing was consumed anywhere
        self.assertEqual(desk.available("I1"), Decimal("10"))
        self.assertEqual(desk.available("I2"), Decimal("4"))
        for d in ("D1", "D2"):
            self.assertEqual(desk.held(d), Decimal("0"))

    def test_delay_past_transit_or_exposure_budget_aborts(self):
        items = [item("I1", max_transit=H(4), exposure=2.0)]
        dsts = [dest("D1", transit=H(2), exposure=1.0)]
        desk = RelocationDesk(dsts, items)
        prepared = desk.prepare(plan_relocation(items, dsts), "R1")
        desk.report_delay("D1", H(8), 3.0)
        with self.assertRaises(RelocationCommitFailed) as caught:
            desk.commit(prepared)
        codes = {r.code for r in caught.exception.rejections}
        self.assertEqual(codes, {"transit", "exposure"})
        self.assertEqual(desk.available("I1"), Decimal("10"))

    def test_two_plans_cannot_overbook_destination(self):
        dsts = [dest("D1", cap="12")]
        desk = RelocationDesk(dsts, [item("I1", qty="10"), item("I2", qty="10")])
        desk.commit(desk.prepare(plan_relocation([item("I1", qty="10")], dsts), "A"))
        pb = desk.prepare(plan_relocation([item("I2", qty="10")], dsts), "B")
        with self.assertRaises(RelocationCommitFailed) as caught:
            desk.commit(pb)
        self.assertIn("capacity", {r.code for r in caught.exception.rejections})
        # rejected plan leaves no partial holds behind
        self.assertEqual(desk.available("I2"), Decimal("10"))
        self.assertEqual(desk.held("D1"), Decimal("10"))

    def test_release_is_atomic_compensation(self):
        items = [item("I1")]
        dsts = [dest("D1", cap="20")]
        desk, locks = execute(items, dsts, "R1")
        self.assertEqual(desk.available("I1"), Decimal("0"))
        self.assertEqual(desk.held("D1"), Decimal("10"))
        released = desk.release("R1", reason="power restored")
        self.assertEqual(released, locks)
        self.assertEqual(desk.available("I1"), Decimal("10"))
        self.assertEqual(desk.held("D1"), Decimal("0"))
        with self.assertRaises(KeyError):
            desk.release("R1")

    def test_execute_convenience(self):
        desk, locks = execute([item("I1")], [dest("D1")], "R1",
                              at=datetime(2026, 10, 1, tzinfo=UTC))
        self.assertEqual(locks[0].destination_id, "D1")

    def test_events_audit_prepare_commit_abort(self):
        events = EventStore()
        items = [item("I1")]
        dsts = [dest("D1"), dest("D2")]
        desk = RelocationDesk(dsts, items, events=events)
        prepared = desk.prepare(plan_relocation(items, dsts), "R1")
        desk.disable_destination(prepared.plan.assignments[0].destination_id)
        with self.assertRaises(RelocationCommitFailed):
            desk.commit(prepared)
        kinds = [e.kind for e in events.read("relocation:R1")]
        self.assertEqual(kinds, ["relocation_prepared", "relocation_aborted"])

    def test_duplicate_reference_rejected(self):
        dsts = [dest("D1")]
        desk = RelocationDesk(dsts, [item("I1")])
        desk.commit(desk.prepare(plan_relocation([item("I1")], dsts), "R1"))
        with self.assertRaises(ValueError):
            desk.prepare(plan_relocation([item("I1")], dsts), "R1")


class ServiceFacadeTests(unittest.TestCase):
    def test_facade_plan(self):
        from coldledger.api import ColdLedgerService
        plan = ColdLedgerService().relocation_plan(
            [item("I1")], [dest("D1")])
        self.assertEqual(plan.assignments[0].destination_id, "D1")


if __name__ == "__main__":
    unittest.main()
