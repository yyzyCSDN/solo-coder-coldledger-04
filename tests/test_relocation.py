import unittest
from decimal import Decimal

from coldledger.api import ColdLedgerService
from coldledger.warehouse.relocation import (
    Pallet,
    Destination,
    plan_relocation,
    evaluate,
    feasible,
    validate_plan,
    InfeasibleRelocation,
)
from coldledger.warehouse.relocation_commit import (
    RelocationCoordinator,
    DestinationUnavailable,
    InventoryConflict,
    CommitError,
)

Q = Decimal


def sample():
    pallets = [
        # frozen goods on the dead circuit C1; 90 min exposure budget
        Pallet("F1", "frozen", Q("4"), exposure_minutes=90, circuit="C1", priority=5),
        Pallet("F2", "frozen", Q("3"), exposure_minutes=120, circuit="C1", priority=1),
        # chilled good, only 30 min tolerance
        Pallet("H1", "chilled", Q("2"), exposure_minutes=30, circuit="C2", priority=4),
    ]
    destinations = [
        # D1 frozen+chilled, live circuit, 30 min away, 2 units pre-reserved
        Destination("D1", frozenset({"frozen", "chilled"}), "L1",
                    capacity=Q("10"), reserved=Q("2"), transit_minutes=30),
        # D2 frozen only, live, far (100 min)
        Destination("D2", frozenset({"frozen"}), "L2",
                    capacity=Q("10"), transit_minutes=100),
        # D3 on a dead receiving circuit
        Destination("D3", frozenset({"frozen", "chilled"}), "C1",
                    capacity=Q("20"), transit_minutes=10),
    ]
    return pallets, destinations


class PlanTests(unittest.TestCase):
    def test_feasible_plan_respects_zone_circuit_time_reservation(self):
        pallets, destinations = sample()
        plan = plan_relocation(pallets, destinations, failed_circuits={"C1"})
        mapping = {a.pallet_id: a.destination_id for a in plan.assignments}
        # Every pallet placed exactly once.
        self.assertEqual(sorted(mapping), ["F1", "F2", "H1"])
        # D3 is on a dead receiving circuit and never usable despite huge volume.
        self.assertNotIn("D3", plan.totals())
        # H1 (chilled, 30-min budget) cannot use frozen-only/far D2: only D1 fits.
        self.assertEqual(mapping["H1"], "D1")
        # F2's 120-min budget lets it reach D2 (100 min); F1 (90 min) cannot, so D1.
        self.assertEqual(mapping["F1"], "D1")
        self.assertEqual(mapping["F2"], "D2")
        # D1 usable = 10 - 2 reserved = 8; only F1(4)+H1(2)=6 lands there.
        self.assertLessEqual(plan.totals()["D1"], Q("8"))
        validate_plan(plan, pallets, destinations, failed_circuits={"C1"})

    def test_volume_alone_is_not_enough(self):
        # D3 has the most free volume but sits on the failed circuit.
        pallets, destinations = sample()
        blocked = evaluate(pallets, destinations, failed_circuits={"C1"})
        self.assertEqual(blocked, {})  # the set above is feasible as a whole
        # remove D1/D2: only the dead-circuit D3 remains -> infeasible
        only_dead = [destinations[2]]
        blocked = evaluate(pallets, only_dead, failed_circuits={"C1"})
        self.assertEqual(set(blocked), {"F1", "F2", "H1"})
        with self.assertRaises(InfeasibleRelocation) as ctx:
            plan_relocation(pallets, only_dead, failed_circuits={"C1"})
        self.assertTrue(any("no power" in r for r in ctx.exception.reasons))

    def test_exposure_budget_rejects_too_far_destination(self):
        pallets = [Pallet("X", "frozen", Q("1"), exposure_minutes=60)]
        far = [Destination("FAR", frozenset({"frozen"}), "L9", Q("5"), transit_minutes=120)]
        with self.assertRaises(InfeasibleRelocation) as ctx:
            plan_relocation(pallets, far)
        self.assertTrue(any("exposure allowance" in r for r in ctx.exception.reasons))

    def test_zone_mismatch(self):
        pallets = [Pallet("X", "chilled", Q("1"), exposure_minutes=60)]
        frozen_only = [Destination("F", frozenset({"frozen"}), "L9", Q("5"), transit_minutes=10)]
        with self.assertRaises(InfeasibleRelocation):
            plan_relocation(pallets, frozen_only)

    def test_capacity_after_reservation(self):
        pallets = [Pallet("X", "frozen", Q("5"), exposure_minutes=60)]
        tight = [Destination("T", frozenset({"frozen"}), "L9",
                             capacity=Q("6"), reserved=Q("2"), transit_minutes=10)]
        # usable = 4 < 5 -> infeasible even though raw capacity says 6
        self.assertFalse(feasible(pallets, tight))
        with self.assertRaises(InfeasibleRelocation):
            plan_relocation(pallets, tight)

    def test_priority_urgent_first(self):
        pallets = [
            Pallet("LOW", "frozen", Q("6"), exposure_minutes=120, priority=0),
            Pallet("HOT", "frozen", Q("6"), exposure_minutes=120, priority=9),
        ]
        dests = [
            Destination("NEAR", frozenset({"frozen"}), "L1", Q("7"),
                        reserved=Q("2"), transit_minutes=20),  # usable 5
            Destination("FAR", frozenset({"frozen"}), "L2", Q("10"), transit_minutes=40),
        ]
        # Only one pallet fits at NEAR (usable 5 < 6) so nobody fits NEAR; both go FAR.
        # Tighten: make NEAR usable exactly 6 -> the high-priority pallet takes it.
        dests[0] = Destination("NEAR", frozenset({"frozen"}), "L1", Q("8"),
                               reserved=Q("2"), transit_minutes=20)  # usable 6
        plan = plan_relocation(pallets, dests)
        mapping = {a.pallet_id: a.destination_id for a in plan.assignments}
        self.assertEqual(mapping["HOT"], "NEAR")
        self.assertEqual(mapping["LOW"], "FAR")


class AtomicCommitTests(unittest.TestCase):
    def test_plan_then_commit_locks_everything(self):
        pallets, dests = sample()
        svc = ColdLedgerService()
        plan = svc.plan_relocation(pallets, dests, failed_circuits={"C1"})
        coord = svc.relocation_coordinator(pallets, dests, failed_circuits={"C1"})
        receipt = svc.commit_relocation(coord, plan)
        self.assertEqual(sorted(r.pallet_id for r in receipt.transfers), ["F1", "F2", "H1"])
        # All source stock consumed.
        self.assertTrue(all(r.available == 0 for r in coord.inventory.rows.values()))
        # Reserved capacity matches the plan totals.
        for dest_id, used in plan.totals().items():
            dest = next(d for d in dests if d.destination_id == dest_id)
            self.assertEqual(coord.capacity.rows[dest_id].available, dest.usable - used)
        # Ledger holds exactly the committed set once.
        self.assertEqual(len(coord.records()), 3)

    def test_mid_commit_destination_failure_compensates_all(self):
        pallets, dests = sample()
        coord = RelocationCoordinator(pallets, dests, failed_circuits={"C1"})
        plan = plan_relocation(pallets, dests, failed_circuits={"C1"})

        before_inv = {k: (r.available, r.version) for k, r in coord.inventory.rows.items()}
        before_cap = {k: (r.available, r.version) for k, r in coord.capacity.rows.items()}

        def fail_on_h1(a):
            # D1 dies after earlier pallets (F1) have already been prepared there.
            if a.pallet_id == "H1":
                coord.capacity.disable("D1")

        with self.assertRaises(DestinationUnavailable) as ctx:
            coord.commit(plan, on_prepare=fail_on_h1)
        self.assertEqual(ctx.exception.destination_id, "D1")

        # No half transfer survives: stock and capacity fully restored...
        self.assertEqual(
            {k: (r.available, r.version) for k, r in coord.inventory.rows.items()}, before_inv
        )
        self.assertEqual(
            {k: (r.available, r.version) for k, r in coord.capacity.rows.items()}, before_cap
        )
        # ...and no transfer record was ever written.
        self.assertEqual(coord.records(), ())

    def test_successful_retry_after_failure(self):
        # Dedicated scenario: PRIME is the shortest route; BACKUP is the contingency.
        pallets = [
            Pallet("A", "frozen", Q("4"), exposure_minutes=120, priority=2),
            Pallet("B", "frozen", Q("3"), exposure_minutes=120, priority=1),
        ]
        dests = [
            Destination("PRIME", frozenset({"frozen"}), "L1", Q("10"), transit_minutes=20),
            Destination("BACKUP", frozenset({"frozen"}), "L2", Q("10"), transit_minutes=40),
        ]
        coord = RelocationCoordinator(pallets, dests)
        plan = plan_relocation(pallets, dests)
        mapping = {a.pallet_id: a.destination_id for a in plan.assignments}
        self.assertEqual(mapping, {"A": "PRIME", "B": "PRIME"})

        with self.assertRaises(DestinationUnavailable):
            coord.commit(
                plan,
                on_prepare=lambda a: coord.capacity.disable("PRIME") if a.pallet_id == "B" else None,
            )
        # Everything returned, so a fresh coordinator can re-plan without PRIME and commit.
        live = [d for d in dests if d.destination_id == "BACKUP"]
        coord2 = RelocationCoordinator(pallets, live)
        plan2 = plan_relocation(pallets, live)
        receipt = coord2.commit(plan2)
        self.assertEqual(len(receipt.transfers), 2)
        self.assertTrue(all(r.destination_id == "BACKUP" for r in receipt.transfers))

    def test_stale_inventory_version_conflicts_and_rolls_back(self):
        pallets, dests = sample()
        coord = RelocationCoordinator(pallets, dests, failed_circuits={"C1"})
        plan = plan_relocation(pallets, dests, failed_circuits={"C1"})
        mapping = {a.pallet_id: a.destination_id for a in plan.assignments}

        # Optimistic snapshots captured before concurrent activity.
        expected_inv = coord.inventory.versions(p.pallet_id for p in pallets)
        expected_cap = coord.capacity.versions(a.destination_id for a in plan.assignments)

        # Concurrent committer moves F1 (the first pallet prepared): inventory
        # version advances. Conflict fires on the very first step.
        coord.inventory.hold("F1", coord.inventory.rows["F1"].version)
        before_cap = {k: (r.available, r.version) for k, r in coord.capacity.rows.items()}
        with self.assertRaises(InventoryConflict):
            coord.commit(plan, expected_inventory=expected_inv, expected_capacity=expected_cap)
        # Capacity was never touched because the first hold failed.
        self.assertEqual({k: (r.available, r.version) for k, r in coord.capacity.rows.items()}, before_cap)
        self.assertEqual(coord.records(), ())

        # Mid-list conflict: a concurrent reservation bumps D2 (F2, last pallet).
        coord2 = RelocationCoordinator(pallets, dests, failed_circuits={"C1"})
        d2_now = coord2.capacity.rows["D2"].version
        coord2.capacity.reserve("D2", Q("0.1"), d2_now)  # version -> 1
        before = {k: (r.available, r.version) for k, r in coord2.capacity.rows.items()}
        with self.assertRaises(CommitError):
            coord2.commit(
                plan,
                expected_capacity={"D2": d2_now},  # stale: says 0, current is 1
            )
        # F1/H1 already held capacity at D1, then F2 conflicted: all fully compensated.
        self.assertEqual(
            {k: (r.available, r.version) for k, r in coord2.capacity.rows.items()}, before
        )
        # Every pallet is back on the shelf at its original footprint and version 0.
        for p in pallets:
            row = coord2.inventory.rows[p.pallet_id]
            self.assertEqual(row.available, p.footprint)
            self.assertEqual(row.version, 0)

    def test_validate_plan_catches_drift(self):
        pallets, dests = sample()
        plan = plan_relocation(pallets, dests, failed_circuits={"C1"})
        # Circuit L1 (D1) drops after planning: the plan is no longer valid.
        with self.assertRaises(ValueError):
            validate_plan(plan, pallets, dests, failed_circuits={"C1", "L1"})


if __name__ == "__main__":
    unittest.main()
