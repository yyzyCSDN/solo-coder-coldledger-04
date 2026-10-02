"""Emergency relocation for a stricken (e.g. power-out) cold store.

Sizing a relocation by free volume alone is unsafe.  Every move has to
satisfy five constraints simultaneously:

* temperature zone (the destination must hold the required zone),
* electrical circuit (the load must land on a live, compatible circuit),
* transit time (the move must fit inside the load's time budget),
* destination reservation (reserved volume can never be promised away),
* exposure allowance (the cumulative excursion cost of the move(s)
  must stay inside the load's exposure budget).

:func:`plan_relocation` first produces an executable :class:`Plan`; the
plan is then committed in one atomic step through :class:`RelocationDesk`,
which locks inventory and destination capacity together.  If any
destination fails between planning and commit the whole relocation is
rejected - partial transfer results are impossible - and a committed
relocation can be released atomically via :meth:`RelocationDesk.release`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from threading import RLock
from typing import Iterable

# --------------------------------------------------------------------- domain


@dataclass(frozen=True)
class RelocationItem:
    item_id: str
    zone: str
    quantity: Decimal
    circuits_allowed: frozenset[str]
    max_transit: timedelta
    exposure_budget: float
    priority: int = 0
    redundant: bool = False

    def validate(self) -> None:
        if not self.item_id:
            raise ValueError("item_id required")
        if not self.zone:
            raise ValueError("zone required")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if not self.circuits_allowed:
            raise ValueError("circuits_allowed required")
        if self.max_transit < timedelta(0):
            raise ValueError("max_transit cannot be negative")
        if self.exposure_budget < 0:
            raise ValueError("exposure_budget cannot be negative")


@dataclass(frozen=True)
class Destination:
    destination_id: str
    zones: frozenset[str]
    circuits: frozenset[str]
    capacity: Decimal
    transit: timedelta
    exposure_cost: float
    reserve: Decimal = Decimal("0")
    enabled: bool = True

    @property
    def usable(self) -> Decimal:
        """Volume that may actually be promised (capacity minus reserve)."""
        return max(Decimal("0"), self.capacity - self.reserve)

    def validate(self) -> None:
        if not self.destination_id:
            raise ValueError("destination_id required")
        if not self.zones:
            raise ValueError("destination must serve at least one zone")
        if not self.circuits:
            raise ValueError("destination must have a live circuit")
        if self.capacity < 0 or self.reserve < 0:
            raise ValueError("negative volume")
        if self.reserve > self.capacity:
            raise ValueError("reserve exceeds capacity")
        if self.transit < timedelta(0) or self.exposure_cost < 0:
            raise ValueError("transit and exposure cannot be negative")


@dataclass(frozen=True)
class Assignment:
    item_id: str
    destination_id: str
    quantity: Decimal
    circuit: str
    transit: timedelta
    exposure_cost: float


@dataclass(frozen=True)
class Plan:
    assignments: tuple[Assignment, ...] = ()

    def quantities_by_item(self) -> dict[str, Decimal]:
        out: dict[str, Decimal] = {}
        for a in self.assignments:
            out[a.item_id] = out.get(a.item_id, Decimal("0")) + a.quantity
        return out

    def quantities_by_destination(self) -> dict[str, Decimal]:
        out: dict[str, Decimal] = {}
        for a in self.assignments:
            out[a.destination_id] = out.get(a.destination_id, Decimal("0")) + a.quantity
        return out


@dataclass(frozen=True)
class Rejection:
    item_id: str
    destination_id: str | None
    code: str
    detail: str


class RelocationInfeasible(ValueError):
    """Raised at planning time when a load cannot satisfy every constraint."""

    def __init__(self, rejections: Iterable[Rejection]):
        self.rejections = tuple(rejections)
        codes = sorted({r.code for r in self.rejections})
        msg = "; ".join(f"{r.item_id}->{r.destination_id or '*'}: {r.code}"
                        for r in self.rejections)
        super().__init__(f"relocation infeasible [{','.join(codes)}] {msg}")


class RelocationCommitFailed(RuntimeError):
    """Raised when a prepared plan can no longer be locked atomically.

    No inventory or capacity locks survive this error.
    """

    def __init__(self, rejections: Iterable[Rejection]):
        self.rejections = tuple(rejections)
        msg = "; ".join(f"{r.item_id}->{r.destination_id or '*'}: {r.code}"
                        for r in self.rejections)
        super().__init__(f"relocation commit aborted: {msg}")


# --------------------------------------------------------------------- planning


def _evaluate(item: RelocationItem, dest: Destination, take: Decimal,
              exposure_left: float, circuits_used: set[str]
              ) -> tuple[str | None, list[Rejection]]:
    """Return ``(chosen circuit, rejections)``.  Circuit is ``None`` on reject.

    Every failed dimension is reported - volume alone never decides.
    """
    reasons: list[Rejection] = []
    if not dest.enabled:
        reasons.append(Rejection(item.item_id, dest.destination_id, "disabled",
                                 "destination unavailable"))
        return None, reasons
    if item.zone not in dest.zones:
        reasons.append(Rejection(item.item_id, dest.destination_id, "zone",
                                 f"{item.zone} not served by {sorted(dest.zones)}"))
    compatible = item.circuits_allowed & dest.circuits
    if item.redundant:
        compatible = {c for c in compatible if c not in circuits_used}
    if not compatible:
        reasons.append(Rejection(item.item_id, dest.destination_id, "circuit",
                                 f"no live compatible circuit "
                                 f"(need {sorted(item.circuits_allowed)})"))
    if dest.transit > item.max_transit:
        reasons.append(Rejection(item.item_id, dest.destination_id, "transit",
                                 f"{dest.transit} exceeds {item.max_transit}"))
    if dest.exposure_cost > exposure_left:
        reasons.append(Rejection(item.item_id, dest.destination_id, "exposure",
                                 f"cost {dest.exposure_cost:g} > "
                                 f"budget left {exposure_left:g}"))
    if take <= 0:
        reasons.append(Rejection(item.item_id, dest.destination_id, "capacity",
                                 "no free volume"))
    if reasons:
        return None, reasons
    return min(compatible), reasons


def plan_relocation(items: Iterable[RelocationItem],
                    destinations: Iterable[Destination]) -> Plan:
    """Build a fully constrained relocation plan or raise.

    Loads are handled in priority / urgency order and a load may be split
    across destinations; each split of a ``redundant`` load must land on a
    different electrical circuit.
    """
    its = list(items)
    dsts: dict[str, Destination] = {}
    for d in destinations:
        d.validate()
        if d.destination_id in dsts:
            raise ValueError(f"duplicate destination {d.destination_id}")
        dsts[d.destination_id] = d
    for item in its:
        item.validate()
        if sum(1 for x in its if x.item_id == item.item_id) > 1:
            raise ValueError(f"duplicate item {item.item_id}")
    if not its:
        return Plan()
    if not dsts:
        raise RelocationInfeasible(
            [Rejection(i.item_id, None, "no_destination",
                       "no destinations supplied") for i in its])

    remaining = {d.destination_id: d.usable for d in dsts.values() if d.enabled}
    exposure_left = {i.item_id: float(i.exposure_budget) for i in its}
    circuits_used: dict[str, set[str]] = {i.item_id: set() for i in its}
    out: list[Assignment] = []

    order = sorted(its,
                   key=lambda i: (-i.priority, i.max_transit,
                                  i.exposure_budget, i.item_id))
    for item in order:
        outstanding = item.quantity
        while outstanding > 0:
            options = []
            rejections: list[Rejection] = []
            for dest in sorted(dsts.values(), key=lambda d: d.destination_id):
                take = min(remaining.get(dest.destination_id, Decimal("0")),
                           outstanding)
                circuit, reasons = _evaluate(item, dest, take,
                                             exposure_left[item.item_id],
                                             circuits_used[item.item_id])
                if circuit is None:
                    rejections.extend(reasons)
                    continue
                slack = remaining[dest.destination_id] - take
                options.append(((slack, dest.transit, dest.exposure_cost,
                                 dest.destination_id), dest, circuit, take))
            if not options:
                dedup = {(r.destination_id, r.code): r for r in rejections}
                raise RelocationInfeasible(list(dedup.values()))
            _, dest, circuit, take = min(options, key=lambda o: o[0])
            out.append(Assignment(item.item_id, dest.destination_id, take,
                                  circuit, dest.transit, dest.exposure_cost))
            remaining[dest.destination_id] -= take
            exposure_left[item.item_id] -= dest.exposure_cost
            circuits_used[item.item_id].add(circuit)
            outstanding -= take

    plan = Plan(tuple(out))
    validate_plan(plan, its, dsts.values())
    return plan


def validate_plan(plan: Plan, items: Iterable[RelocationItem],
                  destinations: Iterable[Destination],
                  require_all_items: bool = True) -> None:
    """Independent re-check of every constraint and quantity in the plan.

    When ``require_all_items`` is False (desk prepare) registered stock
    outside the plan is simply ignored; every item the plan does touch
    must still be covered exactly.
    """
    item_map = {i.item_id: i for i in items}
    dest_map = {d.destination_id: d for d in destinations}
    item_qty: dict[str, Decimal] = {}
    dest_qty: dict[str, Decimal] = {}
    item_exposure: dict[str, float] = {}
    for a in plan.assignments:
        if a.item_id not in item_map:
            raise ValueError(f"plan references unknown item {a.item_id}")
        if a.destination_id not in dest_map:
            raise ValueError(f"plan references unknown destination {a.destination_id}")
        if a.quantity <= 0:
            raise ValueError("assignment must be positive")
        item = item_map[a.item_id]
        dest = dest_map[a.destination_id]
        if not dest.enabled:
            raise ValueError(f"assignment to disabled destination {dest.destination_id}")
        if item.zone not in dest.zones:
            raise ValueError(f"zone mismatch for {a.item_id}")
        if a.circuit not in item.circuits_allowed or a.circuit not in dest.circuits:
            raise ValueError(f"circuit mismatch for {a.item_id}")
        if a.transit > item.max_transit:
            raise ValueError(f"transit budget exceeded for {a.item_id}")
        if a.transit != dest.transit or a.exposure_cost != dest.exposure_cost:
            raise ValueError(f"assignment terms stale for {a.item_id}")
        item_qty[a.item_id] = item_qty.get(a.item_id, Decimal("0")) + a.quantity
        dest_qty[a.destination_id] = dest_qty.get(a.destination_id, Decimal("0")) + a.quantity
        item_exposure[a.item_id] = item_exposure.get(a.item_id, 0.0) + a.exposure_cost
    for key, item in item_map.items():
        planned = item_qty.get(key, Decimal("0"))
        # a plan touches an item either not at all, or covers it fully -
        # splitting across destinations is fine, leaving part behind is not
        if planned > item.quantity:
            raise ValueError(f"item {key} over-covered")
        if planned > 0 and planned != item.quantity:
            raise ValueError(f"item {key} only partially covered")
        if require_all_items and planned == 0:
            raise ValueError(f"item {key} not exactly covered")
        if item_exposure.get(key, 0.0) > item.exposure_budget + 1e-9:
            raise ValueError(f"item {key} exposure budget exceeded")
        if item.redundant:
            used = [a.circuit for a in plan.assignments if a.item_id == key]
            if len(used) != len(set(used)):
                raise ValueError(f"redundant item {key} shares a circuit")
    for key, used in dest_qty.items():
        if used > dest_map[key].usable:
            raise ValueError(f"destination {key} overbooked past reserve")


def summarize(plan: Plan) -> dict[str, Decimal]:
    return plan.quantities_by_destination()


def explain(plan: Plan) -> list[str]:
    return [f"{a.item_id} -> {a.destination_id} ({a.circuit}): {a.quantity} "
            f"transit={a.transit} exposure={a.exposure_cost:g}"
            for a in plan.assignments]


# --------------------------------------------------------------- atomic commit


@dataclass(frozen=True)
class StockRow:
    item: RelocationItem
    available: Decimal
    version: int = 0


@dataclass(frozen=True)
class DestinationRow:
    destination: Destination
    version: int = 0


@dataclass(frozen=True)
class Lock:
    reference: str
    item_id: str
    destination_id: str
    quantity: Decimal
    circuit: str


@dataclass(frozen=True)
class PreparedRelocation:
    reference: str
    plan: Plan
    item_versions: dict[str, int]
    destination_versions: dict[str, int]
    prepared_at: datetime


class RelocationDesk:
    """Versioned, all-or-nothing lock desk for relocation plans.

    Inventory holds and destination-capacity holds are taken inside one
    critical section.  Commit failures (a destination went down, a delay
    changed transit/exposure terms, a version moved, capacity shrank)
    reject the whole plan; nothing is left half-locked.  A committed
    relocation is undone the same way via :meth:`release`.
    """

    def __init__(self, destinations: Iterable[Destination],
                 items: Iterable[RelocationItem] = (), events=None):
        self._dests: dict[str, DestinationRow] = {}
        for d in destinations:
            d.validate()
            self._dests[d.destination_id] = DestinationRow(d)
        self._items: dict[str, StockRow] = {}
        for item in items:
            item.validate()
            self._items[item.item_id] = StockRow(item, item.quantity)
        self._holds: dict[str, tuple[Lock, ...]] = {}
        self._open: set[str] = set()
        self._lock = RLock()
        self._events = events

    # ---- live-state administration (a destination can fail at any time) ----

    def disable_destination(self, destination_id: str) -> None:
        with self._lock:
            row = self._dests[destination_id]
            self._dests[destination_id] = DestinationRow(
                replace(row.destination, enabled=False), row.version + 1)

    def report_delay(self, destination_id: str, transit: timedelta,
                     exposure_cost: float) -> None:
        with self._lock:
            row = self._dests[destination_id]
            updated = replace(row.destination, transit=transit,
                              exposure_cost=exposure_cost)
            updated.validate()
            self._dests[destination_id] = DestinationRow(updated, row.version + 1)

    # ---- reads ----

    def versions(self) -> tuple[dict[str, int], dict[str, int]]:
        with self._lock:
            return ({k: r.version for k, r in self._items.items()},
                    {k: r.version for k, r in self._dests.items()})

    def held(self, destination_id: str) -> Decimal:
        with self._lock:
            return sum((l.quantity for locks in self._holds.values()
                        for l in locks if l.destination_id == destination_id),
                       Decimal("0"))

    def available(self, item_id: str) -> Decimal:
        with self._lock:
            return self._items[item_id].available

    def locks(self, reference: str) -> tuple[Lock, ...]:
        with self._lock:
            return self._holds[reference]

    # ---- two-phase commit ----

    def prepare(self, plan: Plan, reference: str,
                at: datetime | None = None) -> PreparedRelocation:
        if not reference:
            raise ValueError("reference required")
        if not plan.assignments:
            raise ValueError("empty plan cannot be committed")
        with self._lock:
            if reference in self._holds or reference in self._open:
                raise ValueError(f"reference {reference} already used")
            validate_plan(plan, (r.item for r in self._items.values()),
                          (r.destination for r in self._dests.values()),
                          require_all_items=False)
            item_versions = {k: self._items[k].version
                             for k in plan.quantities_by_item()}
            dest_versions = {k: self._dests[k].version
                             for k in plan.quantities_by_destination()}
            prepared = PreparedRelocation(
                reference, plan, item_versions, dest_versions,
                (at or datetime.now(timezone.utc)).astimezone(timezone.utc))
            self._open.add(reference)
        self._append(reference, 0, "relocation_prepared",
                     {"assignments": len(plan.assignments)})
        return prepared

    def commit(self, prepared: PreparedRelocation) -> tuple[Lock, ...]:
        ref = prepared.reference
        with self._lock:
            if ref in self._holds:  # idempotent re-commit
                return self._holds[ref]
            if ref not in self._open:
                raise ValueError(f"reference {ref} was not prepared")
            rejections = self._revalidate(prepared)
            if rejections:
                self._open.discard(ref)
                self._append(ref, 1, "relocation_aborted",
                             {"reasons": sorted({r.code for r in rejections})})
                raise RelocationCommitFailed(rejections)
            locks = self._apply(prepared)
            self._holds[ref] = locks
            self._open.discard(ref)
        self._append(ref, 1, "relocation_committed", {"locks": len(locks)})
        return locks

    def release(self, reference: str, reason: str = "") -> tuple[Lock, ...]:
        """Compensation: undo a committed relocation atomically."""
        with self._lock:
            if reference not in self._holds:
                raise KeyError(reference)
            locks = self._holds.pop(reference)
            for lock in locks:
                row = self._items[lock.item_id]
                self._items[lock.item_id] = StockRow(
                    row.item, row.available + lock.quantity, row.version + 1)
                drow = self._dests[lock.destination_id]
                self._dests[lock.destination_id] = DestinationRow(
                    drow.destination, drow.version + 1)
        self._append(reference, None, "relocation_released", {"reason": reason})
        return locks

    # ---- internals ----

    def _revalidate(self, prepared: PreparedRelocation) -> list[Rejection]:
        plan = prepared.plan
        rejections: list[Rejection] = []
        # Inventory version conflicts alone are decisive: stock may have
        # been reserved for someone else.  Destination version bumps are
        # recorded but never mask the concrete reason a live destination
        # now fails (disabled, delay, capacity shrink, ...).
        for item_id, version in prepared.item_versions.items():
            if self._items[item_id].version != version:
                rejections.append(Rejection(item_id, None, "stale_version",
                                            "inventory moved"))
        changed_destinations = {
            dest_id for dest_id, version in prepared.destination_versions.items()
            if self._dests[dest_id].version != version}
        if rejections:
            return rejections

        item_held: dict[str, Decimal] = {}
        dest_held: dict[str, Decimal] = {}
        for other_ref, locks in self._holds.items():
            if other_ref == prepared.reference:
                continue
            for l in locks:
                item_held[l.item_id] = item_held.get(l.item_id, Decimal("0")) + l.quantity
                dest_held[l.destination_id] = dest_held.get(l.destination_id, Decimal("0")) + l.quantity

        flagged: set[tuple[str, str, str]] = set()

        def flag(item_id: str, dest_id: str | None, code: str, detail: str) -> None:
            key = (item_id, dest_id or "", code)
            if key not in flagged:
                flagged.add(key)
                rejections.append(Rejection(item_id, dest_id, code, detail))

        exposure: dict[str, float] = {}
        for a in plan.assignments:
            item = self._items[a.item_id].item
            dest = self._dests[a.destination_id].destination
            exposure[a.item_id] = exposure.get(a.item_id, 0.0) + dest.exposure_cost
            if not dest.enabled:
                flag(a.item_id, dest.destination_id, "disabled",
                     "destination failed before commit")
            if item.zone not in dest.zones:
                flag(a.item_id, dest.destination_id, "zone",
                     "destination no longer serves zone")
            if a.circuit not in item.circuits_allowed or a.circuit not in dest.circuits:
                flag(a.item_id, dest.destination_id, "circuit",
                     "circuit no longer live or compatible")
            if dest.transit > item.max_transit:
                flag(a.item_id, dest.destination_id, "transit",
                     f"live transit {dest.transit} exceeds {item.max_transit}")
            if dest.exposure_cost > item.exposure_budget + 1e-9:
                flag(a.item_id, dest.destination_id, "exposure",
                     f"live cost {dest.exposure_cost:g} > "
                     f"budget {item.exposure_budget:g}")
        # cumulative exposure across all legs of one item
        for item_id, cost in exposure.items():
            budget = self._items[item_id].item.exposure_budget
            if cost > budget + 1e-9 and not any(
                    r.code == "exposure" and r.item_id == item_id
                    for r in rejections):
                flag(item_id, None, "exposure",
                     f"cumulative live cost {cost:g} > budget {budget:g}")
        for item_id, qty in plan.quantities_by_item().items():
            row = self._items[item_id]
            if qty > row.available - item_held.get(item_id, Decimal("0")):
                flag(item_id, None, "inventory", "insufficient stock")
        for dest_id, qty in plan.quantities_by_destination().items():
            dest = self._dests[dest_id].destination
            free = dest.usable - dest_held.get(dest_id, Decimal("0"))
            if qty > free:
                flag("*", dest_id, "capacity", f"need {qty}, free {free}")
            elif dest_id in changed_destinations and not any(
                    r.destination_id == dest_id for r in rejections):
                flag("*", dest_id, "stale_version", "destination terms moved")
        return rejections

    def _apply(self, prepared: PreparedRelocation) -> tuple[Lock, ...]:
        locks = tuple(Lock(prepared.reference, a.item_id, a.destination_id,
                           a.quantity, a.circuit)
                      for a in prepared.plan.assignments)
        for item_id, qty in prepared.plan.quantities_by_item().items():
            row = self._items[item_id]
            self._items[item_id] = StockRow(
                row.item, row.available - qty, row.version + 1)
        for dest_id in prepared.plan.quantities_by_destination():
            drow = self._dests[dest_id]
            self._dests[dest_id] = DestinationRow(drow.destination, drow.version + 1)
        return locks

    def _append(self, reference: str, expected: int | None, kind: str,
                payload: dict) -> None:
        if self._events is None:
            return
        stream = f"relocation:{reference}"
        if expected is None:
            expected = len(self._events.read(stream))
        self._events.append(stream, expected, kind, payload)


def execute(items: Iterable[RelocationItem],
            destinations: Iterable[Destination], reference: str,
            at: datetime | None = None, events=None):
    """Plan and atomically commit a relocation in one call.

    Returns ``(desk, locks)`` so the caller can later :meth:`release`.
    """
    its = list(items)
    dsts = list(destinations)
    desk = RelocationDesk(dsts, its, events=events)
    prepared = desk.prepare(plan_relocation(its, dsts), reference, at)
    locks = desk.commit(prepared)
    return desk, locks
