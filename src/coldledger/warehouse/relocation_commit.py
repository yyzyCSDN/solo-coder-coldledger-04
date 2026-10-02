from __future__ import annotations
from dataclasses import dataclass, replace
from decimal import Decimal
from threading import RLock
from typing import Callable, Iterable

from .relocation import (
    Assignment,
    RelocationPlan,
    validate_plan,
)


class CommitError(RuntimeError):
    """Raised when an evacuation commit cannot complete atomically."""


class DestinationUnavailable(CommitError):
    """A destination fails (circuit drops / goes disabled) mid-commit."""

    def __init__(self, destination_id: str, reason: str = "unavailable"):
        self.destination_id = destination_id
        self.reason = reason
        super().__init__(f"destination {destination_id} {reason}")


class InventoryConflict(CommitError):
    def __init__(self, pallet_id: str):
        self.pallet_id = pallet_id
        super().__init__(f"optimistic version conflict on pallet {pallet_id}")


@dataclass(frozen=True)
class StockRow:
    pallet_id: str
    available: Decimal
    version: int = 0


class InventoryBook:
    """Atomic source inventory. A pallet is an atomic unit: held once, released once."""

    def __init__(self, rows: Iterable[StockRow]):
        self.rows = {r.pallet_id: r for r in rows}
        self.lock = RLock()

    def snapshot(self, ids: Iterable[str]) -> dict[str, StockRow]:
        with self.lock:
            return {i: self.rows[i] for i in ids}

    def versions(self, ids: Iterable[str]) -> dict[str, int]:
        return {k: v.version for k, v in self.snapshot(ids).items()}

    def hold(self, pallet_id: str, version: int) -> StockRow:
        with self.lock:
            r = self.rows[pallet_id]
            if r.version != version:
                raise InventoryConflict(pallet_id)
            if r.available <= 0:
                raise CommitError(f"pallet {pallet_id} has no free stock")
            nxt = replace(r, available=Decimal("0"), version=version + 1)
            self.rows[pallet_id] = nxt
            return nxt

    def release(self, pallet_id: str, footprint: Decimal, version: int) -> StockRow:
        with self.lock:
            r = self.rows[pallet_id]
            if r.version != version:
                raise InventoryConflict(pallet_id)
            nxt = replace(r, available=r.available + footprint, version=version + 1)
            self.rows[pallet_id] = nxt
            return nxt

    def restore(self, pallet_id: str, footprint: Decimal, version: int) -> None:
        """Compensation path: put the pallet back *and* return to the prior version,
        so an aborted commit leaves no revision trace."""
        with self.lock:
            r = self.rows[pallet_id]
            if r.version != version:
                raise InventoryConflict(pallet_id)
            self.rows[pallet_id] = replace(r, available=r.available + footprint, version=version - 1)


@dataclass(frozen=True)
class CapacityRow:
    destination_id: str
    available: Decimal
    version: int = 0
    enabled: bool = True


class CapacityBook:
    """Atomic receiving capacity, net of reservations, with live circuit status."""

    def __init__(self, rows: Iterable[CapacityRow]):
        self.rows = {r.destination_id: r for r in rows}
        self.lock = RLock()

    def snapshot(self, ids: Iterable[str]) -> dict[str, CapacityRow]:
        with self.lock:
            return {i: self.rows[i] for i in ids}

    def versions(self, ids: Iterable[str]) -> dict[str, int]:
        return {k: v.version for k, v in self.snapshot(ids).items()}

    def disable(self, destination_id: str, reason: str = "circuit lost") -> None:
        with self.lock:
            r = self.rows[destination_id]
            self.rows[destination_id] = replace(r, enabled=False)

    def reserve(self, destination_id: str, qty: Decimal, version: int) -> CapacityRow:
        with self.lock:
            r = self.rows[destination_id]
            if r.version != version:
                raise CommitError(f"optimistic version conflict on destination {destination_id}")
            if not r.enabled:
                raise DestinationUnavailable(destination_id)
            if qty <= 0:
                raise CommitError("reserve quantity must be positive")
            if r.available < qty:
                raise CommitError(f"destination {destination_id} over capacity")
            nxt = replace(r, available=r.available - qty, version=version + 1)
            self.rows[destination_id] = nxt
            return nxt

    def restore(self, destination_id: str, qty: Decimal, version: int) -> None:
        """Compensation path: give capacity back and roll the revision back too."""
        with self.lock:
            r = self.rows[destination_id]
            if r.version != version:
                raise CommitError(f"optimistic version conflict on destination {destination_id}")
            self.rows[destination_id] = replace(r, available=r.available + qty, version=version - 1)


@dataclass(frozen=True)
class TransferRecord:
    plan_id: str
    pallet_id: str
    destination_id: str
    footprint: Decimal
    sequence: int


@dataclass(frozen=True)
class CommitReceipt:
    plan_id: str
    transfers: tuple[TransferRecord, ...]
    inventory_versions: dict[str, int]
    capacity_versions: dict[str, int]


# A participant that can fail at commit time, e.g. a circuit dropping mid-move.
FailureHook = Callable[[Assignment], None]


class RelocationCoordinator:
    """Plans first, then locks inventory + capacity in one atomic saga.

    Each assignment is prepared (source pallet held, destination capacity reserved).
    If any participant — including a destination that dies halfway through — fails,
    every earlier preparation is compensated. No half transfer can survive.
    """

    def __init__(self, pallets, destinations, failed_circuits=()):
        self.pallets = list(pallets)
        self.destinations = list(destinations)
        self.failed_circuits = frozenset(failed_circuits)
        self.inventory = InventoryBook(
            StockRow(p.pallet_id, p.footprint) for p in self.pallets
        )
        self.capacity = CapacityBook(
            CapacityRow(d.destination_id, d.usable, enabled=d.enabled
                        and d.circuit not in self.failed_circuits)
            for d in self.destinations
        )
        self.ledger: list[TransferRecord] = []
        self._commit_lock = RLock()

    # ---- saga steps -------------------------------------------------------
    def _hold(self, a: Assignment, inv_versions: dict[str, int],
              cap_versions: dict[str, int]) -> tuple[int, int]:
        # Hold source stock first. If this step itself fails there is nothing to undo;
        # it is registered before the capacity call so a later failure still compensates it.
        inv = self.inventory.hold(a.pallet_id, inv_versions.get(a.pallet_id, self.inventory.rows[a.pallet_id].version))
        inv_versions[a.pallet_id] = inv.version
        try:
            cap = self.capacity.reserve(
                a.destination_id, a.footprint,
                cap_versions.get(a.destination_id, self.capacity.rows[a.destination_id].version),
            )
        except BaseException:
            # Undo the just-taken inventory so a half step never leaks out.
            self.inventory.restore(a.pallet_id, a.footprint, inv.version)
            inv_versions[a.pallet_id] = inv.version - 1
            raise
        cap_versions[a.destination_id] = cap.version
        return inv.version, cap.version

    def _compensate(self, prepared: list[tuple[Assignment, int, int]]) -> None:
        for a, inv_version, cap_version in reversed(prepared):
            # Reverse the version bumps made by prepare, restoring pre-commit state.
            self.capacity.restore(a.destination_id, a.footprint, cap_version)
            self.inventory.restore(a.pallet_id, a.footprint, inv_version)

    # ---- public API -------------------------------------------------------
    def commit(self, plan: RelocationPlan, on_prepare: FailureHook | None = None,
               expected_inventory: dict[str, int] | None = None,
               expected_capacity: dict[str, int] | None = None) -> CommitReceipt:
        with self._commit_lock:
            # Re-validate against the *live* world: a destination may have changed
            # between planning and locking.
            validate_plan(plan, self.pallets, self.destinations, self.failed_circuits)
            used = {a.destination_id for a in plan.assignments}
            for d in self.destinations:
                if d.destination_id in used and not self.capacity.rows[d.destination_id].enabled:
                    raise DestinationUnavailable(d.destination_id, "already down")

            # Optimistic-concurrency baselines captured at lock time.
            inv_versions = dict(expected_inventory or {})
            cap_versions = dict(expected_capacity or {})
            prepared: list[tuple[Assignment, int, int]] = []
            try:
                for a in plan.assignments:
                    if on_prepare is not None:
                        on_prepare(a)
                    inv_v, cap_v = self._hold(a, inv_versions, cap_versions)
                    prepared.append((a, inv_v, cap_v))
            except BaseException:
                # Any failure — circuit drop, conflict, capacity race — must undo
                # every preparation already made, then propagate. No leftovers.
                self._compensate(prepared)
                raise

            base = len(self.ledger)
            for seq, a in enumerate(plan.assignments, start=base + 1):
                self.ledger.append(
                    TransferRecord(plan.plan_id(), a.pallet_id, a.destination_id, a.footprint, seq)
                )
            return CommitReceipt(
                plan.plan_id(),
                tuple(self.ledger[base:]),
                dict(sorted(inv_versions.items())),
                dict(sorted(cap_versions.items())),
            )

    def records(self, plan_id: str | None = None) -> tuple[TransferRecord, ...]:
        rows = self.ledger if plan_id is None else [r for r in self.ledger if r.plan_id == plan_id]
        return tuple(rows)
