from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable


class RelocationError(ValueError):
    """Base class for cold-store evacuation failures."""


class InfeasibleRelocation(RelocationError):
    def __init__(self, pallet_id: str, reasons: Iterable[str]):
        self.pallet_id = pallet_id
        self.reasons: tuple[str, ...] = tuple(reasons)
        super().__init__(f"pallet {pallet_id} cannot be relocated: {'; '.join(self.reasons)}")


@dataclass(frozen=True)
class Pallet:
    """A unit load in the failed cold store.

    zone            required temperature zone (温区), e.g. 'frozen' / 'chilled'
    footprint       capacity units the pallet occupies at a destination
    circuit         source power circuit (informational; the outage trigger)
    exposure_minutes cumulative out-of-zone allowance for the whole move (暴露额度)
    priority        higher value is evacuated first
    """

    pallet_id: str
    zone: str
    footprint: Decimal
    exposure_minutes: float
    circuit: str = ""
    priority: int = 0


@dataclass(frozen=True)
class Destination:
    """A receiving freezer.

    zones           temperature zones it can hold (温区)
    circuit         receiving power circuit (电路); must be live during outage
    capacity        physical volume
    reserved        volume already pre-held for others (目的地预留)
    transit_minutes travel time from the failed site (转运时间)
    """

    destination_id: str
    zones: frozenset[str]
    circuit: str
    capacity: Decimal
    reserved: Decimal = Decimal("0")
    transit_minutes: float = 0.0
    enabled: bool = True

    @property
    def usable(self) -> Decimal:
        """Free volume net of pre-existing reservations — never the raw capacity."""
        return max(Decimal("0"), self.capacity - self.reserved)


@dataclass(frozen=True)
class Assignment:
    pallet_id: str
    destination_id: str
    footprint: Decimal
    transit_minutes: float
    exposure_margin_minutes: float


@dataclass(frozen=True)
class RelocationPlan:
    assignments: tuple[Assignment, ...]

    @property
    def pallet_ids(self) -> tuple[str, ...]:
        return tuple(a.pallet_id for a in self.assignments)

    def totals(self) -> dict[str, Decimal]:
        out: dict[str, Decimal] = {}
        for a in self.assignments:
            out[a.destination_id] = out.get(a.destination_id, Decimal("0")) + a.footprint
        return out

    def plan_id(self) -> str:
        tokens = sorted(f"{a.pallet_id}>{a.destination_id}" for a in self.assignments)
        return "RX:" + "|".join(tokens)


def _reasons(pallet: Pallet, dest: Destination, dead_circuits: frozenset[str],
             remaining: dict[str, Decimal]) -> list[str]:
    """All five dimensions that raw free volume alone would hide."""
    out: list[str] = []
    if not dest.enabled:
        out.append("destination disabled")
    if dest.circuit in dead_circuits:
        out.append(f"destination circuit {dest.circuit} has no power")
    if pallet.zone not in dest.zones:
        out.append(f"zone {pallet.zone} not supported (offers {sorted(dest.zones)})")
    if dest.transit_minutes > pallet.exposure_minutes:
        out.append(
            f"transit {dest.transit_minutes:g} min exceeds exposure allowance "
            f"{pallet.exposure_minutes:g} min"
        )
    if pallet.footprint > remaining.get(dest.destination_id, Decimal("0")):
        out.append("insufficient free capacity after reservation")
    return out


def _run(pallets: Iterable[Pallet], destinations: Iterable[Destination],
         dead_circuits: frozenset[str], raise_on_fail: bool):
    pallets = list(pallets)
    dests = list(destinations)
    by_id = {d.destination_id: d for d in dests}
    remaining = {
        d.destination_id: d.usable
        for d in dests
        if d.enabled and d.circuit not in dead_circuits
    }
    assignments: list[Assignment] = []
    blocked: dict[str, tuple[str, ...]] = {}
    # Most fragile / most urgent first: priority desc, tight exposure, then id.
    for pallet in sorted(pallets, key=lambda p: (-p.priority, p.exposure_minutes, p.pallet_id)):
        candidates = []
        reasons: dict[str, list[str]] = {}
        for dest in dests:
            found = _reasons(pallet, dest, dead_circuits, remaining)
            if found:
                reasons[dest.destination_id] = found
            else:
                slack = remaining[dest.destination_id] - pallet.footprint
                candidates.append((dest.transit_minutes, slack, dest.destination_id))
        if not candidates:
            flat = tuple(
                f"{did}: {why}"
                for did in sorted(reasons)
                for why in reasons[did]
            )
            blocked[pallet.pallet_id] = flat
            if raise_on_fail:
                raise InfeasibleRelocation(pallet.pallet_id, flat)
            continue
        # Shortest transit first (spend the least exposure allowance), then tightest fit.
        _, _, chosen = min(candidates)
        dest = by_id[chosen]
        remaining[chosen] -= pallet.footprint
        assignments.append(
            Assignment(
                pallet.pallet_id,
                chosen,
                pallet.footprint,
                dest.transit_minutes,
                pallet.exposure_minutes - dest.transit_minutes,
            )
        )
    return RelocationPlan(tuple(assignments)), blocked


def plan_relocation(pallets: Iterable[Pallet], destinations: Iterable[Destination],
                    failed_circuits: Iterable[str] = ()) -> RelocationPlan:
    """Pure, executable evacuation plan. Raises InfeasibleRelocation if any pallet
    has no destination that simultaneously satisfies all five constraints."""
    plan, _ = _run(pallets, destinations, frozenset(failed_circuits), raise_on_fail=True)
    return plan


def evaluate(pallets: Iterable[Pallet], destinations: Iterable[Destination],
             failed_circuits: Iterable[str] = ()) -> dict[str, tuple[str, ...]]:
    """Non-raising feasibility report: pallet_id -> per-destination blocker reasons.
    An empty mapping means the evacuation is fully feasible."""
    _, blocked = _run(pallets, destinations, frozenset(failed_circuits), raise_on_fail=False)
    return blocked


def feasible(pallets: Iterable[Pallet], destinations: Iterable[Destination],
             failed_circuits: Iterable[str] = ()) -> bool:
    return not evaluate(pallets, destinations, failed_circuits)


def validate_plan(plan: RelocationPlan, pallets: Iterable[Pallet],
                  destinations: Iterable[Destination],
                  failed_circuits: Iterable[str] = ()) -> None:
    """Defence in depth before locking: a plan must still satisfy every dimension
    and stay within each destination's reserved capacity."""
    pallet_map = {p.pallet_id: p for p in pallets}
    dest_map = {d.destination_id: d for d in destinations}
    dead = frozenset(failed_circuits)
    used: dict[str, Decimal] = {}
    for a in plan.assignments:
        p = pallet_map.get(a.pallet_id)
        d = dest_map.get(a.destination_id)
        if p is None:
            raise ValueError(f"plan references unknown pallet {a.pallet_id}")
        if d is None:
            raise ValueError(f"plan references unknown destination {a.destination_id}")
        if a.footprint <= 0:
            raise ValueError("assignment footprint must be positive")
        if a.footprint != p.footprint:
            raise ValueError(f"assignment footprint for {a.pallet_id} drifts from pallet")
        if not d.enabled:
            raise ValueError(f"destination {d.destination_id} disabled")
        if d.circuit in dead:
            raise ValueError(f"destination {d.destination_id} is on a failed circuit")
        if p.zone not in d.zones:
            raise ValueError(f"destination {d.destination_id} cannot hold zone {p.zone}")
        if d.transit_minutes > p.exposure_minutes:
            raise ValueError(f"pallet {p.pallet_id} would blow its exposure allowance")
        used[d.destination_id] = used.get(d.destination_id, Decimal("0")) + a.footprint
    for dest_id, qty in used.items():
        if qty > dest_map[dest_id].usable:
            raise ValueError(f"destination {dest_id} over reserved capacity")


def explain(plan: RelocationPlan) -> list[str]:
    return [
        f"{a.pallet_id} -> {a.destination_id} "
        f"({a.footprint}, {a.transit_minutes:g} min transit, "
        f"{a.exposure_margin_minutes:g} min exposure margin)"
        for a in plan.assignments
    ]
