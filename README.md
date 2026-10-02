# ColdLedger Pharmaceutical Cold-Chain Operations Lab

Dependency-free Python reference implementation for temperature-sensitive pharmaceutical distribution. It models inventory expiry, chain-of-custody, telemetry correction, packaging qualification, dry-ice depletion, freezer allocation, customs holds, serialization, recalls, release decisions, vehicle compartment compatibility, returns, SLA calendars and cross-dock state.

## Cold-store power-outage evacuation (`warehouse.relocation`, `warehouse.relocation_commit`)

Evacuation is never decided on free volume alone. `plan_relocation(pallets, destinations, failed_circuits)`
produces an executable plan only when every pallet simultaneously satisfies **five** dimensions:

1. temperature zone (destination supports the pallet's zone),
2. power circuit (the receiving circuit is not in the failed set / destination enabled),
3. transit time vs. the pallet's exposure allowance,
4. destination capacity net of pre-existing reservations (`capacity - reserved`, never raw capacity),
5. residual free volume after placement.

Pallets are ordered by priority (urgent/tight-exposure first) and routed to the shortest-feasible
destination; infeasible pallets raise `InfeasibleRelocation` with per-destination reasons
(`evaluate()` / `relocation_blockers()` return a non-raising report).

Locking happens in a separate, second step via `RelocationCoordinator.commit(plan)`: source
inventory holds and destination capacity reservations are prepared together with optimistic
version checks. If any participant fails mid-commit — including a destination whose circuit drops
halfway (`on_prepare` hook) — every earlier preparation is compensated in reverse, capacity and
inventory return to their exact prior versions, and no transfer record is written. The evacuation
is all-or-nothing; a failed commit can be re-planned and retried cleanly.

Runtime third-party dependencies: none.
