from datetime import timedelta
from .inventory.fefo import reserve
from .quality.release import may_release
from .transport.connections import choose
from .warehouse.relocation import (
    evaluate as _evaluate_relocation,
    feasible as _relocation_feasible,
    plan_relocation as _plan_relocation,
    validate_plan as _validate_relocation,
    explain as _explain_relocation,
)
from .warehouse.relocation_commit import RelocationCoordinator


class ColdLedgerService:
    def reserve_inventory(self,*args,**kwargs): return reserve(*args,**kwargs)
    def release_decision(self,*args,**kwargs): return may_release(*args,**kwargs)
    def next_connection(self,inbound,candidates,minutes): return choose(inbound,candidates,timedelta(minutes=minutes))

    # --- Cold-store power-outage evacuation ----------------------------------
    def plan_relocation(self,pallets,destinations,failed_circuits=()):
        return _plan_relocation(pallets,destinations,failed_circuits)

    def relocation_feasible(self,pallets,destinations,failed_circuits=()):
        return _relocation_feasible(pallets,destinations,failed_circuits)

    def relocation_blockers(self,pallets,destinations,failed_circuits=()):
        return _evaluate_relocation(pallets,destinations,failed_circuits)

    def validate_relocation(self,plan,pallets,destinations,failed_circuits=()):
        return _validate_relocation(plan,pallets,destinations,failed_circuits)

    def relocation_coordinator(self,pallets,destinations,failed_circuits=()):
        return RelocationCoordinator(pallets,destinations,failed_circuits)

    def commit_relocation(self,coordinator,plan,on_prepare=None):
        return coordinator.commit(plan,on_prepare=on_prepare)

    def explain_relocation(self,plan):
        return _explain_relocation(plan)

    def health(self): return {'service':'coldledger','status':'ok','runtime_dependencies':0}
