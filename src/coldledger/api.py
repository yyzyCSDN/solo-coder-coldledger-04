from datetime import timedelta
from .inventory.fefo import reserve
from .quality.release import may_release
from .transport.connections import choose
from .warehouse.relocation import (
    plan_relocation, RelocationDesk, RelocationInfeasible,
    RelocationCommitFailed, execute as execute_relocation,
)
class ColdLedgerService:
    def reserve_inventory(self,*args,**kwargs): return reserve(*args,**kwargs)
    def release_decision(self,*args,**kwargs): return may_release(*args,**kwargs)
    def next_connection(self,inbound,candidates,minutes): return choose(inbound,candidates,timedelta(minutes=minutes))
    def relocation_plan(self,items,destinations): return plan_relocation(items,destinations)
    def relocation_execute(self,items,destinations,reference,at=None,events=None):
        return execute_relocation(items,destinations,reference,at,events)
    def health(self): return {'service':'coldledger','status':'ok','runtime_dependencies':0}
