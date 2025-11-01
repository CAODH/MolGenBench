
## hit rediscovery metric for molecular generation
from typing import Dict, Type, Any, List, Optional
from molgenbench.metrics.base import MetricBase
from molgenbench.types import MoleculeRecord

class hitRediscoveryMetric(MetricBase):
    name = "HitRediscovery"

    def compute_one(self, record: MoleculeRecord, ref_record: Optional[MoleculeRecord] = None) -> Dict[str, Any]:
        # 计算命中重发现指标的逻辑
        
        return {"hit_rediscovery_score": 0.8}  # 示例返回值
