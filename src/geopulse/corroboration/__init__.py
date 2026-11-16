"""Independent source lineage and corroboration package."""

from geopulse.corroboration.lineage import (
    WIRE_SERVICE_CANONICAL,
    build_cluster_lineages,
    extract_article_lineage,
)
from geopulse.corroboration.models import (
    ArticleLineage,
    ClaimCorroboration,
    CorroborationStatus,
    EventCorroboration,
    LineageClassification,
)
from geopulse.corroboration.service import (
    corroborate_claim,
    corroborate_event,
)

__all__ = [
    "WIRE_SERVICE_CANONICAL",
    "ArticleLineage",
    "ClaimCorroboration",
    "CorroborationStatus",
    "EventCorroboration",
    "LineageClassification",
    "build_cluster_lineages",
    "corroborate_claim",
    "corroborate_event",
    "extract_article_lineage",
]
