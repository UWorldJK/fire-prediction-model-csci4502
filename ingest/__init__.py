"""Bronze ingestion: download raw sources into the lake and integrate them across years."""
from __future__ import annotations

from dataclasses import dataclass, field

from lakehouse.manifest import ManifestRecord


@dataclass
class IngestResult:
    records: list[ManifestRecord] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)

    def merge(self, other: "IngestResult") -> "IngestResult":
        return IngestResult(
            self.records + other.records, self.skipped + other.skipped, self.failed + other.failed
        )
