from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tracelm.context import generate_span_id, generate_trace_id
from tracelm.exporters.chrome_exporter import export_trace_to_chrome
from tracelm.span import Span
from tracelm.storage.sqlite_store import init_db, save_trace
from tracelm.trace import Trace


@dataclass
class TraceLMLogger:
    enabled: bool
    output_dir: Path

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._traces: dict[str, Any] = {}
        self._spans: dict[tuple[str, str], Any] = {}
        init_db()

    def start_trace(self, name: str, metadata: dict[str, Any] | None = None) -> tuple[str, str]:
        if not self.enabled:
            return ("", "")
        trace_id = generate_trace_id()
        root_span_id = generate_span_id()
        trace = Trace(trace_id=trace_id)
        root = Span(
            span_id=root_span_id,
            trace_id=trace_id,
            parent_id=None,
            name=name,
            metadata=metadata or {},
        )
        trace.add_span(root)
        self._traces[trace_id] = trace
        self._spans[(trace_id, root_span_id)] = root
        return trace_id, root_span_id

    def start_span(
        self,
        trace_id: str,
        parent_span_id: str,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if not self.enabled or not trace_id:
            return ""
        trace = self._traces.get(trace_id)
        if trace is None:
            return ""
        span_id = generate_span_id()
        span = Span(
            span_id=span_id,
            trace_id=trace_id,
            parent_id=parent_span_id or None,
            name=name,
            metadata=metadata or {},
        )
        trace.add_span(span)
        self._spans[(trace_id, span_id)] = span
        return span_id

    def finish_span(
        self,
        trace_id: str,
        span_id: str,
        metadata_update: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        if not self.enabled or not trace_id or not span_id:
            return
        span = self._spans.get((trace_id, span_id))
        if span is None:
            return
        if metadata_update:
            span.metadata.update(metadata_update)
        if error:
            span.error = error
        span.finish()

    def finalize_trace(self, trace_id: str, root_span_id: str, metadata_update: dict[str, Any] | None = None) -> None:
        if not self.enabled or not trace_id:
            return
        self.finish_span(trace_id, root_span_id, metadata_update=metadata_update)
        trace = self._traces.get(trace_id)
        if trace is None:
            return
        trace.validate()
        save_trace(trace)
        out_file = self.output_dir / f"trace_{trace_id}.json"
        export_trace_to_chrome(trace, str(out_file))

