"""Framework primitives shared by every agent loop.

These modules are *not* agent loops — they are the base class, state
dataclasses, graph scaffolding, and control-flow tracing that the
``generation``/``search``/``tool`` loop packages are built from.

``graph_base`` is intentionally left as a submodule (import it via
``src.agents.core.graph_base``): its ``AgentState`` TypedDict collides with
``state.AgentState``, so re-exporting both here would be ambiguous.
"""

from .base import AgentLoopBase as AgentLoopBase
from .base import AgentLoopConfig as AgentLoopConfig
from .base import AgentLoopOutput as AgentLoopOutput
from .base import OnTurnCallback as OnTurnCallback
from .base import RolloutStep as RolloutStep
from .base import register as register
from .base import simple_timer as simple_timer
from .control_flow_trace import ControlFlowEvent as ControlFlowEvent
from .control_flow_trace import ControlFlowRecorder as ControlFlowRecorder
from .control_flow_trace import EventSink as EventSink
from .state import AgentState as AgentState
from .state import PerformanceMetrics as PerformanceMetrics
from .state import TaskStatus as TaskStatus
from .state import ToolExecutionResult as ToolExecutionResult
from .state import UserRequest as UserRequest
