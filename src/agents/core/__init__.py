"""Framework primitives shared by every agent loop.

These modules are *not* agent loops — they are the base class, the state
dataclasses, and the control-flow tracing that the
``generation``/``search``/``tool`` loop packages are built from.
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
from .state import Plan as Plan
from .state import PlanStep as PlanStep
from .state import RetrievedDocument as RetrievedDocument
from .state import RouteDecision as RouteDecision
from .state import TaskNode as TaskNode
from .state import TaskStatus as TaskStatus
from .state import TaskType as TaskType
from .state import ToolCall as ToolCall
from .state import ToolExecutionResult as ToolExecutionResult
from .state import ToolResult as ToolResult
from .state import ToolType as ToolType
from .state import UserRequest as UserRequest
