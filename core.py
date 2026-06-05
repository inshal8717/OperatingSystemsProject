"""
core.py — Virtual Memory Simulation Engine: Domain Entities & Interfaces
=========================================================================
Defines the canonical hardware-level data structures and the abstract base
class that every page-replacement algorithm must implement.

Architecture mirrors a real MMU: each physical frame slot carries all flags
the hardware would maintain — reference bit, dirty bit, access timestamp,
frequency counter, and a bounded history of recent reference timestamps.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional


# ---------------------------------------------------------------------------
# Hardware-level page frame descriptor
# ---------------------------------------------------------------------------

@dataclass
class VirtualPageFrame:
    """
    Represents one physical memory frame slot in the simulated MMU.

    Attributes
    ----------
    page_id : Optional[int]
        The virtual page number currently occupying this frame.
        None means the frame is empty (free).
    reference_bit : int
        R-bit as set by hardware on every memory access (0 or 1).
        Cleared periodically by the paging daemon (Second-Chance / WSClock).
    dirty_bit : int
        M-bit (modified / dirty flag). Set to 1 on any write access.
        Signals that the page must be written back to disk before eviction.
    last_accessed : int
        Logical clock tick at which this page was most recently referenced.
        Used by LRU, WSClock age computations, etc.
    frequency : int
        Cumulative reference counter.  Incremented on every access.
        Used by MFU and serves as a tiebreaker in other algorithms.
    history : Deque[int]
        Bounded queue of the most recent K reference timestamps in FIFO order
        (oldest first).  Used by LRU-K to compute backward k-distance.
        Capacity is set by the algorithm at initialisation time.
    load_order : int
        Monotonically increasing counter recording when this page was first
        loaded into this frame.  Used by FIFO as its sort key and by MFU as
        a tiebreaker.
    """

    page_id: Optional[int] = None
    reference_bit: int = 0
    dirty_bit: int = 0
    last_accessed: int = 0
    frequency: int = 0
    history: Deque[int] = field(default_factory=deque)
    load_order: int = 0

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def is_empty(self) -> bool:
        """Return True when no page occupies this frame."""
        return self.page_id is None

    def reset(self) -> None:
        """
        Evict the current resident and reset all hardware flags to the
        power-on default state, preserving only the frame's position in the
        circular array (managed externally).
        """
        self.page_id = None
        self.reference_bit = 0
        self.dirty_bit = 0
        self.last_accessed = 0
        self.frequency = 0
        self.history.clear()
        self.load_order = 0

    def load(
        self,
        page_id: int,
        clock_tick: int,
        load_order: int,
        is_write: bool,
        history_capacity: int = 0,
    ) -> None:
        """
        Bring a new page into this frame, initialising all state correctly.

        Parameters
        ----------
        page_id : int
            Virtual page number being loaded.
        clock_tick : int
            Current logical clock value at the moment of the page fault.
        load_order : int
            Global monotonic load counter (for FIFO ordering).
        is_write : bool
            True if the faulting access is a store/write operation.
        history_capacity : int
            Maximum number of historical timestamps the history deque will
            retain.  0 means unbounded (not used by most algorithms).
        """
        self.page_id = page_id
        self.reference_bit = 1
        self.dirty_bit = 1 if is_write else 0
        self.last_accessed = clock_tick
        self.frequency = 1
        self.load_order = load_order
        self.history = deque(maxlen=history_capacity if history_capacity > 0 else None)
        self.history.append(clock_tick)

    def touch(self, clock_tick: int, is_write: bool) -> None:
        """
        Record a hit on a page already resident in this frame.

        Updates all relevant hardware flags without evicting the page.

        Parameters
        ----------
        clock_tick : int
            Current logical clock value at the moment of the access.
        is_write : bool
            True if the access is a write; sets or preserves the dirty bit.
        """
        self.reference_bit = 1
        if is_write:
            self.dirty_bit = 1
        self.last_accessed = clock_tick
        self.frequency += 1
        self.history.append(clock_tick)

    def needs_writeback(self) -> bool:
        """
        Return True if evicting this page requires a disk write-back.

        In a real OS this triggers an I/O operation to flush the modified
        page to the backing store (swap partition or mapped file).
        """
        return self.dirty_bit == 1

    def __repr__(self) -> str:
        status = f"P{self.page_id}" if self.page_id is not None else "---"
        return (
            f"VirtualPageFrame("
            f"page={status}, R={self.reference_bit}, M={self.dirty_bit}, "
            f"t={self.last_accessed}, freq={self.frequency})"
        )


# ---------------------------------------------------------------------------
# Step-level trace record
# ---------------------------------------------------------------------------

@dataclass
class SimulationStep:
    """
    Immutable snapshot of the memory state after processing one reference.

    Collected into a list by every algorithm to power the trace-log viewer
    and the per-step frame-state table in the Streamlit dashboard.

    Attributes
    ----------
    step : int
        1-based index of this reference in the input sequence.
    page_id : int
        The virtual page number referenced at this step.
    is_fault : bool
        True → page fault occurred (page was not resident).
    evicted_page : Optional[int]
        The page evicted to make room, or None if a free frame was used.
    dirty_eviction : bool
        True when the evicted page had its dirty bit set (requires I/O).
    frame_snapshot : List[Optional[int]]
        Ordered list of page IDs resident after this step completes.
        None entries represent empty frame slots.
    algorithm_name : str
        Human-readable label of the owning algorithm.
    """

    step: int
    page_id: int
    is_fault: bool
    evicted_page: Optional[int]
    dirty_eviction: bool
    frame_snapshot: List[Optional[int]]
    algorithm_name: str


# ---------------------------------------------------------------------------
# Abstract base class — every replacement algorithm implements this contract
# ---------------------------------------------------------------------------

class BasePageReplacementAlgorithm(ABC):
    """
    Abstract base class for all page-replacement algorithms.

    Enforces a uniform interface so the Streamlit layer can drive any
    algorithm identically without knowing its internal mechanics.

    Subclasses must:
      • Call ``super().__init__(capacity)`` to initialise shared state.
      • Implement ``access_page`` with the documented semantics.
      • Implement ``get_frame_state`` to expose current frame contents.

    Shared State (managed by this base class)
    ------------------------------------------
    frames : List[VirtualPageFrame]
        Physical frame array.  Length == capacity.
    capacity : int
        Number of physical frames allocated to this process.
    clock : int
        Logical clock tick.  Incremented once per ``access_page`` call.
    page_faults : int
        Running total of page faults observed during the simulation.
    page_hits : int
        Running total of page hits (accesses that found the page resident).
    load_counter : int
        Monotonically increasing counter used to establish FIFO load order.
    trace : List[SimulationStep]
        Ordered list of per-step snapshots for the trace-log viewer.
    """

    # Human-readable label displayed in the dashboard.  Subclasses override.
    name: str = "BaseAlgorithm"

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"Frame capacity must be ≥ 1, got {capacity!r}.")
        self.capacity: int = capacity
        self.frames: List[VirtualPageFrame] = [VirtualPageFrame() for _ in range(capacity)]
        self.clock: int = 0
        self.page_faults: int = 0
        self.page_hits: int = 0
        self.load_counter: int = 0
        self.trace: List[SimulationStep] = []
        # Running sum of occupied frame slots sampled after every reference.
        # Divided by (total_steps * capacity) to yield U_mem.
        self._occupied_slot_sum: int = 0

    # ------------------------------------------------------------------
    # Abstract interface — must be implemented by every subclass
    # ------------------------------------------------------------------

    @abstractmethod
    def access_page(self, page_id: int, is_write: bool = False) -> bool:
        """
        Process a single memory reference.

        Parameters
        ----------
        page_id : int
            The virtual page number being accessed.
        is_write : bool
            True if the operation is a store/write.  Affects the dirty bit.

        Returns
        -------
        bool
            True  → page fault (the page was not resident and had to be loaded).
            False → page hit  (the page was already resident in a frame).
        """

    @abstractmethod
    def get_frame_state(self) -> List[Optional[int]]:
        """
        Return an ordered snapshot of which page IDs currently reside in
        each physical frame slot.  Empty (free) slots are represented as None.
        """

    # ------------------------------------------------------------------
    # Shared helpers available to all subclasses
    # ------------------------------------------------------------------

    def _find_resident(self, page_id: int) -> Optional[int]:
        """
        Linear scan to locate a page in the frame array.

        Returns the frame index if found, else None.
        """
        for idx, frame in enumerate(self.frames):
            if frame.page_id == page_id:
                return idx
        return None

    def _find_free_frame(self) -> Optional[int]:
        """
        Return the index of the first empty frame slot, or None if full.
        """
        for idx, frame in enumerate(self.frames):
            if frame.is_empty():
                return idx
        return None

    def _record_step(
        self,
        page_id: int,
        is_fault: bool,
        evicted_page: Optional[int] = None,
        dirty_eviction: bool = False,
    ) -> None:
        """
        Append an immutable snapshot to the trace log after each access
        and accumulate the occupied-slot count for U_mem computation.
        """
        snapshot = self.get_frame_state()
        # Count non-None slots in the current snapshot and add to the
        # running sum used later by mem_utilization().
        self._occupied_slot_sum += sum(1 for pid in snapshot if pid is not None)
        step = SimulationStep(
            step=self.clock,
            page_id=page_id,
            is_fault=is_fault,
            evicted_page=evicted_page,
            dirty_eviction=dirty_eviction,
            frame_snapshot=snapshot,
            algorithm_name=self.name,
        )
        self.trace.append(step)

    def _next_load_order(self) -> int:
        """Increment and return the monotonic load counter."""
        self.load_counter += 1
        return self.load_counter

    def fault_rate(self) -> float:
        """Return the page-fault rate as a percentage [0.0, 100.0]."""
        total = self.page_faults + self.page_hits
        if total == 0:
            return 0.0
        return (self.page_faults / total) * 100.0

    def mem_utilization(self) -> float:
        """
        Compute Memory Utilization Efficiency (U_mem) as a percentage.

        Formula
        -------
        U_mem = (Total Occupied Frame Slots Across All Steps)
                / (Total Reference Steps * Capacity) * 100.0

        Interpretation
        --------------
        A value of 100 % means every frame was occupied on every reference
        step — the allocator made full use of physical RAM throughout
        execution.  Lower values indicate that frames sat empty for a
        significant portion of the simulation (common during warm-up or
        when the working set is smaller than the frame capacity).
        """
        total_steps = self.page_faults + self.page_hits   # == self.clock
        denominator = total_steps * self.capacity
        if denominator == 0:
            return 0.0
        return (self._occupied_slot_sum / denominator) * 100.0

    def summary(self) -> dict:
        """
        Return a dictionary of high-level performance metrics suitable for
        constructing the comparative DataFrame in the dashboard.
        """
        return {
            "Strategy": self.name,
            "Total Hits": self.page_hits,
            "Total Faults": self.page_faults,
            "Page Fault Rate (%)": round(self.fault_rate(), 2),
            "Memory Utilization (%)": round(self.mem_utilization(), 2),
        }

    def reset(self) -> None:
        """
        Restore the algorithm to its initial state so the same instance can
        be re-run with a different reference string without re-instantiation.
        """
        for frame in self.frames:
            frame.reset()
        self.clock = 0
        self.page_faults = 0
        self.page_hits = 0
        self.load_counter = 0
        self.trace.clear()
        self._occupied_slot_sum = 0