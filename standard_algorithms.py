"""
standard_algorithms.py — Classic Textbook Page-Replacement Algorithms
======================================================================
Implements the three canonical textbook algorithms taught in every OS course:

  1. FIFO   — First-In, First-Out
  2. LRU    — Least Recently Used
  3. Clock  — Second-Chance (approximation of LRU with a rotating clock hand)

Each class inherits from ``BasePageReplacementAlgorithm`` in ``core.py`` and
satisfies the full abstract interface: ``access_page`` and ``get_frame_state``.

Design notes
------------
* All state lives inside the inherited ``self.frames`` list of
  ``VirtualPageFrame`` objects; no duplicate bookkeeping structures are
  maintained unnecessarily.
* The trace log (``self.trace``) is populated via the inherited
  ``_record_step`` helper so the Streamlit dashboard receives identical
  data shapes from all algorithms.
* Belady's Anomaly commentary is embedded in the FIFO docstring because FIFO
  is the only algorithm among the three that can exhibit the anomaly.
"""

from __future__ import annotations

from typing import List, Optional

from core import BasePageReplacementAlgorithm, VirtualPageFrame


# ---------------------------------------------------------------------------
# 1. FIFO — First-In, First-Out
# ---------------------------------------------------------------------------

class FIFOAlgorithm(BasePageReplacementAlgorithm):
    """
    First-In, First-Out page replacement.

    Eviction policy
    ---------------
    The page that has been resident in memory for the longest continuous span
    of time is the victim.  Implemented by comparing ``frame.load_order``
    (a monotonically increasing counter stamped at load time).

    Belady's Anomaly
    ----------------
    FIFO is one of the few replacement policies that can exhibit Belady's
    Anomaly: **adding more physical frames can sometimes *increase* the total
    number of page faults** for certain reference strings.

    Classic example (from Belady 1969) with the string
        1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5:
      • 3 frames → 9 faults
      • 4 frames → 10 faults   ← anomaly!

    The trace log produced by this class lets users step through the execution
    and visually verify whether the anomaly occurs for their chosen frame count
    and reference string.
    """

    name: str = "FIFO"

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def access_page(self, page_id: int, is_write: bool = False) -> bool:
        """
        Process one memory reference under the FIFO replacement policy.

        Algorithm
        ---------
        1. Increment the logical clock.
        2. Search all frames for the requested page.
           a. Hit  → update R-bit and dirty bit; record a hit.
           b. Miss → find a free frame or evict the oldest loaded page;
                     load the new page; record a fault.
        3. Append a trace snapshot and return the fault flag.

        Parameters
        ----------
        page_id : int
            Virtual page number being referenced.
        is_write : bool
            True if the access is a write operation.

        Returns
        -------
        bool
            True on page fault, False on page hit.
        """
        self.clock += 1

        # --- Hit path ---------------------------------------------------
        resident_idx = self._find_resident(page_id)
        if resident_idx is not None:
            # Update hardware flags even on a hit.
            self.frames[resident_idx].touch(self.clock, is_write)
            self.page_hits += 1
            self._record_step(page_id, is_fault=False)
            return False

        # --- Fault path -------------------------------------------------
        self.page_faults += 1
        evicted_page: Optional[int] = None
        dirty_eviction: bool = False

        free_idx = self._find_free_frame()
        if free_idx is not None:
            # A free slot exists — no eviction needed.
            target_idx = free_idx
        else:
            # All frames occupied — evict the page with the smallest
            # load_order (the one that has been in memory the longest).
            target_idx = min(
                range(self.capacity),
                key=lambda i: self.frames[i].load_order,
            )
            evicted_page = self.frames[target_idx].page_id
            dirty_eviction = self.frames[target_idx].needs_writeback()
            self.frames[target_idx].reset()

        # Load the new page into the chosen frame slot.
        self.frames[target_idx].load(
            page_id=page_id,
            clock_tick=self.clock,
            load_order=self._next_load_order(),
            is_write=is_write,
        )

        self._record_step(
            page_id,
            is_fault=True,
            evicted_page=evicted_page,
            dirty_eviction=dirty_eviction,
        )
        return True

    def get_frame_state(self) -> List[Optional[int]]:
        """Return the page ID currently occupying each frame (None if empty)."""
        return [frame.page_id for frame in self.frames]


# ---------------------------------------------------------------------------
# 2. LRU — Least Recently Used
# ---------------------------------------------------------------------------

class LRUAlgorithm(BasePageReplacementAlgorithm):
    """
    Least-Recently-Used page replacement.

    Eviction policy
    ---------------
    The page whose most recent reference is furthest in the past is chosen as
    the eviction victim.  Implemented by comparing ``frame.last_accessed``
    timestamps (logical clock ticks).

    Optimality property
    -------------------
    LRU is provably free of Belady's Anomaly — it belongs to the class of
    *stack algorithms* (Mattson et al.), meaning that the set of pages held
    in memory with N frames is always a superset of those held with N-1 frames
    for every prefix of the reference string.  More frames can only help.

    Hardware cost
    -------------
    Exact LRU requires either a hardware counter updated on every memory
    reference (expensive) or an OS-maintained ordered structure.  This
    simulation uses the timestamp approach for pedagogical clarity.
    """

    name: str = "LRU"

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def access_page(self, page_id: int, is_write: bool = False) -> bool:
        """
        Process one memory reference under the LRU replacement policy.

        Algorithm
        ---------
        1. Increment the logical clock.
        2. Scan frames for the requested page.
           a. Hit  → update timestamp (``last_accessed``) and dirty bit.
           b. Miss → find a free frame or evict the frame whose
                     ``last_accessed`` is minimum (oldest reference);
                     load the new page.
        3. Append trace snapshot and return fault flag.

        Parameters
        ----------
        page_id : int
            Virtual page number being referenced.
        is_write : bool
            True if the access is a write operation.

        Returns
        -------
        bool
            True on page fault, False on page hit.
        """
        self.clock += 1

        # --- Hit path ---------------------------------------------------
        resident_idx = self._find_resident(page_id)
        if resident_idx is not None:
            self.frames[resident_idx].touch(self.clock, is_write)
            self.page_hits += 1
            self._record_step(page_id, is_fault=False)
            return False

        # --- Fault path -------------------------------------------------
        self.page_faults += 1
        evicted_page: Optional[int] = None
        dirty_eviction: bool = False

        free_idx = self._find_free_frame()
        if free_idx is not None:
            target_idx = free_idx
        else:
            # Evict the page with the smallest last_accessed timestamp —
            # the least recently used page.
            target_idx = min(
                range(self.capacity),
                key=lambda i: self.frames[i].last_accessed,
            )
            evicted_page = self.frames[target_idx].page_id
            dirty_eviction = self.frames[target_idx].needs_writeback()
            self.frames[target_idx].reset()

        self.frames[target_idx].load(
            page_id=page_id,
            clock_tick=self.clock,
            load_order=self._next_load_order(),
            is_write=is_write,
        )

        self._record_step(
            page_id,
            is_fault=True,
            evicted_page=evicted_page,
            dirty_eviction=dirty_eviction,
        )
        return True

    def get_frame_state(self) -> List[Optional[int]]:
        """Return the page ID currently occupying each frame (None if empty)."""
        return [frame.page_id for frame in self.frames]


# ---------------------------------------------------------------------------
# 3. Second-Chance (Clock) Algorithm
# ---------------------------------------------------------------------------

class SecondChanceAlgorithm(BasePageReplacementAlgorithm):
    """
    Second-Chance (Clock) page replacement.

    Overview
    --------
    An efficient hardware-friendly approximation of LRU.  Pages are arranged
    logically in a circular buffer.  A rotating "clock hand" pointer scans the
    frames in order:

      • If the current frame's R-bit == 1:
            Clear the R-bit to 0 (give a second chance) and advance the hand.
      • If the current frame's R-bit == 0:
            Evict this page — it has not been referenced since the hand last
            passed.  Load the new page here and advance the hand.

    The R-bit is set by the ``touch`` / ``load`` calls (simulating the hardware
    page-table bit) and cleared by the paging daemon (the clock sweep).

    Implementation details
    ----------------------
    * ``self._hand`` is the clock pointer: an integer index in [0, capacity).
    * The circular advance is: ``self._hand = (self._hand + 1) % self.capacity``.
    * The worst case (all R-bits == 1) degenerates to FIFO for that round.

    Hardware model
    --------------
    On real hardware the MMU sets the R-bit on every TLB hit.  The OS clears
    all R-bits during periodic clock interrupts.  This simulation clears the
    R-bit *during the eviction sweep* (per the textbook model) rather than on
    a separate daemon tick, which is equivalent for the purposes of the fault
    rate calculation.
    """

    name: str = "Second-Chance (Clock)"

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        # Clock hand starts at frame 0.
        self._hand: int = 0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def access_page(self, page_id: int, is_write: bool = False) -> bool:
        """
        Process one memory reference under the Second-Chance clock policy.

        Algorithm
        ---------
        1. Increment the logical clock.
        2. Search all frames for the requested page.
           a. Hit  → set R-bit; record hit.
           b. Miss → run the clock sweep to select an eviction victim;
                     load the new page; advance the hand past the new page.
        3. Append trace snapshot and return fault flag.

        The clock sweep
        ~~~~~~~~~~~~~~~
        Starting from ``self._hand`` and cycling through all frames:
          • R == 1 → clear R to 0; advance hand; continue sweep.
          • R == 0 → this is the victim; break.

        Because at most ``capacity`` frames can all have R == 1, the sweep
        terminates within 2 * capacity steps in the worst case.

        Parameters
        ----------
        page_id : int
            Virtual page number being referenced.
        is_write : bool
            True if the access is a write operation.

        Returns
        -------
        bool
            True on page fault, False on page hit.
        """
        self.clock += 1

        # --- Hit path ---------------------------------------------------
        resident_idx = self._find_resident(page_id)
        if resident_idx is not None:
            # Hardware sets the R-bit on every successful TLB lookup.
            self.frames[resident_idx].touch(self.clock, is_write)
            self.page_hits += 1
            self._record_step(page_id, is_fault=False)
            return False

        # --- Fault path -------------------------------------------------
        self.page_faults += 1
        evicted_page: Optional[int] = None
        dirty_eviction: bool = False

        # Prefer a free frame to avoid unnecessary eviction.
        free_idx = self._find_free_frame()
        if free_idx is not None:
            # Use the free slot directly; do not alter the clock hand.
            target_idx = free_idx
        else:
            # Clock sweep: find the first frame with R-bit == 0.
            # Frames with R-bit == 1 receive a second chance (R cleared).
            scanned = 0
            while True:
                current_frame: VirtualPageFrame = self.frames[self._hand]

                if current_frame.reference_bit == 0:
                    # Found the eviction victim.
                    target_idx = self._hand
                    evicted_page = current_frame.page_id
                    dirty_eviction = current_frame.needs_writeback()
                    current_frame.reset()
                    # Advance past the eviction slot before breaking.
                    self._hand = (self._hand + 1) % self.capacity
                    break
                else:
                    # Give this page a second chance: clear R-bit.
                    current_frame.reference_bit = 0
                    self._hand = (self._hand + 1) % self.capacity

                scanned += 1
                if scanned > 2 * self.capacity:
                    # Safety guard — should never trigger in correct code.
                    raise RuntimeError(
                        "Clock sweep exceeded 2×capacity iterations without "
                        "finding a victim.  This indicates a logic error."
                    )

        self.frames[target_idx].load(
            page_id=page_id,
            clock_tick=self.clock,
            load_order=self._next_load_order(),
            is_write=is_write,
        )

        self._record_step(
            page_id,
            is_fault=True,
            evicted_page=evicted_page,
            dirty_eviction=dirty_eviction,
        )
        return True

    def get_frame_state(self) -> List[Optional[int]]:
        """Return the page ID currently occupying each frame (None if empty)."""
        return [frame.page_id for frame in self.frames]

    def reset(self) -> None:
        """Reset base state and rewind the clock hand to frame 0."""
        super().reset()
        self._hand = 0
