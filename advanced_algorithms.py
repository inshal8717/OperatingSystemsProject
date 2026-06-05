"""
advanced_algorithms.py — Advanced Page-Replacement Algorithms
=============================================================
Implements three production-grade algorithms that go beyond the canonical
textbook trio and model more sophisticated OS/hardware behaviours:

  1. LRU-K (K=2)  — Backward k-distance using penultimate reference history.
  2. MFU          — Most-Frequently-Used eviction with FIFO tiebreaker.
  3. WSClock      — Working-Set Clock with age-based thrashing prevention.

Design decisions
----------------
* All three inherit ``BasePageReplacementAlgorithm`` from ``core.py``.
* Out-of-frame page history is maintained for LRU-K to model the "correlated
  reference problem" — history is preserved even after eviction so that a
  page re-loaded after a period of absence does not spuriously get a
  near-zero backward distance.
* WSClock's working-set window (τ) is a constructor parameter, enabling the
  Streamlit dashboard to expose it as a tunable sidebar control.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from core import BasePageReplacementAlgorithm, VirtualPageFrame


# ---------------------------------------------------------------------------
# 1. LRU-K (K = 2)
# ---------------------------------------------------------------------------

class LRUKAlgorithm(BasePageReplacementAlgorithm):
    """
    LRU-K page replacement with K=2.

    Theoretical background
    ----------------------
    Classical LRU uses the *most recent* reference time as its eviction key.
    O'Neil, O'Neil & Weikum (1993) extended this to the K-th most recent
    reference: the **backward K-distance** of a page at time t is the elapsed
    time since its K-th to last reference.

    For K=2 the eviction key is the interval since the *penultimate* (second-
    to-last) access:

        backward_2_distance(p, t) =
            t − hist[p][-2]     if |hist[p]| >= 2
            +∞                   otherwise

    Pages with backward distance +∞ (fewer than 2 historical references) are
    always evicted before pages with finite distances.  Among pages with
    infinite distance the algorithm falls back to FIFO ordering (load_order).

    Out-of-frame history tracking
    -----------------------------
    ``self._page_history`` maps every page ID ever seen to a bounded deque
    (maxlen=K=2) of its most recent reference timestamps.  This persists across
    evictions so that a page re-entering memory after a period of absence
    carries its correct historical context instead of starting fresh.

    This faithfully models the "correlated reference problem" fix that LRU-2
    was designed to solve: pages accessed in bursts (sequential scans) will
    have a long backward-2-distance and will be evicted before frequently
    reused hot pages.
    """

    name: str = "LRU-2 (K=2)"
    K: int = 2

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        # Global history map: page_id → deque of up to K recent timestamps.
        # Persists across evictions.
        self._page_history: Dict[int, Deque[int]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_history(self, page_id: int) -> Deque[int]:
        """Return the history deque for a page, creating one if absent."""
        if page_id not in self._page_history:
            self._page_history[page_id] = deque(maxlen=self.K)
        return self._page_history[page_id]

    def _backward_k_distance(self, page_id: int) -> float:
        """
        Compute the backward-K distance for a page at the current clock tick.

        Returns
        -------
        float
            The backward-2 distance, or ``math.inf`` if the page has been
            referenced fewer than K times in its entire history.
        """
        hist = self._page_history.get(page_id)
        if hist is None or len(hist) < self.K:
            return math.inf
        # hist[-K] is the K-th most recent (oldest of the K retained entries).
        penultimate_time = hist[-self.K]
        return float(self.clock - penultimate_time)

    def _eviction_key(self, frame_idx: int) -> Tuple[float, int]:
        """
        Composite sort key for choosing the eviction victim.

        Primary   : backward-2 distance (descending → largest first is victim).
        Secondary : load_order (ascending → oldest loaded page breaks ties).

        A tuple (dist, load_order) is returned so Python's ``max`` can work
        directly using default tuple comparison.
        """
        frame = self.frames[frame_idx]
        dist = self._backward_k_distance(frame.page_id)
        return (dist, frame.load_order)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def access_page(self, page_id: int, is_write: bool = False) -> bool:
        """
        Process one memory reference under LRU-2 replacement.

        Algorithm
        ---------
        1. Increment the logical clock and record the reference in the
           global page history (regardless of residency).
        2. Search frames for the page.
           a. Hit  → update frame hardware flags; record hit.
           b. Miss → select victim via backward-2-distance (largest victim,
                     with +∞ distances preferred for eviction); load page.
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

        # Always update global history first — before checking residency.
        hist = self._get_or_create_history(page_id)
        hist.append(self.clock)

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
            # Select the frame whose resident page has the largest backward-2
            # distance.  ``max`` with the composite key handles both infinite
            # and finite distances correctly via Python's tuple comparison
            # (math.inf > any finite float, and load_order breaks ties among
            # multiple infinite-distance pages by preferring the oldest).
            target_idx = max(range(self.capacity), key=self._eviction_key)
            evicted_page = self.frames[target_idx].page_id
            dirty_eviction = self.frames[target_idx].needs_writeback()
            self.frames[target_idx].reset()

        self.frames[target_idx].load(
            page_id=page_id,
            clock_tick=self.clock,
            load_order=self._next_load_order(),
            is_write=is_write,
            history_capacity=self.K,
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
        """Reset base state and clear the out-of-frame history map."""
        super().reset()
        self._page_history.clear()


# ---------------------------------------------------------------------------
# 2. MFUI — Most-Frequently-Used with Age-Incentive Decay
# ---------------------------------------------------------------------------

class MFUIAlgorithm(BasePageReplacementAlgorithm):
    """
    MFUI: Most-Frequently-Used with Age-Incentive decay.

    Eviction policy
    ---------------
    Evict the resident page carrying the **highest decayed frequency counter**.
    The decay mechanism prevents pages that were heavily referenced during
    the program's startup phase from permanently occupying physical RAM — a
    known pathology of plain MFU on workloads with phase transitions.

    Decay mechanics
    ---------------
    Every ``decay_interval`` references (counted globally via ``self.clock``)
    all resident frame frequencies are multiplied by ``decay_factor`` and
    truncated to an integer:

        frame.frequency = int(frame.frequency * decay_factor)

    With ``decay_factor = 0.95`` and ``decay_interval = 5`` a page at count 100
    drops to 95 → 90 → 85 … giving recently active pages a competitive
    advantage over historically hot but now idle pages.

    Tiebreaker
    ----------
    When multiple frames share the maximum decayed frequency the victim is the
    frame with the smallest ``load_order`` (oldest-loaded first — FIFO break).

    Parameters
    ----------
    capacity : int
        Number of physical frames.
    decay_factor : float
        Multiplicative decay applied to all frame frequencies every
        ``decay_interval`` references.  Must be in (0.0, 1.0).
        Default: 0.95.
    decay_interval : int
        Number of references between successive decay sweeps.
        Default: 5.
    """

    name: str = "MFUI"

    def __init__(
        self,
        capacity: int,
        decay_factor: float = 0.95,
        decay_interval: int = 5,
    ) -> None:
        super().__init__(capacity)
        if not (0.0 < decay_factor < 1.0):
            raise ValueError(f"decay_factor must be in (0, 1), got {decay_factor!r}.")
        if decay_interval < 1:
            raise ValueError(f"decay_interval must be ≥ 1, got {decay_interval!r}.")
        self.decay_factor: float = decay_factor
        self.decay_interval: int = decay_interval

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_decay_if_due(self) -> None:
        """
        Apply the frequency decay sweep when the clock is a multiple of
        ``decay_interval``.  Each resident frame's frequency counter is
        multiplied by ``decay_factor`` and cast to int (floor truncation).
        Empty frames are skipped — they carry no meaningful frequency.
        """
        if self.clock % self.decay_interval == 0:
            for frame in self.frames:
                if not frame.is_empty():
                    # int() truncates toward zero (floor for positive values).
                    frame.frequency = int(frame.frequency * self.decay_factor)

    def _eviction_key(self, frame_idx: int) -> Tuple[int, int]:
        """
        Composite sort key for victim selection.

        Primary   : decayed frequency  (max → victim).
        Secondary : load_order negated (smallest load_order wins ties,
                    meaning the oldest-loaded page is evicted first).
        """
        frame = self.frames[frame_idx]
        return (frame.frequency, -frame.load_order)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def access_page(self, page_id: int, is_write: bool = False) -> bool:
        """
        Process one memory reference under MFUI replacement.

        Algorithm
        ---------
        1. Increment the logical clock.
        2. Apply frequency decay sweep if the clock tick is a multiple of
           ``decay_interval`` (before the hit/fault decision so the decay
           is visible to the eviction logic on this same step).
        3. Scan frames for the requested page.
           a. Hit  → increment frequency; update timestamps and dirty bit.
           b. Miss → select victim (max decayed frequency, FIFO tiebreak);
                     load new page with frequency = 1.
        4. Append trace snapshot and return fault flag.

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

        # Apply periodic decay before making the hit/fault decision so that
        # decayed counters are current when the eviction key is evaluated.
        self._apply_decay_if_due()

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
            # Evict the page with the highest decayed frequency.
            # Ties are broken by oldest load_order (FIFO).
            target_idx = max(range(self.capacity), key=self._eviction_key)
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
# 3. WSClock — Working-Set Clock
# ---------------------------------------------------------------------------

class WSClockAlgorithm(BasePageReplacementAlgorithm):
    """
    WSClock page-replacement algorithm.

    Theoretical background
    ----------------------
    Denning's Working-Set model (1968) defines the working set W(t, τ) at
    time t as the set of pages referenced during the window (t−τ, t].  If
    all physical frames are occupied by working-set pages, no page can be
    safely evicted without increasing the fault rate — the system is
    *thrashing*.

    WSClock (Aho, Denning & Ullman 1971) combines the clock (circular buffer)
    mechanics of the Second-Chance algorithm with a working-set age test to
    prevent thrashing:

        A page is eligible for eviction if:
            R-bit == 0   AND   age > τ
        where  age = current_time − last_accessed_time.

    Clock sweep mechanics
    ---------------------
    The hand scans frames in circular order.  For each frame:

      Case 1 — R-bit == 1:
        The page was recently used.  Clear R-bit; update last_accessed to the
        current time (re-anchoring the age calculation).  Advance hand.

      Case 2 — R-bit == 0, age ≤ τ:
        The page is in the working set.  Skip it (do not give it a second
        chance; R is already 0).  Advance hand.

      Case 3 — R-bit == 0, age > τ:
        The page is NOT in the working set and has no recent reference.
        This is the eviction victim.  If dirty, schedule a write-back.
        Load the new page here; advance hand past the new page.

    Full-circle fallback
    --------------------
    If after a complete revolution no eligible victim was found (all pages
    are in the working set), two sub-cases apply:
      • If any dirty page was encountered during the sweep:
            The hand has already initiated write-backs.  On the next tick the
            clean version of one of those pages will become the victim.
            For this simulation we simply select the *oldest* (minimum
            last_accessed) dirty page as the victim.
      • If all pages are clean and in the working set:
            Pick the page with the smallest last_accessed timestamp (oldest
            working-set member) as the least-likely-to-be-needed-next.

    Parameters
    ----------
    capacity : int
        Number of physical frames.
    tau : int
        Working-set window size (τ).  A page whose age exceeds τ logical
        ticks is considered outside the working set and may be evicted.
    """

    name: str = "WSClock"

    def __init__(self, capacity: int, tau: int = 4) -> None:
        super().__init__(capacity)
        if tau < 1:
            raise ValueError(f"WSClock τ must be ≥ 1, got {tau!r}.")
        self.tau: int = tau
        # Circular clock hand pointer (index into self.frames).
        self._hand: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _age(self, frame_idx: int) -> int:
        """
        Compute the structural age of the page in frame ``frame_idx``.

        Age = current_time − last_accessed_time.

        An age > τ means the page lies outside the current working set.
        """
        return self.clock - self.frames[frame_idx].last_accessed

    def _select_victim(self) -> Tuple[int, Optional[int], bool]:
        """
        Run a full WSClock sweep starting from ``self._hand`` and return
        the index of the frame chosen as the eviction victim, the evicted
        page ID, and the dirty-eviction flag.

        The hand advances during this method; on return it points to the
        frame *after* the victim (ready for the next access cycle).

        Returns
        -------
        target_idx : int
            Frame index of the eviction victim.
        evicted_page : Optional[int]
            Page ID of the evicted page.
        dirty_eviction : bool
            True if the evicted page had its dirty bit set.
        """
        # Track whether any dirty-but-eligible pages were seen during the sweep.
        # We may need these as fallback victims if no clean eligible page found.
        dirty_candidate_idx: Optional[int] = None
        dirty_candidate_age: int = -1

        # Track the oldest page overall as a last-resort fallback.
        oldest_idx: int = min(
            range(self.capacity),
            key=lambda i: self.frames[i].last_accessed,
        )

        # Bound the sweep to at most capacity steps to guarantee termination.
        for _ in range(self.capacity):
            frame: VirtualPageFrame = self.frames[self._hand]
            current_hand = self._hand
            current_age = self._age(current_hand)

            # Advance hand immediately; will be corrected if this is the victim.
            self._hand = (self._hand + 1) % self.capacity

            if frame.reference_bit == 1:
                # Case 1: Recently referenced.  Clear R-bit and re-anchor age.
                frame.reference_bit = 0
                frame.last_accessed = self.clock
                # Do NOT yet evict; give this page another chance.
                continue

            # R-bit == 0 from here.
            if current_age <= self.tau:
                # Case 2: In the working set.  Record as potential dirty candidate
                # for fallback, but do not evict.
                if frame.dirty_bit == 1:
                    if dirty_candidate_idx is None or current_age > dirty_candidate_age:
                        dirty_candidate_idx = current_hand
                        dirty_candidate_age = current_age
                continue

            # Case 3: R-bit == 0 AND age > τ — outside working set.
            if frame.dirty_bit == 0:
                # Clean page: evict immediately.
                evicted = frame.page_id
                dirty = False
                frame.reset()
                # Hand already advanced past this slot above.
                return current_hand, evicted, dirty
            else:
                # Dirty page outside working set: schedule write-back.
                # In a real OS this would be async; here we treat it as an
                # immediate (synchronous) write-back to keep the model simple.
                evicted = frame.page_id
                dirty = True
                frame.reset()
                return current_hand, evicted, dirty

        # --- Full-circle fallback ------------------------------------------
        # No victim found in one revolution.  Use fallback heuristics.

        if dirty_candidate_idx is not None:
            # A dirty working-set page was seen.  Evict it (simulating the
            # write-back completing after the sweep).
            frame = self.frames[dirty_candidate_idx]
            evicted = frame.page_id
            dirty = True
            frame.reset()
            self._hand = (dirty_candidate_idx + 1) % self.capacity
            return dirty_candidate_idx, evicted, dirty

        # All pages are clean and in the working set — evict the oldest.
        frame = self.frames[oldest_idx]
        evicted = frame.page_id
        dirty = frame.dirty_bit == 1
        frame.reset()
        self._hand = (oldest_idx + 1) % self.capacity
        return oldest_idx, evicted, dirty

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def access_page(self, page_id: int, is_write: bool = False) -> bool:
        """
        Process one memory reference under WSClock replacement.

        Algorithm
        ---------
        1. Increment the logical clock.
        2. Scan frames for the requested page.
           a. Hit  → set R-bit; update dirty bit; record hit.
           b. Miss → run WSClock sweep to find victim; load new page.
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
            target_idx, evicted_page, dirty_eviction = self._select_victim()

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