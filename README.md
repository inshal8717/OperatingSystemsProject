# OperatingSystemsProject
# Virtual Memory Simulation Engine & Performance Dashboard

An advanced, production-grade Operating Systems simulation engine that models a hardware-level Memory Management Unit (MMU) and evaluates virtual page replacement strategies. This project provides a comparative architectural analysis of canonical textbook policies against sophisticated database-grade and window-based algorithms, instrumented via an interactive Streamlit analytics dashboard.

## 🖥️ System Architecture & Core Entities

The project is built around a strict separation of concerns, decoupling low-level kernel physics from the graphical presentation layer:

* **`core.py` (Domain Framework):** Defines the canonical hardware-level primitives. Each physical frame slot is modeled via a `VirtualPageFrame` metadata dataclass tracking the absolute hardware bit flags required for advanced paging diagnostics:
    * **Valid/Invalid ($V$) Bit:** Reflects active page presence in physical RAM.
    * **Reference ($R$) Bit:** Set on read/write hits; cleared periodically by clock daemons.
    * **Dirty ($M$) Bit:** Flagged on write access to track costly disk write-backs upon eviction.
    * **Temporal/Frequency Trackers:** Maintains high-resolution logical time histories.
* **`standard_algorithms.py` (Canonical Suite):** Implements the foundational course mechanics: First-In, First-Out (**FIFO**), Least Recently Used (**LRU**), and Second-Chance (**Clock**).
* **`advanced_algorithms.py` (Sophisticated Suite):** Implements non-trivial, optimized eviction strategies to demonstrate complex OS concepts:
    * **LRU-K ($K=2$):** Solves the "one-hit wonder" problem by measuring the backward distance to a page's penultimate reference.
    * **MFU (Most Frequently Used):** Tracks access frequency paired with a classic chronological tiebreaker.
    * **WSClock (Working-Set Clock):** Bridges the gap between structural clock pointer sweeps and dynamic window-based ($\tau$) **thrashing prevention**.
* **`app.py` (Streamlit Dashboard):** A decoupled front-end web client that captures simulation parameters, drives the core engines instantly to completion, and visualizes system behaviors.

---

## 📊 Analytical Performance Metrics

The engine continuously profiles every memory reference loop to compute two primary KPIs required for evaluation:

1.  **Page Fault Rate ($PFF$):**
    $$PFF = \left( \frac{\text{Total Page Faults}}{\text{Total Memory References}} \right) \times 100\%$$
2.  **Memory Utilization Efficiency ($U_{mem}$):**
    $$U_{mem} = \frac{1}{\text{Clock}_{\text{total}}} \sum_{t=1}^{\text{Clock}_{\text{total}}} \left( \frac{\text{Occupied Frame Slots}_t}{\text{Total Allocated Capacity}} \times 100\% \right)$$

---

## 🛠️ Installation & Setup

### Prerequisites
Ensure you have **Python 3.8+** installed on your system.

### 1. Clone or Extract the Project Files
Ensure your project workspace contains the following file layout:
```text
├── core.py                 # Abstract base classes & hardware descriptors
├── standard_algorithms.py  # FIFO, LRU, Second-Chance Clock
├── advanced_algorithms.py  # LRU-2, MFU, WSClock
└── app.py                  # Streamlit graphical dashboard interface
