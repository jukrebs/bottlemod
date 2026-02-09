# Storage Hierarchy Modeling in BottleMod-SH

## Overview

The BottleMod-SH extension enhances the original BottleMod framework by enabling detailed modeling of storage hierarchy effects, such as caches and tiered storage (e.g., DRAM, SSD, HDD). This allows for more accurate bottleneck analysis in data-intensive workloads where storage is a critical factor.

---

## How Storage Hierarchy is Modeled

### 1. Logical Access Profiles
- **Purpose:** Describe how a process accesses data, independent of time.
- **Class:** `LogicalAccessProfile`
- **Attributes:**
  - `A_read`, `A_write`: Cumulative bytes read/written as a function of progress.
  - `Q_read`, `Q_write`: Cumulative operations (e.g., IOPS) as a function of progress.
- **Example:**
  - Sequential read: 1GB over 100 progress units.

### 2. Storage Tiers
- **Purpose:** Represent physical or logical storage layers (e.g., DRAM, NVMe SSD, HDD).
- **Class:** `StorageTier`
- **Attributes:**
  - Bandwidth and IOPS functions (as `PPoly` objects, over time)
  - Capacity (for cache modeling)
  - Tier index (0 = fastest, higher = slower)
- **Predefined tiers:** `memory`, `nvme_ssd`, `sata_ssd`, `hdd`

### 3. Tier Mappings
- **Purpose:** Specify how accesses are distributed across tiers (e.g., cache hit/miss rates).
- **Class:** `TierMapping`
- **Attributes:**
  - For each dataset, maps what fraction of accesses are served by each tier as a function of progress.

### 4. Cache Behavior Models
- **Purpose:** Abstract how cache/tier hit rates are computed (e.g., direct hit rate, stack distance, phase-based).
- **Class:** `CacheBehaviorModel` and its subclasses

### 5. Resource Derivation
- **Purpose:** Convert the above abstractions into standard resource requirement and input functions for the BottleMod progress engine.
- **Functions:** `derive_tier_resources`, `derive_all_tier_resources`

### 6. StorageHierarchyTask
- **Purpose:** Convenience wrapper to combine all the above and produce a standard `Task` for simulation/analysis.
- **Class:** `StorageHierarchyTask`

---

## Required Inputs to Use Storage Hierarchy

1. **Logical Access Profiles** (`LogicalAccessProfile`)
   - For each dataset, specify how much data is read/written and how (sequential/random).
   - Example: `LogicalAccessProfile.sequential_read("dataset1", total_bytes=1e9, max_progress=100)`

2. **Storage Tiers** (`StorageTier`)
   - Define the storage layers in your environment.
   - Example: `StorageTier.memory()`, `StorageTier.nvme_ssd()`, etc.

3. **Tier Mappings** (`TierMapping`)
   - For each dataset, specify how accesses are distributed across tiers (e.g., all from disk, or a mix based on cache hit rates).
   - Example: `TierMapping.all_from_tier("dataset1", tier_index=2, progress_range=(0, 100))`

4. **(Optional) CPU/Data Requirement Functions**
   - If you want to combine storage with CPU or other requirements.

5. **(Optional) Cache Behavior Model**
   - If you want to automatically compute tier mappings from cache models.

---

## Example Usage

```python
# 1. Define logical access profiles for your datasets
access_profiles = [
    LogicalAccessProfile.sequential_read("dataset1", total_bytes=1e9, max_progress=100)
]

# 2. Define your storage tiers (fastest to slowest)
tiers = [
    StorageTier.memory(),
    StorageTier.nvme_ssd(),
    StorageTier.hdd()
]

# 3. Define tier mappings (e.g., all reads from HDD)
tier_mappings = [
    TierMapping.all_from_tier("dataset1", tier_index=2, progress_range=(0, 100))
]

# 4. Create a StorageHierarchyTask
task = StorageHierarchyTask(
    access_profiles=access_profiles,
    tier_mappings=tier_mappings,
    tiers=tiers
)

# 5. Convert to standard Task for simulation
bottlemod_task = task.to_task()
```

---

## Changes Made to the Original BottleMod Implementation

1. **New Abstractions:**
   - Added `LogicalAccessProfile`, `StorageTier`, `TierMapping`, and `CacheBehaviorModel` classes to represent storage hierarchy concepts.

2. **Resource Derivation:**
   - Introduced functions to convert storage hierarchy models into standard resource requirement/input functions compatible with BottleMod's progress engine.

3. **Task Extension:**
   - Added `StorageHierarchyTask` as a wrapper to combine storage and CPU/data requirements, and to produce a standard `Task` object.

4. **Integration Points:**
   - The core progress calculation algorithm of BottleMod remains unchanged; the extension works by generating compatible resource functions.
   - No changes to the core simulation/solver logic were required—only the way resource requirements/inputs are constructed was extended.

5. **Convenience and Validation:**
   - Added validation and utility methods for easier modeling and debugging (e.g., checking that tier mappings sum to 1).

