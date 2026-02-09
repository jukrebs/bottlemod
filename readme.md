# Bottlemod I/O extension

> Cache and main memory are synonyms in the further explanation.

This extension to bottlemod implements a simple version of I/O dependency. Currently resources and data can be modeled. Resources are currently CPU time/share only. This can be transferred to all hardware resources. Data on the other hand is modelled by availability. A load is currently linear increasing. Data is loaded once and assumed to be present during the whole computation.

## Simple I/O extension
The main idea is to model data usage/load as a resource. The general data availability is keep as before. If a file or similar is not present, it still cant be loaded.
The data requirement is modelled as unit/s (can be bytes). This is the required data for workflow to progress. This required rate is than split/modelled into requirements for data rate form:
- disk (low data rate)
- cache (memory, high data rate)

For this to work over progress (as a time unit) a split factor needs to be calculated, also over progress. E.g. at a progress of 0.5 (50%) 30% of data needs to be loaded from disk, therefore 70% can come from memory. The needed data rate (requirements function) for the computation is 10 GB/s. This can than be split into two requirements functions, disk and cache. They are now matched against the input data rate(what the respective resource is capable of).

## Split factor

The split factor is basically the hit rate of data. Is the required data in memory or should it be loaded from disk.

Parameters to consider when calculating this hit rate:
- Spatial locality
    - Access patterns:
        - Sequential, Random, Strided, often used in ML

- Temporal locality
    - Was the data accessed before? If so, is it evicted or not.
    - Evictions policies come into play: FIFO, LFU, LRU, random

- Cache size

- Application footprint (in size)
    - Working set size WSS

## Linux Memory and Cache
In memory there is:
- Memory allocated to the systems kernel
- Cache of file system Inodes etc
- Page cache, here are files stored that are used
    - Divided into Dirty and not Dirty, Dirty pages need to be synced with the storage at some point
    - Divided onto active and inactive
    - LRU to evict data form the page cache.

2 important principles:
- readahead -> when reading sequential data, data that is read to be in the future can be loaded into the page cache ahead of use
- write back caching -> data is not directly written to disk, it is written to memory(marked as dirty) and later written to disk in the background

Important: Some applications implement caching them self. They use DirectI/O.


## Hit-Rate h:
Dependent on:
    - spatial locality (data access to elements near the target), access pattern
    - temporal locality (when was the data last used, eviction)

The size of unique data used by a program/or similar is the WSS working set size. The relation between WSS and cache to hit rate is very high:
    - If WSS is smaller than cache size c: After initial cache load/warm up the hit rate is very high
    - If WSS is larger than cache size c: thrashing and cache misses will occur -> eviction

**Sequential access** patterns
    - High spatial locality
    - If data is accessed linear, a data access is very likely when data before was just accessed
    - Theoretically prefetching is possible to further increase hit rate
    - If C > WSS: high hit rate
    - If C < WSS: mediocre hit rate, because prefetching would be possible

**Random access** are unpredictable lookups (uniform random)
    - Spatial locality is low
    - Temporal locality varies and is dependent on the data
    - If C > WSS: high hit rate
    - If C < WSS: very low hit rate, no prediction possible

## How to derive the hit rate h

### Option 1
First option is to *just* input it as a function. Here we could argue that just like the cpu requirements function it has to be derived outside of the Bottlemod model.

We would than input hit rate h as a function over progress. Define cache and disk speeds over time (most likely linear) as input functions. Than we specify a data load of a dataset as a requirement.

Then we can compute the requirements over time for disk and cache by splitting the overall data requirement into two requirement functions (disk and cache).


### Option 2
Derive the hit rate from additional parameters:
These are:
    - data size
    - cache size
    - access pattern (based on this we can empirically infer a hit rate)
        - locality:
            - spatial is very high when we read sequential
            - temporal are we reading the same data again and again
                - this leads to cold->warm cache

Algorithm looks as follows:
We input the i/o environment (cache size), we input data size and access pattern.
Apply heuristics (maybe from experiments). We compute hit rate h over [0,1]


**Deriving heuristics**

Create benchmark or mini experiments to derive/characterize the hit rate for sequential and random reads for a specific I/O env (cache, disk combination and wss/data size)

For sequential:
    - Reading a "cold" file of 0.5x, 1x, 2x the cache from disk
    - Rereading a file of these sizes, cache moves from "cold" to "warm" for that file

For random (uniform):
    - Random read

Real world scenarios:

Sequential -> video encoding
Random -> database query/dataset query

These give us basic hit rates.


## Experiment

Assumptions:
Linux, single user (no shared cache), two level storage (disk and cache, no cpu cache).


1. Video encoding/ encoding of a genome/ dataset -> very sequential

2. Database lookups/ dataset lookups random -> very random


Scenarios:
WSS is larger than Cache, data gets evicted from cache. -> slow down
    - We increase cache/memory size -> no data eviction/thrashing -> speedup

Memory or disk speed is the bottleneck, a faster disk or staggered loads speedup the process
 - Similar to network bandwidth in original paper




# Further ideas
- Investigate the "Power law of cache misses"
- More sophisticated access patterns:
    - Strided access patter
    - Consider prefetching
- IOPS to capture tasks where a lot of small files are read
    - This can model hardware more accurate
- Multi layered cache. The current extension only considers main system memory/cache. There is no logic for L1, L2 and more cache layers that can be found in modern systems.

Writes can be modelled as 100% cache speed. Because of writeback caching.

Todo:
Wie funktioniert cache unter linux. Docker assigned z.b. 2gb, was passiert wenn mehr als 2gb datein gealden wird. wer handled den cache. betriebsystem oder container?


Bottlemod Bugs:
1. Program crash

2. Danger to iterate over ever increasing increments -> increments of progress are getting smaller and smaller.
