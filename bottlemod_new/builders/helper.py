"""
Helper Functions for BottleMod Builders

Provides utility functions to express common units and patterns more easily.
"""

# ============================================================================
# FLOPS (Floating Point Operations)
# ============================================================================


def KFLOPS(value: float) -> float:
    """Convert kilo-FLOPS to FLOPS."""
    return value * 1e3


def MFLOPS(value: float) -> float:
    """Convert mega-FLOPS to FLOPS."""
    return value * 1e6


def GFLOPS(value: float) -> float:
    """Convert giga-FLOPS to FLOPS."""
    return value * 1e9


def TFLOPS(value: float) -> float:
    """Convert tera-FLOPS to FLOPS."""
    return value * 1e12


def PFLOPS(value: float) -> float:
    """Convert peta-FLOPS to FLOPS."""
    return value * 1e15


# ============================================================================
# Bytes
# ============================================================================


def KB(value: float) -> float:
    """Convert kilobytes to bytes."""
    return value * 1024


def MB(value: float) -> float:
    """Convert megabytes to bytes."""
    return value * 1024**2


def GB(value: float) -> float:
    """Convert gigabytes to bytes."""
    return value * 1024**3


def TB(value: float) -> float:
    """Convert terabytes to bytes."""
    return value * 1024**4


def PB(value: float) -> float:
    """Convert petabytes to bytes."""
    return value * 1024**5


# ============================================================================
# Bandwidth (Bytes per second)
# ============================================================================


def KBps(value: float) -> float:
    """Convert kilobytes per second to bytes per second."""
    return value * 1024


def MBps(value: float) -> float:
    """Convert megabytes per second to bytes per second."""
    return value * 1024**2


def GBps(value: float) -> float:
    """Convert gigabytes per second to bytes per second."""
    return value * 1024**3


def TBps(value: float) -> float:
    """Convert terabytes per second to bytes per second."""
    return value * 1024**4


# ============================================================================
# Time
# ============================================================================


def ms(value: float) -> float:
    """Convert milliseconds to seconds."""
    return value * 1e-3


def us(value: float) -> float:
    """Convert microseconds to seconds."""
    return value * 1e-6


def ns(value: float) -> float:
    """Convert nanoseconds to seconds."""
    return value * 1e-9


def minutes(value: float) -> float:
    """Convert minutes to seconds."""
    return value * 60


def hours(value: float) -> float:
    """Convert hours to seconds."""
    return value * 3600


# ============================================================================
# Frequency (for bandwidth conversion)
# ============================================================================


def KHz(value: float) -> float:
    """Convert kilohertz to hertz."""
    return value * 1e3


def MHz(value: float) -> float:
    """Convert megahertz to hertz."""
    return value * 1e6


def GHz(value: float) -> float:
    """Convert gigahertz to hertz."""
    return value * 1e9


# ============================================================================
# Common Patterns
# ============================================================================


def percent(value: float) -> float:
    """
    Convert percentage (0-100) to fraction (0-1).

    Examples
    --------
    >>> percent(80)  # 80% -> 0.8
    0.8
    >>> percent(5.5)  # 5.5% -> 0.055
    0.055
    """
    return value / 100.0


def hit_rate(value: float) -> float:
    """
    Alias for percent() - convert cache hit rate percentage to fraction.

    Examples
    --------
    >>> hit_rate(90)  # 90% hit rate -> 0.9
    0.9
    """
    return percent(value)


def miss_rate(value: float) -> float:
    """
    Convert cache miss rate percentage to fraction.

    Examples
    --------
    >>> miss_rate(10)  # 10% miss rate -> 0.1
    0.1
    """
    return percent(value)
