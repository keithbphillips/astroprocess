"""Guess the processing profile (galaxy / nebula / cluster) from an object name.

Covers the Messier catalogue and common NGC/IC/Sharpless/LDN prefixes. Anything unknown defaults to
"nebula", which is the safest general-purpose look; override with ``--type``.
"""
from __future__ import annotations

import re

MESSIER = {
    "galaxy": {31, 32, 33, 49, 51, 58, 59, 60, 61, 63, 64, 65, 66, 74, 77, 81, 82, 83, 84, 85, 86, 87, 88, 89,
               90, 91, 94, 95, 96, 98, 99, 100, 101, 102, 104, 105, 106, 108, 109, 110},
    "nebula": {1, 8, 16, 17, 20, 27, 42, 43, 45, 57, 76, 78, 97},  # M45: the reflection nebula is the show
    # everything else in Messier is a globular or open cluster (or M40/M73 asterisms)
}

NGC_GALAXIES = {55, 247, 253, 300, 891, 925, 1300, 1365, 2403, 2683, 2841, 2903, 3115, 3184, 3344, 3521, 3628,
                4038, 4216, 4244, 4395, 4449, 4490, 4517, 4559, 4565, 4631, 4656, 4725, 5005, 5033, 5128, 5907,
                6503, 6822, 6946, 7331, 7814}
NGC_CLUSTERS = {457, 663, 752, 869, 884, 1502, 1528, 2158, 2169, 2232, 2244, 2301, 2362, 2392, 6231, 6633,
                6705, 6791, 6819, 6939, 7243, 7789}
IC_GALAXIES = {10, 342, 1613, 2574}
IC_CLUSTERS = {2391, 2602, 4665, 4756}


def guess_type(name: str) -> str:
    n = name.upper().replace("_", " ")
    if re.search(r"\b(SH ?2|SHARPLESS|LDN|LBN|VDB|BARNARD|B ?\d|ABELL|PN|RCW|CED|GUM)\b", n):
        return "nebula"
    if any(w in n for w in ("GALAXY", "LEO TRIPLET", "MARKARIAN")):
        return "galaxy"
    if any(w in n for w in ("CLUSTER", "BEEHIVE", "DOUBLE CLUSTER")):
        return "cluster"
    if any(w in n for w in ("NEBULA", "VEIL", "CRESCENT", "IRIS", "HEART", "SOUL", "ROSETTE", "PLEIADES")):
        return "nebula"
    m = re.match(r"^\s*M\s*0*(\d+)\b", n)
    if m:
        num = int(m.group(1))
        for t, nums in MESSIER.items():
            if num in nums:
                return t
        return "cluster"
    m = re.match(r"^\s*NGC\s*0*(\d+)", n)
    if m:
        num = int(m.group(1))
        if num in NGC_GALAXIES:
            return "galaxy"
        if num in NGC_CLUSTERS:
            return "cluster"
        return "nebula"
    m = re.match(r"^\s*IC\s*0*(\d+)", n)
    if m:
        num = int(m.group(1))
        if num in IC_GALAXIES:
            return "galaxy"
        if num in IC_CLUSTERS:
            return "cluster"
        return "nebula"
    return "nebula"


def is_solar_system(name: str) -> bool:
    """Comets, Moon, Sun, planets: not handled by the deep-sky pipeline."""
    n = name.upper()
    return bool(re.match(r"^\s*(\d+P|C[-/ ]?\d{4}|LUNAR|MOON|SOLAR|SUN|JUPITER|SATURN|MARS|VENUS)", n))
