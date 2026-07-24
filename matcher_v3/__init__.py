"""
matcher_v3: Three-stage fabric pattern matching engine.

Stages:
  0. KD-tree compact signature screening (O(log n), scales to 100K+)
  1. Full feature comparison (color + texture + FFT)
  2. ORB geometric verification + re-ranking

Usage:
    from matcher_v3.indexer import FabricIndex
    from matcher_v3.matcher import FabricMatcher

    index = FabricIndex().build("素材/fabric/")
    matcher = FabricMatcher(index)
    results = matcher.match("素材/photo/1.png")
"""
from .indexer import FabricIndex
from .matcher import FabricMatcher
