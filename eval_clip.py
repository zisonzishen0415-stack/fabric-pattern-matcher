"""Run CLIP matching eval."""
import sys, os
sys.path.insert(0, '.')
from matcher_v3.clip_matcher import FabricIndex, FabricMatcher

d = r'D:\zisonzishen\development\fabric\dir'
index = FabricIndex()
index.build(os.path.join(d, 'fabric'))
matcher = FabricMatcher(index)
matcher.eval_all(os.path.join(d, 'photo'), os.path.join(d, 'fabric'))
