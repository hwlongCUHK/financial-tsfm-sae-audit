#!/usr/bin/env python3
"""Apply 3 fixes to pipeline.py."""
path = '/home/houwanlong/marketing_kg/marketing_kg/pipeline.py'
with open(path) as f:
    code = f.read()

# Fix 1: build_matrix - add S_b lookup and proper entry dicts
code = code.replace(
    'def build_matrix(self, pharma, sb_top20, pool_size=80):',
    'def build_matrix(self, pharma, sb_pairs, pool_size=80):')

code = code.replace(
    'sb_set = {s[0] for s in sb_top20}',
    'sb_top20 = sb_pairs[:20]\n        sb_set = {s[0] for s in sb_top20}\n        sb_map = dict(sb_pairs)')

code = code.replace(
    "for p in pharma:\n            if p['intensity'] in ('高', '中'):\n                if p['name'] in sb_set: core.append(p)\n                else: threats.append(p)\n            elif p['name'] in sb_set:\n                co_pur.append(p)",
    "for p in pharma:\n            entry = {'name': p['name'], 'S_b': round(sb_map.get(p['name'], 0.0), 6), 'llm_intensity': p.get('intensity', ''), 'llm_reason': p.get('reason', '')}\n            if p['intensity'] in ('高', '中'):\n                if p['name'] in sb_set: core.append(entry)\n                else: threats.append(entry)\n            elif p['name'] in sb_set:\n                co_pur.append(entry)")

# Fix 2: build_kg call site
code = code.replace(
    'matrix = self.build_matrix(pharma, sb_pairs[:20])',
    'matrix = self.build_matrix(pharma, sb_pairs)')

# Fix 3: Prompt 1 - detailed intensity definitions
old_line = '从中选出{drug}的药理竞品，最多10个。'
new_line = '从中选出{drug}的药理竞品（适应症或治疗效果重叠，消费者理论上可在两者之间替代的药品），最多10个，按竞争强度排序。'
code = code.replace(old_line, new_line)

# Add intensity definitions before the JSON output instruction
old_json_hint = '对每个竞品输出：名称（必须与候选列表一致）、竞争理由（一句话）、竞争强度（高/中/低）。'
new_json_hint = '对每个竞品输出：名称（必须与候选列表完全一致）、竞争理由（一句话）、竞争强度（高/中/低）。\n\n竞争强度定义：\n- 高：同一适应症、同一成分，直接替代关系\n- 中：适应症部分重叠，或同类别不同成分\n- 低：仅有辅助症状重叠，或共病相关但非药理替代（如抗生素与感冒药仅在上呼吸道感染场景中共存，不应标为高/中）'
code = code.replace(old_json_hint, new_json_hint)

with open(path, 'w') as f:
    f.write(code)

print("3 fixes applied to pipeline.py")

# Verify
import re
if 'sb_map' in code:
    print("  [OK] Fix 1: build_matrix uses sb_map")
if 'entry = {' in code:
    print("  [OK] Fix 2: matrix entries have S_b + llm_reason fields")
if '共病相关但非药理替代' in code:
    print("  [OK] Fix 3: detailed intensity definitions")
if 'sb_pairs[:20]' not in code:
    print("  [OK] Fix 4: build_kg passes full sb_pairs")
