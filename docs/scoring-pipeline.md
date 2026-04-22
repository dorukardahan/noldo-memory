# NoldoMem Scoring Pipeline

End-to-end guide for how memories are scored at capture and recall.

## 1. Capture Scoring (Write-Time)

### Hook Path (realtime-capture)
```
Base: 0.45
+ Decision detected:  → max(0.45, 0.90)
+ Feedback detected:  → max(0.45, 0.85)
+ Length > 80ch:      + 0.05
+ Length > 200ch:     + 0.05
+ Confirmation (>30ch): + 0.10  ("tamam", "onaylıyorum", "approved", etc.)
+ Error keyword:      + 0.10  ("hata", "bug", "fix", "error", etc.)
= Range: 0.45 – 1.00 (capped)
```

User messages stored if: length > 15ch AND importance ≥ 0.30  
Assistant messages stored if: length ≥ 80ch OR decision/plan keywords

### API Path (triggers.py)
```
Base: 0.35
+ Conversation source: + 0.10
+ Has question mark:   + 0.05
+ Importance markers:  + 0.25
+ Decision markers:    + 0.20
+ Turkish decision:    + 0.20
+ Task markers:        + 0.15
+ Ops markers:         + 0.20
+ Entity names:        + 0.02 per (max 0.10)
+ Word count >150:     + 0.08
+ Word count >80:      + 0.04
+ Word count <8:       - 0.05
+ User role:           + 0.05
+ QA pair:             + 0.08
+ Cron source:         capped at 0.30
+ Decision override:   min(score, 0.70)
= Range: 0.05 – 1.00
```

## 2. Search Floor
- `importance >= 0.05` — effectively no hard filter
- Let ranking determine visibility

## 3. Recall Scoring (Read-Time)

### 6 Candidate Sources
1. **Semantic** (weight 0.50) — sqlite-vec cosine similarity
2. **Keyword/BM25** (weight 0.25) — FTS5 full-text search
3. **Recency** (weight 0.10) — exponential decay from created_at
4. **Strength** (weight 0.07) — Ebbinghaus retention score
5. **Importance** (weight 0.08) — write-time importance score
6. **KG Entity** — candidate pool only (200ms timeout, no RRF weight)

Plus: **Metadata type search** — direct DB lookup for intent-matching types

### RRF Fusion
```
score = Σ weight_i * (1 / (k + rank_i))   where k = 60
```

### Memory Type Bonuses (added after RRF)
| Type | Normal Bonus | Inferred Intent Bonus |
|------|-------------|----------------------|
| lesson | 0.0058 | 0.0117 (2x, min 0.012) |
| rule | 0.0050 | 0.0100 |
| preference | 0.0033 | 0.0067 |
| fact | 0.002 | 0.002 |

### Intent Detection
- Explicit `memory_type` param → hard filter at DB level
- Inferred intent (keyword detection) → soft boost only, cross-type preserved
- Keywords use word boundary matching (regex)
- `memory_type` is limited to `fact`, `preference`, `rule`, `conversation`, `lesson`, and `other`. Operational labels such as incidents or deployments should stay in `category`, `source`, `namespace`, or text.

### Typical Score Ranges
| Scenario | Expected Score |
|----------|---------------|
| Highly relevant, recent, strong | 0.018 – 0.022 |
| Relevant, recent | 0.015 – 0.018 |
| Somewhat relevant | 0.012 – 0.015 |
| Weak / old | 0.008 – 0.012 |

## 4. Decay (Maintenance)

### Regular Memories
- Very recent (72h): boost +0.1 (cap 5.0)
- Recent (21d): maintain ≥ 0.8
- Mid-stale (21–90d): gentle decay 0.07 * (1 + (1 - importance))
- Very stale (90d+): drop to floor (0.3)
- GC: soft-delete at floor + low importance + 60d unaccessed

### Lessons (Immune)
- Recent (72h): boost +0.1
- Protected (60d): maintain ≥ 0.8
- Mid-stale (60–270d): slow decay 0.02
- Floor: **0.8** (never drops below useful threshold)
- Very stale (270d+): drop to 0.8 (still usable)

### Auto Maintenance (runs during decay)
1. `cleanup_envelope_noise()` — strip Slack metadata + refresh FTS + flag re-embed
2. `sync_fts_missing()` — add memories missing from FTS index
