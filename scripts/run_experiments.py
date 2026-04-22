"""
AeroLex — Chunking Strategy Comparison
Compares all 3 strategies and picks the best one.
"""

import json
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_chunks(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def analyze_chunks(chunks: list, strategy: str) -> dict:
    """Analyze chunk quality metrics."""
    if not chunks:
        return {}

    sizes = [c["char_count"] for c in chunks]
    words = [c["word_count"] for c in chunks]

    # Count chunks that are too small (< 50 chars) — bad for RAG
    too_small = sum(1 for s in sizes if s < 50)

    # Count chunks that are too large (> 1000 chars) — too much noise
    too_large = sum(1 for s in sizes if s > 1000)

    # Ideal chunks: 100-600 chars
    ideal = sum(1 for s in sizes if 100 <= s <= 600)
    ideal_pct = round(ideal / len(chunks) * 100, 1)

    return {
        "strategy":       strategy,
        "total_chunks":   len(chunks),
        "avg_chars":      round(sum(sizes) / len(sizes)),
        "min_chars":      min(sizes),
        "max_chars":      max(sizes),
        "avg_words":      round(sum(words) / len(words)),
        "too_small":      too_small,
        "too_large":      too_large,
        "ideal_range":    ideal,
        "ideal_pct":      ideal_pct,
        "has_citations":  all("citation" in c for c in chunks),
        "has_hierarchy":  all("hierarchy" in c for c in chunks),
    }

def print_comparison(results: list[dict]) -> None:
    """Print formatted comparison table."""
    print("\n" + "="*70)
    print("📊 AEROLEX — CHUNKING STRATEGY COMPARISON")
    print("="*70)

    headers = ["Metric", "Recursive", "Semantic", "Hierarchical"]
    metrics = [
        ("Total Chunks",     "total_chunks"),
        ("Avg Size (chars)", "avg_chars"),
        ("Min Size",         "min_chars"),
        ("Max Size",         "max_chars"),
        ("Avg Words",        "avg_words"),
        ("Too Small (<50)",  "too_small"),
        ("Too Large (>1000)","too_large"),
        ("Ideal Range",      "ideal_range"),
        ("Ideal % (100-600)","ideal_pct"),
    ]

    print(f"\n{'Metric':<25} {'Recursive':>12} {'Semantic':>12} {'Hierarchical':>14}")
    print("-"*65)

    for label, key in metrics:
        vals = [str(r.get(key, "N/A")) for r in results]
        print(f"{label:<25} {vals[0]:>12} {vals[1]:>12} {vals[2]:>14}")

    print("\n" + "="*70)
    print("🏆 WINNER ANALYSIS")
    print("="*70)

    # Find best strategy
    ideal_pcts = [(r["ideal_pct"], r["strategy"]) for r in results]
    best = max(ideal_pcts, key=lambda x: x[0])

    print(f"\n✅ Best % of ideal-size chunks: {best[1]} ({best[0]}%)")
    print(f"\n📝 Strategy Notes:")
    print(f"   Recursive    — Fast, consistent size, good baseline")
    print(f"   Semantic     — Meaning-aware, but slow + variable size")
    print(f"   Hierarchical — Aviation-specific, exact paragraph citations")
    print(f"\n💡 AeroLex Recommendation:")
    print(f"   Use HIERARCHICAL for regulation queries (exact citations)")
    print(f"   Use RECURSIVE as fallback for non-regulation docs (SKYbrary)")
    print("="*70)


if __name__ == "__main__":
    print("\n--- AeroLex Chunking Strategy Comparison ---\n")

    # Load all chunk files
    base = Path("data/processed/chunks")

    recursive_path    = base / "part_91_recursive_512.json"
    semantic_path     = base / "part_91_semantic_05.json"
    hierarchical_path = base / "part_91_hierarchical.json"

    results = []

    if recursive_path.exists():
        chunks = load_chunks(str(recursive_path))
        results.append(analyze_chunks(chunks, "Recursive"))
        print(f"✅ Recursive: {len(chunks)} chunks loaded")
    else:
        print("❌ Recursive chunks not found")

    if semantic_path.exists():
        chunks = load_chunks(str(semantic_path))
        results.append(analyze_chunks(chunks, "Semantic"))
        print(f"✅ Semantic: {len(chunks)} chunks loaded (20 sections only)")
    else:
        print("❌ Semantic chunks not found")

    if hierarchical_path.exists():
        chunks = load_chunks(str(hierarchical_path))
        results.append(analyze_chunks(chunks, "Hierarchical"))
        print(f"✅ Hierarchical: {len(chunks)} chunks loaded")
    else:
        print("❌ Hierarchical chunks not found")

    if results:
        print_comparison(results)