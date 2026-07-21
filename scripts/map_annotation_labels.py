"""
Build label mappings between annotators based on their "Document type"
annotations of a shared set of 5 PDFs, and report where mapping breaks down.

For each pair of annotators, labels are linked whenever they co-occur on the
same page of a shared PDF. Connected components of the resulting bipartite
graph are then classified as:

    1-to-1       : one label on each side
    1-to-many    : one label on one side, several on the other (the other
                   annotator subdivides that category)
    many-to-many : several labels on both sides -> agreement issue, the
                   annotations involved need to be reconciled

These pairwise mappings are then combined into a single mapping spanning all
all annotators. Labels that were never used on the shared PDFs (so have no
co-occurrence evidence) are still merged into a group if another annotator
used the exact same spelling (case/whitespace-insensitive) elsewhere in their
data; these groups are flagged as resting on "identical label" evidence only.
Whatever is left over - a label unique to one annotator with no matching
spelling anywhere else - is reported as unmapped.

Usage:
    python scripts/map_annotation_labels.py [--data-dir data/annotations] [--out-dir output]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx
import pandas as pd

from map_singletons import map_to_preferred


ANNOTATION_FILES = {
    "Marijke": "annotations_Marijke.tsv.gz",
    "Marijn": "annotations_Marijn.tsv.gz",
    "Rik": "annotations_Rik.tsv.gz",
    "Yeqian": "annotations_Yeqian.tsv.gz",
}

LABEL_COL = "Document type"
IMAGE_COL = "image path"
PAGE_COL = "page number"


def normalize_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Collapse case/whitespace variants of the same label into one canonical
    spelling (the most frequently used variant), so that e.g. "Approval
    notice" and "approval notice" are treated as one label instead of being
    mistaken for a genuine subdivision. Returns the updated df and a log of
    the variants that got collapsed, for transparency.
    """
    labels = df[LABEL_COL].dropna()
    key = labels.str.strip().str.casefold()
    canonical = {}
    variant_log = []
    for norm_key, group in labels.groupby(key):
        variants = group.value_counts()
        canonical_form = variants.idxmax()
        canonical[norm_key] = canonical_form
        if len(variants) > 1:
            variant_log.append({"canonical": canonical_form, "variants": dict(variants)})

    def apply_canonical(label):
        if pd.isna(label):
            return label
        return canonical[label.strip().casefold()]

    df = df.copy()
    df[LABEL_COL] = df[LABEL_COL].apply(apply_canonical)
    return df, variant_log


def load_annotations(data_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, list[dict]]]:
    dfs = {}
    variant_logs = {}
    for name, fname in ANNOTATION_FILES.items():
        df = pd.read_csv(data_dir / fname, sep="\t", dtype=str)
        df = df.dropna(subset=[IMAGE_COL])
        df = df[df[IMAGE_COL].str.strip() != ""]
        df = df[[IMAGE_COL, PAGE_COL, LABEL_COL]].copy()
        df, variant_log = normalize_labels(df)
        dfs[name] = df
        variant_logs[name] = variant_log
    return dfs, variant_logs


def get_shared_pdfs(dfs: dict[str, pd.DataFrame]) -> set[str]:
    pdf_sets = [set(df[IMAGE_COL].unique()) for df in dfs.values()]
    shared = set.intersection(*pdf_sets)
    return shared


def build_shared_table(dfs: dict[str, pd.DataFrame], shared_pdfs: set[str]) -> pd.DataFrame:
    """One row per (image path, page number) with a Document-type column per annotator."""
    merged = None
    for name, df in dfs.items():
        sub = df[df[IMAGE_COL].isin(shared_pdfs)].copy()
        sub = sub.rename(columns={LABEL_COL: f"label_{name}"})
        sub = sub.drop_duplicates(subset=[IMAGE_COL, PAGE_COL])
        if merged is None:
            merged = sub
        else:
            merged = merged.merge(sub, on=[IMAGE_COL, PAGE_COL], how="outer")
    return merged


def build_pair_graph(shared_table: pd.DataFrame, a: str, b: str) -> tuple[nx.Graph, pd.DataFrame]:
    """Bipartite co-occurrence graph + crosstab for a pair of annotators."""
    cols = [f"label_{a}", f"label_{b}"]
    pair_df = shared_table[cols].dropna()
    crosstab = pd.crosstab(pair_df[f"label_{a}"], pair_df[f"label_{b}"])

    g = nx.Graph()
    for x1 in crosstab.index:
        g.add_node((a, x1))
    for x2 in crosstab.columns:
        g.add_node((b, x2))
    for x1 in crosstab.index:
        for x2 in crosstab.columns:
            count = crosstab.loc[x1, x2]
            if count > 0:
                g.add_edge((a, x1), (b, x2), weight=int(count))
    return g, crosstab


def classify_components(g: nx.Graph, a: str, b: str) -> list[dict]:
    components = []
    for comp in nx.connected_components(g):
        side_a = sorted(n[1] for n in comp if n[0] == a)
        side_b = sorted(n[1] for n in comp if n[0] == b)
        if len(side_a) <= 1 and len(side_b) <= 1:
            kind = "1-to-1"
        elif len(side_a) == 1 and len(side_b) > 1:
            kind = "1-to-many"
        elif len(side_a) > 1 and len(side_b) == 1:
            kind = "many-to-1"
        elif len(side_a) > 1 and len(side_b) > 1:
            kind = "many-to-many"
        else:
            # one side empty -> label never co-occurred with a valid label
            # on the other side (e.g. missing Document type on that page)
            kind = "unmatched"
        components.append({"type": kind, a: side_a, b: side_b})
    return components


def edge_support(g: nx.Graph, comp: dict, a: str, b: str) -> list[tuple[str, str, int]]:
    edges = []
    for x1 in comp[a]:
        for x2 in comp[b]:
            if g.has_edge((a, x1), (b, x2)):
                edges.append((x1, x2, g[(a, x1)][(b, x2)]["weight"]))
    return edges


def unused_labels(dfs: dict[str, pd.DataFrame], shared_pdfs: set[str]) -> dict[str, set[str]]:
    """Labels each annotator used, but never on the 5 shared PDFs."""
    result = {}
    for name, df in dfs.items():
        all_labels = set(df[LABEL_COL].dropna().unique())
        shared_labels = set(df.loc[df[IMAGE_COL].isin(shared_pdfs), LABEL_COL].dropna().unique())
        result[name] = all_labels - shared_labels
    return result


def normalize_key(label: str) -> str:
    return label.strip().casefold()


def compute_label_counts(dfs: dict[str, pd.DataFrame]) -> dict[tuple[str, str], int]:
    """How often each annotator actually used each (canonical) label, across
    all of their annotated pages (shared and unique PDFs alike)."""
    counts = {}
    for name, df in dfs.items():
        for label, n in df[LABEL_COL].value_counts().items():
            counts[(name, label)] = int(n)
    return counts


def preferred_label(labels_by_annotator: dict[str, list[str]], label_counts: dict[tuple[str, str], int]) -> str:
    """The most frequently used exact spelling for a mapped group, summed
    across annotators - used as the controlled-vocabulary label to adopt."""
    text_counts: dict[str, int] = defaultdict(int)
    for annotator, labels in labels_by_annotator.items():
        for label in labels:
            text_counts[label] += label_counts.get((annotator, label), 0)
    return max(text_counts, key=text_counts.get)


def build_unified_graph(
    dfs: dict[str, pd.DataFrame], shared_table: pd.DataFrame, annotators: list[str]
) -> nx.Graph:
    """One graph spanning all annotators. Two kinds of edges:

    - "co-occurrence": labels used on the same page of a shared PDF (strong
      evidence, carries a page count as weight).
    - "identical-label": labels with the exact same spelling (after
      case/whitespace normalization) used by different annotators, even if
      that label was only ever seen outside the shared PDFs. This is weaker
      evidence (no shared page to confirm it), so it is tagged separately.
    """
    g = nx.Graph()
    for name, df in dfs.items():
        for label in df[LABEL_COL].dropna().unique():
            g.add_node((name, label))

    for a, b in combinations(annotators, 2):
        _, crosstab = build_pair_graph(shared_table, a, b)
        for x1 in crosstab.index:
            for x2 in crosstab.columns:
                count = crosstab.loc[x1, x2]
                if count > 0:
                    g.add_edge((a, x1), (b, x2), weight=int(count), source="co-occurrence")

    by_key = defaultdict(list)
    for name, df in dfs.items():
        for label in df[LABEL_COL].dropna().unique():
            by_key[normalize_key(label)].append((name, label))

    for nodes in by_key.values():
        for node1, node2 in combinations(nodes, 2):
            if node1[0] == node2[0]:
                continue
            if g.has_edge(node1, node2):
                g[node1][node2]["source"] += "+identical-label"
            else:
                g.add_edge(node1, node2, weight=0, source="identical-label")

    return g


def build_unified_mapping(g: nx.Graph, annotators: list[str]) -> list[dict]:
    groups = []
    for comp in nx.connected_components(g):
        by_annotator = {a: sorted(n[1] for n in comp if n[0] == a) for a in annotators}
        counts = [len(labels) for labels in by_annotator.values() if labels]
        n_annotators_present = sum(1 for labels in by_annotator.values() if labels)
        if max(counts, default=0) <= 1:
            kind = "1-to-1" if n_annotators_present == len(annotators) else "partial-1-to-1"
        elif sum(1 for c in counts if c > 1) == 1:
            subdividing = [a for a in annotators if len(by_annotator[a]) > 1]
            kind = f"1-to-many ({subdividing[0]} subdivides)"
        else:
            kind = "many-to-many (unresolved agreement issue)"

        has_identity_only = any(
            "co-occurrence" not in g[u][v]["source"]
            for u, v in combinations(comp, 2)
            if g.has_edge(u, v)
        )

        groups.append(
            {
                "type": kind,
                "labels": by_annotator,
                "identity_only_evidence": has_identity_only,
            }
        )
    return groups


def format_mapping(kind: str, a: str, side_a: list[str], b: str, side_b: list[str]) -> str:
    def fmt(labels):
        if len(labels) == 1:
            return labels[0]
        return "(" + ", ".join(labels) + ")"

    return f"  [{kind}] {a}: {fmt(side_a)}  <->  {b}: {fmt(side_b)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/annotations"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/labels"))
    args = parser.parse_args()

    dfs, variant_logs = load_annotations(args.data_dir)

    print("Case/whitespace label variants collapsed per annotator (treated as one label):")
    any_variants = False
    for name, log in variant_logs.items():
        for entry in log:
            any_variants = True
            variants = ", ".join(f"{v!r}:{n}" for v, n in entry["variants"].items())
            print(f"  {name}: {entry['canonical']!r}  <-  {variants}")
    if not any_variants:
        print("  (none)")
    print()

    shared_pdfs = get_shared_pdfs(dfs)
    print(f"Shared PDFs used for mapping ({len(shared_pdfs)}):")
    for pdf in sorted(shared_pdfs):
        print(f"  - {pdf}")
    print()

    shared_table = build_shared_table(dfs, shared_pdfs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_mapping_rows = []
    all_agreement_issue_rows = []

    annotators = list(ANNOTATION_FILES.keys())
    for a, b in combinations(annotators, 2):
        print("=" * 70)
        print(f"Pair: {a} <-> {b}")
        print("=" * 70)

        g, crosstab = build_pair_graph(shared_table, a, b)
        components = classify_components(g, a, b)

        counts = {"1-to-1": 0, "1-to-many": 0, "many-to-1": 0, "many-to-many": 0, "unmatched": 0}
        for comp in components:
            counts[comp["type"]] += 1

        print(
            f"{counts['1-to-1']} one-to-one, {counts['1-to-many']} one-to-many, "
            f"{counts['many-to-1']} many-to-one, {counts['many-to-many']} many-to-many, "
            f"{counts['unmatched']} unmatched"
        )
        print()

        for comp in sorted(components, key=lambda c: (c["type"] != "many-to-many", c[a])):
            print(format_mapping(comp["type"], a, comp[a], b, comp[b]))
            edges = edge_support(g, comp, a, b)
            support = ", ".join(f"{x1}~{x2}:{n}" for x1, x2, n in sorted(edges))
            if support:
                print(f"      support: {support}")

            all_mapping_rows.append(
                {
                    "annotator_1": a,
                    "annotator_2": b,
                    "type": comp["type"],
                    f"labels_{a}": "; ".join(comp[a]),
                    f"labels_{b}": "; ".join(comp[b]),
                }
            )
            if comp["type"] == "many-to-many":
                all_agreement_issue_rows.append(
                    {
                        "annotator_1": a,
                        "annotator_2": b,
                        f"labels_{a}": "; ".join(comp[a]),
                        f"labels_{b}": "; ".join(comp[b]),
                    }
                )
        print()

    print("=" * 70)
    print(f"Unified mapping across all {len(annotators)} annotators")
    print("=" * 70)
    print(
        "Combines the shared-PDF co-occurrence evidence with a second, weaker\n"
        "rule: labels that different annotators spelled identically (even if\n"
        "never seen on a shared page) are also merged. Groups relying only on\n"
        "that second rule are marked '(identical label only, no shared-PDF\n"
        "evidence)'.\n"
    )

    g = build_unified_graph(dfs, shared_table, annotators)
    groups = build_unified_mapping(g, annotators)
    label_counts = compute_label_counts(dfs)

    # Multi-annotator groups (the actual mapping); singletons are handled below.
    mapped_groups = [gr for gr in groups if sum(len(v) for v in gr["labels"].values()) > 1]
    singleton_groups = [gr for gr in groups if sum(len(v) for v in gr["labels"].values()) == 1]
    unified_rows = []
    for gr in sorted(mapped_groups, key=lambda x: ("many-to-many" not in x["type"], str(x["labels"]))):
        parts = []
        for a in annotators:
            labels = gr["labels"][a]
            if not labels:
                continue
            shown = labels[0] if len(labels) == 1 else "(" + ", ".join(labels) + ")"
            parts.append(f"{a}: {shown}")
        note = "  [identical label only, no shared-PDF evidence]" if gr["identity_only_evidence"] else ""
        preferred = preferred_label(gr["labels"], label_counts)
        print(f"  [{gr['type']}] " + "  <->  ".join(parts) + f"  =>  preferred: {preferred}" + note)

        row = {"type": gr["type"], "preferred_label": preferred, "identity_only_evidence": gr["identity_only_evidence"]}
        for a in annotators:
            row[f"labels_{a}"] = "; ".join(gr["labels"][a])
        unified_rows.append(row)
    print()

    remaining_agreement_issues = [gr for gr in mapped_groups if "many-to-many" in gr["type"]]
    if remaining_agreement_issues:
        print(f"WARNING: {len(remaining_agreement_issues)} unresolved many-to-many group(s) remain.")
    else:
        print("No many-to-many (agreement issue) groups remain.")
    print()

    print("=" * 70)
    print("Labels that cannot be mapped (unique to one annotator: not used on")
    print("the shared PDFs, and not spelled identically by another annotator)")
    print("=" * 70)
    final_unmapped_rows = []
    for a in annotators:
        labels = sorted(
            gr["labels"][a][0] for gr in singleton_groups if gr["labels"][a]
        )
        print(f"{a} ({len(labels)}):")
        for label in labels:
            print(f"  - {label}")
            final_unmapped_rows.append({"annotator": a, "label": label})
        print()

    mapping_df = pd.DataFrame(all_mapping_rows)
    mapping_df.to_csv(args.out_dir / "label_mappings_pairwise.tsv", index=False, sep='\t')

    agreement_df = pd.DataFrame(all_agreement_issue_rows)
    agreement_df.to_csv(args.out_dir / "agreement_issues_pairwise.tsv", index=False, sep='\t')

    unified_df = pd.DataFrame(unified_rows)
    unified_df.to_csv(args.out_dir / "label_mapping_unified.tsv", index=False, sep='\t')

    unmapped_df = pd.DataFrame(final_unmapped_rows)
    unmapped_df.to_csv(args.out_dir / "unmapped_labels.tsv", index=False, sep='\t')

    unmapped_df['preferred_label'] = unmapped_df.label.apply(map_to_preferred)
    unmapped_df.to_csv(args.out_dir / "mapped_single_annotator_labels.tsv", index=False, sep='\t')


    print(f"Wrote {args.out_dir / 'label_mappings_pairwise.tsv'}")
    print(f"Wrote {args.out_dir / 'agreement_issues_pairwise.tsv'}")
    print(f"Wrote {args.out_dir / 'label_mapping_unified.tsv'}")
    print(f"Wrote {args.out_dir / 'unmapped_labels.tsv'}")


if __name__ == "__main__":
    main()
