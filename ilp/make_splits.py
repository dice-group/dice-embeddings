"""Generate transductive / inductive / semi-inductive splits from a triples file.

The output layouts match exactly what ``ilp.train`` / ``ilp.eval`` already
consume (see TRIPLE_FORMATS and the ``--obs-file`` flow in eval.py):

  transductive    KGs/{name}_transductive/{train,valid,test}.txt
                  every entity & relation in valid/test also appears in train.

  inductive       KGs/{name}_inductive/{train,valid}.txt        (entity set A)
                  KGs/{name}_inductive_ind/{train,test}.txt     (entity set B, A∩B=∅)
                  Train on graph A; at eval the queries live on a disjoint
                  inference graph B whose observed part is *_ind/train.txt.

  semi_inductive  KGs/{name}_semi_inductive/{train,valid,test,context}.txt
                  each valid/test triple has exactly one unseen endpoint; train
                  has none. context.txt = train + held-out observed triples of
                  the unseen entities (pass it as --obs-file at eval time).

Usage:
    python -m ilp.make_splits --triples ilp/kg_triples.tsv --name kg --regime all

Then train (auto-evaluates the test split when training finishes), e.g.:
    python -m ilp.train --config ilp/configs/kg_transductive.yaml --save runs/kg.pt

Standalone eval of a saved bundle (transductive: rank test against itself):
    python -m ilp.eval --model <bundle> \
        --obs-file KGs/kg_transductive/test.txt \
        --test-file KGs/kg_transductive/test.txt \
        --filter-file KGs/kg_transductive/train.txt KGs/kg_transductive/valid.txt

Eval (inductive: observed context = the disjoint inference graph's train.txt):
    python -m ilp.eval --model <bundle> \
        --obs-file  KGs/kg_inductive_ind/train.txt \
        --test-file KGs/kg_inductive_ind/test.txt

Eval (semi-inductive, with observed context):
    python -m ilp.eval --model <bundle> \
        --obs-file  KGs/kg_semi_inductive/context.txt \
        --test-file KGs/kg_semi_inductive/test.txt \
        --filter-file KGs/kg_semi_inductive/train.txt KGs/kg_semi_inductive/valid.txt
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from .dataset import TRIPLE_FORMATS, Triple

# Header tokens we skip if the first row of the source file is a header.
_HEADER_TOKENS = {"head", "relation", "tail", "h", "r", "t",
                  "subject", "predicate", "object"}


def read_triples_skip_header(path: str | Path, fmt: str) -> list[Triple]:
    """Like dataset.read_triples but tolerant of a single header row.

    Returns canonical (head, relation, tail) triples.
    """
    h_idx, r_idx, t_idx = TRIPLE_FORMATS[fmt]
    out: list[Triple] = []
    for i, line in enumerate(Path(path).read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"Bad triple line {i}: {line!r}")
        if i == 0 and all(p.lower() in _HEADER_TOKENS for p in parts):
            continue  # header row
        out.append((parts[h_idx], parts[r_idx], parts[t_idx]))
    return out


def entities_of(triples: Iterable[Triple]) -> set[str]:
    ents: set[str] = set()
    for h, _, t in triples:
        ents.add(h)
        ents.add(t)
    return ents


def relations_of(triples: Iterable[Triple]) -> set[str]:
    return {r for _, r, _ in triples}


def write_triples(path: Path, triples: Sequence[Triple]) -> None:
    """Write triples as ``head <TAB> relation <TAB> tail`` (head_relation_tail)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{h}\t{r}\t{t}\n" for h, r, t in triples))


# --- transductive ---------------------------------------------------------

def split_transductive(
    triples: list[Triple], ratios: tuple[float, float, float], rng: random.Random,
) -> dict[str, list[Triple]]:
    """Random triple split, then repair coverage so every valid/test entity and
    relation is present in train (otherwise the model has no token for it)."""
    _, valid_r, test_r = ratios
    pool = list(triples)
    rng.shuffle(pool)
    n = len(pool)
    n_test = round(n * test_r)
    n_valid = round(n * valid_r)
    test = pool[:n_test]
    valid = pool[n_test:n_test + n_valid]
    train = pool[n_test + n_valid:]

    train_ents = entities_of(train)
    train_rels = relations_of(train)

    def repair(split: list[Triple]) -> list[Triple]:
        kept: list[Triple] = []
        for tr in split:
            h, r, t = tr
            if h in train_ents and t in train_ents and r in train_rels:
                kept.append(tr)
            else:
                train.append(tr)  # demote to train to preserve coverage
                train_ents.update((h, t))
                train_rels.add(r)
        return kept

    valid = repair(valid)
    test = repair(test)
    return {"train": train, "valid": valid, "test": test}


# --- fully inductive (disjoint train / inference entity sets) -------------

def split_inductive(
    triples: list[Triple],
    ratios: tuple[float, float, float],
    entity_frac: float,
    obs_frac: float,
    rng: random.Random,
) -> dict[str, dict[str, list[Triple]]]:
    """Partition entities into disjoint A (train graph) and B (inference graph).

    Train graph = triples with both endpoints in A.
    Inference graph = triples with both endpoints in B, restricted to relations
    seen in A (the model has no embedding for B-only relations). Cross-set edges
    are dropped. The inference graph is split into observed/query parts.
    """
    _, valid_r, test_r = ratios
    ents = sorted(entities_of(triples))
    rng.shuffle(ents)
    n_b = max(1, round(len(ents) * entity_frac))
    set_b = set(ents[:n_b])

    train_graph: list[Triple] = []
    inf_graph: list[Triple] = []
    dropped_cross = 0
    for h, r, t in triples:
        hb, tb = h in set_b, t in set_b
        if not hb and not tb:
            train_graph.append((h, r, t))
        elif hb and tb:
            inf_graph.append((h, r, t))
        else:
            dropped_cross += 1

    train_rels = relations_of(train_graph)
    inf_kept = [tr for tr in inf_graph if tr[1] in train_rels]
    dropped_rel = len(inf_graph) - len(inf_kept)

    # main dir: train graph split into train/valid (transductive holdout over A)
    rng.shuffle(train_graph)
    n_v = round(len(train_graph) * valid_r)
    main_valid = train_graph[:n_v]
    main_train = train_graph[n_v:]
    # repair valid coverage against main_train
    mt_ents, mt_rels = entities_of(main_train), relations_of(main_train)
    kept_valid: list[Triple] = []
    for tr in main_valid:
        h, r, t = tr
        if h in mt_ents and t in mt_ents and r in mt_rels:
            kept_valid.append(tr)
        else:
            main_train.append(tr)
    main_valid = kept_valid

    # _ind dir: inference graph split into observed (train.txt) and query (test.txt)
    rng.shuffle(inf_kept)
    n_obs = round(len(inf_kept) * obs_frac)
    ind_train = inf_kept[:n_obs]   # observed context
    ind_test = inf_kept[n_obs:]    # ranking queries

    return {
        "main": {"train": main_train, "valid": main_valid},
        "ind": {"train": ind_train, "test": ind_test},
        "_stats": {"dropped_cross": dropped_cross, "dropped_rel": dropped_rel,
                   "|A|": len(ents) - n_b, "|B|": n_b},
    }


# --- semi-inductive -------------------------------------------------------

def split_semi_inductive(
    triples: list[Triple],
    ratios: tuple[float, float, float],
    entity_frac: float,
    obs_frac: float,
    rng: random.Random,
) -> dict[str, list[Triple]]:
    """Designate a fraction of entities as 'unseen' (U).

    train   = triples with no endpoint in U.
    boundary= triples with exactly one endpoint in U (the eval material).
    (triples with both endpoints in U are dropped — not "exactly one".)

    Each unseen entity's boundary triples are split: a fraction become observed
    context (kept in context.txt), the rest become valid/test queries. Only
    queries whose relation is seen in train are kept.
    """
    _, valid_r, test_r = ratios
    ents = sorted(entities_of(triples))
    rng.shuffle(ents)
    n_u = max(1, round(len(ents) * entity_frac))
    set_u = set(ents[:n_u])

    train: list[Triple] = []
    dropped_both = 0
    boundary_by_unseen: dict[str, list[Triple]] = {e: [] for e in set_u}
    for h, r, t in triples:
        hu, tu = h in set_u, t in set_u
        if not hu and not tu:
            train.append((h, r, t))
        elif hu and tu:
            dropped_both += 1
        else:
            boundary_by_unseen[h if hu else t].append((h, r, t))

    train_rels = relations_of(train)
    train_ents = entities_of(train)

    observed: list[Triple] = []
    queries: list[Triple] = []
    dropped_rel = 0
    for e, trs in boundary_by_unseen.items():
        rng.shuffle(trs)
        n_obs = int(len(trs) * obs_frac)
        for i, tr in enumerate(trs):
            h, r, t = tr
            if r not in train_rels:
                dropped_rel += 1
                continue
            # The non-unseen endpoint must genuinely appear in train, else this
            # is effectively a two-unseen-entity triple (no seen anchor). Keep
            # such triples as observed context rather than as queries.
            seen_end = t if e == h else h
            if i < n_obs or seen_end not in train_ents:
                observed.append(tr)
            else:
                queries.append(tr)

    rng.shuffle(queries)
    q_total = valid_r + test_r
    n_valid = round(len(queries) * (valid_r / q_total)) if q_total else 0
    valid = queries[:n_valid]
    test = queries[n_valid:]

    context = train + observed
    return {"train": train, "valid": valid, "test": test, "context": context,
            "_stats": {"dropped_both": dropped_both, "dropped_rel": dropped_rel,
                       "|U|": n_u, "|observed|": len(observed)}}


# --- config emission ------------------------------------------------------

def emit_config(name: str, regime: str, data_dir: str, configs_dir: Path) -> Path:
    """Write a ready-to-train YAML based on configs/default.yaml."""
    default = yaml.safe_load((configs_dir / "default.yaml").read_text())
    default["data_dir"] = data_dir
    default["triple_format"] = "head_relation_tail"
    default["run_name"] = f"{name}_{regime}"
    # kg_triples.tsv has rdf:type (17 Klasse:* tails) — a clean schema set worth
    # promoting to [VAL_*] tokens. Inherited from default.yaml; override if needed.
    default["type_relation"] = "rdf:type"
    out = configs_dir / f"{name}_{regime}.yaml"
    out.write_text(yaml.safe_dump(default, sort_keys=False))
    return out


# --- verification ---------------------------------------------------------

def _verify_transductive(s: dict[str, list[Triple]]) -> list[str]:
    te = entities_of(s["train"]); tr = relations_of(s["train"])
    issues = []
    for k in ("valid", "test"):
        for h, r, t in s[k]:
            if h not in te or t not in te or r not in tr:
                issues.append(f"{k} triple references unseen ent/rel: {(h, r, t)}")
                break
    return issues


def _verify_inductive(s: dict) -> list[str]:
    a = entities_of(s["main"]["train"]) | entities_of(s["main"]["valid"])
    b = entities_of(s["ind"]["train"]) | entities_of(s["ind"]["test"])
    issues = []
    overlap = a & b
    if overlap:
        issues.append(f"A∩B nonempty: {len(overlap)} shared entities")
    a_rels = relations_of(s["main"]["train"])
    for h, r, t in s["ind"]["test"] + s["ind"]["train"]:
        if r not in a_rels:
            issues.append(f"inference relation {r!r} unseen in train graph")
            break
    return issues


def _verify_semi(s: dict) -> list[str]:
    seen = entities_of(s["train"])
    issues = []
    for k in ("valid", "test"):
        for h, r, t in s[k]:
            hu, tu = h not in seen, t not in seen
            if hu == tu:  # both seen or both unseen — violates "exactly one"
                issues.append(f"{k} triple lacks exactly one unseen endpoint: {(h, r, t)}")
                break
    return issues


# --- driver ---------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    triples = read_triples_skip_header(args.triples, args.triple_format)
    ratios = tuple(args.ratios)  # train, valid, test
    print(f"Loaded {len(triples)} triples, "
          f"{len(entities_of(triples))} entities, "
          f"{len(relations_of(triples))} relations from {args.triples}")
    out_root = Path(args.out_root)
    configs_dir = Path(args.configs_dir)
    regimes = (["transductive", "inductive", "semi_inductive"]
               if args.regime == "all" else [args.regime])

    for regime in regimes:
        print(f"\n=== {regime} ===")
        if regime == "transductive":
            s = split_transductive(triples, ratios, random.Random(args.seed))
            d = out_root / f"{args.name}_transductive"
            for k in ("train", "valid", "test"):
                write_triples(d / f"{k}.txt", s[k])
                print(f"  {k}.txt: {len(s[k])}")
            issues = _verify_transductive(s)
            cfg_dir = str(d)

        elif regime == "inductive":
            s = split_inductive(triples, ratios, args.ind_entity_frac,
                                args.obs_frac, random.Random(args.seed))
            d = out_root / f"{args.name}_inductive"
            d_ind = out_root / f"{args.name}_inductive_ind"
            for k in ("train", "valid"):
                write_triples(d / f"{k}.txt", s["main"][k])
                print(f"  {d.name}/{k}.txt: {len(s['main'][k])}")
            for k in ("train", "test"):
                write_triples(d_ind / f"{k}.txt", s["ind"][k])
                print(f"  {d_ind.name}/{k}.txt: {len(s['ind'][k])}")
            st = s["_stats"]
            print(f"  |A|={st['|A|']} |B|={st['|B|']} "
                  f"dropped_cross={st['dropped_cross']} dropped_rel={st['dropped_rel']}")
            issues = _verify_inductive(s)
            cfg_dir = str(d)

        else:  # semi_inductive
            s = split_semi_inductive(triples, ratios, args.semi_entity_frac,
                                     args.obs_frac, random.Random(args.seed))
            d = out_root / f"{args.name}_semi_inductive"
            for k in ("train", "valid", "test", "context"):
                write_triples(d / f"{k}.txt", s[k])
                print(f"  {k}.txt: {len(s[k])}")
            st = s["_stats"]
            print(f"  |U|={st['|U|']} |observed|={st['|observed|']} "
                  f"dropped_both={st['dropped_both']} dropped_rel={st['dropped_rel']}")
            issues = _verify_semi(s)
            cfg_dir = str(d)

        print("  verify: " + ("OK" if not issues else "; ".join(issues)))
        if args.emit_config:
            cfg_path = emit_config(args.name, regime, cfg_dir, configs_dir)
            print(f"  config: {cfg_path}  (type_relation=rdf:type; set device as needed)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--triples", required=True, help="Source triples file (e.g. ilp/kg_triples.tsv)")
    ap.add_argument("--name", default="kg", help="Dataset name prefix for output dirs/configs")
    ap.add_argument("--regime", default="all",
                    choices=["all", "transductive", "inductive", "semi_inductive"])
    ap.add_argument("--out-root", default="KGs", help="Root dir for output splits")
    ap.add_argument("--configs-dir", default=str(Path(__file__).parent / "configs"))
    ap.add_argument("--triple-format", default="head_relation_tail",
                    choices=list(TRIPLE_FORMATS), help="Column layout of the source file")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                    metavar=("TRAIN", "VALID", "TEST"))
    ap.add_argument("--ind-entity-frac", type=float, default=0.15,
                    help="Fraction of entities placed in the disjoint inference graph (inductive)")
    ap.add_argument("--semi-entity-frac", type=float, default=0.10,
                    help="Fraction of entities marked unseen (semi-inductive)")
    ap.add_argument("--obs-frac", type=float, default=0.5,
                    help="Fraction of inference/boundary triples kept as observed context")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-config", dest="emit_config", action="store_false",
                    help="Do not emit ready-to-train YAML configs")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
