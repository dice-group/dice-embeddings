import random
import networkx as nx


# -----------------------------
# 1) Loading triples from a file
# -----------------------------

def load_triples(path):
    """
    Each line:  h r t   (space-separated)
    Only the first three tokens are used.
    """
    triples = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            h, r, t = parts[:3]
            triples.append((h, r, t))
    return triples


# -----------------------------
# 2) Build a graph from triples
# -----------------------------

def triples_to_graph(triples, directed=False):
    """
    Build a graph from (h, r, t) triples.
    Nodes: entities (h and t)
    Edges: between h and t (relation r stored as edge attribute, but
           for modularity / clustering we only care about the endpoints).
    """
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    for h, r, t in triples:
        if G.has_edge(h, t):
            # If multiple relations: store as a set (optional)
            G[h][t].setdefault("relations", set()).add(r)
        else:
            G.add_edge(h, t, relation=r)

    return G


# ----------------------------------------------
# 3) Heuristics on modularity & clustering
# ----------------------------------------------

def is_intra(u, v, node2comm):
    return node2comm.get(u) == node2comm.get(v)


def triangle_count(u, v, G):
    """Number of common neighbors between u and v."""
    Nu = set(G.neighbors(u))
    Nv = set(G.neighbors(v))
    return len(Nu & Nv)


def improve_modularity_reduce_clustering(
    G,
    budget=20,
    max_candidates_per_step=50,
    seed=None,
):
    """
    Heuristic algorithm that tries to increase modularity and reduce
    the average clustering coefficient of a given graph by edge rewiring.
    """
    rng = random.Random(seed)
    G = G.copy()
    G = G.to_undirected()

    # --- Step 1: initial community detection (fixed partition) ---
    communities = list(nx.algorithms.community.greedy_modularity_communities(G))
    node2comm = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            node2comm[node] = cid

    # Compute initial metrics
    if len(communities) > 0 and G.number_of_edges() > 0:
        Q = nx.algorithms.community.modularity(G, communities)
    else:
        Q = float("nan")
    C = nx.average_clustering(G) if G.number_of_edges() > 0 else float("nan")

    history = [{"step": 0, "Q": Q, "C": C}]

    for step in range(1, budget + 1):
        improved = False

        edges = list(G.edges())
        if not edges:
            break

        # Classify edges into inter- and intra-community
        inter_edges = [(u, v) for (u, v) in edges if not is_intra(u, v, node2comm)]
        intra_edges = [(u, v) for (u, v) in edges if is_intra(u, v, node2comm)]

        # --- Move Type A: Rewire inter-community edge to intra-community edge ---
        if inter_edges:
            rng.shuffle(inter_edges)
            inter_edges = inter_edges[:max_candidates_per_step]

            for (u, v) in inter_edges:
                # Try keeping u or v as the "anchor"
                for a, b in ((u, v), (v, u)):
                    comm_a = node2comm.get(a)
                    # candidates in same community as a, not already connected, not b
                    cand_nodes = [
                        w for w in G.nodes()
                        if node2comm.get(w) == comm_a
                        and w not in (a, b)
                        and not G.has_edge(a, w)
                    ]
                    if not cand_nodes:
                        continue

                    # Prefer nodes that will create few triangles
                    cand_nodes.sort(key=lambda w: triangle_count(a, w, G))
                    # Try a few best candidates
                    for w in cand_nodes[:5]:
                        # Propose: remove (a, b), add (a, w)
                        G.remove_edge(a, b)
                        G.add_edge(a, w)

                        Q_new = nx.algorithms.community.modularity(G, communities) \
                            if G.number_of_edges() > 0 else float("nan")
                        C_new = nx.average_clustering(G) if G.number_of_edges() > 0 else float("nan")

                        # Enforce: modularity not worse, clustering not higher
                        if (not (Q_new != Q and (Q_new < Q))) and (C_new <= C):
                            Q, C = Q_new, C_new
                            history.append({"step": step, "Q": Q, "C": C})
                            improved = True
                            break
                        else:
                            # revert
                            G.remove_edge(a, w)
                            G.add_edge(a, b)

                    if improved:
                        break
                if improved:
                    break

        # --- If no success, try Move Type B: intra edge rewiring to reduce triangles ---
        if not improved and intra_edges:
            # sort intra edges by how many triangles they participate in (high first)
            intra_edges_sorted = sorted(
                intra_edges,
                key=lambda e: triangle_count(e[0], e[1], G),
                reverse=True,
            )
            intra_edges_sorted = intra_edges_sorted[:max_candidates_per_step]

            for (u, v) in intra_edges_sorted:
                a, b = u, v
                comm_a = node2comm.get(a)

                cand_nodes = [
                    w for w in G.nodes()
                    if node2comm.get(w) == comm_a
                    and w not in (a, b)
                    and not G.has_edge(a, w)
                ]
                if not cand_nodes:
                    continue

                # Prefer nodes that create fewer triangles
                cand_nodes.sort(key=lambda w: triangle_count(a, w, G))

                for w in cand_nodes[:5]:
                    # Propose: remove (a, b), add (a, w)
                    G.remove_edge(a, b)
                    G.add_edge(a, w)

                    Q_new = nx.algorithms.community.modularity(G, communities) \
                        if G.number_of_edges() > 0 else float("nan")
                    C_new = nx.average_clustering(G) if G.number_of_edges() > 0 else float("nan")

                    # Here we tolerate tiny changes in Q if C improves
                    if (Q_new >= Q and C_new <= C) or (C_new < C and (Q_new + 1e-6) >= Q):
                        Q, C = Q_new, C_new
                        history.append({"step": step, "Q": Q, "C": C})
                        improved = True
                        break
                    else:
                        # revert
                        G.remove_edge(a, w)
                        G.add_edge(a, b)

                if improved:
                    break

        # If no improving move found, stop early
        if not improved:
            break

    return G, history


def reduce_modularity_increase_clustering(
    G,
    budget=20,
    max_candidates_per_step=50,
    seed=None,
):
    """
    Heuristic algorithm that tries to REDUCE modularity and INCREASE
    the average clustering coefficient of a given graph by edge rewiring.
    """
    rng = random.Random(seed)
    G = G.copy()
    G = G.to_undirected()

    # --- Step 1: initial community detection (fixed partition) ---
    communities = list(nx.algorithms.community.greedy_modularity_communities(G))
    node2comm = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            node2comm[node] = cid

    # Compute initial metrics
    if len(communities) > 0 and G.number_of_edges() > 0:
        Q = nx.algorithms.community.modularity(G, communities)
    else:
        Q = float("nan")
    C = nx.average_clustering(G) if G.number_of_edges() > 0 else float("nan")

    history = [{"step": 0, "Q": Q, "C": C}]

    for step in range(1, budget + 1):
        improved = False

        edges = list(G.edges())
        if not edges:
            break

        # Classify edges into intra- and inter-community
        intra_edges = [(u, v) for (u, v) in edges if is_intra(u, v, node2comm)]
        inter_edges = [(u, v) for (u, v) in edges if not is_intra(u, v, node2comm)]

        # -------------------------------------------------
        # Move Type A: Rewire intra edge -> inter edge
        # Goal: lower modularity (mix communities), but try to raise clustering
        # -------------------------------------------------
        if intra_edges:
            rng.shuffle(intra_edges)
            intra_edges = intra_edges[:max_candidates_per_step]

            for (u, v) in intra_edges:
                # Try keeping u or v as the "anchor"
                for a, b in ((u, v), (v, u)):
                    comm_a = node2comm.get(a)

                    # Candidates in a *different* community than a, not already connected
                    cand_nodes = [
                        w for w in G.nodes()
                        if node2comm.get(w) != comm_a
                        and w not in (a, b)
                        and not G.has_edge(a, w)
                    ]
                    if not cand_nodes:
                        continue

                    # Prefer nodes that will create *more* triangles with a
                    cand_nodes.sort(key=lambda w: -triangle_count(a, w, G))

                    # Try a few best candidates
                    for w in cand_nodes[:5]:
                        # Propose: remove (a, b) [intra], add (a, w) [inter]
                        G.remove_edge(a, b)
                        G.add_edge(a, w)

                        Q_new = nx.algorithms.community.modularity(G, communities) \
                            if G.number_of_edges() > 0 else float("nan")
                        C_new = nx.average_clustering(G) if G.number_of_edges() > 0 else float("nan")

                        # We want: modularity not higher, clustering not lower
                        if (Q_new <= Q + 1e-9) and (C_new >= C - 1e-9):
                            Q, C = Q_new, C_new
                            history.append({"step": step, "Q": Q, "C": C})
                            improved = True
                            break
                        else:
                            # revert
                            G.remove_edge(a, w)
                            G.add_edge(a, b)

                    if improved:
                        break
                if improved:
                    break

        # -------------------------------------------------
        # Move Type B: Rewire any edge to increase triangles
        # Goal: raise clustering; allow small modularity changes but
        #       try not to increase Q much.
        # -------------------------------------------------
        if not improved and edges:
            # Focus on edges that currently participate in *few* triangles
            edges_sorted = sorted(
                edges,
                key=lambda e: triangle_count(e[0], e[1], G),
            )
            edges_sorted = edges_sorted[:max_candidates_per_step]

            for (u, v) in edges_sorted:
                a, b = u, v

                # Candidates that are not neighbors of a yet
                cand_nodes = [
                    w for w in G.nodes()
                    if w not in (a, b) and not G.has_edge(a, w)
                ]
                if not cand_nodes:
                    continue

                # Prefer nodes that create *more* triangles with a
                cand_nodes.sort(key=lambda w: -triangle_count(a, w, G))

                for w in cand_nodes[:5]:
                    # Propose: remove (a, b), add (a, w)
                    G.remove_edge(a, b)
                    G.add_edge(a, w)

                    Q_new = nx.algorithms.community.modularity(G, communities) \
                        if G.number_of_edges() > 0 else float("nan")
                    C_new = nx.average_clustering(G) if G.number_of_edges() > 0 else float("nan")

                    # Here we prioritize increasing clustering,
                    # and we try not to raise modularity too much.
                    if (C_new > C + 1e-9) and (Q_new <= Q + 1e-6):
                        Q, C = Q_new, C_new
                        history.append({"step": step, "Q": Q, "C": C})
                        improved = True
                        break
                    else:
                        # revert
                        G.remove_edge(a, w)
                        G.add_edge(a, b)

                if improved:
                    break

        # If no improving move found, stop early
        if not improved:
            break

    return G, history


# ----------------------------------------------
# 4) Compute removed / added / remained parts
# ----------------------------------------------

def compute_diff_triples(original_triples, G_original, G_new):
    """
    Compare original vs new graph and classify triples as:
      - removed_triples: triples whose (h, t) edge no longer exists
      - remained_triples: triples whose (h, t) edge still exists
      - added_edges: new (u, v) edges that had no original triple
    """
    # Edge sets as undirected pairs
    orig_edges = {frozenset((u, v)) for u, v in G_original.to_undirected().edges()}
    new_edges = {frozenset((u, v)) for u, v in G_new.to_undirected().edges()}

    removed_triples = []
    remained_triples = []

    for h, r, t in original_triples:
        e = frozenset((h, t))
        if e in new_edges:
            remained_triples.append((h, r, t))
        else:
            removed_triples.append((h, r, t))

    added_edge_pairs = new_edges - orig_edges
    added_edges = []
    for e in added_edge_pairs:
        u, v = tuple(e)
        added_edges.append((u, v))

    return removed_triples, remained_triples, added_edges


# ----------------------------------------------
# 5) Example: wire it all together & save outputs
# ----------------------------------------------

if __name__ == "__main__":

    DB = "NELL-995-h100"
    path = f"./KGs/{DB}/train.txt"

    # Load triples
    triples = load_triples(path)
    print(f"Loaded {len(triples)} triples.")

    # Build original graph (undirected)
    G_orig = triples_to_graph(triples, directed=False)
    print(f"Original graph: {G_orig.number_of_nodes()} nodes, {G_orig.number_of_edges()} edges")

    # Initial metrics
    if G_orig.number_of_edges() > 0:
        comms0 = list(nx.algorithms.community.greedy_modularity_communities(G_orig))
        Q0 = nx.algorithms.community.modularity(G_orig, comms0) if comms0 else float("nan")
        C0 = nx.average_clustering(G_orig)
    else:
        Q0, C0 = float("nan"), float("nan")

    print(f"Original modularity Q0 = {Q0:.6f}")
    print(f"Original clustering  C0 = {C0:.6f}")

    # === Choose which heuristic to apply ===
    # For *increasing* modularity & *reducing* clustering:
    # G_new, history = improve_modularity_reduce_clustering(
    #     G_orig,
    #     budget=1000,
    #     max_candidates_per_step=50,
    #     seed=0,
    # )

    # For *reducing* modularity & *increasing* clustering:
    G_new, history = reduce_modularity_increase_clustering(
        G_orig,
        budget=1000,
        max_candidates_per_step=50,
        seed=0,
    )

    # New metrics
    if G_new.number_of_edges() > 0:
        comms1 = list(nx.algorithms.community.greedy_modularity_communities(G_new))
        Q1 = nx.algorithms.community.modularity(G_new, comms1) if comms1 else float("nan")
        C1 = nx.average_clustering(G_new)
    else:
        Q1, C1 = float("nan"), float("nan")

    print(f"\nNew modularity Q1 = {Q1:.6f}")
    print(f"New clustering  C1 = {C1:.6f}")

    # Compute removed / remained / added
    removed_triples, remained_triples, added_edges = compute_diff_triples(
        triples, G_orig, G_new
    )

    print(f"\nRemoved triples:  {len(removed_triples)}")
    print(f"Remained triples: {len(remained_triples)}")
    print(f"Added edges:      {len(added_edges)}")

    # Save to files
    with open(f"./modu/{DB}/removed.txt", "w") as f:
        for h, r, t in removed_triples:
            f.write(f"{h} {r} {t}\n")

    with open(f"./modu/{DB}/train.txt", "w") as f:
        for h, r, t in remained_triples:
            f.write(f"{h} {r} {t}\n")

    with open(f"./modu/{DB}/added.txt", "w") as f:
        for u, v in added_edges:
            f.write(f"{u} {v}\n")

    print("\nDiff saved to:")
    print(f"  ./modu/{DB}/removed.txt")
    print(f"  ./modu/{DB}/train.txt")
    print(f"  ./modu/{DB}/added.txt")
