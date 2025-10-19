import random
import math
moduleCapacity = 1  # M

links = {
    1: {"A": 1, "Z": 2, "modules": 4, "cost": 1},
    2: {"A": 1, "Z": 3, "modules": 4, "cost": 1},
    3: {"A": 2, "Z": 3, "modules": 2, "cost": 1},
    4: {"A": 2, "Z": 4, "modules": 4, "cost": 1},
    5: {"A": 3, "Z": 4, "modules": 4, "cost": 1},
}

demands = {
    1: {"A": 1, "Z": 2, "volume": 3, "paths": 3},
    2: {"A": 1, "Z": 3, "volume": 4, "paths": 3},
    3: {"A": 1, "Z": 4, "volume": 5, "paths": 2},
    4: {"A": 2, "Z": 3, "volume": 2, "paths": 3},
    5: {"A": 2, "Z": 4, "volume": 3, "paths": 3},
    6: {"A": 3, "Z": 4, "volume": 4, "paths": 3},
}

# (d, p) -> lista krawędzi
demand_paths = {
    (1, 1): [1],
    (1, 2): [2, 3],
    (1, 3): [2, 4, 5],
    (2, 1): [2],
    (2, 2): [1, 3],
    (2, 3): [1, 4, 5],
    (3, 1): [1, 4],
    (3, 2): [2, 5],
    (4, 1): [3],
    (4, 2): [1, 2],
    (4, 3): [4, 5],
    (5, 1): [4],
    (5, 2): [3, 5],
    (5, 3): [1, 2, 5],
    (6, 1): [5],
    (6, 2): [3, 4],
    (6, 3): [1, 2, 4],
}

# ===============================
# Parametry EA
# ===============================
N = 20           # populacja
K = 10           # liczba par krzyżowań → 2K potomków
p_mut = 0.1      # p (mutacja osobnika)
q_gene = 0.1     # q (szansa mutacji danego popytu)
GENERATIONS = 100
SEED = 0

# ===============================
# Chromosom: d -> [x(d,1),...,x(d,P(d))], suma = h(d)
# ===============================

def random_chromosome():
    chrom = {}
    for d, info in demands.items():
        P = info["paths"]
        h = info["volume"]
        flows = [0] * P
        for _ in range(h):
            flows[random.randrange(P)] += 1
        chrom[d] = flows
    return chrom

def repair(chrom):
    new = {}
    for d, info in demands.items():
        P = info["paths"]
        h = info["volume"]
        g = chrom.get(d, [0]*P)[:P]
        g = [max(0, int(round(x))) for x in g]
        s = sum(g)
        if s == h:
            new[d] = g
            continue
        if s == 0:
            ng = [0]*P
            ng[0] = h
            new[d] = ng
            continue
        scaled = [int(math.floor(x * h / s)) for x in g]
        diff = h - sum(scaled)
        i = 0
        while diff > 0:
            scaled[i % P] += 1
            diff -= 1
            i += 1
        while diff < 0:
            for j in range(P):
                if scaled[j] > 0 and diff < 0:
                    scaled[j] -= 1
                    diff += 1
                    if diff == 0:
                        break
        new[d] = scaled
    return new

def link_loads_from_chrom(chrom):
    loads = {e: 0 for e in links}
    for d, flow_list in chrom.items():
        for p_idx, units in enumerate(flow_list, start=1):
            for e in demand_paths[(d, p_idx)]:
                loads[e] += units
    return loads

# ===============================
# Funkcje celu
# ===============================

def objective_DAP(chrom):
    """
    DAP: F(x) = max_e O(e,x), O(e,x) = l(e,x) - C(e), C(e)=modules*M..
    """
    chrom = repair(chrom)
    loads = link_loads_from_chrom(chrom)
    O_raw = {}
    O_pos = {}
    for e, load in loads.items():
        cap = links[e]["modules"] * moduleCapacity
        o = load - cap
        O_raw[e] = o
        O_pos[e] = max(0, o)
    F_raw = max(O_raw.values()) if O_raw else 0
    return F_raw, loads, O_pos

def objective_DDAP(chrom):
    """
    DDAP: F(x) = Σ_e ξ(e) * y(e,x), y = ceil(l/M).
    """
    chrom = repair(chrom)
    loads = link_loads_from_chrom(chrom)
    y = {e: (loads[e] + moduleCapacity - 1) // moduleCapacity for e in loads}
    cost = sum(links[e]["cost"] * y[e] for e in loads)
    return cost, loads, y

# ===============================
# Operatory EA
# ===============================

def crossover(p1, p2):
    """50/50 per populacja """
    c1, c2 = {}, {}
    for d in demands:
        if random.random() < 0.5:
            c1[d] = p1[d][:]
            c2[d] = p2[d][:]
        else:
            c1[d] = p2[d][:]
            c2[d] = p1[d][:]
    return repair(c1), repair(c2)

def mutate(chrom, q=q_gene):
    """Przesunięcie 1 jednostki przepływu (dla każdego popytu z prawdopodobieństwem q)."""
    c = repair(chrom)
    for d, flows in c.items():
        if random.random() < q:
            P = len(flows)
            if P < 2:
                continue
            nonzero = [i for i, v in enumerate(flows) if v > 0]
            if not nonzero:
                continue
            i = random.choice(nonzero)
            j = random.choice([k for k in range(P) if k != i])
            flows[i] -= 1
            flows[j] += 1
    return c

# ===============================
# EA (μ+λ) z losowym doborem par
# ===============================

def run_EA(mode="DAP", generations=GENERATIONS, verbose=False):
    if SEED is not None:
        random.seed(SEED)

    # Inicjalizacja (N osobników)
    population = [random_chromosome() for _ in range(N)]

    def evaluate(ind):
        if mode == "DAP":
            return objective_DAP(ind)  # (F, loads, overloads_pos)
        else:
            return objective_DDAP(ind)  # (cost, loads, y)

    # Start
    scored = []
    for ch in population:
        F, loads, extra = evaluate(ch)
        scored.append((F, ch, loads, extra))
    scored.sort(key=lambda x: x[0])
    best = scored[0]
    history = [best[0]]

    if verbose:
        print(f"\n[{mode}] start best = {best[0]}")

    # Generacja
    for gen in range(1, generations + 1):
        offspring = []

        # K par → 2K potomków, rodzice dobierani losowo
        for _ in range(K):
            p1, p2 = random.sample(population, 2)
            c1, c2 = crossover(p1, p2)
            if random.random() < p_mut:
                c1 = mutate(c1, q_gene)
            if random.random() < p_mut:
                c2 = mutate(c2, q_gene)
            offspring.append(c1)
            offspring.append(c2)

        # Potomkowie
        off_scored = []
        for ch in offspring:
            F, loads, extra = evaluate(ch)
            off_scored.append((F, ch, loads, extra))
        # (μ+λ): rodzice + potomkowie -> N najlepszych
        all_scored = scored + off_scored
        all_scored.sort(key=lambda x: x[0])
        next_scored = all_scored[:N]
        scored = next_scored
        population = [x[1] for x in scored]

        best = scored[0]
        history.append(best[0])

        if verbose and gen % 20 == 0:
            print(f"[{mode}] gen {gen}: best = {best[0]}")

    # Trajektoria
    print("\n" + "="*60)
    print(f"{mode} — Trajektoria")
    print("="*60)
    print(" -> ".join(str(v) for v in history))

    print_solution(mode, best)

    return best, history


def print_solution(mode, best_tuple):
    val, chrom, loads, extra = best_tuple
    print("\n" + "="*60)
    print(f"{mode} — Wyniki:")
    print("="*60)

    if mode == "DAP":
        print(f"- Funkcja celu F(x) = max_e (l-C) = {val}")
    else:
        print(f"- Funkcja celu F(x) = Σ ξ(e)·y(e,x) = {val}")

    print("\nŁącza:")
    if mode == "DAP":
        print(" id | (u-v) | load | cap | overload")
        print("-"*40)
        overloads_pos = extra  # max(0, l-C)
        for e in sorted(links):
            u, v = links[e]["A"], links[e]["Z"]
            L = loads[e]
            C = links[e]["modules"] * moduleCapacity
            O = overloads_pos[e]
            print(f"{e:>3} | ({u}-{v}) | {L:>4} | {C:>3} | {O:>8}")
    else:
        y = extra
        print(" id | (u-v) | load | y | link_cost")
        print("-"*40)
        for e in sorted(links):
            u, v = links[e]["A"], links[e]["Z"]
            L = loads[e]
            Y = y[e]
            xi = links[e]["cost"]
            print(f"{e:>3} | ({u}-{v}) | {L:>4} | {Y:>1} | {xi*Y:>9}")

    print("\nPopyty i przepływy po ścieżkach:")
    for d in sorted(demands):
        A, Z, h, P = demands[d]["A"], demands[d]["Z"], demands[d]["volume"], demands[d]["paths"]
        flows = repair(chrom)[d]
        print(f"- Demand {d}: {A}->{Z}, h={h}")
        for p_idx in range(1, P+1):
            path = demand_paths[(d, p_idx)]
            path_str = "-".join(str(e) for e in path)
            print(f"    ścieżka {p_idx}: [{path_str}]  flow={flows[p_idx-1]}")
    print("="*60 + "\n")

# ===============================
# Uruchomienia
# ===============================
if __name__ == "__main__":
    if SEED is not None:
        random.seed(SEED)

    print("=== EA dla DAP (maksymalne przeciążenie) ===")
    best_DAP, hist_DAP = run_EA(mode="DAP", generations=GENERATIONS, verbose=True)

    print("=== EA dla DDAP (maksymalne przeciążenie) ===")
    best_DDAP, hist_DDAP = run_EA(mode="DDAP", generations=GENERATIONS, verbose=True)