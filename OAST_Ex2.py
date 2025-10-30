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
    3: {"A": 1, "Z": 4, "volume": 2, "paths": 2},
    4: {"A": 2, "Z": 3, "volume": 2, "paths": 3},
    5: {"A": 2, "Z": 4, "volume": 3, "paths": 3},
    6: {"A": 3, "Z": 4, "volume": 4, "paths": 3},
}

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

# --- Parametry EA ---
N = 20           # populacja
K = 10           # liczba par rodziców
p_mut = 0.1      # prawdopodobieństwo mutacji całego chromosomu
q_gene = 0.1     # prawdopodobieństwo zmiany genu
GENERATIONS = 100
SEED = 0

# --- Funkcje pomocnicze ---
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

# --- Funkcje celu ---
def objective_DAP(chrom):
    chrom = repair(chrom)
    loads = link_loads_from_chrom(chrom)
    O_pos = {}
    for e, load in loads.items():
        cap = links[e]["modules"] * moduleCapacity
        O_pos[e] = max(0, load - cap)
    F = max(O_pos.values()) if O_pos else 0
    return F, loads, O_pos

def objective_DDAP(chrom):
    chrom = repair(chrom)
    loads = link_loads_from_chrom(chrom)
    y = {e: (loads[e] + moduleCapacity - 1) // moduleCapacity for e in loads}
    cost = sum(links[e]["cost"] * y[e] for e in loads)
    return cost, loads, y

# --- Operatory krzyżowania ---
def crossover_uniform(p1, p2):
    c1, c2 = {}, {}
    for d in demands:
        if random.random() < 0.5:
            c1[d] = p1[d][:]
            c2[d] = p2[d][:]
        else:
            c1[d] = p2[d][:]
            c2[d] = p1[d][:]
    return repair(c1), repair(c2)

def crossover_one_point(p1, p2):
    c1, c2 = {}, {}
    for d in demands:
        P = len(p1[d])
        point = random.randint(1, P-1) if P>1 else 1
        c1[d] = p1[d][:point] + p2[d][point:]
        c2[d] = p2[d][:point] + p1[d][point:]
    return repair(c1), repair(c2)

def crossover_uniform_gene(p1, p2):
    c1, c2 = {}, {}
    for d in demands:
        c1[d], c2[d] = [], []
        for g1, g2 in zip(p1[d], p2[d]):
            if random.random()<0.5:
                c1[d].append(g1)
                c2[d].append(g2)
            else:
                c1[d].append(g2)
                c2[d].append(g1)
    return repair(c1), repair(c2)

# --- Operatory mutacji ---
def mutate_shift(chrom):
    c = repair(chrom)
    for d, flows in c.items():
        if random.random()<q_gene and len(flows)>1:
            nonzero = [i for i,v in enumerate(flows) if v>0]
            if not nonzero: continue
            i = random.choice(nonzero)
            j = random.choice([k for k in range(len(flows)) if k!=i])
            flows[i]-=1
            flows[j]+=1
    return c

def mutate_swap_demands(chrom):
    c = repair(chrom)
    d1, d2 = random.sample(list(c.keys()),2)
    c[d1], c[d2] = c[d2][:], c[d1][:]
    return c

def mutate_random_reset(chrom):
    c = repair(chrom)
    d = random.choice(list(c.keys()))
    h = sum(c[d])
    P = len(c[d])
    new = [0]*P
    for _ in range(h):
        new[random.randrange(P)]+=1
    c[d]=new
    return c

# --- EA ---
def run_EA(mode="DAP", generations=GENERATIONS, crossover_op=crossover_uniform, mutation_op=mutate_shift):
    if SEED is not None: random.seed(SEED)
    population = [random_chromosome() for _ in range(N)]

    def evaluate(ind):
        return objective_DAP(ind) if mode=="DAP" else objective_DDAP(ind)

    scored = [(evaluate(ch)[0], ch) for ch in population]
    scored.sort(key=lambda x:x[0])
    best = scored[0]

    print(f"\n[{mode}] start best = {best[0]}")

    for gen in range(1, generations+1):
        offspring=[]
        for _ in range(K):
            p1,p2=random.sample(population,2)
            c1,c2=crossover_op(p1,p2)
            if random.random()<p_mut: c1=mutation_op(c1)
            if random.random()<p_mut: c2=mutation_op(c2)
            offspring.extend([c1,c2])

        off_scored=[(evaluate(ch)[0], ch) for ch in offspring]
        all_scored=scored+off_scored
        all_scored.sort(key=lambda x:x[0])
        scored = all_scored[:N]
        population = [x[1] for x in scored]
        best = scored[0]

        print(f"Generacja {gen:3d} | Najlepsza wartość: {best[0]}")
        for d in sorted(best[1].keys()):
            print(f"  D{d}: {best[1][d]}")

    return best

# --- MAIN ---
if __name__ == "__main__":
    print("=== Porównanie operatorów krzyżowania dla DAP i DDAP ===")

    testy_krzyzowania = [
        ("Jednopunktowe krzyżowanie", crossover_one_point),
        ("Równomierne krzyżowanie", crossover_uniform),
        ("Równomierne krzyżowanie po genach", crossover_uniform_gene),
    ]

    wyniki = []

    for opis, crossover_op in testy_krzyzowania:
        print(f"\n--- {opis} (DAP) ---")
        best_dap = run_EA("DAP", generations=100,
                          crossover_op=crossover_op, mutation_op=mutate_shift)
        print(f"Najlepszy wynik DAP dla {opis}: {best_dap[0]}")
        wyniki.append(("DAP", opis, best_dap[0]))

        print(f"\n--- {opis} (DDAP) ---")
        best_ddap = run_EA("DDAP", generations=100,
                           crossover_op=crossover_op, mutation_op=mutate_shift)
        print(f"Najlepszy wynik DDAP dla {opis}: {best_ddap[0]}")
        wyniki.append(("DDAP", opis, best_ddap[0]))

    # --- Ładne podsumowanie ---
    print("\n" + "=" * 60)
    print(f"{'Tryb':<8} | {'Operator krzyżowania':<40} | {'F(x)':>5}")
    print("-" * 60)
    for tryb, opis, val in wyniki:
        print(f"{tryb:<8} | {opis:<40} | {val:>5}")
    print("=" * 60)