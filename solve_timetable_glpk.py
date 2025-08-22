import pandas as pd
import pulp
import json
from pathlib import Path

# ---------- Leitura dos dados ----------
def read_sets():
    P = pd.read_csv("P.csv")["p"].astype(str).tolist()
    T = pd.read_csv("T.csv")["t"].astype(str).tolist()
    D = pd.read_csv("D.csv")["d"].astype(str).tolist()
    H = pd.read_csv("H.csv")["h"].astype(str).tolist()
    return P, T, D, H

def read_prefD(P, D):
    df = pd.read_csv("PrefDpd.csv").set_index(df_first_col("PrefDpd.csv"))
    # garante cobertura das labels e ausentes=0
    df = df.reindex(index=P, columns=D, fill_value=0.0).astype(float)
    return df

def read_prefph(P, H):
    df = pd.read_csv("Prefph.csv").set_index(df_first_col("Prefph.csv"))
    df = df.reindex(index=P, columns=H, fill_value=0.0).astype(float)
    return df

def read_CH(P):
    ch = pd.read_csv("CH.csv")
    ch["p"] = ch["p"].astype(str)
    ch = ch.set_index("p").reindex(P)
    return ch["CHM"].astype(float).to_dict(), ch["CHA"].astype(float).to_dict()

def read_aulas(T, D):
    a = pd.read_csv("Aulas_td.csv")
    a["t"] = a["t"].astype(str)
    a["d"] = a["d"].astype(str)
    demanda = {(row.t, row.d): float(row.aulas) for row in a.itertuples(index=False)}
    # completa faltantes com 0 (sem demanda)
    for t in T:
        for d in D:
            demanda.setdefault((t, d), 0.0)
    return demanda

def df_first_col(fname):
    # retorna o nome da 1ª coluna (para CSVs no formato " ,col2,col3..." com índice nomeado ou não)
    probe = pd.read_csv(fname, nrows=0)
    return probe.columns[0]

# ---------- Modelo ----------
def solve_model():
    P, T, D, H = read_sets()

    PrefD = read_prefD(P, D)   # habilitação + peso por disciplina
    PrefH = read_prefph(P, H)  # preferência por horário
    CHM, CHA = read_CH(P)
    Aulas = read_aulas(T, D)

    # Problema: maximizar 
    prob = pulp.LpProblem("Tabelamento_Escolar", pulp.LpMaximize)

    # Variáveis binárias X[p,t,d,h]; se PrefD[p,d]==0 e and PrefH.loc[p, h] = 0, variáveis ficam com ub=0 (bloqueadas)
    X = {}
    for p in P:
        for t in T:
            for d in D:
                ub = 1.0 if (PrefD.loc[p, d] > 0 and PrefH.loc[p, h] > 0) else 0.0
                for h in H:
                    X[(p, t, d, h)] = pulp.LpVariable(f"X_{p}_{t}_{d}_{h}", lowBound=0, upBound=ub, cat=pulp.LpBinary)

    # Função Objetivo: sum X * PrefH[p,h] * PrefD[p,d]
    prob += pulp.lpSum(
        X[(p, t, d, h)] * PrefH.loc[p, h] * PrefD.loc[p, d]
        for p in P for t in T for d in D for h in H
    )

    # (1) Professor não pode dar duas aulas no mesmo horário: sum_{t,d} X <= 1  ∀p,h
    for p in P:
        for h in H:
            prob += pulp.lpSum(X[(p, t, d, h)] for t in T for d in D) <= 1, f"prof_1_at_a_time_{p}_{h}"

    # (2) Turma tem no máximo 1 aula por horário: sum_{p,d} X <= 1  ∀t,h
    for t in T:
        for h in H:
            prob += pulp.lpSum(X[(p, t, d, h)] for p in P for d in D) <= 1, f"turma_1_class_{t}_{h}"

    # (4) Carga horária por professor: CHM <= sum X <= CHA  ∀p
    for p in P:
        carga = pulp.lpSum(X[(p, t, d, h)] for t in T for d in D for h in H)
        prob += carga >= CHM[p], f"CH_min_{p}"
        prob += carga <= CHA[p], f"CH_max_{p}"

    # (5) Atender demanda de aulas por turma/disciplina: sum_{p,h} X = Aulas_td
    for t in T:
        for d in D:
            demanda = Aulas[(t, d)]
            prob += pulp.lpSum(X[(p, t, d, h)] for p in P for h in H) == demanda, f"demanda_{t}_{d}"

    # Resolver com GLPK
    status = prob.solve(pulp.GLPK_CMD(msg=False))
    status_str = pulp.LpStatus[status]
    obj = pulp.value(prob.objective)

    # Exportar TXT
    with open("resultado.txt", "w", encoding="utf-8") as f:
        f.write(f"Status: {status_str}\n")
        f.write(f"Objetivo: {obj}\n\n")
        f.write("Variáveis (X=1):\n")
        for (p, t, d, h), var in X.items():
            if var.value() > 0.5:
                f.write(f"{p};{t};{d};{h} = 1\n")

    # Exportar JSON
    sol = {
        "status": status_str,
        "objective": obj,
        "assignments": [
            {"p": p, "t": t, "d": d, "h": h}
            for (p, t, d, h), var in X.items() if var.value() > 0.5
        ],
        "slacks": {name: cons.slack for name, cons in prob.constraints.items()}
    }
    with open("resultado.json", "w", encoding="utf-8") as f:
        json.dump(sol, f, ensure_ascii=False, indent=2)

    print("Solução escrita em resultado.txt e resultado.json")

if __name__ == "__main__":
    # checagem simples de arquivos
    required = ["P.csv", "T.csv", "D.csv", "H.csv", "PrefDpd.csv", "Prefph.csv", "CH.csv", "Aulas_td.csv"]
    missing = [r for r in required if not Path(r).exists()]
    if missing:
        raise FileNotFoundError(f"Arquivos ausentes: {missing}")
    solve_model()
