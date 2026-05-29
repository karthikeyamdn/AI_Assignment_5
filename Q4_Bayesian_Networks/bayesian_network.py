"""
Q4: Bayesian Networks – Modelling, Representation & Inference
=============================================================
This file implements a Bayesian Network (BN) engine from scratch.

What is a Bayesian Network?
───────────────────────────
A Bayesian Network is a probabilistic graphical model that represents
a set of variables and their conditional dependencies via a directed
acyclic graph (DAG).

Each node represents a random variable (e.g. "Disease", "Fever").
Each directed edge A → B means "A influences B" (A is a parent of B).
Each node stores a Conditional Probability Table (CPT):
  P(node | parents)

Given observations (evidence), we can query the probability of
any unobserved variable using inference.

Common Tools for Bayesian Networks
───────────────────────────────────
| Tool          | Language | Notes                                  |
|---------------|----------|----------------------------------------|
| pgmpy         | Python   | Full BN library (VE, BP, MCMC, etc.)  |
| pomegranate   | Python   | GPU-accelerated probabilistic models  |
| Netica        | GUI      | Commercial BN tool with visual editor  |
| GeNIe/SMILE   | GUI      | Free BN modelling tool (BayesFusion)  |
| Hugin         | GUI/API  | Commercial BN platform                |
| BayesiaLab    | GUI      | Advanced machine-learning BN          |
| bnlearn (R)   | R        | Learning & inference in BNs           |

Example: Medical Diagnosis Network
────────────────────────────────────
We model the following causal structure:

  Flu ──► Fever ──► Headache
   │
   └────► Cough

  SpO2_Low ──► Breathlessness

  Flu ──► Breathlessness

Variables (all binary: True / False):
  Flu           – patient has influenza
  Fever         – elevated temperature
  Headache      – patient reports headache
  Cough         – patient has cough
  SpO2_Low      – blood-oxygen saturation is low
  Breathlessness – patient has difficulty breathing
"""

from __future__ import annotations
from itertools import product
from typing import Iterable
import math, random


# ─────────────────────────────────────────────────────────
#  CONDITIONAL PROBABILITY TABLE  (CPT)
# ─────────────────────────────────────────────────────────

class CPT:
    """
    Conditional Probability Table for a single binary variable.

    The table maps each combination of parent values to P(node=True).
    P(node=False) = 1 - P(node=True) (Bernoulli).

    Parameters
    ----------
    variable : name of the node this CPT belongs to
    parents  : ordered list of parent variable names
    table    : dict mapping tuple(parent_values) → P(variable=True)
               e.g. {(True, False): 0.8}
               Use {(): p} for a node with no parents (prior).
    """

    def __init__(self, variable: str, parents: list[str], table: dict):
        self.variable = variable
        self.parents  = parents
        self.table    = table   # {(parent_val_1, ..., parent_val_n): P(True)}

    def probability(self, value: bool,
                    parent_values: dict[str, bool]) -> float:
        """
        Return P(variable = value | parent assignments).

        Parameters
        ----------
        value         : True or False
        parent_values : dict of {parent_name: bool}
        """
        key = tuple(parent_values[p] for p in self.parents)
        p_true = self.table[key]
        return p_true if value else (1.0 - p_true)


# ─────────────────────────────────────────────────────────
#  BAYESIAN NETWORK
# ─────────────────────────────────────────────────────────

class BayesianNetwork:
    """
    A discrete (binary) Bayesian Network.

    Supports:
      • Exact inference via Variable Elimination
      • Approximate inference via Likelihood Weighting (Monte Carlo)
      • Prior / posterior sampling
    """

    def __init__(self, name: str = "BayesNet"):
        self.name   = name
        self.nodes: dict[str, CPT] = {}        # variable → CPT
        self._order: list[str] = []            # topological order

    def add_node(self, cpt: CPT):
        """Register a node (variable + CPT) in the network."""
        self.nodes[cpt.variable] = cpt
        self._order.append(cpt.variable)

    # ── topological sort ────────────────────────────────────

    def topological_order(self) -> list[str]:
        """
        Return variables in topological (ancestral) order using
        Kahn's algorithm.  Nodes with no parents come first.
        """
        in_degree = {v: 0 for v in self.nodes}
        children  = {v: [] for v in self.nodes}
        for var, cpt in self.nodes.items():
            for p in cpt.parents:
                children[p].append(var)
                in_degree[var] += 1

        queue = [v for v, d in in_degree.items() if d == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in children[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return order

    # ── prior sample ────────────────────────────────────────

    def prior_sample(self) -> dict[str, bool]:
        """
        Generate one sample from the joint prior P(X_1,...,X_n)
        by ancestral sampling (forward sampling).
        """
        sample = {}
        for var in self.topological_order():
            cpt = self.nodes[var]
            p_true = cpt.probability(True, sample)
            sample[var] = random.random() < p_true
        return sample

    # ── likelihood-weighted sampling (approximate inference) ──

    def likelihood_weighting(self, query: str,
                              evidence: dict[str, bool],
                              n_samples: int = 10000) -> dict[bool, float]:
        """
        Approximate P(query | evidence) using likelihood weighting.

        Algorithm
        ---------
        For each sample:
          1. Sample non-evidence variables ancestrally.
          2. Fix evidence variables to their observed values.
          3. Compute sample weight = Π P(e_i | parents(e_i)).
        Normalise the weighted counts to get P(query=True | evidence).

        Parameters
        ----------
        query     : variable name to query
        evidence  : {var: observed_value}
        n_samples : number of weighted samples

        Returns
        -------
        dict {True: P(query=True | e), False: P(query=False | e)}
        """
        weights = {True: 0.0, False: 0.0}

        for _ in range(n_samples):
            sample = {}
            weight = 1.0

            for var in self.topological_order():
                cpt = self.nodes[var]
                if var in evidence:
                    # fix to observed value & adjust weight
                    val = evidence[var]
                    sample[var] = val
                    weight *= cpt.probability(val, sample)
                else:
                    p_true = cpt.probability(True, sample)
                    sample[var] = random.random() < p_true

            weights[sample[query]] += weight

        total = weights[True] + weights[False]
        if total == 0:
            return {True: 0.5, False: 0.5}
        return {True: weights[True] / total,
                False: weights[False] / total}

    # ── variable elimination (exact inference) ───────────────

    def variable_elimination(self, query: str,
                              evidence: dict[str, bool]) -> dict[bool, float]:
        """
        Exact inference via Variable Elimination.

        Only supported for small, fully-binary networks.
        Enumerates all hidden variable combinations.

        This is equivalent to:
          P(Q|e) = α Σ_{hidden} P(Q, hidden, e)

        Parameters
        ----------
        query    : variable name to query
        evidence : {var: observed_value}

        Returns
        -------
        dict {True: P(query=True | e), False: P(query=False | e)}
        """
        hidden = [v for v in self.nodes if v != query and v not in evidence]
        weights = {True: 0.0, False: 0.0}

        # iterate over all assignments to hidden variables
        for combo in product([True, False], repeat=len(hidden)):
            assignment = dict(zip(hidden, combo))
            assignment.update(evidence)

            for q_val in [True, False]:
                assignment[query] = q_val
                prob = 1.0
                for var in self.topological_order():
                    cpt = self.nodes[var]
                    parent_vals = {p: assignment[p] for p in cpt.parents}
                    prob *= cpt.probability(assignment[var], parent_vals)
                weights[q_val] += prob

        total = weights[True] + weights[False]
        return {True: weights[True] / total,
                False: weights[False] / total}

    # ── MAP / Most Probable Explanation ─────────────────────

    def map_estimate(self, evidence: dict[str, bool]) -> dict[str, bool]:
        """
        Find the Most Probable Explanation (MPE):
        the joint assignment to all unobserved variables that
        maximises P(assignment | evidence).

        Uses exhaustive enumeration (suitable for small networks).
        """
        hidden = [v for v in self.nodes if v not in evidence]
        best_prob = -1.0
        best_assignment = {}

        for combo in product([True, False], repeat=len(hidden)):
            assignment = dict(zip(hidden, combo))
            assignment.update(evidence)
            prob = 1.0
            for var in self.topological_order():
                cpt = self.nodes[var]
                parent_vals = {p: assignment[p] for p in cpt.parents}
                prob *= cpt.probability(assignment[var], parent_vals)
            if prob > best_prob:
                best_prob = prob
                best_assignment = {k: v for k, v in assignment.items()
                                   if k not in evidence}

        return best_assignment, best_prob

    def display_cpts(self):
        """Print all CPTs in the network."""
        print(f"\n{'═'*55}")
        print(f"  Bayesian Network: {self.name}")
        print(f"{'═'*55}")
        for var in self.topological_order():
            cpt = self.nodes[var]
            parents = cpt.parents
            print(f"\n  {var}")
            print(f"  Parents: {parents if parents else 'none (prior)'}")
            print(f"  {'Parent values':<30} P({var}=True)")
            print("  " + "─" * 45)
            for key, p in sorted(cpt.table.items()):
                label = str(dict(zip(parents, key))) if parents else "(prior)"
                print(f"  {label:<30} {p:.3f}")


# ─────────────────────────────────────────────────────────
#  BUILD THE MEDICAL DIAGNOSIS NETWORK
# ─────────────────────────────────────────────────────────

def build_medical_bn() -> BayesianNetwork:
    """
    Medical Diagnosis Bayesian Network.

    Structure:
      Flu ──► Fever ──► Headache
      Flu ──► Cough
      Flu ──► Breathlessness
      SpO2_Low ──► Breathlessness
    """
    bn = BayesianNetwork("MedicalDiagnosis")

    # P(Flu)  – prior: 10% base rate of flu in population
    bn.add_node(CPT("Flu", [],
                    {(): 0.10}))

    # P(SpO2_Low)  – prior: 5% have low blood oxygen
    bn.add_node(CPT("SpO2_Low", [],
                    {(): 0.05}))

    # P(Fever | Flu)
    #   If flu  → 85 % chance of fever
    #   No flu  → 5 % chance (other causes)
    bn.add_node(CPT("Fever", ["Flu"],
                    {(True,): 0.85,
                     (False,): 0.05}))

    # P(Headache | Fever)
    bn.add_node(CPT("Headache", ["Fever"],
                    {(True,): 0.70,
                     (False,): 0.10}))

    # P(Cough | Flu)
    bn.add_node(CPT("Cough", ["Flu"],
                    {(True,): 0.80,
                     (False,): 0.10}))

    # P(Breathlessness | Flu, SpO2_Low)
    bn.add_node(CPT("Breathlessness", ["Flu", "SpO2_Low"],
                    {(True,  True):  0.95,
                     (True,  False): 0.60,
                     (False, True):  0.85,
                     (False, False): 0.02}))

    return bn


# ─────────────────────────────────────────────────────────
#  DEMO
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    random.seed(42)
    bn = build_medical_bn()
    bn.display_cpts()

    print("\n" + "═" * 55)
    print("  INFERENCE QUERIES")
    print("═" * 55)

    # ── Exact inference ──────────────────────────────────────

    # Q1: Prior probability of flu
    q1_exact = bn.variable_elimination("Flu", {})
    print(f"\nQ1 (exact):  P(Flu)              = {q1_exact[True]:.4f}")

    # Q2: P(Flu | Fever=True, Cough=True)
    q2_exact = bn.variable_elimination(
        "Flu", {"Fever": True, "Cough": True})
    print(f"Q2 (exact):  P(Flu | Fever, Cough) = {q2_exact[True]:.4f}")

    # Q3: P(Flu | Fever=True, Cough=True, Headache=True)
    q3_exact = bn.variable_elimination(
        "Flu", {"Fever": True, "Cough": True, "Headache": True})
    print(f"Q3 (exact):  P(Flu | Fever+Cough+Headache) = {q3_exact[True]:.4f}")

    # Q4: P(Breathlessness | SpO2_Low=True)
    q4_exact = bn.variable_elimination(
        "Breathlessness", {"SpO2_Low": True})
    print(f"Q4 (exact):  P(Breathless | SpO2Low)    = {q4_exact[True]:.4f}")

    # ── Approximate inference ────────────────────────────────
    print("\n── Likelihood Weighting (n=20 000) ──")

    q1_lw = bn.likelihood_weighting("Flu", {}, 20000)
    print(f"Q1 (approx): P(Flu)              ≈ {q1_lw[True]:.4f}")

    q2_lw = bn.likelihood_weighting(
        "Flu", {"Fever": True, "Cough": True}, 20000)
    print(f"Q2 (approx): P(Flu | Fever, Cough) ≈ {q2_lw[True]:.4f}")

    # ── MAP ──────────────────────────────────────────────────
    print("\n── Most Probable Explanation ──")
    mpe, prob = bn.map_estimate({"Fever": True, "Cough": True})
    print(f"Evidence: Fever=True, Cough=True")
    print(f"Most probable hidden state: {mpe}")
    print(f"Joint probability: {prob:.6f}")

    # ── Prior sampling demo ──────────────────────────────────
    print("\n── Prior Sampling (5 samples) ──")
    for i in range(5):
        s = bn.prior_sample()
        print(f"  Sample {i+1}: {s}")
