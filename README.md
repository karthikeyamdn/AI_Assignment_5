# AI Programming Assignment

> **97 tests across 4 questions — all passing ✅**
> No AI-assisted tools were used in writing the algorithms.

---

## Repository Structure

```
AI_Assignment/
├── Q1_Search_Algorithms/
│   ├── search_algorithms.py       # Minimax, Alpha-Beta, Heuristic A-B, MCTS
│   └── test_search_algorithms.py  # 28 tests
│
├── Q2_Travel_Planner/
│   ├── travel_planner.py          # KB + Inference Engine + Planner
│   └── test_travel_planner.py     # 23 tests
│
├── Q3_Knowledge_Graphs/
│   ├── knowledge_graph.py         # Triple-store KG engine + Tourism KG
│   └── test_knowledge_graph.py    # 25 tests
│
├── Q4_Bayesian_Networks/
│   ├── bayesian_network.py        # BN engine + Medical Diagnosis model
│   └── test_bayesian_network.py   # 21 tests
│
└── README.md
```

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/AI_Assignment.git
cd AI_Assignment

# No external libraries needed — pure Python 3.8+

# Run each question's tests
python Q1_Search_Algorithms/test_search_algorithms.py
python Q2_Travel_Planner/test_travel_planner.py
python Q3_Knowledge_Graphs/test_knowledge_graph.py
python Q4_Bayesian_Networks/test_bayesian_network.py

# Or run the demos
python Q1_Search_Algorithms/search_algorithms.py
python Q2_Travel_Planner/travel_planner.py
python Q3_Knowledge_Graphs/knowledge_graph.py
python Q4_Bayesian_Networks/bayesian_network.py
```

---

## Q1 — Game Tree Search Algorithms

**File:** `Q1_Search_Algorithms/search_algorithms.py`  
**Tests:** 28 (all pass)  
**Domain:** Tic-Tac-Toe

### Algorithms Implemented

#### 1. Minimax Search
The classic adversarial search algorithm for two-player zero-sum games.
- **MAX player** (X) tries to maximise the score (+1 win, 0 draw, -1 loss).
- **MIN player** (O) tries to minimise the score.
- Searches the **entire game tree** recursively to terminal nodes.
- Guarantees **optimal play** but is computationally expensive (exponential in depth).

```
         MAX (X's turn)
        /       \
      MIN        MIN  (O's turn)
     /   \      /   \
   MAX  MAX   MAX  MAX  (X's turn)
   ...
```

#### 2. Alpha-Beta Pruning
An enhancement of Minimax that **prunes branches** that cannot affect the final decision.
- **α (alpha):** best value MAX can guarantee — lower bound.
- **β (beta):** best value MIN can guarantee — upper bound.
- When `α ≥ β`, the branch is **pruned** (no further exploration needed).
- Produces **identical results** to Minimax but explores far fewer nodes.
- In the best case, reduces branching factor from b to √b.

#### 3. Heuristic Alpha-Beta (Depth-Limited)
Extends Alpha-Beta with a **depth limit** and **evaluation function**.
- When maximum depth is reached before a terminal node, calls `evaluate()`.
- **Evaluation heuristic** for Tic-Tac-Toe:
  - For each of the 8 lines (rows, columns, diagonals):
    - +10 if line has 2 X's and no O (X nearly wins)
    - +1 if line has 1 X and no O
    - -10 if line has 2 O's and no X (O nearly wins)
    - -1 if line has 1 O and no X
- Practical for large state spaces where full search is infeasible.

#### 4. Monte-Carlo Tree Search (MCTS)
A sampling-based approach that builds a search tree guided by random rollouts.

Four phases per iteration:
1. **Selection** — walk the tree using the UCB1 formula to balance exploitation vs exploration:
   ```
   UCB1 = (wins / visits) + c × √(ln(parent.visits) / visits)
   ```
2. **Expansion** — add one new child node for an untried move.
3. **Simulation** — play out the game randomly (rollout) until terminal.
4. **Back-propagation** — update win counts and visit counts up the tree.

After all iterations, the **most-visited child** (robust child) is selected as the best move.

### Test Results
```
TestTicTacToe        7/7  ✅
TestMinimax          5/5  ✅
TestAlphaBeta        5/5  ✅
TestHeuristicAlphaBeta 5/5 ✅
TestMCTS             4/4  ✅
TestIntegration      2/2  ✅
──────────────────────────
Total               28/28  ✅
```

---

## Q2 — AI-Based Travel Planner

**File:** `Q2_Travel_Planner/travel_planner.py`  
**Tests:** 23 (all pass)  
**Domain:** India Tourism (Hyderabad, Goa, Jaipur)

### Architecture

```
UserProfile
    │
    ▼
InferenceEngine ──► derives implicit facts (e.g. insurance, transport mode)
    │
    ▼
TravelPlanner
    ├── KnowledgeBase.query_places(city, category, month, rating)
    ├── KnowledgeBase.query_foods(city, dietary)
    └── KnowledgeBase.query_cost(city, style)
    │
    ▼
TourPlan  (day-by-day schedule + meal plan + cost breakdown)
```

### Knowledge Bases Used

| KB | Content |
|----|---------|
| **Tourist Places KB** | 16 attractions across 3 cities with category, season, rating, entry fee, visit duration |
| **Food KB** | 16 local dishes with cuisine type, dietary tags, average cost, meal type |
| **Cost KB** | 9 profiles (3 cities × 3 budget styles) with hotel, transport, misc costs |
| **Rules KB** | Forward-chaining rules (luxury→insurance, budget→shared transport, veg→strict filter) |

### Inference Engine
A forward-chaining rule engine that fires IF-THEN rules until no new facts are derived.

### Sample Output
```
══════════════════════════════════════════════════════════
  PERSONALISED TOUR PLAN FOR ARJUN
══════════════════════════════════════════════════════════
  Duration : 7 days | Budget: Moderate | Month: Jan
  Interests: heritage, religious

  ── HYDERABAD (4 days) ──
    Day 1:
      🏛  Charminar (heritage) ~1.5h  ₹25
      🏛  Salar Jung Museum (heritage) ~2.0h  ₹20
      🍽  Breakfast: Pesarattu ~₹80/person
      🍽  Lunch: Mirchi ka Salan ~₹120/person
      ...
```

---

## Q3 — Knowledge Graphs

**File:** `Q3_Knowledge_Graphs/knowledge_graph.py`  
**Tests:** 25 (all pass)

### What is a Knowledge Graph?
A Knowledge Graph (KG) represents real-world entities and their relationships as a directed graph. Each piece of knowledge is a **triple**: (Subject, Predicate, Object) following the RDF model.

```
("Charminar",  "locatedIn", "Hyderabad")
("Hyderabad",  "locatedIn", "India")
("Charminar",  "category",  "heritage")
```

### Implementation Details

**Triple-Store with 3 indexes** for O(1) lookups:
- `spo[subject][predicate]` → set of objects
- `pos[predicate][object]` → set of subjects  
- `osp[object][subject]` → set of predicates

**Features:**
- `add_triple / remove_triple / has_triple`
- `match(s, p, o)` — wildcard pattern matching
- `select(patterns)` — SPARQL-like conjunctive queries with variable binding
- `find_path(start, end)` — BFS shortest path across any predicates
- `neighbours(entity)` — directly connected entities
- `to_turtle()` — serialize to Turtle-like notation
- `stats()` — triple/entity/predicate counts

### Tools for Building Knowledge Graphs

| Tool | Type | Use Case |
|------|------|----------|
| **Protégé** | Desktop GUI | Ontology editing (OWL/RDF) |
| **Neo4j** | Database | Property graphs, Cypher queries |
| **Apache Jena** | Java framework | RDF storage and SPARQL |
| **RDFLib** | Python library | RDF triples and SPARQL in Python |
| **NetworkX** | Python library | Graph analysis and algorithms |
| **Wikidata** | Open KG | 100M+ real-world entities |
| **GraphDB** | RDF store | SPARQL endpoint, visual explorer |
| **YAGO/DBpedia** | Open KG | Wikipedia-derived structured data |

---

## Q4 — Bayesian Networks

**File:** `Q4_Bayesian_Networks/bayesian_network.py`  
**Tests:** 21 (all pass)  
**Domain:** Medical Diagnosis

### What is a Bayesian Network?
A BN is a probabilistic graphical model (DAG) where:
- **Nodes** = random variables
- **Directed edges** = causal/conditional dependencies
- **CPT** (Conditional Probability Table) = P(node | parents) at each node

### Medical Diagnosis Network

```
  Flu ──► Fever ──► Headache
   │
   ├────► Cough
   │
   └────► Breathlessness ◄── SpO2_Low
```

**Variables:** Flu, SpO2_Low, Fever, Headache, Cough, Breathlessness (all binary)

**CPT Example — P(Fever | Flu):**
| Flu | P(Fever=True) |
|-----|---------------|
| True | 0.85 |
| False | 0.05 |

### Inference Methods

#### 1. Variable Elimination (Exact)
Enumerates all hidden variable combinations:
`P(Q|e) = α Σ_{hidden} P(Q, hidden, e)`

#### 2. Likelihood Weighting (Approximate)
Monte Carlo sampling where each sample is weighted by `Π P(evidence | parents)`, making it efficient even when exact inference is expensive.

#### 3. MAP — Most Probable Explanation
Finds the joint assignment to unobserved variables that maximises `P(assignment | evidence)`.

### Sample Queries
```
P(Flu)                              = 0.1000  (prior)
P(Flu | Fever=T, Cough=T)          = 0.6561  (diagnosis)
P(Flu | Fever=T, Cough=T, Headache=T) = 0.6726
P(Breathlessness | SpO2_Low=T)     = 0.8075
```

### Tools for Bayesian Networks

| Tool | Language | Notes |
|------|----------|-------|
| **pgmpy** | Python | Full BN library (VE, Belief Propagation, MCMC) |
| **pomegranate** | Python | GPU-accelerated probabilistic models |
| **Netica** | GUI | Commercial BN tool |
| **GeNIe/SMILE** | GUI/API | Free BN modeller (BayesFusion) |
| **Hugin** | GUI/API | Commercial BN platform |
| **bnlearn** | R | Structure learning & inference |

---

## Test Summary

| Question | Tests | Result |
|----------|-------|--------|
| Q1 Search Algorithms | 28 | ✅ All Pass |
| Q2 Travel Planner | 23 | ✅ All Pass |
| Q3 Knowledge Graphs | 25 | ✅ All Pass |
| Q4 Bayesian Networks | 21 | ✅ All Pass |
| **Total** | **97** | **✅ 97/97** |

---

## Requirements
- Python 3.8 or higher
- No external libraries required (pure Python standard library)
