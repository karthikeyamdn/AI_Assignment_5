"""
Q3: Knowledge Graphs – Description & Implementation
=====================================================
This file:
  1. Implements a Knowledge Graph engine from scratch (no external libs).
  2. Demonstrates it with a Tourism domain graph.
  3. Supports: add triples, SPARQL-style SELECT queries, path finding,
     graph statistics, and serialisation to Turtle-like notation.

What is a Knowledge Graph?
──────────────────────────
A Knowledge Graph (KG) is a structured representation of real-world
entities and the relationships between them.  It uses a graph data
model where:

  • Nodes  – represent entities (e.g. "Hyderabad", "Charminar")
             or literal values (e.g. the string "heritage").
  • Edges  – represent typed relationships called predicates
             (e.g. hasAttraction, locatedIn, servesCuisine).

Every piece of knowledge is stored as a triple:
       (Subject, Predicate, Object)  ← the RDF model

Example triples:
  ("Charminar",   "locatedIn",      "Hyderabad")
  ("Hyderabad",   "isCapitalOf",    "Telangana")
  ("Charminar",   "category",       "heritage")
  ("Hyderabad",   "hasCuisine",     "HyderabadiFood")

Common Tools for Building Knowledge Graphs
──────────────────────────────────────────
| Tool            | Purpose                                      |
|-----------------|----------------------------------------------|
| Protégé         | Ontology editor (OWL/RDF), visual graph edit |
| Neo4j           | Property-graph database (Cypher query lang)  |
| Apache Jena     | Java framework; stores/queries RDF triples   |
| GraphDB         | RDF store with SPARQL endpoint               |
| NetworkX        | Python library for graph analysis            |
| RDFLib          | Python library for RDF triples & SPARQL      |
| Wikidata        | Open KG with 100M+ entities                  |
| Google KG       | Entity search & disambiguation API          |
| YAGO / DBpedia  | KGs derived from Wikipedia                  |
| OpenKE          | Knowledge graph embedding (KGE) toolkit      |
"""

from __future__ import annotations
from collections import defaultdict, deque
from typing import Any


# ─────────────────────────────────────────────────────────
#  KNOWLEDGE GRAPH ENGINE
# ─────────────────────────────────────────────────────────

class KnowledgeGraph:
    """
    An in-memory, triple-based Knowledge Graph.

    Internal storage
    ────────────────
    Three hash-map indexes are maintained for O(1) look-ups:

      spo[s][p]  → set of objects   (given subject & predicate)
      pos[p][o]  → set of subjects  (given predicate & object)
      osp[o][s]  → set of predicates (given object & subject)

    This mirrors the triple-store design used by systems such as
    Apache Jena, AllegroGraph, and Stardog.
    """

    def __init__(self, name: str = "KnowledgeGraph"):
        self.name = name
        # SPO index
        self.spo: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
        # POS index
        self.pos: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
        # OSP index
        self.osp: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
        self._triple_count = 0

    # ── Triple manipulation ──────────────────────────────────

    def add_triple(self, subject: str, predicate: str, obj: str):
        """
        Add a (subject, predicate, object) triple.

        Parameters
        ----------
        subject   : source entity
        predicate : relationship type / property name
        obj       : target entity or literal value
        """
        if obj not in self.spo[subject][predicate]:
            self.spo[subject][predicate].add(obj)
            self.pos[predicate][obj].add(subject)
            self.osp[obj][subject].add(predicate)
            self._triple_count += 1

    def remove_triple(self, subject: str, predicate: str, obj: str):
        """Remove a triple if it exists."""
        if obj in self.spo.get(subject, {}).get(predicate, set()):
            self.spo[subject][predicate].discard(obj)
            self.pos[predicate][obj].discard(subject)
            self.osp[obj][subject].discard(predicate)
            self._triple_count -= 1

    def has_triple(self, subject: str, predicate: str, obj: str) -> bool:
        return obj in self.spo.get(subject, {}).get(predicate, set())

    # ── Pattern matching (SPARQL-style) ──────────────────────

    def match(self, subject: str = None, predicate: str = None,
              obj: str = None) -> list[tuple[str, str, str]]:
        """
        Return all triples matching the given pattern.
        Pass None as a wildcard.

        Examples
        --------
        kg.match(subject="Charminar")           # all facts about Charminar
        kg.match(predicate="locatedIn")         # all location facts
        kg.match(predicate="locatedIn", obj="Hyderabad")  # what's in Hyderabad
        """
        results = []

        if subject and predicate and obj:
            if self.has_triple(subject, predicate, obj):
                results.append((subject, predicate, obj))

        elif subject and predicate:
            for o in self.spo.get(subject, {}).get(predicate, set()):
                results.append((subject, predicate, o))

        elif subject and obj:
            for p in self.osp.get(obj, {}).get(subject, set()):
                results.append((subject, p, obj))

        elif predicate and obj:
            for s in self.pos.get(predicate, {}).get(obj, set()):
                results.append((s, predicate, obj))

        elif subject:
            for p, objs in self.spo.get(subject, {}).items():
                for o in objs:
                    results.append((subject, p, o))

        elif predicate:
            for o, subjs in self.pos.get(predicate, {}).items():
                for s in subjs:
                    results.append((s, predicate, o))

        elif obj:
            for s, preds in self.osp.get(obj, {}).items():
                for p in preds:
                    results.append((s, p, obj))

        else:
            for s, preds in self.spo.items():
                for p, objs in preds.items():
                    for o in objs:
                        results.append((s, p, o))

        return results

    # ── SPARQL-like SELECT query ──────────────────────────────

    def select(self, patterns: list[tuple],
               filters: dict = None) -> list[dict]:
        """
        Execute a conjunctive query over multiple triple patterns.

        Parameters
        ----------
        patterns : list of (s, p, o) tuples; use '?' prefix for variables
                   e.g. [('?place', 'locatedIn', 'Hyderabad'),
                          ('?place', 'category',  '?cat')]
        filters  : optional dict of variable → required value

        Returns
        -------
        list of binding dicts, e.g. [{'?place': 'Charminar', '?cat': 'heritage'}]
        """
        bindings = [{}]

        for (s, p, o) in patterns:
            new_bindings = []
            for binding in bindings:
                # Resolve variables
                rs = binding.get(s, s) if s.startswith('?') else s
                rp = binding.get(p, p) if p.startswith('?') else p
                ro = binding.get(o, o) if o.startswith('?') else o

                # Wildcard = None if still a variable
                ms = None if (isinstance(rs, str) and rs.startswith('?')) else rs
                mp = None if (isinstance(rp, str) and rp.startswith('?')) else rp
                mo = None if (isinstance(ro, str) and ro.startswith('?')) else ro

                for (fs, fp, fo) in self.match(ms, mp, mo):
                    new_b = dict(binding)
                    if s.startswith('?'):  new_b[s] = fs
                    if p.startswith('?'):  new_b[p] = fp
                    if o.startswith('?'):  new_b[o] = fo
                    new_bindings.append(new_b)
            bindings = new_bindings

        if filters:
            bindings = [b for b in bindings
                        if all(b.get(k) == v for k, v in filters.items())]

        return bindings

    # ── Path finding (BFS) ───────────────────────────────────

    def find_path(self, start: str, end: str,
                  max_hops: int = 5) -> list[tuple] | None:
        """
        BFS shortest path between two entities across any predicate.

        Returns
        -------
        list of (entity, predicate, entity) steps, or None if no path.
        """
        if start == end:
            return []

        # BFS: queue holds (current_node, path_so_far)
        queue = deque([(start, [])])
        visited = {start}

        while queue:
            node, path = queue.popleft()
            if len(path) >= max_hops:
                continue

            # Explore all outgoing edges
            for p, objs in self.spo.get(node, {}).items():
                for o in objs:
                    step = (node, p, o)
                    if o == end:
                        return path + [step]
                    if o not in visited:
                        visited.add(o)
                        queue.append((o, path + [step]))

        return None   # no path found

    # ── Entity neighbours ────────────────────────────────────

    def neighbours(self, entity: str) -> dict[str, list[str]]:
        """Return all directly connected entities grouped by predicate."""
        result = {}
        for p, objs in self.spo.get(entity, {}).items():
            result[p] = list(objs)
        return result

    # ── Statistics ───────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "triples":    self._triple_count,
            "entities":   len(self.spo),
            "predicates": len(self.pos),
        }

    # ── Serialisation (Turtle-like) ──────────────────────────

    def to_turtle(self) -> str:
        """
        Serialize the graph in a Turtle-like notation:
            <Subject> <Predicate> "Object" .
        """
        lines = [f"# Knowledge Graph: {self.name}\n"]
        for s in sorted(self.spo):
            for p in sorted(self.spo[s]):
                for o in sorted(self.spo[s][p]):
                    lines.append(f'<{s}> <{p}> "{o}" .')
        return '\n'.join(lines)

    def display_entity(self, entity: str):
        """Pretty-print all facts about a given entity."""
        print(f"\n── Entity: {entity} ──")
        facts = self.match(subject=entity)
        if not facts:
            print("  (no facts found)")
            return
        for s, p, o in sorted(facts, key=lambda x: x[1]):
            print(f"  {p:25s} → {o}")


# ─────────────────────────────────────────────────────────
#  BUILD TOURISM KNOWLEDGE GRAPH
# ─────────────────────────────────────────────────────────

def build_tourism_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph("TourismKG")

    triples = [
        # Cities
        ("Hyderabad", "type",       "City"),
        ("Hyderabad", "locatedIn",  "Telangana"),
        ("Hyderabad", "locatedIn",  "India"),
        ("Hyderabad", "nickname",   "CityOfPearls"),
        ("Hyderabad", "language",   "Telugu"),
        ("Hyderabad", "language",   "Urdu"),

        ("Goa",       "type",       "State"),
        ("Goa",       "locatedIn",  "India"),
        ("Goa",       "nickname",   "ParadiseOfEast"),
        ("Goa",       "language",   "Konkani"),

        ("Jaipur",    "type",       "City"),
        ("Jaipur",    "locatedIn",  "Rajasthan"),
        ("Jaipur",    "locatedIn",  "India"),
        ("Jaipur",    "nickname",   "PinkCity"),
        ("Jaipur",    "language",   "Hindi"),

        # Attractions
        ("Charminar",          "type",        "Monument"),
        ("Charminar",          "locatedIn",   "Hyderabad"),
        ("Charminar",          "category",    "heritage"),
        ("Charminar",          "builtBy",     "MuhammadQuliQutbShah"),
        ("Charminar",          "builtIn",     "1591"),
        ("Charminar",          "isPartOf",    "UNESCO_WorldHeritage"),

        ("GolcondaFort",       "type",        "Fort"),
        ("GolcondaFort",       "locatedIn",   "Hyderabad"),
        ("GolcondaFort",       "category",    "heritage"),
        ("GolcondaFort",       "builtBy",     "QutbShahiDynasty"),

        ("CalangutéBeach",     "type",        "Beach"),
        ("CalangutéBeach",     "locatedIn",   "Goa"),
        ("CalangutéBeach",     "category",    "beach"),
        ("CalangutéBeach",     "bestSeason",  "November-March"),

        ("AmberFort",          "type",        "Fort"),
        ("AmberFort",          "locatedIn",   "Jaipur"),
        ("AmberFort",          "category",    "heritage"),
        ("AmberFort",          "isPartOf",    "UNESCO_WorldHeritage"),
        ("AmberFort",          "builtBy",     "RajaManSingh"),

        ("HawaMahal",          "type",        "Palace"),
        ("HawaMahal",          "locatedIn",   "Jaipur"),
        ("HawaMahal",          "category",    "heritage"),
        ("HawaMahal",          "nickname",    "PalaceOfWinds"),

        # Foods
        ("HyderabadiFood",     "type",        "Cuisine"),
        ("HyderabadiFood",     "includes",    "HyderabadiBiryani"),
        ("HyderabadiFood",     "includes",    "Haleem"),
        ("HyderabadiFood",     "includes",    "IraniChai"),
        ("HyderabadiFood",     "servedIn",    "Hyderabad"),

        ("GoanFood",           "type",        "Cuisine"),
        ("GoanFood",           "includes",    "FishCurryRice"),
        ("GoanFood",           "includes",    "Bebinca"),
        ("GoanFood",           "servedIn",    "Goa"),

        ("RajasthaniFood",     "type",        "Cuisine"),
        ("RajasthaniFood",     "includes",    "DalBaatiChurma"),
        ("RajasthaniFood",     "includes",    "LaalMaas"),
        ("RajasthaniFood",     "servedIn",    "Jaipur"),

        # UNESCO Heritage Sites
        ("UNESCO_WorldHeritage","type",       "Organisation"),
        ("UNESCO_WorldHeritage","recognises", "Charminar"),
        ("UNESCO_WorldHeritage","recognises", "AmberFort"),
    ]

    for s, p, o in triples:
        kg.add_triple(s, p, o)

    return kg


# ─────────────────────────────────────────────────────────
#  DEMO
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    kg = build_tourism_kg()

    print("=" * 60)
    print("  TOURISM KNOWLEDGE GRAPH")
    print("=" * 60)
    stats = kg.stats()
    print(f"  Triples: {stats['triples']}  |  "
          f"Entities: {stats['entities']}  |  "
          f"Predicates: {stats['predicates']}")

    # Entity details
    kg.display_entity("Charminar")
    kg.display_entity("Jaipur")

    # Pattern matching
    print("\n── Heritage sites in Hyderabad ──")
    results = kg.match(predicate="locatedIn", obj="Hyderabad")
    heritage = [s for s, p, o in results
                if kg.has_triple(s, "category", "heritage")]
    for h in heritage:
        print(f"  {h}")

    # SPARQL-like SELECT
    print("\n── SELECT: places in India with category = heritage ──")
    bindings = kg.select([
        ('?place', 'locatedIn',  '?city'),
        ('?city',  'locatedIn',  'India'),
        ('?place', 'category',   'heritage'),
    ])
    for b in bindings:
        print(f"  {b.get('?place', '?')} in {b.get('?city', '?')}")

    # Path finding
    print("\n── Shortest path: HyderabadiBiryani → India ──")
    path = kg.find_path("HyderabadiBiryani", "India")
    if path:
        for step in path:
            print(f"  {step[0]} --[{step[1]}]--> {step[2]}")
    else:
        print("  No path found")

    # Serialisation
    print("\n── Turtle Serialisation (first 10 lines) ──")
    turtle = kg.to_turtle()
    print('\n'.join(turtle.split('\n')[:10]))
    print("  ...")
