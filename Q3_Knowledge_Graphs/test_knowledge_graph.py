"""
test_knowledge_graph.py
=======================
Tests for Q3: Knowledge Graph implementation.
Run with: python test_knowledge_graph.py
"""

import sys
sys.path.insert(0, '.')
from knowledge_graph import KnowledgeGraph, build_tourism_kg

def run(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
        return True
    except AssertionError as e:
        print(f"  FAIL  {name}  —  {e}")
        return False
    except Exception as e:
        print(f"  ERROR {name}  —  {e}")
        return False

KG = build_tourism_kg()

class TestTripleStore:

    def test_add_and_has(self):
        g = KnowledgeGraph()
        g.add_triple("A", "rel", "B")
        assert g.has_triple("A", "rel", "B")

    def test_duplicate_not_counted(self):
        g = KnowledgeGraph()
        g.add_triple("A", "rel", "B")
        g.add_triple("A", "rel", "B")
        assert g.stats()["triples"] == 1

    def test_remove_triple(self):
        g = KnowledgeGraph()
        g.add_triple("A", "rel", "B")
        g.remove_triple("A", "rel", "B")
        assert not g.has_triple("A", "rel", "B")
        assert g.stats()["triples"] == 0

    def test_stats(self):
        stats = KG.stats()
        assert stats["triples"] > 0
        assert stats["entities"] > 0
        assert stats["predicates"] > 0

    def test_triple_count_accurate(self):
        g = KnowledgeGraph()
        for i in range(10):
            g.add_triple(f"s{i}", "p", f"o{i}")
        assert g.stats()["triples"] == 10


class TestMatch:

    def test_match_by_subject(self):
        triples = KG.match(subject="Charminar")
        assert len(triples) > 0
        assert all(s == "Charminar" for s, p, o in triples)

    def test_match_by_predicate(self):
        triples = KG.match(predicate="locatedIn")
        assert all(p == "locatedIn" for s, p, o in triples)

    def test_match_by_object(self):
        triples = KG.match(obj="Hyderabad")
        assert all(o == "Hyderabad" for s, p, o in triples)

    def test_match_subject_predicate(self):
        triples = KG.match(subject="Jaipur", predicate="locatedIn")
        locations = [o for s, p, o in triples]
        assert "Rajasthan" in locations or "India" in locations

    def test_match_wildcard_all(self):
        all_triples = KG.match()
        assert len(all_triples) == KG.stats()["triples"]

    def test_match_no_result(self):
        triples = KG.match(subject="NonExistent", predicate="rel")
        assert triples == []


class TestSelect:

    def test_select_single_pattern(self):
        results = KG.select([('?place', 'locatedIn', 'Hyderabad')])
        places = [b['?place'] for b in results]
        assert 'Charminar' in places

    def test_select_multi_pattern(self):
        results = KG.select([
            ('?place', 'locatedIn', '?city'),
            ('?city',  'locatedIn', 'India'),
            ('?place', 'category',  'heritage'),
        ])
        assert len(results) > 0
        for b in results:
            assert '?place' in b and '?city' in b

    def test_select_with_filter(self):
        results = KG.select(
            [('?place', 'category', '?cat')],
            filters={'?cat': 'beach'}
        )
        assert all(b['?cat'] == 'beach' for b in results)

    def test_select_no_match(self):
        results = KG.select([('?x', 'locatedIn', 'Antarctica')])
        assert results == []


class TestPathFinding:

    def test_direct_connection(self):
        path = KG.find_path("Charminar", "Hyderabad")
        assert path is not None and len(path) == 1
        assert path[0] == ("Charminar", "locatedIn", "Hyderabad")

    def test_two_hop_path(self):
        # GolcondaFort → Hyderabad → Telangana
        path = KG.find_path("GolcondaFort", "Telangana")
        assert path is not None
        assert len(path) <= 3

    def test_same_node(self):
        path = KG.find_path("Charminar", "Charminar")
        assert path == []

    def test_unreachable_returns_none(self):
        path = KG.find_path("Charminar", "IsolatedIsland_XYZ")
        assert path is None

    def test_path_is_valid_chain(self):
        """Each step in the path must be a valid triple in the KG."""
        path = KG.find_path("HyderabadiBiryani", "India")
        if path:
            for s, p, o in path:
                assert KG.has_triple(s, p, o), f"Step {(s,p,o)} not in KG"


class TestNeighbours:

    def test_neighbours_not_empty(self):
        n = KG.neighbours("Charminar")
        assert len(n) > 0

    def test_neighbours_content(self):
        n = KG.neighbours("Jaipur")
        assert "locatedIn" in n

    def test_neighbours_unknown(self):
        n = KG.neighbours("Unknown_Entity_XYZ")
        assert n == {}


class TestSerialisation:

    def test_turtle_contains_triples(self):
        turtle = KG.to_turtle()
        assert "<Charminar>" in turtle
        assert "<locatedIn>" in turtle

    def test_turtle_line_count(self):
        turtle = KG.to_turtle()
        data_lines = [l for l in turtle.split('\n') if l.strip() and not l.startswith('#')]
        assert len(data_lines) == KG.stats()["triples"]


# ── runner ──

if __name__ == '__main__':
    suites = [TestTripleStore, TestMatch, TestSelect,
              TestPathFinding, TestNeighbours, TestSerialisation]
    total = passed = 0
    for suite_cls in suites:
        suite = suite_cls()
        print(f"\n{'─'*50}")
        print(f"  {suite_cls.__name__}")
        print(f"{'─'*50}")
        for name in sorted(dir(suite)):
            if name.startswith('test_'):
                total += 1
                ok = run(name, getattr(suite, name))
                if ok: passed += 1
    print(f"\n{'═'*50}")
    print(f"  Results: {passed}/{total} tests passed")
    print(f"{'═'*50}")
