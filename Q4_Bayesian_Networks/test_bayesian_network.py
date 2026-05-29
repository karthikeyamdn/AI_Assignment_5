"""
test_bayesian_network.py
========================
Tests for Q4: Bayesian Network implementation.
Run with: python test_bayesian_network.py
"""

import sys, math, random
sys.path.insert(0, '.')
from bayesian_network import CPT, BayesianNetwork, build_medical_bn

random.seed(42)

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


BN = build_medical_bn()


class TestCPT:

    def test_no_parent_prior(self):
        cpt = CPT("Flu", [], {(): 0.10})
        assert abs(cpt.probability(True, {}) - 0.10) < 1e-9
        assert abs(cpt.probability(False, {}) - 0.90) < 1e-9

    def test_with_parent_true(self):
        cpt = CPT("Fever", ["Flu"], {(True,): 0.85, (False,): 0.05})
        assert abs(cpt.probability(True, {"Flu": True})  - 0.85) < 1e-9
        assert abs(cpt.probability(False, {"Flu": True}) - 0.15) < 1e-9

    def test_with_parent_false(self):
        cpt = CPT("Fever", ["Flu"], {(True,): 0.85, (False,): 0.05})
        assert abs(cpt.probability(True, {"Flu": False}) - 0.05) < 1e-9

    def test_complements_sum_to_one(self):
        cpt = CPT("X", ["A", "B"],
                  {(True,True):0.9,(True,False):0.7,
                   (False,True):0.4,(False,False):0.1})
        for key in cpt.table:
            pa = dict(zip(["A","B"], key))
            assert abs(cpt.probability(True, pa) +
                       cpt.probability(False, pa) - 1.0) < 1e-9


class TestBayesianNetworkStructure:

    def test_all_nodes_present(self):
        for var in ["Flu", "SpO2_Low", "Fever", "Headache",
                    "Cough", "Breathlessness"]:
            assert var in BN.nodes, f"{var} not in network"

    def test_topological_order_parents_before_children(self):
        order = BN.topological_order()
        index = {v: i for i, v in enumerate(order)}
        for var, cpt in BN.nodes.items():
            for parent in cpt.parents:
                assert index[parent] < index[var], \
                    f"Parent {parent} comes after child {var} in topo order"

    def test_prior_sample_valid(self):
        s = BN.prior_sample()
        for var in BN.nodes:
            assert var in s, f"{var} missing from sample"
            assert isinstance(s[var], bool)


class TestExactInference:

    def test_prior_flu_correct(self):
        result = BN.variable_elimination("Flu", {})
        assert abs(result[True] - 0.10) < 0.001, \
            f"Prior P(Flu) should be ~0.10, got {result[True]:.4f}"

    def test_prior_spo2_correct(self):
        result = BN.variable_elimination("SpO2_Low", {})
        assert abs(result[True] - 0.05) < 0.001

    def test_probs_sum_to_one(self):
        result = BN.variable_elimination("Flu", {"Fever": True})
        assert abs(result[True] + result[False] - 1.0) < 1e-9

    def test_fever_raises_flu_probability(self):
        prior = BN.variable_elimination("Flu", {})
        posterior = BN.variable_elimination("Flu", {"Fever": True})
        assert posterior[True] > prior[True], \
            "Observing Fever must increase P(Flu)"

    def test_cough_and_fever_raise_flu_more(self):
        p_fever = BN.variable_elimination("Flu", {"Fever": True})
        p_both  = BN.variable_elimination("Flu", {"Fever": True, "Cough": True})
        assert p_both[True] > p_fever[True], \
            "Adding Cough evidence must further increase P(Flu)"

    def test_breathlessness_given_spo2_low(self):
        result = BN.variable_elimination("Breathlessness", {"SpO2_Low": True})
        assert result[True] > 0.5, \
            "P(Breathlessness | SpO2Low=True) should be > 0.5"

    def test_no_evidence_fever_prior(self):
        # P(Fever) = P(F|Flu)*P(Flu) + P(F|¬Flu)*P(¬Flu)
        #          = 0.85*0.10 + 0.05*0.90 = 0.085 + 0.045 = 0.13
        result = BN.variable_elimination("Fever", {})
        expected = 0.85 * 0.10 + 0.05 * 0.90
        assert abs(result[True] - expected) < 0.001, \
            f"P(Fever) should be ~{expected:.3f}, got {result[True]:.4f}"


class TestApproximateInference:

    def test_prior_flu_approx(self):
        result = BN.likelihood_weighting("Flu", {}, n_samples=30000)
        assert abs(result[True] - 0.10) < 0.03, \
            f"Approx P(Flu) = {result[True]:.3f}, expected ~0.10"

    def test_probs_sum_to_one(self):
        result = BN.likelihood_weighting("Flu", {}, 10000)
        assert abs(result[True] + result[False] - 1.0) < 1e-6

    def test_fever_raises_flu_approx(self):
        prior = BN.likelihood_weighting("Flu", {}, 20000)
        post  = BN.likelihood_weighting("Flu", {"Fever": True}, 20000)
        assert post[True] > prior[True], \
            "Likelihood weighting: Fever must raise P(Flu)"

    def test_breathlessness_approx(self):
        result = BN.likelihood_weighting(
            "Breathlessness", {"SpO2_Low": True}, 20000)
        assert result[True] > 0.5


class TestMAP:

    def test_map_returns_all_hidden(self):
        evidence = {"Fever": True, "Cough": True}
        mpe, prob = BN.map_estimate(evidence)
        hidden = [v for v in BN.nodes if v not in evidence]
        for h in hidden:
            assert h in mpe, f"{h} missing from MAP estimate"

    def test_map_prob_is_positive(self):
        _, prob = BN.map_estimate({"Fever": True})
        assert prob > 0.0

    def test_flu_true_in_map_given_symptoms(self):
        """With strong flu symptoms, MAP should assign Flu=True."""
        mpe, _ = BN.map_estimate(
            {"Fever": True, "Cough": True, "Headache": True})
        assert mpe.get("Flu") is True, \
            "MAP should infer Flu=True given Fever+Cough+Headache"


# ── runner ──

if __name__ == '__main__':
    suites = [TestCPT, TestBayesianNetworkStructure,
              TestExactInference, TestApproximateInference, TestMAP]
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
