"""
test_travel_planner.py
======================
Test cases for Q2: AI-Based Travel Planner.
Run with:  python test_travel_planner.py
"""

import sys, random
sys.path.insert(0, '.')
from travel_planner import (
    KnowledgeBase, TravelPlanner, UserProfile,
    build_knowledge_base, Place, Food, CostProfile
)

random.seed(0)

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


KB = build_knowledge_base()


class TestKnowledgeBase:

    def test_places_loaded(self):
        assert len(KB.places) > 0, "KB must contain places"

    def test_foods_loaded(self):
        assert len(KB.foods) > 0, "KB must contain foods"

    def test_costs_loaded(self):
        assert len(KB.costs) > 0, "KB must contain cost profiles"

    def test_query_places_by_city(self):
        results = KB.query_places(city="Goa")
        assert all(p.city == "Goa" for p in results), \
            "All results must be in Goa"

    def test_query_places_by_category(self):
        results = KB.query_places(category="heritage")
        assert all(p.category == "heritage" for p in results)

    def test_query_places_by_month(self):
        results = KB.query_places(month="Dec")
        assert all("Dec" in p.best_season for p in results)

    def test_query_places_rating_filter(self):
        results = KB.query_places(min_rating=4.5)
        assert all(p.rating >= 4.5 for p in results)

    def test_query_foods_by_city(self):
        results = KB.query_foods(city="Hyderabad")
        assert all(f.city == "Hyderabad" for f in results)

    def test_query_foods_dietary_veg(self):
        results = KB.query_foods(city="Hyderabad", dietary=["veg"])
        assert all("veg" in f.dietary for f in results)

    def test_query_cost_profile(self):
        cp = KB.query_cost("Jaipur", "moderate")
        assert cp is not None, "Jaipur moderate profile must exist"
        assert cp.hotel_per_night > 0

    def test_query_cost_missing(self):
        cp = KB.query_cost("NonExistentCity", "luxury")
        assert cp is None, "Missing city must return None"


class TestInferenceEngine:

    def test_luxury_triggers_insurance(self):
        from travel_planner import InferenceEngine
        rules = KB._rules
        engine = InferenceEngine(rules)
        facts = engine.infer({"budget_style": "luxury"})
        assert facts.get("recommend_travel_insurance") is True

    def test_budget_triggers_shared_transport(self):
        from travel_planner import InferenceEngine
        engine = InferenceEngine(KB._rules)
        facts = engine.infer({"budget_style": "budget"})
        assert facts.get("transport_mode") == "shared"

    def test_no_spurious_inference(self):
        from travel_planner import InferenceEngine
        engine = InferenceEngine(KB._rules)
        facts = engine.infer({"budget_style": "moderate"})
        assert facts.get("recommend_travel_insurance") is None
        assert facts.get("transport_mode") is None


class TestTravelPlanner:

    def _make_user(self, **kwargs):
        defaults = dict(
            name="Test User",
            budget_style="moderate",
            interests=["heritage"],
            dietary=["veg"],
            travel_month="Jan",
            num_days=3,
            cities=["Hyderabad"],
        )
        defaults.update(kwargs)
        return UserProfile(**defaults)

    def test_plan_returns_tour_plan(self):
        from travel_planner import TourPlan
        planner = TravelPlanner(KB)
        plan = planner.plan(self._make_user())
        assert isinstance(plan, TourPlan)

    def test_plan_has_correct_num_city_plans(self):
        planner = TravelPlanner(KB)
        user = self._make_user(cities=["Hyderabad", "Jaipur"], num_days=6)
        plan = planner.plan(user)
        assert len(plan.city_plans) == 2

    def test_total_days_match(self):
        planner = TravelPlanner(KB)
        user = self._make_user(cities=["Hyderabad", "Goa"], num_days=5)
        plan = planner.plan(user)
        total = sum(cp.num_days for cp in plan.city_plans)
        assert total == 5, f"Total days must be 5, got {total}"

    def test_no_places_outside_travel_month(self):
        # Travel in June (monsoon) – only Dudhsagar should appear in Goa
        planner = TravelPlanner(KB)
        user = self._make_user(
            cities=["Goa"], travel_month="Jun",
            interests=["nature", "beach", "heritage"])
        plan = planner.plan(user)
        for cp in plan.city_plans:
            for day in cp.daily_schedule:
                for place in day:
                    assert "Jun" in place.best_season, \
                        f"{place.name} is not recommended in June"

    def test_veg_user_only_gets_veg_food(self):
        planner = TravelPlanner(KB)
        user = self._make_user(dietary=["veg"])
        plan = planner.plan(user)
        for cp in plan.city_plans:
            for meal_day in cp.meal_plan:
                for food in meal_day.values():
                    if food:
                        assert "veg" in food.dietary, \
                            f"{food.name} is not vegetarian"

    def test_cost_is_positive(self):
        planner = TravelPlanner(KB)
        plan = planner.plan(self._make_user())
        assert plan.total_cost > 0

    def test_luxury_costs_more_than_budget(self):
        planner = TravelPlanner(KB)
        luxury_user = self._make_user(budget_style="luxury")
        budget_user = self._make_user(budget_style="budget")
        luxury_plan = planner.plan(luxury_user)
        budget_plan  = planner.plan(budget_user)
        assert luxury_plan.total_cost > budget_plan.total_cost

    def test_daily_schedule_within_8_hours(self):
        planner = TravelPlanner(KB)
        user = self._make_user(interests=["heritage","nature","beach","adventure"], num_days=3)
        plan = planner.plan(user)
        for cp in plan.city_plans:
            for day in cp.daily_schedule:
                total_hrs = sum(p.duration_hrs for p in day)
                assert total_hrs <= 8.0, f"Day exceeds 8h: {total_hrs}h"

    def test_display_does_not_crash(self):
        planner = TravelPlanner(KB)
        plan = planner.plan(self._make_user(cities=["Hyderabad","Jaipur"], num_days=6))
        # Should not raise
        import io, sys
        old = sys.stdout
        sys.stdout = io.StringIO()
        try:
            plan.display()
        finally:
            sys.stdout = old


# ── runner ──────────────────────────────────────────────

if __name__ == '__main__':
    suites = [TestKnowledgeBase, TestInferenceEngine, TestTravelPlanner]
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
                if ok:
                    passed += 1
    print(f"\n{'═'*50}")
    print(f"  Results: {passed}/{total} tests passed")
    print(f"{'═'*50}")
