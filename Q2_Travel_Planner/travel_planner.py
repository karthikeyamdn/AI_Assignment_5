"""
Q2: AI-Based Travel Planner with Knowledge Bases
=================================================
Reuses existing knowledge in the domain:
  • Tourist Places KB   – attractions, category, city, best season
  • Food KB             – local dishes, cuisine type, dietary tags
  • Cost KB             – per-day budget ranges by city and style
  • Personalisation KB  – preference → attraction/food matching rules

Architecture
────────────
  KnowledgeBase  – dict-backed store with SPARQL-like query API
  InferenceEngine – forward-chaining rule engine
  TravelPlanner  – orchestrates KB + IE to generate tour plans
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import random, textwrap

# ─────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────

@dataclass
class Place:
    name: str
    city: str
    category: str          # heritage | nature | adventure | religious | beach
    best_season: list[str] # e.g. ["Oct","Nov","Dec","Jan","Feb"]
    entry_fee: float       # INR
    duration_hrs: float    # suggested visit hours
    rating: float          # 1-5
    tags: list[str] = field(default_factory=list)


@dataclass
class Food:
    name: str
    city: str
    cuisine: str
    dietary: list[str]     # veg | non-veg | vegan | gluten-free
    avg_cost: float        # INR per person per meal
    meal_type: list[str]   # breakfast | lunch | dinner | snack


@dataclass
class CostProfile:
    city: str
    style: str             # budget | moderate | luxury
    hotel_per_night: float # INR
    transport_per_day: float
    misc_per_day: float


@dataclass
class UserProfile:
    name: str
    budget_style: str                    # budget | moderate | luxury
    interests: list[str]                 # categories of places
    dietary: list[str]                   # veg | non-veg | vegan
    travel_month: str                    # 3-letter month e.g. "Jan"
    num_days: int
    cities: list[str]                    # preferred cities to visit


# ─────────────────────────────────────────────────────────
#  KNOWLEDGE BASES
# ─────────────────────────────────────────────────────────

class KnowledgeBase:
    """
    Simple triple-store-style KB.
    Entities are stored as typed dicts; queries use Python filtering.
    """

    def __init__(self):
        self.places: list[Place] = []
        self.foods:  list[Food]  = []
        self.costs:  list[CostProfile] = []
        self._rules: list[dict]  = []   # for inference engine

    # ── loading ─────────────────────────────────────────────

    def load_places(self, records: list[Place]):
        self.places.extend(records)

    def load_foods(self, records: list[Food]):
        self.foods.extend(records)

    def load_costs(self, records: list[CostProfile]):
        self.costs.extend(records)

    def add_rule(self, rule: dict):
        """Add an IF-THEN rule for the inference engine."""
        self._rules.append(rule)

    # ── queries ─────────────────────────────────────────────

    def query_places(self, city: str = None, category: str = None,
                     month: str = None, min_rating: float = 0.0) -> list[Place]:
        results = self.places
        if city:
            results = [p for p in results if p.city == city]
        if category:
            results = [p for p in results if p.category == category]
        if month:
            results = [p for p in results if month in p.best_season]
        return [p for p in results if p.rating >= min_rating]

    def query_foods(self, city: str = None, dietary: list[str] = None) -> list[Food]:
        results = self.foods
        if city:
            results = [f for f in results if f.city == city]
        if dietary:
            results = [f for f in results
                       if any(d in f.dietary for d in dietary)]
        return results

    def query_cost(self, city: str, style: str) -> CostProfile | None:
        for c in self.costs:
            if c.city == city and c.style == style:
                return c
        return None


# ─────────────────────────────────────────────────────────
#  INFERENCE ENGINE  (forward-chaining rule base)
# ─────────────────────────────────────────────────────────

class InferenceEngine:
    """
    Forward-chaining rule engine.

    Rules have the shape:
        { "if": {"field": value, ...}, "then": {"field": value, ...} }

    The engine matches the IF part against a fact dict and adds the
    THEN part to the working memory when all conditions are satisfied.
    """

    def __init__(self, rules: list[dict]):
        self.rules = rules

    def infer(self, facts: dict) -> dict:
        """
        Run forward chaining until no new facts are derived.

        Parameters
        ----------
        facts : initial fact dictionary

        Returns
        -------
        dict : enriched fact dictionary after all applicable rules fire
        """
        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                conditions = rule["if"]
                conclusions = rule["then"]
                # check if all conditions are met
                if all(facts.get(k) == v for k, v in conditions.items()):
                    for k, v in conclusions.items():
                        if facts.get(k) != v:
                            facts[k] = v
                            changed = True
        return facts


# ─────────────────────────────────────────────────────────
#  TRAVEL PLANNER
# ─────────────────────────────────────────────────────────

class TravelPlanner:
    """
    Orchestrates the KB and IE to produce a personalised tour plan.

    Steps
    ─────
    1. Build the user's fact dict and run the inference engine to
       derive any implicit preferences.
    2. For each city in the user's itinerary, query the KB for
       matching places (filtered by month, interest, rating).
    3. Assign places to days respecting visit-duration constraints.
    4. Recommend local foods that match dietary preferences.
    5. Compute cost breakdown per city and total trip cost.
    6. Return a structured TourPlan.
    """

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.engine = InferenceEngine(kb._rules)

    def plan(self, user: UserProfile) -> "TourPlan":
        # ── 1. Inference ──────────────────────────────────────
        facts = {
            "budget_style":   user.budget_style,
            "dietary":        ",".join(user.dietary),
            "travel_month":   user.travel_month,
            "num_days":       user.num_days,
        }
        facts = self.engine.infer(facts)

        days_per_city = self._split_days(user.cities, user.num_days)

        city_plans = []
        total_cost = 0.0

        for city, num_days in days_per_city.items():
            # ── 2. Select places ─────────────────────────────
            candidates = []
            for interest in user.interests:
                candidates += self.kb.query_places(
                    city=city, category=interest,
                    month=user.travel_month, min_rating=3.5)
            # de-dup preserving order
            seen = set()
            unique = []
            for p in candidates:
                if p.name not in seen:
                    seen.add(p.name)
                    unique.append(p)

            # ── 3. Day-by-day schedule ───────────────────────
            daily_schedule = self._schedule(unique, num_days)

            # ── 4. Food recommendations ──────────────────────
            foods = self.kb.query_foods(city=city, dietary=user.dietary)
            meal_plan = self._assign_meals(foods, num_days)

            # ── 5. Cost ──────────────────────────────────────
            cost_profile = self.kb.query_cost(city, user.budget_style)
            city_cost = self._compute_cost(cost_profile, unique, num_days)
            total_cost += city_cost

            city_plans.append(CityPlan(
                city=city,
                num_days=num_days,
                daily_schedule=daily_schedule,
                meal_plan=meal_plan,
                cost_profile=cost_profile,
                city_total_cost=city_cost,
            ))

        return TourPlan(user=user, city_plans=city_plans,
                        total_cost=total_cost, inferred_facts=facts)

    # ── internal helpers ────────────────────────────────────

    def _split_days(self, cities: list[str], total: int) -> dict[str, int]:
        """Distribute total days roughly equally across cities."""
        n = len(cities)
        base, extra = divmod(total, n)
        return {c: base + (1 if i < extra else 0)
                for i, c in enumerate(cities)}

    def _schedule(self, places: list[Place],
                  num_days: int) -> list[list[Place]]:
        """
        Greedily pack places into days (≤ 8 h per day).
        """
        days: list[list[Place]] = [[] for _ in range(num_days)]
        hours = [0.0] * num_days
        MAX_HOURS = 8.0

        for place in places:
            for d in range(num_days):
                if hours[d] + place.duration_hrs <= MAX_HOURS:
                    days[d].append(place)
                    hours[d] += place.duration_hrs
                    break

        return days

    def _assign_meals(self, foods: list[Food],
                      num_days: int) -> list[dict[str, Food | None]]:
        """Assign breakfast, lunch, dinner for each day."""
        meals = []
        for _ in range(num_days):
            day_meals: dict[str, Food | None] = {}
            for meal_type in ("breakfast", "lunch", "dinner"):
                options = [f for f in foods if meal_type in f.meal_type]
                day_meals[meal_type] = random.choice(options) if options else None
            meals.append(day_meals)
        return meals

    def _compute_cost(self, profile: CostProfile | None,
                      places: list[Place], num_days: int) -> float:
        if profile is None:
            return 0.0
        accommodation = profile.hotel_per_night * num_days
        transport     = profile.transport_per_day * num_days
        misc          = profile.misc_per_day * num_days
        entry_fees    = sum(p.entry_fee for p in places)
        return accommodation + transport + misc + entry_fees


# ─────────────────────────────────────────────────────────
#  OUTPUT DATA CLASSES
# ─────────────────────────────────────────────────────────

@dataclass
class CityPlan:
    city: str
    num_days: int
    daily_schedule: list[list[Place]]
    meal_plan: list[dict]
    cost_profile: CostProfile | None
    city_total_cost: float


@dataclass
class TourPlan:
    user: UserProfile
    city_plans: list[CityPlan]
    total_cost: float
    inferred_facts: dict

    def display(self):
        sep = "═" * 60
        print(f"\n{sep}")
        print(f"  PERSONALISED TOUR PLAN FOR {self.user.name.upper()}")
        print(sep)
        print(f"  Duration : {self.user.num_days} days "
              f"| Budget: {self.user.budget_style.title()}"
              f"| Month : {self.user.travel_month}")
        print(f"  Interests: {', '.join(self.user.interests)}")
        print(f"  Dietary  : {', '.join(self.user.dietary)}")
        if self.inferred_facts.get("recommend_travel_insurance"):
            print("  ⚠  Inference: Travel insurance recommended "
                  "(international travel detected)")
        print()

        for cp in self.city_plans:
            print(f"  ── {cp.city.upper()} ({cp.num_days} days) ──")
            for d, places in enumerate(cp.daily_schedule, 1):
                print(f"\n    Day {d}:")
                if places:
                    for p in places:
                        print(f"      🏛  {p.name} ({p.category}) "
                              f"~ {p.duration_hrs}h  ₹{p.entry_fee:.0f}")
                else:
                    print("      (leisure / travel day)")
                meals = cp.meal_plan[d - 1] if d - 1 < len(cp.meal_plan) else {}
                for meal, food in meals.items():
                    if food:
                        print(f"      🍽  {meal.title()}: {food.name} "
                              f"({food.cuisine}) ~₹{food.avg_cost:.0f}/person")

            if cp.cost_profile:
                print(f"\n    Cost Breakdown ({cp.city}):")
                p = cp.cost_profile
                print(f"      Hotel      : ₹{p.hotel_per_night:.0f}/night × "
                      f"{cp.num_days}  = ₹{p.hotel_per_night * cp.num_days:,.0f}")
                print(f"      Transport  : ₹{p.transport_per_day:.0f}/day × "
                      f"{cp.num_days}   = ₹{p.transport_per_day * cp.num_days:,.0f}")
                print(f"      Entry fees : ₹"
                      f"{sum(pl.entry_fee for day in cp.daily_schedule for pl in day):,.0f}")
                print(f"      City Total : ₹{cp.city_total_cost:,.0f}")

        print(f"\n{sep}")
        print(f"  ESTIMATED TOTAL TRIP COST : ₹{self.total_cost:,.0f}")
        print(f"{sep}\n")


# ─────────────────────────────────────────────────────────
#  POPULATE THE KNOWLEDGE BASE  (India tourism domain)
# ─────────────────────────────────────────────────────────

def build_knowledge_base() -> KnowledgeBase:
    kb = KnowledgeBase()

    # ── Tourist Places ───────────────────────────────────────
    places = [
        # Hyderabad
        Place("Charminar",         "Hyderabad", "heritage",   ["Oct","Nov","Dec","Jan","Feb","Mar"], 25,  1.5, 4.5, ["UNESCO","iconic"]),
        Place("Golconda Fort",     "Hyderabad", "heritage",   ["Oct","Nov","Dec","Jan","Feb"],       15,  2.5, 4.4, ["fort","history"]),
        Place("Hussain Sagar Lake","Hyderabad", "nature",     ["Oct","Nov","Dec","Jan","Feb","Mar"], 0,   1.0, 4.0, ["lake","boating"]),
        Place("Salar Jung Museum", "Hyderabad", "heritage",   ["Oct","Nov","Dec","Jan","Feb","Mar"], 20,  2.0, 4.3, ["museum","art"]),
        Place("Ramoji Film City",  "Hyderabad", "adventure",  ["Oct","Nov","Dec","Jan","Feb","Mar"], 1500,6.0, 4.2, ["entertainment"]),
        Place("Birla Mandir",      "Hyderabad", "religious",  ["Oct","Nov","Dec","Jan","Feb","Mar"], 0,   1.0, 4.1, ["temple","hilltop"]),
        # Goa
        Place("Calangute Beach",   "Goa",       "beach",      ["Nov","Dec","Jan","Feb","Mar"],       0,   3.0, 4.3, ["beach","water-sports"]),
        Place("Baga Beach",        "Goa",       "beach",      ["Nov","Dec","Jan","Feb","Mar"],       0,   2.5, 4.1, ["beach","nightlife"]),
        Place("Basilica of Bom Jesus","Goa",    "heritage",   ["Oct","Nov","Dec","Jan","Feb","Mar"], 0,   1.5, 4.5, ["church","UNESCO"]),
        Place("Dudhsagar Falls",   "Goa",       "nature",     ["Jun","Jul","Aug","Sep","Oct"],       400, 4.0, 4.6, ["waterfall","trek"]),
        Place("Fort Aguada",       "Goa",       "heritage",   ["Oct","Nov","Dec","Jan","Feb","Mar"], 0,   1.5, 4.2, ["fort","view"]),
        # Jaipur
        Place("Amber Fort",        "Jaipur",    "heritage",   ["Oct","Nov","Dec","Jan","Feb","Mar"], 200, 3.0, 4.7, ["fort","UNESCO"]),
        Place("Hawa Mahal",        "Jaipur",    "heritage",   ["Oct","Nov","Dec","Jan","Feb"],       50,  1.0, 4.5, ["palace","iconic"]),
        Place("Jaigarh Fort",      "Jaipur",    "heritage",   ["Oct","Nov","Dec","Jan","Feb","Mar"], 35,  2.0, 4.3, ["fort","cannons"]),
        Place("Nahargarh Fort",    "Jaipur",    "heritage",   ["Oct","Nov","Dec","Jan","Feb","Mar"], 50,  2.0, 4.4, ["fort","sunset"]),
        Place("Jal Mahal",         "Jaipur",    "nature",     ["Oct","Nov","Dec","Jan","Feb","Mar"], 0,   0.5, 4.2, ["lake","palace"]),
    ]
    kb.load_places(places)

    # ── Food KB ──────────────────────────────────────────────
    foods = [
        # Hyderabad
        Food("Hyderabadi Biryani","Hyderabad","Mughlai",    ["non-veg"],              350, ["lunch","dinner"]),
        Food("Haleem",            "Hyderabad","Mughlai",    ["non-veg"],              200, ["lunch","dinner"]),
        Food("Mirchi ka Salan",   "Hyderabad","Hyderabadi", ["veg","vegan"],          120, ["lunch","dinner"]),
        Food("Osmania Biscuits",  "Hyderabad","Bakery",     ["veg"],                  50,  ["breakfast","snack"]),
        Food("Irani Chai",        "Hyderabad","Beverage",   ["veg","vegetarian"],     30,  ["breakfast","snack"]),
        Food("Pesarattu",         "Hyderabad","South Indian",["veg","vegan","gluten-free"], 80, ["breakfast"]),
        # Goa
        Food("Fish Curry Rice",   "Goa",      "Goan",       ["non-veg"],              250, ["lunch","dinner"]),
        Food("Prawn Balchão",     "Goa",      "Goan",       ["non-veg"],              300, ["dinner"]),
        Food("Bebinca",           "Goa",      "Dessert",    ["veg"],                  120, ["snack","dinner"]),
        Food("Pão de Queijo",     "Goa",      "Bakery",     ["veg"],                  60,  ["breakfast"]),
        Food("Veg Xacuti",        "Goa",      "Goan",       ["veg","vegan"],          180, ["lunch","dinner"]),
        # Jaipur
        Food("Dal Baati Churma",  "Jaipur",   "Rajasthani", ["veg"],                  150, ["lunch","dinner"]),
        Food("Laal Maas",         "Jaipur",   "Rajasthani", ["non-veg"],              300, ["dinner"]),
        Food("Pyaaz ki Kachori",  "Jaipur",   "Street",     ["veg"],                  30,  ["breakfast","snack"]),
        Food("Mawa Kachori",      "Jaipur",   "Street",     ["veg"],                  40,  ["breakfast","snack"]),
        Food("Ker Sangri",        "Jaipur",   "Rajasthani", ["veg","vegan","gluten-free"], 130, ["lunch","dinner"]),
    ]
    kb.load_foods(foods)

    # ── Cost Profiles ─────────────────────────────────────────
    costs = [
        CostProfile("Hyderabad", "budget",   800,  200, 300),
        CostProfile("Hyderabad", "moderate", 2500, 500, 600),
        CostProfile("Hyderabad", "luxury",   8000, 1500, 1500),
        CostProfile("Goa",       "budget",   1200, 300, 400),
        CostProfile("Goa",       "moderate", 4000, 700, 800),
        CostProfile("Goa",       "luxury",   12000,2000, 2000),
        CostProfile("Jaipur",    "budget",   900,  250, 300),
        CostProfile("Jaipur",    "moderate", 3000, 600, 700),
        CostProfile("Jaipur",    "luxury",   10000,1800, 1800),
    ]
    kb.load_costs(costs)

    # ── Inference Rules ──────────────────────────────────────
    # Rule 1: Luxury travellers get travel-insurance recommendation
    kb.add_rule({
        "if":  {"budget_style": "luxury"},
        "then": {"recommend_travel_insurance": True}
    })
    # Rule 2: Budget travellers → suggest shared transport
    kb.add_rule({
        "if":  {"budget_style": "budget"},
        "then": {"transport_mode": "shared"}
    })
    # Rule 3: Veg-only travellers get filtered food list (flag)
    kb.add_rule({
        "if":  {"dietary": "veg"},
        "then": {"strict_vegetarian": True}
    })

    return kb


# ─────────────────────────────────────────────────────────
#  DEMO
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    random.seed(42)
    kb = build_knowledge_base()
    planner = TravelPlanner(kb)

    # User 1 – heritage + food lover, moderate budget
    user1 = UserProfile(
        name="Arjun",
        budget_style="moderate",
        interests=["heritage", "religious"],
        dietary=["veg"],
        travel_month="Jan",
        num_days=7,
        cities=["Hyderabad", "Jaipur"],
    )
    plan1 = planner.plan(user1)
    plan1.display()

    # User 2 – beach + adventure, budget
    user2 = UserProfile(
        name="Priya",
        budget_style="budget",
        interests=["beach", "nature", "adventure"],
        dietary=["non-veg"],
        travel_month="Dec",
        num_days=5,
        cities=["Goa"],
    )
    plan2 = planner.plan(user2)
    plan2.display()
