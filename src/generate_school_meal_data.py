"""
Synthetic School-Meal Subscription Dataset Generator (v2)
=========================================================
Fully synthetic data that mirrors the schema of a US school-lunch pre-ordering
platform. Calibrated to aggregate EDA statistics only; no real records used.

Key guarantees
- Every lunch is chosen from that school's published menu for that delivery date.
- No lunch ever conflicts with a student's allergens / dietary preferences.
- At most one live (non-cancelled) lunch per student per delivery date.
- Subscriptions are locked per school year (same days-per-week, same weekdays,
  same included sides); subscription-coded lunches fall only on plan weekdays.
- Menus run in ~2-month cycles; ~30% of each school's line-up changes per cycle
  (items rotate out/back in, or move weekday).
- Food preferences depend on age (grade).

Usage: python generate_school_meal_data.py --seed 42 --out ./data
"""
import argparse, csv, json, os, time
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd

Q = '""""""'  # empty-string export artifact seen in the source sample
EPOCH = date(1970, 1, 1)

# ----------------------------------------------------------------------------- reference
ALLERGENS = [(1, "Milk", "allergen"), (2, "Egg", "allergen"), (3, "Soy", "allergen"),
             (4, "Wheat", "allergen"), (5, "Peanut", "allergen"), (6, "Tree nut", "allergen"),
             (7, "Fish", "allergen"), (8, "Shellfish", "allergen"), (9, "Gluten", "allergen"),
             (10, "Beef", "preference"), (11, "Pork", "preference"),
             (12, "NonVegetarian", "preference"), (13, "NonVegan", "preference"),
             (14, "Sesame", "allergen"), (15, "Coconut", "allergen")]
ALG_PREV = {1: 10, 2: 6, 3: 3, 4: 4, 5: 14, 6: 10, 7: 3, 8: 6, 9: 9, 10: 4, 11: 6,
            12: 9, 13: 2, 14: 4, 15: 2}
ALG_CODE = {"M": 1, "E": 2, "S": 3, "W": 4, "P": 5, "T": 6, "F": 7, "H": 8, "G": 9, "Se": 14, "C": 15}

CUISINES = {1: "Classic American", 2: "Italian", 3: "Breakfast for Lunch", 4: "Asian",
            5: "Sandwich", 6: "Mexican", 7: "Cool Munchies", 8: "Sushi", 9: "Salad", 10: "Campus Cafe"}
CUISINE_POP = {1: 61, 2: 55, 3: 41, 4: 38, 5: 34, 6: 29, 7: 21, 8: 19, 9: 14, 10: 30}

# age profiles: (peak grade, width, base weight, floor)
AGE = {"K": (1.0, 2.6, 1.0, .12), "H2": (2.0, 1.3, 1.0, .08), "M": (5.0, 3.0, 1.0, .15),
       "E8": (8.0, 1.7, 1.0, .12), "T": (11.0, 3.2, 1.0, .10), "A": (5.0, 8.0, 1.0, .30)}
PROTEINS = ["beef", "pork", "chicken", "fish", "veg", "vegan"]

# name | code | cuisine | serving | adventurous | premium | protein | allergens | age | pop | anchor(dow:p) | flags
CAT = """Chicken Tenders|TEND|1|Hot|0|0|chicken|E W G|M|0.62||staple
Mac and Cheese|MAC|1|Hot|0|0|veg|M W G|K|1.1||
All Beef Hot Dog|HOTD|1|Hot|0|0|beef|W G|H2|1.0||
Cheeseburger|BURG|1|Hot|0|0|beef|M W G Se|E8|1.1|2:.80|
Grilled Cheese|GRIL|1|Hot|0|0|veg|M W G|K|0.9||
Fish Sticks|FISH|1|Hot|1|0|fish|W F G|M|0.8|4:.95|
Chicken Pot Pie|PIE|1|Hot|1|0|chicken|M E W G|M|0.6||
Shepherds Pie|SHEP|1|Hot|2|0|beef|M|M|0.5||
Corn Dog Bites|CORN|1|Hot|0|0|beef|E W G|K|0.9||
Popcorn Chicken|POPC|1|Hot|0|0|chicken|E W G|K|0.9||
Sloppy Joe|SLOP|1|Hot|1|0|beef|W G S|M|0.6||
BBQ Chicken Sliders|SLID|1|Hot|1|0|chicken|W G|T|0.7||
Black Bean Veggie Burger|GARD|1|Hot|2|0|veg|S W G|T|0.5||
Gluten Free Hamburger|BURG|1|Hot|1|1|beef|M|T|0.6||
Bacon Potato Skins|POTA|1|Hot|1|0|pork|M|T|0.7||
Turkey Meatloaf|LOAF|1|Hot|2|0|chicken|E W G|M|0.4||
Chicken Nuggets and Fries|NUGG|1|Hot|0|0|chicken|W G|K|1.0||
Cheese Pizza|PIZZ|2|Hot|0|0|veg|M W G|M|0.66||staple
Pepperoni Pizza|PIZZ|2|Hot|0|0|pork|M W G|A|1.2|0:.86|
Buttered Noodles|NOOD|2|Hot|0|0|veg|M W G|K|1.0||
Spaghetti and Meatballs|PAST|2|Hot|1|0|beef|M E W G|M|0.8||
Pesto Pasta|PAST|2|Hot|2|0|veg|M T W G|T|0.6||
Chicken Alfredo|PAST|2|Hot|1|0|chicken|M W G|T|0.8||
Baked Ziti|ZITI|2|Hot|1|0|veg|M W G|M|0.7||
Pepperoni Pizza Pocket|POCK|2|Hot|1|0|pork|M W G|M|0.8||
BBQ Chicken Pizza Pocket|POCK|2|Hot|2|0|chicken|M W G|T|0.6||
Cheese Ravioli|RAVI|2|Hot|1|0|veg|M E W G|M|0.6||
Beef Lasagna|LASA|2|Hot|2|0|beef|M E W G|T|0.6||
Plant Based Meatball Sub|HOT|2|Hot|3|0|veg|M S W G|T|0.4||
Margherita Flatbread|FLAT|2|Hot|2|0|veg|M W G|T|0.6||
Pancakes with Turkey Sausage|CAKE|3|Hot|0|0|chicken|M E W G|K|1.15||
French Toast Sticks|TOAS|3|Hot|0|0|veg|M E W G|K|1.15||
Waffles and Berries|WAFF|3|Hot|0|0|veg|M E W G|K|1.0||
Bacon Egg and Cheese Burrito|RITO|3|Hot|2|0|pork|M E W G|T|0.8||
Breakfast Sandwich|BRKS|3|Hot|1|0|pork|M E W G|T|0.7||
Yogurt Parfait Box|PARF|3|Cold|1|0|veg|M|M|0.5||
Mini Bagels and Cream Cheese|BAGL|3|Cold|0|0|veg|M W G Se|M|0.6||
Chicken Fried Rice|RICE|4|Hot|1|0|chicken|E S|M|0.8||
Teriyaki Chicken Bowl|BOWL|4|Hot|2|0|chicken|S W G Se|T|0.9||
Chicken Potstickers|POTS|4|Hot|2|0|chicken|S W G Se|M|0.8||
Orange Chicken|BOWL|4|Hot|1|0|chicken|E S W G|T|0.9||
Veggie Lo Mein|NOOD|4|Hot|2|0|vegan|S W G Se|T|0.5||
Sesame Noodles|NOOD|4|Hot|2|0|vegan|S W G Se P|T|0.4||
Chicken Zucchini Stir Fry|HOT|4|Hot|3|0|chicken|S|T|0.4||
Coconut Curry Chicken|HOT|4|Hot|3|1|chicken|C|T|0.4||
Beef and Broccoli|BOWL|4|Hot|2|0|beef|S W G|T|0.6||
Pork Bao Buns|BAO|4|Hot|3|0|pork|S W G Se|T|0.4||
Turkey and Cheese Sandwich|COLD|5|Cold|1|0|chicken|M W G|T|0.7||
Ham and Cheese Roll Up|ROLL|5|Cold|0|0|pork|M W G|K|0.8||
Tuna Salad Sandwich|COLD|5|Cold|2|0|fish|E W F G|T|0.5|4:.92|
Chicken Salad Sandwich|COLD|5|Cold|2|0|chicken|E W G|T|0.5|2:.90|
BLTA Sandwich|COLD|5|Cold|2|0|pork|W G|T|0.5|0:.90|
Sunbutter and Jelly Sandwich|COLD|5|Cold|0|0|vegan|W G|K|0.8||
Ham and Cheese Sliders|SLID|5|Cold|1|0|pork|M W G|M|0.8||
Italian Sub|SUB|5|Cold|2|0|pork|M W G|T|0.6||
Turkey Club Wrap|WRAP|5|Cold|1|0|pork|W G|T|0.6||
Caprese Panini|PANI|5|Hot|2|0|veg|M W G|T|0.4||
Bean and Cheese Burrito|RITO|6|Hot|1|0|veg|M W G|M|0.8||
Chicken Quesadilla|QUES|6|Hot|1|0|chicken|M W G|M|0.9||
Beef Tacos|TACO|6|Hot|2|0|beef|M W G|T|0.8|1:.85|
Chicken Fajita Rice Bowl|BOWL|6|Hot|2|0|chicken|M|T|0.7|0:.90|
Nachos Supreme|NACH|6|Hot|1|0|beef|M|T|0.8||
Cheese Enchiladas|ENCH|6|Hot|1|0|veg|M W G|M|0.5||
Chicken Burrito Bowl|BOWL|6|Hot|2|0|chicken|M|T|0.7||
Chicken Taquitos|TAQU|6|Hot|1|0|chicken|M W G|M|0.6||
Cheese and Crackers Box|BOX|7|Cold|0|0|veg|M W G|K|0.8||
Turkey Pepperoni Snack Box|BOX|7|Cold|1|0|chicken|M W G|M|0.7||
Hummus and Veggie Box|BOX|7|Cold|2|0|vegan|W G Se|T|0.4||
Pretzel and Cheese Dip Box|BOX|7|Cold|0|0|veg|M W G|K|0.7||
Chicken Caesar Wrap Box|BOX|7|Cold|2|0|chicken|M E F W G|T|0.5||
California Roll|SUSH|8|Cold|3|1|fish|S W H G Se|T|0.7||
Cucumber Avocado Roll|SUSH|8|Cold|3|1|vegan|S Se|T|0.5||
Teriyaki Chicken Roll|SUSH|8|Cold|3|1|chicken|S W G Se|T|0.6||
Spicy Tuna Roll|SUSH|8|Cold|3|1|fish|S F E Se|T|0.4||
Chicken Caesar Salad|SALA|9|Cold|2|0|chicken|M E F W G|T|0.5||
Garden Salad|SALA|9|Cold|2|0|veg|M|T|0.3||
Cobb Salad|SALA|9|Cold|3|0|pork|M E|T|0.3||
Southwest Chicken Salad|SALA|9|Cold|2|0|chicken|M|T|0.4||
Asian Chicken Salad|SALA|9|Cold|3|0|chicken|S W G Se|T|0.3||
Campus Cafe Turkey Wrap|CAFE|10|Cold|1|0|chicken|W G|A|0.9||excl
Campus Cafe Pasta Bowl|CAFE|10|Hot|1|0|veg|M W G|A|1.0||excl
Campus Cafe Burrito Bowl|CAFE|10|Hot|2|0|beef|M|T|0.8||excl
Campus Cafe Chicken Plate|CAFE|10|Hot|1|0|chicken||A|0.9||excl
Campus Cafe Grain Bowl|CAFE|10|Hot|3|0|vegan|S|T|0.5||excl
Campus Cafe Pizza Slice|CAFE|10|Hot|0|0|veg|M W G|K|1.1||excl
Campus Cafe Noodle Cup|CAFE|10|Hot|1|0|veg|W G S|M|0.8||excl
Campus Cafe Chicken Sandwich|CAFE|10|Hot|1|0|chicken|E W G|M|0.9||excl"""

# name | type | price | allergens | popularity | paired cuisines | season months (blank=all)
COMP = """Whole Grain Chips|Snack|0.90|W G|1.0|1 6 5|
Sea Salt Kettle Chips|Snack|0.90||1.1|1 5 6|
Cheddar Cheese Puffs|Snack|0.90|M|0.9|1 2|
Apple Crisps|Snack|1.25||0.6|3 5|
Organic Apple Sauce|Snack|1.25||0.8|3 1|
Seaweed Snacks|Snack|1.25|S Se|0.5|4 8|
Giant Chocolate Chunk Cookie|Snack|1.50|M E W G|1.2|2 1|
Turkey Pepperoni Sticks|Snack|1.50||0.5|7 5|
Pretzel Twists|Snack|0.90|W G|0.7|5 7|
Yogurt Tube|Snack|1.00|M|0.7|3|
String Cheese|Snack|1.00|M|0.8|7 2|
Hummus Cup|Snack|1.25|Se|0.3|9 7|
Graham Crackers|Snack|0.90|W G|0.6|3|
Veggie Straws|Snack|0.90||0.7|5 1|
Cantaloupe|Fruit|1.00||0.6|3 9|5 6 7 8 9
Honeydew|Fruit|1.00||0.4|3|5 6 7 8 9
Watermelon|Fruit|1.25||0.9|1 6|5 6 7 8 9
Fresh Pineapple|Fruit|1.25||0.7|4 8|
Red Grapes|Fruit|1.25||0.8|5 7|
Apple Slices|Fruit|1.00||0.9|5 1|9 10 11 12 1 2 3 4
Mandarin Oranges|Fruit|1.00||0.7|4|11 12 1 2 3
Strawberries|Fruit|1.25||0.8|3 2|3 4 5 6
Banana|Fruit|0.80||0.6|3|
Broccoli|Fruit|0.80||0.3|4 1|
Baby Carrots|Fruit|0.80||0.5|7 5|
Cucumber Slices|Fruit|0.80||0.3|8 9|
Milk 1 Percent|Drink|1.00|M|1.0|3 1|
Chocolate Milk|Drink|1.00|M|1.3|1 2 3|
Fruit Infused Water|Drink|1.25||0.6|9 8|
Apple Juice Box|Drink|1.00||0.9|3 6|
Sparkling Water|Drink|1.25||0.4|9|
Orange Juice|Drink|1.00||0.6|3|
Mega Mango Smoothie|Smoothie|3.50|M|1.6||
Strawberry Whirl Smoothie|Smoothie|3.50|M|1.7||
Apple n Greens Smoothie|Smoothie|3.50|M|1.1||"""
SMOOTHIE_DOW = {"Mega Mango Smoothie": 1, "Strawberry Whirl Smoothie": 3, "Apple n Greens Smoothie": 1}

FIRST = ("Ava Liam Mia Noah Zoe Ethan Aria Lucas Isla Mateo Nora Kai Leah Owen Maya Ezra Ruby Arjun Ivy Leo "
         "Sofia Jonah Anaya Theo Hana Diego Clara Rohan Elena Felix Priya Miles Layla Kenji Camila Rowan Amara "
         "Jude Mei Oscar Naomi Ari Lena Sami Iris Omar Tessa Nico Wren Eli Ada Rafael Zara Dev Lila Hugo Nina "
         "Arlo Esme Malik Yara Ravi Luca Freya Tomas Aisha Jasper Vera Kian Sana Emil Juno Adrian Talia").split()
LAST = ("Alder Brook Calloway Delgado Ellison Fairbanks Garnett Holloway Ibarra Jansen Kessler Lindqvist "
        "Marlowe Nakamura Okafor Pellegrini Quintero Rasmussen Sorensen Thakkar Underhill Vasquez Whitlock Xu "
        "Yamada Zielinski Ashworth Bhandari Castellanos Dunmore Everly Fontaine Greer Hollis Iyer Kowalski "
        "Lombardi Mendez Novak Ortega Patel Reyes Sato Torres Varga Weston Abara Beaumont Chandra Dorsey").split()
PLACE = ("Oak Cedar Willow Harbor Bay Sunny Mission Redwood Lakeshore Valley Maple Pine Aspen Birch Summit "
         "Meadow River Crest Hillside Brookside Fairview Riverside Eastwood Westfield Northgate Southpark "
         "Greenfield Clearwater Stonebridge Highland Parkside Lincoln Franklin Jefferson Hawthorne Magnolia "
         "Sycamore Laurel Juniper Sequoia Cypress Alamo Vista Monte Sierra Ridgeview Canyon Horizon").split()
CITIES = [("San Jose", "Santa Clara"), ("Palo Alto", "Santa Clara"), ("Fremont", "Alameda"),
          ("Oakland", "Alameda"), ("Pleasanton", "Alameda"), ("San Mateo", "San Mateo"),
          ("Burlingame", "San Mateo"), ("Walnut Creek", "Contra Costa"), ("Orinda", "Contra Costa"),
          ("Sacramento", "Sacramento"), ("Davis", "Yolo"), ("Vacaville", "Solano"),
          ("Santa Rosa", "Sonoma"), ("Los Angeles", "Los Angeles"), ("Pasadena", "Los Angeles")]

# school bands: (label, grades, count, type options)
BANDS = [("Elementary", ["K", "1", "2", "3", "4", "5"], 38, ["Public", "Private - Secular", "Private - Catholic"]),
         ("TK-5", ["TK", "K", "1", "2", "3", "4", "5"], 5, ["Public", "Private - Secular"]),
         ("K-8", ["K", "1", "2", "3", "4", "5", "6", "7", "8"], 27, ["Private - Catholic", "Private - Secular", "Private - Christian"]),
         ("Middle", ["6", "7", "8"], 15, ["Public", "Private - Secular"]),
         ("High", ["9", "10", "11", "12"], 9, ["Private - Secular", "Private - Catholic"]),
         ("K-12", ["K", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"], 8, ["Private - Secular", "Private - Christian"]),
         ("Camp", ["K", "1", "2", "3", "4", "5", "6", "7", "8"], 3, ["WeeklyCamp"])]
GNUM = {"PS": -1.0, "TK": -0.5, "K": 0.0, **{str(i): float(i) for i in range(1, 13)}, "Staff": 15.0}

START, END = date(2023, 8, 14), date(2025, 6, 13)
SY = [(date(2023, 8, 1), date(2024, 6, 14)), (date(2024, 8, 1), date(2025, 6, 13))]
CAMP = (date(2024, 6, 17), date(2024, 8, 9))
CYCLE_STARTS = [date(2023, 8, 1), date(2023, 10, 2), date(2023, 12, 4), date(2024, 2, 5), date(2024, 4, 8),
                date(2024, 6, 10), date(2024, 8, 1), date(2024, 10, 7), date(2024, 12, 2), date(2025, 2, 3),
                date(2025, 4, 7)]


def dnum(d): return (d - EPOCH).days
def ts_str(sec):
    """epoch seconds (int64 array, -1 = blank) -> 'YYYY-mm-dd HH:MM:SS.000' strings"""
    sec = np.asarray(sec, dtype=np.int64)
    out = np.datetime_as_string(sec.astype("datetime64[s]"), unit="s")
    out = np.char.add(np.char.replace(out, "T", " "), ".000")
    return np.where(sec < 0, "", out)


def wday_occ(d, wd):  # first occurrence of weekday in month
    return d.weekday() == wd and d.day <= 7


def third_thursday_before(d):
    m = date(d.year, d.month, 1) - timedelta(days=1)
    first = date(m.year, m.month, 1)
    t = first + timedelta(days=(3 - first.weekday()) % 7 + 14)
    return t


# ============================================================================ main
def main(seed, out, n_students_target):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    gt_dir = os.path.join(out, "_ground_truth")
    os.makedirs(gt_dir, exist_ok=True)

    # ------------------------------------------------------------- items
    items = []
    for k, line in enumerate(CAT.strip().split("\n")):
        name, code, cui, serv, adv, prem, prot, alg, age, pop, anch, flag = line.split("|")
        algs = {ALG_CODE[a] for a in alg.split()} if alg.strip() else set()
        conf = set(algs)
        if prot == "beef": conf.add(10)
        if prot == "pork": conf.add(11)
        if prot in ("beef", "pork", "chicken", "fish"): conf.add(12)
        if prot != "vegan": conf.add(13)
        anchor = {}
        if anch:
            a, p = anch.split(":"); anchor = {int(a): float(p)}
        items.append(dict(idx=k, id=5001 + k, name=name, code=code, cui=int(cui), serv=serv, adv=int(adv),
                          prem=int(prem), prot=prot, alg=algs, conf=conf, age=age, pop=float(pop),
                          anchor=anchor, staple=flag == "staple", excl=flag == "excl"))
    NI = len(items)
    item_cui = np.array([it["cui"] for it in items]); item_adv = np.array([it["adv"] for it in items], float)
    item_prot = np.array([PROTEINS.index(it["prot"]) for it in items])
    item_pop = np.array([it["pop"] for it in items])
    for it in items:
        it["price"] = round(5.25 + .35 * it["adv"] + 1.6 * it["prem"] + rng.uniform(-.25, .6), 2)
        it["cost"] = round(it["price"] * rng.uniform(.22, .42), 2)
        it["itype"] = str(rng.choice(list("ABCJLT")))

    # age appeal matrix A[grade_level_index, item]
    glevels = sorted(set(GNUM.values()))
    G_IDX = {g: i for i, g in enumerate(glevels)}
    A = np.zeros((len(glevels), NI), np.float32)
    for i, it in enumerate(items):
        pk, w, b, fl = AGE[it["age"]]
        for g, gi in G_IDX.items():
            gg = min(g, 13.0)  # staff behave like older teens/adults
            A[gi, i] = np.log(fl + b * np.exp(-0.5 * ((gg - pk) / w) ** 2)) + np.log(it["pop"])
    staff_bonus = np.array([.6 if it["cui"] in (8, 9) or it["age"] == "T" else 0 for it in items], np.float32)
    A[G_IDX[15.0]] += staff_bonus

    # ------------------------------------------------------------- components
    comps = []
    for j, line in enumerate(COMP.strip().split("\n")):
        name, ctype, price, alg, pop, aff, season = line.split("|")
        comps.append(dict(idx=j, id=7001 + j, name=name, type=ctype, price=float(price),
                          alg={ALG_CODE[a] for a in alg.split()}, pop=float(pop),
                          aff=[int(x) for x in aff.split()], season=[int(x) for x in season.split()]))
    NC = len(comps)
    CTYPES = ["Snack", "Fruit", "Drink", "Smoothie"]
    comp_type = np.array([CTYPES.index(c["type"]) for c in comps])
    comp_pop = np.array([c["pop"] for c in comps], np.float32)
    PAIR = np.ones((NI, NC), np.float32)
    for i, it in enumerate(items):
        for j, c in enumerate(comps):
            if it["cui"] in c["aff"]: PAIR[i, j] = 2.4
    smoothies = [j for j, c in enumerate(comps) if c["type"] == "Smoothie"]
    everyday = [j for j, c in enumerate(comps) if c["type"] != "Smoothie"]

    # ------------------------------------------------------------- schools
    schools = []
    sid = 9100001
    name_pool = list(PLACE); rng.shuffle(name_pool)
    for band, grades, count, types in BANDS:
        for c in range(count):
            stype = str(rng.choice(types))
            nm = name_pool[len(schools) % len(name_pool)]
            suffix = {"Elementary": "Elementary", "TK-5": "Primary School", "K-8": "Academy",
                      "Middle": "Middle School", "High": "High School", "K-12": "Day School",
                      "Camp": "Summer Camp"}[band]
            if "Catholic" in stype: nm = "St. " + nm
            city, county = CITIES[int(rng.integers(0, len(CITIES)))]
            schools.append(dict(i=len(schools), id=sid, name=f"{nm} {suffix}" + (f" {city}" if band == "Camp" else ""),
                                band=band, grades=grades, type=stype, city=city, county=county,
                                camp=band == "Camp", breadth=int(rng.integers(6, 11)),
                                exclusive=False, large=band in ("Middle", "High", "K-12", "K-8"),
                                iou=bool(rng.random() < .6)))
            sid += 1
    # one exclusive-menu school (a K-8)
    k8 = [s for s in schools if s["band"] == "K-8"]
    k8[0]["exclusive"] = True
    NS = len(schools)

    # grades, serving groups, packing groups
    grade_rows, sg_rows, pg_rows = [], [], []
    gid, sgid, pgid = 9300001, 9330001, 9400001
    for s in schools:
        gl = s["grades"]
        n_pg = 1 if len(gl) <= 3 or s["camp"] else (2 if len(gl) <= 7 else 3)
        chunks = np.array_split(np.arange(len(gl)), n_pg)
        s["pg"], s["grade_ids"], s["grade_pg"] = [], {}, {}
        for ci, ch in enumerate(chunks):
            pg = dict(id=pgid, sg=sgid, grades=[gl[x] for x in ch],
                      large=s["large"] and GNUM[gl[ch[-1]]] >= 4)
            s["pg"].append(pg)
            pg_rows.append({"PACKINGGROUPID": pgid, "SCHOOLID": s["id"], "PACKINGGROUPCODE": f"PG{pgid % 10000:04d}",
                            "PACKINGGROUPNAME": f"{s['name']} {gl[ch[0]]}-{gl[ch[-1]]}", "AREIOUSALLOWED": s["iou"],
                            "HIDDEN": False})
            sg_rows.append({"SERVINGGROUPID": sgid, "SCHOOLID": s["id"], "PACKINGGROUPID": pgid,
                            "SERVINGGROUPNAME": f"Grades {gl[ch[0]]}-{gl[ch[-1]]}"})
            for x in ch: s["grade_pg"][gl[x]] = ci
            pgid += 1; sgid += 1
        s["grade_pg"]["Staff"] = len(chunks) - 1; s["grade_pg"]["Unknown"] = len(chunks) - 1
        all_g = gl + ["Staff", "Unknown"]
        for k, g in enumerate(all_g):
            s["grade_ids"][g] = gid + k
        for k, g in enumerate(all_g):
            nxt = all_g[k + 1] if g in gl and k + 1 < len(gl) else g
            grade_rows.append({"GRADEID": gid + k, "SCHOOLID": s["id"],
                               "SERVINGGROUPID": s["pg"][s["grade_pg"][g]]["sg"], "GRADE": g,
                               "GRADELEVEL": g.zfill(2) if g.isdigit() else g, "GRADEQUALVALUE": Q,
                               "GRADETYPE": 3, "SORTORDER": k, "GRADUATEDGRADEID": s["grade_ids"][nxt],
                               "ACTIVECODE": True, "ARCHIVED": False})
        gid += 20

    # ------------------------------------------------------------- calendar per school
    def common_closed(d, s, r):
        y = d.year
        fixed = {date(2023, 9, 4), date(2024, 9, 2), date(2024, 1, 15), date(2025, 1, 20), date(2024, 5, 27),
                 date(2025, 5, 26), date(2024, 6, 19), date(2024, 7, 4), date(2024, 7, 5)}
        if d in fixed: return True
        if d in (date(2023, 11, 10), date(2024, 11, 11)) and r["vet"]: return True
        for tw in (date(2023, 11, 20), date(2024, 11, 25)):
            if tw <= d <= tw + timedelta(days=4):
                return True
        if date(2023, 12, 18) + timedelta(days=r["wb"]) <= d <= date(2024, 1, 5) + timedelta(days=r["wb2"]): return True
        if date(2024, 12, 20) <= d <= date(2025, 1, 3) + timedelta(days=r["wb2"]): return True
        for pw in (date(2024, 2, 19), date(2025, 2, 17)):
            if d == pw or (r["pres_week"] and pw <= d <= pw + timedelta(days=4)): return True
        sb = [(date(2024, 3, 25), date(2024, 4, 1)), (date(2025, 4, 14), date(2025, 4, 21))]
        for a, b in sb:
            w0 = a if r["sb"] == 0 else b
            if w0 <= d <= w0 + timedelta(days=4): return True
        if "Catholic" in s["type"] and d in (date(2024, 3, 29), date(2025, 4, 18)): return True
        if d in r["inservice"]: return True
        return False

    school_days = {}  # school idx -> sorted list of open dates
    all_days = [START + timedelta(days=i) for i in range((END - START).days + 1)]
    for s in schools:
        r = dict(vet=rng.random() < .5, thanks_open=rng.random() < .3, wb=int(rng.choice([0, 3])),
                 wb2=int(rng.choice([0, 3])), pres_week=rng.random() < .4, sb=int(rng.integers(0, 2)))
        r["inservice"] = set()
        for (a, b) in SY:
            for _ in range(2):
                r["inservice"].add(a + timedelta(days=int(rng.integers(40, 280))))
        s1 = date(2023, 8, 14) + timedelta(days=int(rng.integers(0, 15)))
        e1 = date(2024, 6, 5) + timedelta(days=int(rng.integers(0, 9)))
        s2 = date(2024, 8, 12) + timedelta(days=int(rng.integers(0, 15)))
        e2 = date(2025, 6, 4) + timedelta(days=int(rng.integers(0, 10)))
        s["sy_bounds"] = [(s1, e1), (s2, e2)]
        no_fri = rng.random() < .06
        open_days = []
        for d in all_days:
            if d.weekday() >= 5: continue
            if s["camp"]:
                ok = CAMP[0] <= d <= CAMP[1]
            else:
                ok = (s1 <= d <= e1) or (s2 <= d <= e2)
                if no_fri and d.weekday() == 4: ok = False
            if ok and not common_closed(d, s, r):
                open_days.append(d)
        school_days[s["i"]] = open_days
        s["first_days"] = {s1, s2, CAMP[0]}

    # ------------------------------------------------------------- menus (cycles)
    regular_idx = [it["idx"] for it in items if not it["excl"]]
    excl_idx = [it["idx"] for it in items if it["excl"]]
    staples = [it["idx"] for it in items if it["staple"]]
    anchored = {it["idx"]: it["anchor"] for it in items if it["anchor"]}

    def cycle_of(d):
        c = 0
        for k, cs in enumerate(CYCLE_STARTS):
            if d >= cs: c = k
        return c

    menu = {}          # (school_i, date) -> np.array of item idx
    change_log = []    # cycle changes per school
    pgm_rows_count = 0
    pgm_path = os.path.join(out, "TBLPACKINGGROUPMENUS.csv")
    pga_path = os.path.join(out, "TBLPACKINGGROUPADDONS.csv")
    pgm_cols = ["CATERERID", "SCHOOLID", "PACKINGGROUPID", "DELIVERYDATE", "ITEMNAME", "ITEMNAMELONG",
                "ITEMDESCRIPTION", "SERVINGTYPE", "MENUITEMID", "ITEMCODE", "PRODUCTTYPEID", "PRODUCTTYPE",
                "PORTIONCODE", "PORTIONNAME", "PORTIONPRICE", "PORTIONREVENUE", "PORTIONCOST", "ITEMTYPE",
                "ISNEW", "MENUSORT"]
    pga_cols = ["CATERERID", "SCHOOLID", "PACKINGGROUPID", "DELIVERYDATE", "COMPONENTNAME", "COMPONENTTYPE",
                "COMPONENTID", "COMPONENTPRICE", "COMPONENTREVENUE", "COMPONENTSORT"]
    fm = open(pgm_path, "w", newline="", encoding="utf-8-sig"); fm.write(",".join(pgm_cols) + "\n")
    fa = open(pga_path, "w", newline="", encoding="utf-8-sig"); fa.write(",".join(pga_cols) + "\n")
    addons_avail = {}
    first_seen = {}

    for s in schools:
        si = s["i"]; b = s["breadth"]
        pool = excl_idx + [i for i in staples] if s["exclusive"] else regular_idx
        lineup_size = min(len(pool), int(round(b * 4.2)) if not s["exclusive"] else len(pool))
        prio = {i: float(rng.random()) for i in pool}          # stable within-school priority
        k_wd = [int(np.clip(b + rng.integers(-1, 2), 6, 10)) for _ in range(5)]
        # initial line-up: staples + anchored (most) + weighted random
        must = [i for i in staples if i in pool]
        anc = [i for i in pool if i in anchored and rng.random() < .8]
        rest = [i for i in pool if i not in must and i not in anc]
        w = np.array([CUISINE_POP[items[i]["cui"]] * items[i]["pop"] for i in rest]); w /= w.sum()
        need = max(0, lineup_size - len(must) - len(anc))
        lineup = must + anc + list(rng.choice(rest, size=min(need, len(rest)), replace=False, p=w))
        lineup = [int(x) for x in lineup]
        # weekday assignment
        def assign(i, prev=None):
            if items[i]["staple"]:
                return sorted(rng.choice(5, size=int(rng.integers(2, 4)), replace=False).tolist())
            if i in anchored:
                return [list(anchored[i].keys())[0]]
            if prev is not None:
                choices = [x for x in range(5) if x not in prev]
                return [int(rng.choice(choices))]
            return [int(rng.integers(0, 5))]
        days_of = {i: assign(i) for i in lineup}
        ab = {i: (int(rng.integers(0, 2)) if (not items[i]["staple"] and rng.random() < .25) else -1) for i in lineup}
        offered = list(rng.choice(everyday, size=int(rng.integers(14, 22)), replace=False))
        cur_cycle = 0
        prev_menu = []
        for d in school_days[si]:
            c = cycle_of(d)
            if c != cur_cycle:
                # ~30% of non-staple line-up changes: half rotate out/in, half move weekday
                cand = [i for i in lineup if not items[i]["staple"]]
                n_change = int(round(.21 * len(lineup)))
                chg = list(rng.choice(cand, size=min(n_change, len(cand)), replace=False))
                n_rot = 0
                for i in chg:
                    i = int(i)
                    if rng.random() < .5:  # rotate out, bring another (often previously used) in
                        outside = [x for x in pool if x not in lineup]
                        if not outside: continue
                        ww = np.array([CUISINE_POP[items[x]["cui"]] * items[x]["pop"] for x in outside], float)
                        new = int(rng.choice(outside, p=ww / ww.sum()))
                        lineup.remove(i); lineup.append(new)
                        days_of.pop(i, None); days_of[new] = assign(new)
                        ab[new] = int(rng.integers(0, 2)) if rng.random() < .25 else -1
                        change_log.append((s["id"], CYCLE_STARTS[c].isoformat(), "rotate_out", items[i]["id"], ""))
                        change_log.append((s["id"], CYCLE_STARTS[c].isoformat(), "rotate_in", items[new]["id"],
                                           ",".join(str(x) for x in days_of[new])))
                    else:  # weekday move (anchored items move rarely)
                        if i in anchored and rng.random() < .55: continue
                        old = days_of[i]
                        days_of[i] = assign(i, prev=old) if i not in anchored else [int(rng.integers(0, 5))]
                        change_log.append((s["id"], CYCLE_STARTS[c].isoformat(), "weekday_move", items[i]["id"],
                                           f"{old}->{days_of[i]}"))
                for i in list(days_of):  # moved anchored items return to their anchor day next cycle
                    if i in anchored and i not in [int(x) for x in chg] and days_of[i] != [list(anchored[i].keys())[0]]:
                        days_of[i] = [list(anchored[i].keys())[0]]
                k_wd = [int(np.clip(x + rng.integers(-1, 2), 6, 10)) if rng.random() < .3 else x for x in k_wd]
                cur_cycle = c
            wd = d.weekday()
            week_par = d.isocalendar()[1] % 2
            todays = [i for i in lineup if wd in days_of[i] and (ab[i] < 0 or ab[i] == week_par)]
            todays = [i for i in todays if rng.random() < .97]  # occasional supply swap-out
            k_day = k_wd[wd]
            if len(todays) > k_day:
                keep = [i for i in todays if items[i]["staple"] or i in anchored]
                others = sorted([i for i in todays if i not in keep], key=lambda i: -prio[i])
                todays = keep + others[:max(0, k_day - len(keep))]
            if len(todays) < k_day:
                for allow_prev in (False, True):
                    if len(todays) >= k_day: break
                    fill = [i for i in lineup if i not in todays and (allow_prev or i not in prev_menu)
                            and (i not in anchored or anchored[i].get(wd) or rng.random() < .06)]
                    ww = np.array([1.0 if i not in anchored else (.5 if anchored[i].get(wd) else .035) for i in fill])
                    if fill:
                        fill = sorted(fill, key=lambda i: -(prio[i] * (1 if i not in anchored else .05)))
                        todays += fill[:k_day - len(todays)]
                if len(todays) < 6:  # tiny pools (exclusive school): allow any pool item
                    extra = [i for i in pool if i not in todays]
                    todays += [int(x) for x in rng.choice(extra, size=min(6 - len(todays), len(extra)), replace=False)]
            todays = [int(x) for x in dict.fromkeys(todays)]
            menu[(si, d)] = np.array(todays, dtype=np.int32)
            prev_menu = todays
            ds = d.isoformat() + " 00:00:00.000"
            for pg in s["pg"]:
                for sort, i in enumerate(todays):
                    it = items[i]
                    key = (si, i)
                    if key not in first_seen: first_seen[key] = d
                    isnew = int((d - first_seen[key]).days < 14 and d > school_days[si][0] + timedelta(days=20))
                    portions = [(1, "Regular", it["price"], it["cost"])]
                    if pg["large"]:
                        portions.append((2, "Large", round(it["price"] * 1.38, 2), round(it["cost"] * 1.45, 2)))
                    for pc, pn, pr, co in portions:
                        fm.write(f"1,{s['id']},{pg['id']},{ds},{it['name']},{it['name']},"
                                 f"Synthetic menu description for {it['name'].lower()},{it['serv']},{it['id']},"
                                 f"{it['code']},1,ENTREE,{pc},{pn},{pr:.4f},{pr:.4f},{co:.4f},{it['itype']},"
                                 f"{isnew},{sort}\n")
                        pgm_rows_count += 1
            # add-ons available today (fixed offering + seasonal fruit + monthly smoothies)
            av = [j for j in offered if not comps[j]["season"] or d.month in comps[j]["season"]]
            for j in smoothies:
                if wday_occ(d, SMOOTHIE_DOW[comps[j]["name"]]): av.append(j)
            addons_avail[(si, d)] = np.array(av, dtype=np.int32)
            for pg in s["pg"]:
                for sort, j in enumerate(av):
                    c_ = comps[j]
                    fa.write(f"1,{s['id']},{pg['id']},{ds},{c_['name']},{c_['type']},{c_['id']},"
                             f"{c_['price']:.4f},{c_['price']:.4f},{sort}\n")
    fm.close(); fa.close()
    print(f"menus done {time.time()-t0:.0f}s  menu rows {pgm_rows_count:,}")

    # ------------------------------------------------------------- students & families
    regular = [s for s in schools if not s["camp"]]
    camps = [s for s in schools if s["camp"]]
    n_camp = 3 * 200
    n_reg = n_students_target - n_camp
    sizes = np.clip(rng.normal(1, .38, len(regular)), .3, 2.2)
    sizes = np.round(sizes / sizes.sum() * n_reg).astype(int)
    sizes[-1] += n_reg - sizes.sum()

    S = []  # list of dicts (converted to arrays later)
    fam_id = 9600001
    fams = []

    def new_family(school_i):
        nonlocal fam_id
        f = dict(id=fam_id, style=int(rng.choice(4, p=[.30, .33, .27, .10])),  # 0 monthly 1 weekly 2 daily 3 mixed
                 channel=str(rng.choice(["app iphone", "app android", "web calendar", "web mobile"],
                                        p=[.42, .16, .30, .12])),
                 taste=rng.normal(0, .45, NI).astype(np.float32),
                 sub_prop=float(rng.beta(1.3, 5)), auto=rng.random() < .20,
                 n_gaps=int(rng.choice(4, p=[.45, .30, .17, .08])))
        fam_id += 1
        fams.append(f)
        return len(fams) - 1

    def school_for_grade(gnum_next):
        cands = [s for s in regular if any(GNUM[g] == gnum_next for g in s["grades"]) and not s["exclusive"]]
        return cands[int(rng.integers(0, len(cands)))]

    for s, n in zip(regular, sizes):
        k = 0
        while k < n:
            fsize = int(rng.choice([1, 2, 3], p=[.62, .30, .08]))
            fi = new_family(s["i"])
            for _ in range(min(fsize, n - k)):
                cat = rng.choice(["returning", "join_sy1", "new_sy2"], p=[.60, .09, .31])
                staff = rng.random() < .02
                if staff:
                    g1 = g2 = "Staff"
                else:
                    if cat == "new_sy2":
                        g2 = s["grades"][0] if rng.random() < .45 else str(rng.choice(s["grades"]))
                        g1 = None
                    else:
                        g1 = str(rng.choice(s["grades"]))
                        g2 = None
                S.append(dict(fam=fi, school=s["i"], cat=cat, staff=staff, g1=g1, g2=g2))
                k += 1
    for s in camps:
        for _ in range(n_camp // 3):
            fi = new_family(s["i"])
            S.append(dict(fam=fi, school=s["i"], cat="camp", staff=False,
                          g1=str(rng.choice(s["grades"])), g2=None))
    N = len(S)
    print(f"students {N:,}  families {len(fams):,}")

    # periods: 0 = SY1, 1 = camp summer 2024, 2 = SY2
    sch_p = np.full((N, 3), -1, np.int32)
    gr_p = np.full((N, 3), None, dtype=object)
    reg_dt = np.zeros(N, np.int64)
    ord_start = np.zeros(N, np.int64); ord_end = np.zeros(N, np.int64)
    leave_reason = np.array([""] * N, dtype=object)
    for n, st in enumerate(S):
        s = schools[st["school"]]
        if st["cat"] == "camp":
            sch_p[n, 1] = s["i"]; gr_p[n, 1] = st["g1"]
            ord_start[n] = dnum(CAMP[0]); ord_end[n] = dnum(CAMP[1])
            reg = datetime(2024, 4, 1) + timedelta(days=int(rng.integers(0, 75)))
        elif st["cat"] in ("returning", "join_sy1"):
            sch_p[n, 0] = s["i"]; gr_p[n, 0] = st["g1"]
            if st["cat"] == "returning":
                reg = datetime(2015, 8, 1) + timedelta(days=int(rng.integers(0, 2900)))
                ord_start[n] = dnum(s["sy_bounds"][0][0])
            else:
                j = s["sy_bounds"][0][0] + timedelta(days=int(rng.integers(20, 250)))
                reg = datetime(j.year, j.month, j.day) - timedelta(days=int(rng.integers(0, 5)))
                ord_start[n] = dnum(j)
            # SY2 continuation
            if st["staff"]:
                if rng.random() < .85: sch_p[n, 2] = s["i"]; gr_p[n, 2] = "Staff"
            else:
                gi = s["grades"].index(st["g1"])
                if gi + 1 < len(s["grades"]):
                    if rng.random() < .93: sch_p[n, 2] = s["i"]; gr_p[n, 2] = s["grades"][gi + 1]
                    else: leave_reason[n] = "moved_away"
                else:
                    nxt = GNUM[st["g1"]] + 1
                    if nxt <= 12 and rng.random() < .55:
                        s2 = school_for_grade(nxt)
                        sch_p[n, 2] = s2["i"]; gr_p[n, 2] = str(int(nxt))
                    else:
                        leave_reason[n] = "graduated"
            ord_end[n] = dnum(schools[sch_p[n, 2]]["sy_bounds"][1][1]) if sch_p[n, 2] >= 0 else dnum(s["sy_bounds"][0][1])
        else:  # new in SY2
            sch_p[n, 2] = s["i"]; gr_p[n, 2] = st["g2"]
            j = s["sy_bounds"][1][0] if rng.random() < .72 else s["sy_bounds"][1][0] + timedelta(days=int(rng.integers(15, 230)))
            reg = datetime(j.year, j.month, j.day) - timedelta(days=int(rng.integers(0, 20)))
            ord_start[n] = dnum(j); ord_end[n] = dnum(s["sy_bounds"][1][1])
        reg = reg + timedelta(seconds=int(rng.integers(0, 86400)))
        reg_dt[n] = int((reg - datetime(1970, 1, 1)).total_seconds())

    # churn (stops ordering, account often left active) and never-orderers
    never = rng.random(N) < .085
    churn = (rng.random(N) < .24) & ~never
    span = ord_end - ord_start
    churn_at = ord_start + (rng.random(N) * np.maximum(span, 1)).astype(np.int64)
    ord_end = np.where(churn, np.maximum(ord_start + 10, churn_at), ord_end)
    trial = (rng.random(N) < .21) & ~never & ~churn   # tried the service a handful of times, then stopped
    ord_end = np.where(trial, ord_start + rng.integers(3, 30, N), ord_end)
    churn = churn | trial

    # ------------------------------------------------------------- latent traits
    gnum_now = np.array([GNUM[(S[n]["g1"] or S[n]["g2"])] for n in range(N)])
    young = gnum_now <= 2
    loyalty = rng.gamma(1.9, 1.35, N) * np.where(gnum_now <= 3, 1.25, np.where(gnum_now >= 9, .7, 1.0))
    adv = np.clip(rng.normal(.6 + .12 * np.clip(gnum_now, 0, 13), .7), 0, 3)
    cpw = np.array([CUISINE_POP[c] for c in sorted(CUISINES)], float) / 12
    cpref = np.log(rng.dirichlet(cpw, N) + .01).astype(np.float32) * .55
    ppref = rng.normal(0, .35, (N, len(PROTEINS))).astype(np.float32)
    idio = np.zeros((N, NI), np.float32)
    for n in range(N):
        idio[n] = .55 * fams[S[n]["fam"]]["taste"] + rng.normal(0, .45, NI).astype(np.float32)
    fatigue = rng.gamma(2, .5, N) / (1 + .5 * loyalty)
    novelty = rng.normal(0, .4, N) + .25 * (adv - 1)
    noise = .55 + .35 * rng.random(N)
    thr = rng.normal(.8, .6, N)
    freq = rng.choice([1, 2, 3, 4, 5], N, p=[.46, .27, .14, .08, .05])
    dow_w = rng.dirichlet([3] * 5, N) * 5
    side_rate = rng.choice([0, .35, 1.0, 1.8], N, p=[.40, .22, .20, .18])
    fav_side = rng.choice(everyday, N)
    side_type_pref = rng.dirichlet([2, 2, 2], N)  # snack/fruit/drink
    large_p = np.where(gnum_now >= 9, .5, np.where(gnum_now >= 6, .3, np.where(gnum_now >= 4, .12, 0.0)))
    large_p = np.where(gnum_now == 15, .45, large_p)
    stu_type_base = np.where(gnum_now == 15, "Staff", np.where(rng.random(N) < .025, "Free", "Paid"))

    # allergens (~15% of students)
    alg_mask = np.zeros((N, NI), bool)
    calg_mask = np.zeros((N, NC), bool)
    sa_rows = []
    ks, ws = zip(*ALG_PREV.items()); ws = np.array(ws, float) / sum(ws)
    for n in range(N):
        if rng.random() < .15:
            chosen = set(int(a) for a in rng.choice(ks, size=int(rng.choice([1, 1, 1, 2, 2, 3])), replace=False, p=ws))
            for a in chosen:
                added = reg_dt[n] + int(rng.integers(0, 200 * 86400))
                active, deleted = True, -1
                if rng.random() < .05:
                    active = False
                    del_t = added + int(rng.integers(30, 500)) * 86400
                    deleted = del_t if del_t >= dnum(date(2025, 1, 1)) * 86400 else -1  # systematic only from Jan 2025
                sa_rows.append([0, n, a, added, active, deleted])
                if active:
                    for i, it in enumerate(items):
                        if a in it["conf"]: alg_mask[n, i] = True
                    for j, c in enumerate(comps):
                        if a in c["alg"]: calg_mask[n, j] = True
    # quirk: 4 records deleted at creation timestamp
    for r in sa_rows[:4]:
        r[4] = False; r[5] = r[3]
    # those 4 still need unmasking? they're inactive -> recompute masks for affected students
    for r in sa_rows[:4]:
        n = r[1]
        act = {x[2] for x in sa_rows if x[1] == n and x[4]}
        alg_mask[n] = [bool(act & it["conf"]) for it in items]
        calg_mask[n] = [bool(act & c["alg"]) for c in comps]

    # ------------------------------------------------------------- subscriptions (locked per school year)
    PLAN_D = [1, 2, 3, 4, 5]; PLAN_DP = [.06, .13, .16, .15, .50]
    sub_on = np.zeros((N, 3), bool)
    sub_days = np.zeros((N, 3, 5), bool)
    sub_S = np.zeros((N, 3), np.int8)
    sub_code = np.full((N, 3), "", dtype=object)
    sub_start = np.zeros((N, 3), np.int64); sub_end = np.zeros((N, 3), np.int64)
    sub_auto = np.zeros((N, 3), bool)
    util = np.where(rng.random(N) < .85, rng.uniform(.86, 1.0, N),
                    np.where(rng.random(N) < .8, rng.uniform(.6, .86, N), rng.uniform(.3, .6, N)))
    extra_p = np.where(rng.random(N) < .32, rng.uniform(.12, .45, N), 0.0)
    for n in range(N):
        if never[n] or S[n]["staff"] or trial[n]: continue
        fam = fams[S[n]["fam"]]
        for p in (0, 2):
            si = sch_p[n, p]
            if si < 0: continue
            sy0, sy1 = schools[si]["sy_bounds"][0 if p == 0 else 1]
            prev = sub_on[n, 0] if p == 2 else False
            prob = .78 if prev else min(.7, .10 + 1.0 * fam["sub_prop"])
            if rng.random() >= prob: continue
            if prev and rng.random() < .75:
                D = int(sub_days[n, 0].sum()); Sd = int(sub_S[n, 0]); days = sub_days[n, 0].copy()
            else:
                D = int(rng.choice(PLAN_D, p=PLAN_DP)); Sd = int(rng.choice(4, p=[.26, .20, .25, .29]))
                days = np.zeros(5, bool)
                days[rng.choice(5, size=D, replace=False, p=np.array([1.1, 1, 1.05, 1, .85]) / 5)] = True
            st = max(dnum(sy0), int(ord_start[n]))
            if rng.random() < .22:  # mid-year sign-up: starts on sign-up date, locked to June
                st = max(st, dnum(sy0) + int(rng.integers(20, 200)))
            if st >= dnum(sy1): continue
            sub_on[n, p] = True; sub_days[n, p] = days; sub_S[n, p] = Sd
            up = "1" if rng.random() < .12 else "0"
            sub_code[n, p] = f"Sub {D}{Sd}{up}0"
            sub_start[n, p] = st; sub_end[n, p] = dnum(sy1)
            sub_auto[n, p] = fam["auto"]
    for n in range(N):  # churn cannot cut a running plan short (except a few dormant subscribers)
        for p in (0, 2):
            if sub_on[n, p] and rng.random() > .03 and not trial[n]:
                ord_end[n] = max(ord_end[n], sub_end[n, p]) if churn[n] and ord_end[n] < sub_end[n, p] else ord_end[n]
    # dormancy gaps (family wallet lapses)
    gaps = np.full((N, 3, 2), -1, np.int64)
    for n in range(N):
        f = fams[S[n]["fam"]]
        if "gaps" not in f:
            g = []
            for _ in range(f["n_gaps"]):
                a = dnum(START) + int(rng.integers(0, (END - START).days))
                g.append((a, a + int(rng.integers(7, 56))))
            f["gaps"] = g
        for k, (a, b) in enumerate(f["gaps"]): gaps[n, k] = (a, b)

    # family ordering sessions
    NF = len(fams)
    fam_of = np.array([S[n]["fam"] for n in range(N)])
    style = np.array([f["style"] for f in fams])[fam_of]
    month_keys = sorted({(d.year, d.month) for d in all_days})
    MI = {m: k for k, m in enumerate(month_keys)}
    fam_msess = rng.uniform(10, 34, (NF, len(month_keys)))           # days before month start
    fam_msess_h = rng.uniform(17, 23.5, (NF, len(month_keys)))
    weeks = sorted({d.isocalendar()[:2] for d in all_days})
    WI = {w: k for k, w in enumerate(weeks)}
    fam_wsess = rng.uniform(1, 3, (NF, len(weeks)))                   # days before Monday
    fam_wsess_h = rng.uniform(8, 22.5, (NF, len(weeks)))
    channel_f = np.array([f["channel"] for f in fams])
    CH = ["app iphone", "app android", "web calendar", "web mobile"]
    ch_code = np.array([CH.index(c) for c in channel_f])[fam_of]

    # ------------------------------------------------------------- simulation state
    CNT = np.zeros((N, NI), np.float32)
    CNTD = np.zeros((5, N, NI), np.float32)
    last1 = np.full(N, -1, np.int32); last2 = np.full(N, -1, np.int32)
    n_orders = np.zeros(N, np.int32)
    SCNT = np.zeros((N, NC), np.float32)

    # school-period index
    idx_by = {}
    for p in range(3):
        for si in range(NS):
            idx_by[(p, si)] = np.where(sch_p[:, p] == si)[0]
    large_ok = np.zeros((N, 3), bool)
    gidx_p = np.full((N, 3), -1, np.int32)
    gid_p = np.full((N, 3), -1, np.int64)
    for n in range(N):
        for p in range(3):
            if sch_p[n, p] >= 0:
                g = gr_p[n, p]
                gidx_p[n, p] = G_IDX[GNUM[g]]
                gid_p[n, p] = schools[sch_p[n, p]]["grade_ids"][g]
                sc_ = schools[sch_p[n, p]]
                large_ok[n, p] = sc_["pg"][sc_["grade_pg"][g]]["large"]

    L = {k: [] for k in ["stu", "gid", "item", "portion", "day", "stype", "ltype", "chan", "created",
                         "cancel", "detail", "sides_n", "school"]}
    LT = ["PreOrder", "AutoOrder", "LastMinute", "IOU"]
    ST_CODES = {}
    def stcode(s):
        if s not in ST_CODES: ST_CODES[s] = len(ST_CODES)
        return ST_CODES[s]

    def push(stu, gidv, item, portion, day, stype, ltype, chan, created, cancel, detail, sides_n, school):
        for k, v in zip(L.keys(), (stu, gidv, item, portion, day, stype, ltype, chan, created, cancel, detail,
                                   sides_n, school)):
            L[k].append(np.asarray(v))

    day_list = sorted({d for si in range(NS) for d in school_days[si]})
    by_day = {}
    for si in range(NS):
        for d in school_days[si]:
            by_day.setdefault(d, []).append(si)

    for d in day_list:
        dn = dnum(d); wd = d.weekday()
        p = 1 if CAMP[0] <= d <= CAMP[1] and d > SY[0][1] else (0 if d <= SY[0][1] else 2)
        mi = MI[(d.year, d.month)]; wi = WI[d.isocalendar()[:2]]
        for si in by_day[d]:
            s = schools[si]
            if s["camp"] and p != 1: continue
            if (not s["camp"]) and p == 1:
                continue
            stu = idx_by[(p, si)]
            if len(stu) == 0: continue
            mitems = menu[(si, d)]
            # eligibility
            act = (ord_start[stu] <= dn) & (dn <= ord_end[stu]) & ~never[stu]
            g = gaps[stu]
            in_gap = ((g[:, :, 0] <= dn) & (dn <= g[:, :, 1])).any(axis=1)
            subnow = sub_on[stu, p] & (sub_start[stu, p] <= dn) & (dn <= sub_end[stu, p]) if p != 1 else np.zeros(len(stu), bool)
            act &= ~(in_gap & ~subnow)
            if not act.any(): continue
            stu = stu[act]
            # utilities
            gi = gidx_p[stu, p]
            U = (A[gi][:, mitems] + cpref[stu][:, item_cui[mitems] - 1] + ppref[stu][:, item_prot[mitems]]
                 - .55 * np.abs(item_adv[mitems][None, :] - adv[stu][:, None]) + idio[stu][:, mitems]
                 + (loyalty[stu][:, None] * 2.2) * (CNT[stu][:, mitems] / (n_orders[stu][:, None] + 4))
                 + .9 * (CNTD[wd][stu][:, mitems] / (CNTD[wd][stu].sum(1)[:, None] + 3))
                 - fatigue[stu][:, None] * ((mitems[None, :] == last1[stu][:, None]) + .5 * (mitems[None, :] == last2[stu][:, None]))
                 + novelty[stu][:, None] * (CNT[stu][:, mitems] == 0) * (n_orders[stu] >= 10)[:, None]
                 + rng.gumbel(size=(len(stu), len(mitems))) * (noise[stu] * np.where(n_orders[stu] < 10, 1.5, 1.0))[:, None])
            U = np.where(alg_mask[stu][:, mitems], -np.inf, U)
            safe_any = np.isfinite(U).any(axis=1)
            Ufin = np.where(np.isfinite(U), U, -50)
            order_rank = np.argsort(-Ufin, axis=1)
            best = mitems[order_rank[:, 0]]
            if len(mitems) > 1:
                second = mitems[order_rank[:, 1]]
                second = np.where(np.isfinite(np.take_along_axis(U, order_rank[:, 1:2], 1)[:, 0]), second, best)
            else:
                second = best.copy()
            maxU = Ufin.max(axis=1)
            # ordering decisions
            subp = sub_on[stu, p] & (sub_start[stu, p] <= dn) & (dn <= sub_end[stu, p]) if p != 1 else np.zeros(len(stu), bool)
            plan_day = subp & sub_days[stu, p, wd] if p != 1 else np.zeros(len(stu), bool)
            first = d in s["first_days"] or (d - timedelta(days=4)) in s["first_days"]
            season = .78 if first else 1.0
            if d.month == 12 and d.day >= 13: season *= .85
            if d.month == 6 or (d.month == 5 and d.day > 25): season *= .85
            attract = .45 + 1.1 / (1 + np.exp(-(maxU - thr[stu])))
            p_non = np.clip(freq[stu] / 5 * dow_w[stu, wd] * season * attract * (.93 if wd == 4 else 1), 0, .97)
            if p == 1: p_non = np.clip(.40 * attract, 0, .95)
            p_ord = np.where(plan_day, util[stu] * (.97 if first else 1.0),
                             np.where(subp, extra_p[stu] * attract, p_non))
            go = (rng.random(len(stu)) < p_ord) & safe_any
            # IOU for non-orderers (emergency lunch, entree only)
            iou = (~go) & safe_any & (rng.random(len(stu)) < (.0035 if s["iou"] else 0)) & (p != 1)
            if go.any():
                o = np.where(go)[0]
                so = stu[o]
                is_plan = plan_day[o]
                auto = is_plan & sub_auto[so, p] if p != 1 else np.zeros(len(o), bool)
                chosen = best[o].copy()
                # legacy auto-order: repeats usual item for weekday, ignores variety
                if auto.any():
                    ao = np.where(auto)[0]
                    sa = so[ao]
                    score = (2 * CNTD[wd][sa][:, mitems] + CNT[sa][:, mitems] + .01 * item_pop[mitems][None, :])
                    score = np.where(alg_mask[sa][:, mitems], -1e9, score)
                    sys_pick = mitems[np.argmax(score, axis=1)]
                    chosen[ao] = sys_pick
                ltype = np.where(auto, 1, 0)
                # creation timestamps
                st_ = style[so]; fo = fam_of[so]
                daystart = dn * 86400
                mon_first = dnum(date(d.year, d.month, 1)) * 86400
                monday = (dn - wd) * 86400
                c_month = mon_first - (fam_msess[fo, mi] * 86400).astype(np.int64) + (fam_msess_h[fo, mi] * 3600).astype(np.int64)
                c_week = monday - (fam_wsess[fo, wi] * 86400).astype(np.int64) + (fam_wsess_h[fo, wi] * 3600).astype(np.int64)
                same_day = rng.random(len(o)) < .35
                c_daily = np.where(same_day, daystart + rng.integers(6 * 3600, 9 * 3600 - 60, len(o)),
                                   daystart - 86400 + rng.integers(16 * 3600, 86399, len(o)))
                mixed_pick = rng.integers(0, 3, len(o))
                eff = np.where(st_ == 3, mixed_pick, st_)
                created = np.where(eff == 0, c_month, np.where(eff == 1, c_week, c_daily))
                lastmin = (eff == 2) & (rng.random(len(o)) < .13) & ~auto
                created = np.where(lastmin, daystart + rng.integers(9 * 3600 + 60, 10 * 3600 + 1800, len(o)), created)
                ltype = np.where(lastmin, 2, ltype)
                created = np.maximum(created, reg_dt[so] + rng.integers(60, 3 * 86400, len(o)))
                created = np.minimum(created, daystart + 10 * 3600 + 1800)
                auto_created = dnum(third_thursday_before(d)) * 86400 + 2 * 3600 + rng.integers(0, 3000, len(o))
                created = np.where(auto, auto_created, created)
                detail = np.where(auto | lastmin, 0, np.where(eff == 2, 2, np.where(rng.random(len(o)) < .85, 1, 0)))
                chan = np.where(auto, -1, ch_code[so])
                chan = np.where(rng.random(len(o)) < .06, rng.integers(0, 4, len(o)), chan)
                chan = np.where(auto, -1, chan)
                # cancellations
                r = rng.random(len(o))
                sys_rank_ok = chosen == best[o]
                p_absent = np.where(auto, .05, .035)
                p_modify = np.where(auto, np.where(sys_rank_ok, .05, .38), .05)
                p_instant = np.where(auto | lastmin, 0, .015)
                absent = r < p_absent
                modify = (~absent) & (r < p_absent + p_modify)
                instant = (~absent) & (~modify) & (r < p_absent + p_modify + p_instant)
                cancel = np.full(len(o), -1, np.int64)
                span_ = np.maximum(daystart - 15 * 3600 - created, 60)
                cancel = np.where(absent, created + (rng.random(len(o)) * span_).astype(np.int64), cancel)
                cancel = np.where(modify, created + (rng.random(len(o)) * span_).astype(np.int64), cancel)
                cancel = np.where(instant, created + rng.integers(5, 90, len(o)), cancel)
                cancel = np.where(cancel >= 0, np.minimum(cancel, daystart + 8 * 3600 + 3000), cancel)
                portion = np.where((rng.random(len(o)) < large_p[so]) & large_ok[so, p], 2, 1)
                stype_codes = np.array([stcode(sub_code[n, p] if (pl and p != 1) else stu_type_base[n])
                                        for n, pl in zip(so, is_plan)])
                # sides count
                plan_S = sub_S[so, p] if p != 1 else np.zeros(len(o), np.int8)
                rr = rng.random(len(o))
                n_sub = np.where(plan_S == 0, (rr < .33).astype(int),
                                 np.where(rr < .80, plan_S, np.where(rr < .92, plan_S + 1, plan_S - 1)))
                n_non = np.minimum(rng.poisson(side_rate[so]), 3)
                sides_n = np.where(is_plan, n_sub, n_non)
                sides_n = np.where(lastmin, 0, np.clip(sides_n, 0, 4))
                push(so, gid_p[so, p], chosen, portion, np.full(len(o), dn), stype_codes, ltype, chan,
                     created, cancel, detail, sides_n, np.full(len(o), si))
                # replacements (modification / instant edit): parent's actual preference
                rep = modify | instant
                if rep.any():
                    rpl = np.where(rep)[0]
                    rep_item = np.where(auto[rpl], best[o][rpl], second[o][rpl])
                    rep_item = np.where(rep_item == chosen[rpl], second[o][rpl], rep_item)
                    ok = rep_item != chosen[rpl]
                    rpl = rpl[ok]; rep_item = rep_item[ok]
                    if len(rpl):
                        rc = cancel[rpl] + rng.integers(3, 40, len(rpl))
                        push(so[rpl], gid_p[so[rpl], p], rep_item, portion[rpl], np.full(len(rpl), dn),
                             stype_codes[rpl], np.zeros(len(rpl), int), np.where(chan[rpl] < 0, ch_code[so[rpl]], chan[rpl]),
                             rc, np.full(len(rpl), -1), np.full(len(rpl), 2), sides_n[rpl], np.full(len(rpl), si))
                        chosen_final = chosen.copy(); chosen_final[rpl] = rep_item
                    else:
                        chosen_final = chosen
                else:
                    chosen_final = chosen
                # update habit state for lunches actually eaten
                eaten = ~absent
                eo = so[eaten]; ei = chosen_final[eaten]
                np.add.at(CNT, (eo, ei), 1); np.add.at(CNTD[wd], (eo, ei), 1)
                last2[eo] = last1[eo]; last1[eo] = ei; n_orders[eo] += 1
            if iou.any():
                io = stu[np.where(iou)[0]]
                pick = mitems[np.argmax(np.where(alg_mask[io][:, mitems], -1e9, A[gidx_p[io, p]][:, mitems]), axis=1)]
                push(io, gid_p[io, p], pick, np.ones(len(io), int), np.full(len(io), dn),
                     np.array([stcode(stu_type_base[n]) for n in io]), np.full(len(io), 3), np.full(len(io), -1),
                     dn * 86400 + rng.integers(11 * 3600, 12 * 3600, len(io)), np.full(len(io), -1),
                     np.zeros(len(io), int), np.zeros(len(io), int), np.full(len(io), si))
        if d.day == 1:
            print(f"  {d}  {time.time()-t0:.0f}s  lunches so far {sum(len(x) for x in L['stu']):,}")

    Ld = {k: np.concatenate(v) for k, v in L.items()}
    NL = len(Ld["stu"])
    print(f"lunches {NL:,}  {time.time()-t0:.0f}s")

    # ------------------------------------------------------------- sides (vectorised by add-on availability groups)
    sides_parent, sides_comp = [], []
    order = np.lexsort((Ld["day"], Ld["school"]))
    keys = Ld["school"][order] * 10 ** 6 + Ld["day"][order]
    bounds = np.flatnonzero(np.diff(keys)) + 1
    for grp in np.split(order, bounds):
        if len(grp) == 0: continue
        si = int(Ld["school"][grp[0]]); d = EPOCH + timedelta(days=int(Ld["day"][grp[0]]))
        need = grp[Ld["sides_n"][grp] > 0]
        if len(need) == 0: continue
        av = addons_avail[(si, d)]
        st_ = Ld["stu"][need]; it = Ld["item"][need]
        base = (np.log(comp_pop[av])[None, :] + np.log(PAIR[it][:, av])
                + np.log(side_type_pref[st_][:, np.minimum(comp_type[av], 2)] * 3 + .05)
                + 1.4 * (av[None, :] == fav_side[st_][:, None]) + .4 * np.log1p(SCNT[st_][:, av])
                + 2.2 * (comp_type[av] == 3)[None, :])
        base = np.where(calg_mask[st_][:, av], -np.inf, base)
        taken_type = np.zeros((len(need), 4), np.int8)
        picked = np.zeros((len(need), len(av)), bool)
        k_need = Ld["sides_n"][need]
        for slot in range(int(k_need.max())):
            active = k_need > slot
            sc = base - 1.6 * taken_type[:, comp_type[av]] + rng.gumbel(size=base.shape) * .8
            sc = np.where(picked, -np.inf, sc)
            ch = np.argmax(sc, axis=1)
            valid = active & np.isfinite(sc[np.arange(len(need)), ch])
            rows = np.where(valid)[0]
            picked[rows, ch[rows]] = True
            taken_type[rows, comp_type[av][ch[rows]]] += 1
            sides_parent.append(need[rows]); sides_comp.append(av[ch[rows]])
            np.add.at(SCNT, (st_[rows], av[ch[rows]]), 1)
    sp = np.concatenate(sides_parent); sc_ = np.concatenate(sides_comp)
    print(f"sides {len(sp):,}  {time.time()-t0:.0f}s")

    # ------------------------------------------------------------- IDs by creation time & write
    o = np.argsort(Ld["created"], kind="stable")
    lunch_id = np.empty(NL, np.int64); lunch_id[o] = 70000001 + np.arange(NL)
    stu_id = 9500001 + np.arange(N)
    ST_LIST = [None] * len(ST_CODES)
    for k_, v in ST_CODES.items(): ST_LIST[v] = k_
    chan_str = np.array(CH + [Q])
    det_str = np.array([Q, "bydate multiple", "bydate single"])
    chan_str = np.array(CH + [Q]); det_str = np.array([Q, "bydate multiple", "bydate single"])
    ST_LIST = [None] * len(ST_CODES)
    for k_, v in ST_CODES.items(): ST_LIST[v] = k_
    def write_chunks(path, build, n_rows, step=250000):
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            for k0 in range(0, n_rows, step):
                part = build(k0, min(n_rows, k0 + step))
                part.to_csv(fh, index=False, header=(k0 == 0), quoting=csv.QUOTE_NONE, quotechar="~")
    item_ids = np.array([it["id"] for it in items]); st_arr = np.array(ST_LIST, dtype=object)
    def build_l(a, b):
        ix = o[a:b]
        return pd.DataFrame({
            "STUDENTLUNCHID": lunch_id[ix], "STUDENTID": stu_id[Ld["stu"][ix]], "GRADEID": Ld["gid"][ix],
            "MENUITEMID": item_ids[Ld["item"][ix]], "PORTIONCODE": Ld["portion"][ix],
            "DELIVERYDATE": np.array(Ld["day"][ix], "datetime64[D]").astype(str),
            "STUDENTTYPE": st_arr[Ld["stype"][ix]], "LUNCHTYPE": np.array(LT)[Ld["ltype"][ix]],
            "ORDERCHANNEL": chan_str[Ld["chan"][ix]], "ISSERVED": "false",
            "LUNCHCREATED": ts_str(Ld["created"][ix]), "LUNCHCANCELED": ts_str(Ld["cancel"][ix]),
            "WCICANCELED": "false", "NSLPELIGIBILITY": Q, "ORDERCHANNELDETAIL": det_str[Ld["detail"][ix]]})
    write_chunks(os.path.join(out, "STUDENTLUNCH.csv"), build_l, NL)
    a_created = Ld["created"][sp] + rng.integers(4, 240, len(sp))
    a_cancel = np.where(Ld["cancel"][sp] >= 0, Ld["cancel"][sp] + rng.integers(0, 3, len(sp)), -1)
    a_chan = np.where(Ld["chan"][sp] < 0, 4, Ld["chan"][sp])
    a_chan_arr = np.array(["app-iphone", "app-android", "web-calendar", "web-mobile", Q])
    comp_ids = np.array([c["id"] for c in comps])
    ao = np.argsort(a_created, kind="stable")
    def build_a(a, b):
        ix = ao[a:b]
        return pd.DataFrame({"ADDONID": 30000001 + np.arange(a, b), "STUDENTLUNCHID": lunch_id[sp[ix]],
                             "COMPONENTID": comp_ids[sc_[ix]], "ORDERCHANNEL": a_chan_arr[a_chan[ix]],
                             "ADDONCREATED": ts_str(a_created[ix]), "ADDONCANCELED": ts_str(a_cancel[ix]),
                             "ADDONCOMBOID": 0})
    write_chunks(os.path.join(out, "STUDENTLUNCHADDON.csv"), build_a, len(sp))
    print(f"written lunches/addons {time.time()-t0:.0f}s")

    # ------------------------------------------------------------- dimension tables
    def wcsv(name, rows, cols=None):
        d_ = pd.DataFrame(rows, columns=cols)
        for c in d_.columns:
            if d_[c].dtype == bool: d_[c] = d_[c].map({True: "true", False: "false"})
        d_.to_csv(os.path.join(out, f"{name}.csv"), index=False, quoting=csv.QUOTE_NONE, quotechar="~",
                  encoding="utf-8-sig")
        return len(d_)

    counts = {}
    counts["ALLERGEN"] = wcsv("ALLERGEN", [{"ALLERGENID": a, "ALLERGENNAME": n, "ACTIVECODE": True, "ALLERGENTYPE": t}
                                           for a, n, t in ALLERGENS])
    counts["CUISINE"] = wcsv("CUISINE", [{"CUISINEID": k, "CUISINENAME": v} for k, v in CUISINES.items()])
    counts["MENUITEM"] = wcsv("MENUITEM", [{
        "MENUITEMID": it["id"], "CATERERID": 1, "CUISINEID": it["cui"], "PRODUCTTYPEID": 1, "ITEMCODE": it["code"],
        "ITEMNAME": it["name"], "ITEMNAMELONG": it["name"],
        "ITEMDESCRIPTION": f"Synthetic menu description for {it['name'].lower()}", "SERVINGTYPE": it["serv"],
        "FILMTYPE": str(rng.choice(["N/A", "Perforated", "Non-perforated"])), "NSLPREQUIREDSNACK": "None",
        "CONDIMENTPROFILEID": int(rng.integers(1000, 1017)), "NSLPTYPE": int(rng.choice([1, 3])),
        "ADVENTUROUSRATING": it["adv"], "ACTIVECODE": True, "ISEVERYDAYITEM": it["staple"],
        "ISALLERGYINFOCOMPLETE": True, "ISNEW": False, "ISPESCETARIAN": it["prot"] in ("fish", "veg", "vegan"),
        "ISPREMIUM": bool(it["prem"]), "SIDEDISHID": 0, "PRICINGCLASS": 0, "ADDONSALLOWED": True} for it in items])
    counts["MENUITEMALLERGEN"] = wcsv("MENUITEMALLERGEN", [{"MENUITEMID": it["id"], "ALLERGENID": a, "ACTIVECODE": True}
                                                           for it in items for a in sorted(it["alg"])])
    counts["COMPONENT"] = wcsv("COMPONENT", [{"COMPONENTID": c["id"], "CATERERID": 1, "COMPONENTTYPE": c["type"],
                                              "COMPONENTNAME": c["name"], "COMPONENTDESCRIPTION": c["name"],
                                              "ACTIVECODE": True, "COMPONENTAVAILABILITY": 1, "GROUPWITH": c["id"],
                                              "COMPONENTSORT": c["idx"]} for c in comps])
    ca = []
    for c in comps:
        for a in sorted(c["alg"]):
            ca.append({"COMPONENTID": c["id"], "ALLEGENID": a, "ACTIVECODE": True,
                       "DATEADDED": ts_str([int(datetime(2021, 6, 1).timestamp()) + int(rng.integers(0, 4e7))])[0]})
    counts["COMPONENTALLERGEN"] = wcsv("COMPONENTALLERGEN", ca)
    counts["SCHOOL"] = wcsv("SCHOOL", [{
        "SCHOOLID": s["id"], "CATERERID": 1, "FACILITYID": int(rng.choice([2, 4, 6])),
        "ACCTMGRCONTACTID": int(rng.integers(1, 40)), "SCHOOLCODE": f"SYN{s['i']:03d}", "SCHOOLNAME": s["name"],
        "ADDRESS1": f"{int(rng.integers(100, 9900))} Example Ave", "ADDRESS2": Q, "CITY": s["city"],
        "COUNTY": s["county"], "STATE": "CA", "ZIP": int(rng.integers(90000, 95999)), "PHONE": 5550100000 + s["i"],
        "FAX": Q, "WEBSITE": f"https://school{s['i']:03d}.example.org", "PRINCIPAL": Q, "SECRETARY": Q,
        "SCHOOLCONTACTNAME": Q, "SCHOOLCONTACTEMAIL": f"office@school{s['i']:03d}.example.org", "HLCCONTACTNAME": Q,
        "HLCCONTACTEMAIL": Q, "HLCCONTACTPHONE": Q, "TROUBLESTATUS": 0, "REGISTRATIONCREDITAMOUNT": 0.0,
        "TYPE": s["type"], "PROGRAMTYPE": "Premium", "GRADELEVELS": f"{s['grades'][0]} {s['grades'][-1]}",
        "ACTIVECODE": True, "MENUACTIVE": True, "LENTENEXCEPTIONS": "Catholic" in s["type"],
        "ONLINEDR": not s["camp"], "HIDEREBATEDISPLAY": False, "ORDERLEADTIME": 9,
        "LASTMINUTEORDERLEADTIME": 9, "CANCELLEADTIME": 15, "CANCELNOPENALTYLEADTIME": int(rng.choice([7, 15, 16])),
        "UNKNOWNACCOUNTID": 9200000 + s["i"], "CRISISCOMMEMAIL": Q} for s in schools])
    counts["GRADE"] = wcsv("GRADE", grade_rows)
    counts["SERVINGGROUP"] = wcsv("SERVINGGROUP", sg_rows)
    counts["PACKINGGROUP"] = wcsv("PACKINGGROUP", pg_rows)

    # STUDENT (current state at end of data)
    last_p = np.where(sch_p[:, 2] >= 0, 2, np.where(sch_p[:, 1] >= 0, 1, 0))
    ever = n_orders > 0
    active = np.where(leave_reason != "", rng.random(N) < .3, rng.random(N) < .96)
    active &= ~(churn & (rng.random(N) < .25))
    cur_type = np.array([sub_code[n, last_p[n]] if sub_on[n, last_p[n]] else stu_type_base[n] for n in range(N)], dtype=object)
    eff_dt = np.array([(EPOCH + timedelta(days=int(sub_start[n, last_p[n]]))).isoformat() if sub_on[n, last_p[n]]
                       else (str(rng.choice(["2024-06-20", "2023-06-20"])) if rng.random() < .5 else "")
                       for n in range(N)], dtype=object)
    dom_portion = np.where(large_p > .3, 2, np.where(rng.random(N) < .5, 1, 0))
    counts["STUDENT"] = wcsv("STUDENT", {
        "STUDENTID": stu_id, "LUNCHACCOUNTID": np.array([f["id"] for f in fams])[fam_of],
        "GRADEID": gid_p[np.arange(N), last_p], "FIRSTNAME": rng.choice(FIRST, N), "LASTNAME": rng.choice(LAST, N),
        "STUDENTTYPE": cur_type, "ACTIVECODE": active, "DEFAULTPORTIONCODE": dom_portion, "DRINKCOMPONENTID": 0,
        "DATEREGISTERED": ts_str(reg_dt), "EXTERNALID": 0, "STUDENTTYPEEFFECTIVE": eff_dt, "NSLPELIGIBILITY": Q,
        "PALETTERATING": np.clip(np.round(adv + rng.normal(0, .6, N)), 0, 3).astype(int), "DATEUPDATED": ""})
    sa = pd.DataFrame(sa_rows, columns=["STUDENTALLERGENSID", "n", "ALLERGENID", "added", "ACTIVECODE", "deleted"])
    sa = sa.sort_values("added")
    counts["STUDENTALLERGENS"] = wcsv("STUDENTALLERGENS", {
        "STUDENTALLERGENSID": 95001 + np.arange(len(sa)), "STUDENTID": stu_id[sa.n.values],
        "ALLERGENID": sa.ALLERGENID.values, "DATEADDED": ts_str(sa.added.values), "ACTIVECODE": sa.ACTIVECODE.values,
        "DATEDELETED": ts_str(sa.deleted.values)})

    # ------------------------------------------------------------- ground truth (hidden; for evaluation only)
    gt = pd.DataFrame({"STUDENTID": stu_id, "LUNCHACCOUNTID": np.array([f["id"] for f in fams])[fam_of],
                       "loyalty": loyalty.round(3), "adventurousness": adv.round(3),
                       "fatigue": fatigue.round(3), "novelty_seeking": novelty.round(3),
                       "nonsub_weekly_freq": freq, "side_rate": side_rate, "favourite_side_id": np.array([c["id"] for c in comps])[fav_side],
                       "ordering_style": np.array(["monthly", "weekly", "daily", "mixed"])[style],
                       "never_orders": never, "churned": churn,
                       "churn_date": np.where(churn, (np.array(ord_end, "datetime64[D]")).astype(str), ""),
                       "leave_reason": leave_reason, "sub_utilisation": util.round(3), "extra_day_rate": extra_p.round(3)})
    for k_, c in enumerate(sorted(CUISINES)):
        gt[f"cuisine_pref_{CUISINES[c].replace(' ', '_')}"] = cpref[:, k_].round(3)
    gt.to_csv(os.path.join(gt_dir, "STUDENT_LATENT_TRAITS.csv"), index=False)
    subs = []
    for n in range(N):
        for p, lab in ((0, "2023-24"), (2, "2024-25")):
            if sub_on[n, p]:
                subs.append({"STUDENTID": stu_id[n], "SCHOOL_YEAR": lab, "PLAN_CODE": sub_code[n, p],
                             "DAYS_PER_WEEK": int(sub_days[n, p].sum()),
                             "PLAN_WEEKDAYS": ",".join(["Mon", "Tue", "Wed", "Thu", "Fri"][k] for k in range(5) if sub_days[n, p, k]),
                             "SIDES_INCLUDED": int(sub_S[n, p]),
                             "START_DATE": (EPOCH + timedelta(days=int(sub_start[n, p]))).isoformat(),
                             "END_DATE": (EPOCH + timedelta(days=int(sub_end[n, p]))).isoformat(),
                             "LEGACY_AUTO_ORDER": bool(sub_auto[n, p])})
    pd.DataFrame(subs).to_csv(os.path.join(gt_dir, "SUBSCRIPTION_PLANS.csv"), index=False)
    pd.DataFrame(change_log, columns=["SCHOOLID", "CYCLE_START", "CHANGE", "MENUITEMID", "DETAIL"]).to_csv(
        os.path.join(gt_dir, "MENU_CYCLE_CHANGES.csv"), index=False)
    counts.update({"STUDENTLUNCH": NL, "STUDENTLUNCHADDON": len(sp), "TBLPACKINGGROUPMENUS": pgm_rows_count,
                   "TBLPACKINGGROUPADDONS": None})
    json.dump({k: v for k, v in counts.items() if v is not None}, open(os.path.join(gt_dir, "row_counts.json"), "w"), indent=1)
    print(f"done {time.time()-t0:.0f}s")
    for k_, v in counts.items(): print(f"{k_:24s} {v if v is None else f'{v:,}'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--students", type=int, default=30000)
    ap.add_argument("--out", default="./data")
    a = ap.parse_args()
    main(a.seed, a.out, a.students)
