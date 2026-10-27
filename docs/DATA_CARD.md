# Synthetic School-Meal Subscription Dataset (v2): Data Card

## Summary
This is a fully synthetic dataset that mirrors the schema of a US school-lunch pre-ordering and subscription platform. It was built for dissertation research on constraint-aware auto-ordering. The schema (table names, column names, value formats and data quirks) follows an approved 10-row sample and the provider's data dictionary. Behavioural parameters are calibrated to aggregate EDA statistics.

- **No real records are included.** All people, schools, addresses and IDs are invented.
- **The provider is not named anywhere.**

| Item | Value |
|---|---|
| Period | 14 Aug 2023 to 13 Jun 2025 (about 22 months, two school years and one summer) |
| Students | 30,000 (about 20,800 lunch accounts or families) |
| Schools | 105 (about 285 students per school, matching the real ratio), 227 packing groups |
| School types | Elementary, TK-5, K-8, Middle, High, K-12, and 3 summer camps |
| Lunch orders | about 2.7M rows (live and cancelled) |
| Side add-ons | about 3.0M rows |
| Menu rows | about 980k (school × packing group × day × item × portion) |
| Reproducible | `python generate_school_meal_data.py --seed 42 --students 30000 --out ./data` (about 2 minutes) |

## Tables
These tables follow the sample schema: ALLERGEN, COMPONENT, COMPONENTALLERGEN (keeps the source typo `ALLEGENID`), GRADE, MENUITEM, SCHOOL, STUDENT, STUDENTALLERGENS, STUDENTLUNCH, STUDENTLUNCHADDON, TBLPACKINGGROUPMENUS and TBLPACKINGGROUPADDONS.

These tables were added from the data dictionary or for usability:
- **CUISINE:** lets you join on `MENUITEM.CUISINEID`.
- **SERVINGGROUP and PACKINGGROUP:** link the chain `GRADE.SERVINGGROUPID → SERVINGGROUP.PACKINGGROUPID → TBLPACKINGGROUPMENUS`.
- **MENUITEMALLERGEN:** entree allergens, which you need for safety filtering. This table is not in the sample.

The `_ground_truth/` folder is **hidden and for evaluation only**. Do not use it as model input.
- `STUDENT_LATENT_TRAITS.csv`: each student's simulated loyalty, adventurousness, cuisine preferences, ordering style and churn flags.
- `SUBSCRIPTION_PLANS.csv`: every plan per school year, with plan weekdays, included sides and the legacy auto-order flag.
- `MENU_CYCLE_CHANGES.csv`: every item rotated out, rotated in, or moved to another weekday, per school and cycle.

## How to join the tables
- A lunch's school is found through `STUDENTLUNCH.GRADEID → GRADE.SCHOOLID`.
- A lunch's menu is found through `GRADEID → SERVINGGROUPID → PACKINGGROUPID`, then matching `TBLPACKINGGROUPMENUS` on PACKINGGROUPID, DELIVERYDATE and MENUITEMID.
- `DELIVERYDATE` in the menu table carries a `00:00:00.000` time, as in the source.

## Hard guarantees (all checked in VALIDATION_REPORT.txt)
1. Every lunch item was on that packing group's published menu for that date. Large portions appear only where Large was offered.
2. Zero lunches conflict with a student's active allergens or dietary preferences.
3. At most one live (non-cancelled) lunch per student per day. A swap means the original row is cancelled and a new row is created.
4. **Subscriptions are locked per school year.** The days per week, the exact weekdays and the included sides never change mid-year.
   - Subscription-coded lunches fall only on plan weekdays, within the plan window.
   - Mid-year sign-ups start on the sign-up date and stay locked until June.
   - Lunches ordered beyond the plan are coded `Paid` (retail).
5. Menus run in 2-month cycles, with about 30% of each school's line-up changing per cycle. Changes are rotations (out and back in) or weekday moves. Weekday-anchored items move rarely and return to their anchor day.
6. Every school offers 6–10 entrees per day, and every day has a menu. Closure days (holidays, breaks, in-service days) have no menu rows, so they have no orders.
7. Prices are fixed for the whole period.

## Behaviour model (what generates the choices)
Each child chooses from today's safe menu items using a utility that combines:
- **Age-dependent appeal.** Younger grades favour mac and cheese, noodles, nuggets and hot dogs (hot dogs peak in grade 2). Cheeseburger peaks around grade 8. Teens and staff favour bowls, sushi, salads and sandwiches.
- **Personal taste.** Cuisine and protein preferences, an item-level taste shared partly with siblings, and an adventurousness match.
- **Habit.** A personal loyalty level times the share of past orders for that item, plus a weekday habit.
- **Short-term fatigue** after repeating an item, and novelty-seeking for untried items.
- **Random noise**, which is higher for new students during their first 10 orders.

Ordering behaviour:
- **Subscribers** order on their plan weekdays at their personal utilisation level. About 30% also order on extra days at retail.
- **Non-subscribers** order based on weekly frequency and weekday habits, and are more likely to order when a favourite item is on the menu.
- **Families** have an ordering style (monthly batch, weekly, day-before or same-day, or mixed) and a usual channel. They also have occasional dormancy gaps, such as lapsed wallet funding.
- **Lunch types:**
  - **PreOrder:** the standard type.
  - **LastMinute:** same-day orders after the 9 AM cutoff, entree only.
  - **IOU:** emergency lunches. They come with no channel and no sides, and **should be excluded from modelling**.
  - **AutoOrder:** a legacy system that runs throughout and repeats the child's usual item for that weekday. Batches are created on the third Thursday of the prior month. Parents cancel or swap these much more often when the system's pick isn't what the child would choose.
- **Cancellations:** absences (no replacement), deliberate swaps (replaced by the child's preferred item), and instant edits (cancelled within seconds).
- **Population dynamics:**
  - Each August, students move up a grade. Graduating students either transfer to another school or leave.
  - New students enrol over time.
  - Some students only try the service briefly, some churn mid-way, and some register but never order.
- **Sides:**
  - Subscribers usually take their plan's included number of sides. 0-side subscribers often buy one anyway.
  - Non-subscribers have a personal side rate.
  - Choices depend on entree pairing, a favourite side, a preference for mixing types, seasonal fruit, and monthly smoothies (first given weekday of the month).

## Calibration against real aggregates
See `VALIDATION_REPORT.txt` for exact numbers from this build. Targets came from the provider's EDA:
- about 85% of students without allergens
- about 90% of active students have ordered
- over 40% dormant for 3 months
- weekly order mix of 27/23/18/16/15%
- variety buckets of 51/23/10/16%
- cuisine ranking
- Pepperoni Pizza on Monday 86%, Fish Sticks and Tuna on Friday 92–95%
- Cheeseburger weekly (most common gap 7 days), Tenders often on consecutive days (gap 1)
- smoothies monthly, one exclusive-menu school
- Thanksgiving week and late-April dips, low summer volume
- about 55% of lunches from subscriptions
- about half of lunches without sides

## Format notes (replicated from the source sample)
- Files are UTF-8 with a byte-order mark.
- Empty text fields appear as the literal `""""""` export artifact in `NSLPELIGIBILITY`, `ORDERCHANNEL` (for auto and IOU lunches), `ORDERCHANNELDETAIL`, `GRADEQUALVALUE` and some SCHOOL fields. Clean them with `df.replace('""', np.nan)` after reading.
- `ISSERVED` is deprecated and always false. **Use `LUNCHCANCELED IS NULL` for live lunches.**
- Channel values have spaces in STUDENTLUNCH (`app iphone`) and hyphens in STUDENTLUNCHADDON (`app-iphone`), as in the source.
- Plan codes in `STUDENTTYPE` look like `Sub 2200`. The digits are days per week, then included sides, then a large-portion upgrade flag, then a reserved 0.
- Inactive allergen records have a `DATEDELETED` only from January 2025 onward, and 4 records were deleted at their creation timestamp. Both are data-quality quirks that exist in the real data.

## Design choices that differ from the real data (state them in your methodology)
- 6–10 entree choices per day everywhere. The real EDA found 2–3 choices at some contract-served schools.
- The legacy auto-order system runs across the whole period. This is plausible but not verified.
- The 30% menu change per 2-month cycle is the researcher's estimate.
- The dataset uses 30k students across 105 schools, a subset of the real scale of about 74k across 261 schools.
- The catalogue is fixed (87 entrees, 35 sides). Menu change happens through rotation and weekday moves, not brand-new items.

## Limitations
- Behaviour comes from a designed generative model calibrated to a dozen marginal statistics. How the variables interact is assumed.
- **Circularity risk:** recommenders that resemble the generator's utility model may look better here than on real children. Mitigations: test robustness by regenerating with different seeds and parameters, and report recovery of the ground-truth traits rather than claims about real behaviour.
- There is no nutrition data, no price variation, and no weather or calendar events beyond school closures.

## Suggested citation text
> "Experiments use a synthetic dataset generated to mirror the schema of a US school-lunch subscription platform. The generator was calibrated to aggregate statistics from an approved data sample. No individual-level real data is included. Because the data-generating process is known, the recovery of planted behaviours can be evaluated directly."
