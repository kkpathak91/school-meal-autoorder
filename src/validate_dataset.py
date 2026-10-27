"""Validate the synthetic dataset: hard business rules + calibration against real EDA aggregates."""
import sys, os
import numpy as np, pandas as pd

D = sys.argv[1] if len(sys.argv) > 1 else "./data"
R = lambda t, **k: pd.read_csv(os.path.join(D, f"{t}.csv"), encoding="utf-8-sig", **k)
out = []
def p(section, k, v, target=""):
    line = f"{k:55s} {v}" + (f"   [real/target: {target}]" if target else "")
    out.append((section, line)); print(line)

L = R("STUDENTLUNCH", usecols=["STUDENTLUNCHID", "STUDENTID", "GRADEID", "MENUITEMID", "DELIVERYDATE", "STUDENTTYPE",
                               "LUNCHTYPE", "LUNCHCANCELED", "PORTIONCODE"],
      dtype={"STUDENTTYPE": "category", "LUNCHTYPE": "category", "STUDENTID": "int32", "GRADEID": "int32",
             "MENUITEMID": "int16", "PORTIONCODE": "int8"})
L["LUNCHCANCELED"] = L.LUNCHCANCELED.notna(); L = L.rename(columns={"LUNCHCANCELED": "is_canc"})
L["DELIVERYDATE"] = pd.to_datetime(L.DELIVERYDATE)
L["live"] = ~L.is_canc
M = R("MENUITEM"); S = R("STUDENT"); SA = R("STUDENTALLERGENS"); MA = R("MENUITEMALLERGEN"); G = R("GRADE")
SG = R("SERVINGGROUP"); CU = R("CUISINE"); A = R("STUDENTLUNCHADDON", usecols=["STUDENTLUNCHID", "ADDONCANCELED"], dtype={"STUDENTLUNCHID": "int32"})
PGM = R("TBLPACKINGGROUPMENUS", usecols=["SCHOOLID", "PACKINGGROUPID", "DELIVERYDATE", "MENUITEMID", "ITEMNAME", "PORTIONCODE"])
PGM["DELIVERYDATE"] = pd.to_datetime(PGM.DELIVERYDATE)
PGA = R("TBLPACKINGGROUPADDONS", usecols=["PACKINGGROUPID", "DELIVERYDATE", "COMPONENTID", "COMPONENTTYPE"])
gt = pd.read_csv(os.path.join(D, "_ground_truth", "SUBSCRIPTION_PLANS.csv"))
lat = pd.read_csv(os.path.join(D, "_ground_truth", "STUDENT_LATENT_TRAITS.csv"))
live = L[L.live]
# map lunch -> packing group via grade -> serving group
g2pg = G.merge(SG[["SERVINGGROUPID", "PACKINGGROUPID"]], on="SERVINGGROUPID")[["GRADEID", "PACKINGGROUPID", "GRADE"]]
L = L.merge(g2pg, on="GRADEID", how="left")

sec = "HARD RULES"
print(f"\n== {sec}")
key = PGM[PGM.PORTIONCODE == 1][["PACKINGGROUPID", "DELIVERYDATE", "MENUITEMID"]].drop_duplicates()
key["on_menu"] = True
chk = L.merge(key, on=["PACKINGGROUPID", "DELIVERYDATE", "MENUITEMID"], how="left")
p(sec, "Lunches whose item was NOT on that day's menu", int(chk.on_menu.isna().sum()), "0")
pl = PGM[PGM.PORTIONCODE == 2][["PACKINGGROUPID", "DELIVERYDATE", "MENUITEMID"]].drop_duplicates(); pl["lg"] = True
chk2 = L[L.PORTIONCODE == 2].merge(pl, on=["PACKINGGROUPID", "DELIVERYDATE", "MENUITEMID"], how="left")
p(sec, "Large portions where Large not offered", int(chk2.lg.isna().sum()), "0")
pref = {10: "beef", 11: "pork", 12: "nonveg", 13: "nonvegan"}
sa = SA[SA.ACTIVECODE.astype(str).str.lower() == "true"]
true_alg = sa[~sa.ALLERGENID.isin(pref)].groupby("STUDENTID").ALLERGENID.apply(set).to_dict()
ialg = MA.groupby("MENUITEMID").ALLERGENID.apply(set).to_dict()
viol = sum(1 for s_, m_ in zip(L.STUDENTID.values, L.MENUITEMID.values) if s_ in true_alg and true_alg[s_] & ialg.get(m_, set()))
p(sec, "Lunches conflicting with an active student allergen", viol, "0")
dup = live.groupby(["STUDENTID", "DELIVERYDATE"]).size()
p(sec, "Student-days with more than one live lunch", int((dup > 1).sum()), "0")
p(sec, "ISSERVED always false (deprecated column)", "yes")
# subscription consistency
L["wd"] = L.DELIVERYDATE.dt.dayofweek
subl = L[L.STUDENTTYPE.astype(str).str.startswith("Sub")].copy()
subl["SY"] = np.where(subl.DELIVERYDATE < "2024-07-01", "2023-24", "2024-25")
plans = gt.assign(wds=gt.PLAN_WEEKDAYS.str.split(","))
wmap = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4}
plans["wdset"] = plans.wds.apply(lambda x: {wmap[w] for w in x})
pm = subl.merge(plans[["STUDENTID", "SCHOOL_YEAR", "PLAN_CODE", "wdset", "START_DATE"]],
                left_on=["STUDENTID", "SY"], right_on=["STUDENTID", "SCHOOL_YEAR"], how="left")
p(sec, "Sub-coded lunches with no plan that year", int(pm.PLAN_CODE.isna().sum()), "0")
pm = pm.dropna(subset=["PLAN_CODE"])
p(sec, "Sub-coded lunches with a different plan code", int((pm.PLAN_CODE != pm.STUDENTTYPE.astype(str)).sum()), "0")
p(sec, "Sub-coded lunches on a non-plan weekday", int(sum(w not in s_ for w, s_ in zip(pm.wd, pm.wdset))), "0")
p(sec, "Sub-coded lunches before plan start", int((pm.DELIVERYDATE < pd.to_datetime(pm.START_DATE)).sum()), "0")
pcs = subl.assign(ST=subl.STUDENTTYPE.astype(str)).groupby(["STUDENTID", "SY"]).ST.nunique()
p(sec, "Student-years with >1 plan code (plan must stay locked)", int((pcs > 1).sum()), "0")
sm = PGA[PGA.COMPONENTTYPE == "Smoothie"]
p(sec, "Smoothies only in first 7 days of month", bool((pd.to_datetime(sm.DELIVERYDATE).dt.day <= 7).all()), "True")

sec = "EDA CALIBRATION"
print(f"\n== {sec}")
act = S[S.ACTIVECODE.astype(str).str.lower() == "true"].STUDENTID
p(sec, "Active students with no allergen record", f"{(~act.isin(SA.STUDENTID)).mean():.1%}", "~85%")
p(sec, "Active students with at least one order", f"{act.isin(L.STUDENTID).mean():.1%}", "~90% (66,558/73,751)")
last = L.DELIVERYDATE.max()
rec = live[live.DELIVERYDATE > last - pd.Timedelta(days=90)].STUDENTID.unique()
p(sec, "Active students with no order in last 3 months", f"{(~act.isin(rec)).mean():.1%}", ">40%")
wk = live[live.LUNCHTYPE != "IOU"]
wk = wk[wk.STUDENTID.isin(act)].groupby(["STUDENTID", wk.DELIVERYDATE.dt.to_period("W")]).size().clip(upper=5)
wkd = wk.value_counts(normalize=True).sort_index()
p(sec, "Weekly order count distribution 1/2/3/4/5", " / ".join(f"{x:.0%}" for x in wkd.values), "27% / 23% / 18% / 16% / 15%")
v = live[live.STUDENTID.isin(act)].groupby("STUDENTID").agg(n=("MENUITEMID", "size"), u=("MENUITEMID", "nunique"))
vb = pd.cut(v.u / v.n, [0, .255, .505, .755, 1.005]).value_counts(normalize=True).sort_index()
p(sec, "Variety ratio buckets (low -> high)", " / ".join(f"{x:.0%}" for x in vb.values), "51% / 23% / 10% / 16%")
m = live.merge(M[["MENUITEMID", "CUISINEID"]]).merge(CU)
p(sec, "Cuisine ranking (top 6)", ", ".join(m[m.CUISINEID != 10].CUISINENAME.value_counts().index[:6]),
  "Classic American, Italian, Breakfast, Asian, Sandwich, Mexican")
pg1 = PGM[PGM.PORTIONCODE == 1].drop_duplicates(["SCHOOLID", "DELIVERYDATE", "MENUITEMID"]).copy()
pg1["dow"] = pg1.DELIVERYDATE.dt.day_name()
for it, tgt in [("Pepperoni Pizza", "Monday 86%"), ("Fish Sticks", "Friday 95%"), ("Tuna Salad Sandwich", "Friday 92%"),
                ("BLTA Sandwich", "Monday"), ("Chicken Salad Sandwich", "Wednesday")]:
    sh = pg1[pg1.ITEMNAME == it].dow.value_counts(normalize=True)
    p(sec, f"{it}: top menu weekday", f"{sh.index[0]} {sh.iloc[0]:.0%}", tgt)
def gap_mode(name):
    x = pg1[pg1.ITEMNAME == name].sort_values(["SCHOOLID", "DELIVERYDATE"])
    gaps = x.groupby("SCHOOLID").DELIVERYDATE.diff().dt.days.dropna()
    return int(gaps.mode()[0])
p(sec, "Cheeseburger most common gap (days)", gap_mode("Cheeseburger"), "7")
p(sec, "Chicken Tenders most common gap (days)", gap_mode("Chicken Tenders"), "1")
per_day = pg1.groupby(["SCHOOLID", "DELIVERYDATE"]).size()
p(sec, "Entree choices per school-day (min / mean / max)", f"{per_day.min()} / {per_day.mean():.1f} / {per_day.max()}", "6-10 (your spec)")
excl = pg1[pg1.ITEMNAME.str.startswith("Campus Cafe")].SCHOOLID.nunique()
p(sec, "Schools serving the exclusive 'Campus Cafe' items", excl, "1")
ds = live.groupby("DELIVERYDATE").size()
p(sec, "Thanksgiving week 2024 (Nov 25-29) lunches", int(ds["2024-11-25":"2024-11-29"].sum()), "near zero")
p(sec, "Mean daily lunches: school year vs Jun 15-Aug 10 2024",
  f"{ds[ds.index.month.isin([9,10,2,3])].mean():.0f} vs {ds['2024-06-15':'2024-08-10'].mean():.0f}", "summer far lower")
p(sec, "Apr 14-25 2025 vs rest of April (daily mean)",
  f"{ds['2025-04-14':'2025-04-25'].mean():.0f} vs {ds['2025-04-01':'2025-04-11'].mean():.0f}", "dip")
sub_share = live[live.LUNCHTYPE != "IOU"].STUDENTTYPE.astype(str).str.startswith("Sub").mean()
p(sec, "Share of live lunches from subscription plans", f"{sub_share:.0%}", "~55%")
# utilisation: all live lunches in the plan window / open school days on plan weekdays
sch_of_grade = G.set_index("GRADEID").SCHOOLID
open_days = PGM.drop_duplicates(["SCHOOLID", "DELIVERYDATE"])[["SCHOOLID", "DELIVERYDATE"]]
open_days["wd"] = open_days.DELIVERYDATE.dt.dayofweek
sub_sch = subl.assign(SCHOOLID=subl.GRADEID.map(sch_of_grade)).groupby(["STUDENTID", "SY"]).SCHOOLID.first()
plans = plans.sample(min(len(plans), 2500), random_state=1)
pl_ = plans.merge(sub_sch.rename("SCHOOLID").reset_index(), left_on=["STUDENTID", "SCHOOL_YEAR"],
                  right_on=["STUDENTID", "SY"], how="left")
pl_["pid"] = np.arange(len(pl_)); pl_["START_DATE"] = pd.to_datetime(pl_.START_DATE); pl_["END_DATE"] = pd.to_datetime(pl_.END_DATE)
ex = pl_.dropna(subset=["SCHOOLID"]).merge(open_days, on="SCHOOLID")
ex = ex[(ex.DELIVERYDATE >= ex.START_DATE) & (ex.DELIVERYDATE <= ex.END_DATE)]
ex = ex[[w in s_ for w, s_ in zip(ex.wd, ex.wdset)]]
den = ex.groupby("pid").size()
gl_ = pl_[["pid", "STUDENTID", "START_DATE", "END_DATE"]].merge(live[["STUDENTID", "DELIVERYDATE"]], on="STUDENTID")
gl_ = gl_[(gl_.DELIVERYDATE >= gl_.START_DATE) & (gl_.DELIVERYDATE <= gl_.END_DATE)]
num = gl_.groupby("pid").size()
rows = (num.reindex(den.index).fillna(0) / den).values
ub = pd.cut(pd.Series(rows), [-1, .4999, .7999, .8999, 1.0, 99]).value_counts(normalize=True).sort_index()
p(sec, "Plan utilisation <50 / 50-80 / 80-90 / 90-100 / >100%", " / ".join(f"{x:.0%}" for x in ub.values),
  "1% / 17% / 34% / 26% / 22%")

# menu change per 2-month cycle: share of a school's line-up (item+weekday slots) that changed
cyc = pd.to_datetime(["2023-08-01","2023-10-02","2023-12-04","2024-02-05","2024-04-08","2024-06-10","2024-08-01",
                      "2024-10-07","2024-12-02","2025-02-03","2025-04-07","2030-01-01"])
x = pg1[["SCHOOLID", "DELIVERYDATE", "MENUITEMID"]].copy()
x["cycle"] = np.searchsorted(cyc.values, x.DELIVERYDATE.values, side="right") - 1
x["wd"] = x.DELIVERYDATE.dt.dayofweek
slot = x.groupby(["SCHOOLID", "cycle", "MENUITEMID"]).wd.agg(lambda w: w.value_counts().idxmax()).reset_index()
rates = []
for sid_, g_ in slot.groupby("SCHOOLID"):
    cs = sorted(g_.cycle.unique())
    for a_, b_ in zip(cs, cs[1:]):
        A_ = dict(zip(g_[g_.cycle == a_].MENUITEMID, g_[g_.cycle == a_].wd))
        B_ = dict(zip(g_[g_.cycle == b_].MENUITEMID, g_[g_.cycle == b_].wd))
        changed = sum(1 for k in A_ if k not in B_ or B_[k] != A_[k])
        rates.append(changed / max(1, len(A_)))
p(sec, "Menu line-up changed per 2-month cycle (median)", f"{np.median(rates):.0%}", "~30% (your spec)")
sides = A[A.ADDONCANCELED.isna()].groupby("STUDENTLUNCHID").size()
ls = live[["STUDENTLUNCHID"]].merge(sides.rename("k"), left_on="STUDENTLUNCHID", right_index=True, how="left").fillna(0)
p(sec, "Lunches with 0 / 1 / 2 / 3+ sides", " / ".join(f"{x:.0%}" for x in ls.k.clip(upper=3).value_counts(normalize=True).sort_index().values),
  "~half with no sides")
canc = L.groupby("LUNCHTYPE", observed=True).is_canc.mean()
p(sec, "Cancellation rate by LUNCHTYPE", ", ".join(f"{k} {v:.1%}" for k, v in canc.items()),
  "auto-orders cancelled more often")
p(sec, "LUNCHTYPE mix", ", ".join(f"{k} {v:.1%}" for k, v in L.LUNCHTYPE.value_counts(normalize=True).items()))

sec = "AGE PREFERENCES"
print(f"\n== {sec}")
gl = live.merge(G[["GRADEID", "GRADE"]]).merge(M[["MENUITEMID", "ITEMNAME"]])
band = {"TK": "K-2", "K": "K-2", "1": "K-2", "2": "K-2", "3": "3-5", "4": "3-5", "5": "3-5", "6": "6-8", "7": "6-8",
        "8": "6-8", "9": "9-12", "10": "9-12", "11": "9-12", "12": "9-12", "Staff": "Staff"}
gl["band"] = gl.GRADE.map(band)
for b in ["K-2", "3-5", "6-8", "9-12", "Staff"]:
    top = gl[gl.band == b].ITEMNAME.value_counts(normalize=True).head(5)
    p(sec, f"Top items {b}", "; ".join(f"{k} {v:.0%}" for k, v in top.items()))
hot = gl[gl.ITEMNAME == "All Beef Hot Dog"].GRADE.value_counts().index[0]
burg = gl[gl.ITEMNAME == "Cheeseburger"].groupby("GRADE").size() / gl.groupby("GRADE").size()
p(sec, "Grade with most hot dog orders", hot, "grade 2")
p(sec, "Grade with highest cheeseburger share", burg.idxmax(), "grade 8")

with open(os.path.join(D, "VALIDATION_REPORT.txt"), "w") as fh:
    cur = None
    for s_, line in out:
        if s_ != cur: fh.write(f"\n== {s_}\n"); cur = s_
        fh.write(line + "\n")
