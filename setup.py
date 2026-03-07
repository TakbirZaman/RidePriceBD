# setup.py — Run this ONCE
# python setup.py

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import math

np.random.seed(42)

# ── Dhaka locations with lat/lng coordinates ──────────────────────────────
LOCATIONS = {
    "Jagannath University":  (23.7066, 90.4070),
    "Motijheel":             (23.7330, 90.4182),
    "Lalbagh":               (23.7203, 90.3871),
    "Shahbagh":              (23.7394, 90.3964),
    "New Market":            (23.7313, 90.3855),
    "Dhanmondi":             (23.7461, 90.3742),
    "Farmgate":              (23.7582, 90.3900),
    "Mohammadpur":           (23.7578, 90.3572),
    "Mohakhali":             (23.7799, 90.4070),
    "BRAC University":       (23.7779, 90.4351),
    "East West University":  (23.7647, 90.4279),
    "Gulshan":               (23.7925, 90.4078),
    "Banani":                (23.7937, 90.4066),
    "UIU":                   (23.7672, 90.4284),
    "Mirpur 10":             (23.8073, 90.3680),
    "Jamuna Future Park":    (23.8130, 90.4224),
    "AIUB":                  (23.8304, 90.4241),
    "NSU":                   (23.8150, 90.4244),
    "Uttara":                (23.8759, 90.3795),
}

# ── Vehicle pricing per km + base fare ───────────────────────────────────
# Based on real Dhaka market rates (2024)
VEHICLES = {
    "Rickshaw": {"base": 30,  "per_km": 15,  "max_km": 6},   # short trips only
    "Bike":     {"base": 35,  "per_km": 14,  "max_km": 999}, # Pathao/Shohoz
    "Car":      {"base": 80,  "per_km": 28,  "max_km": 999}, # Uber/Pathao car
    "Bus":      {"base": 10,  "per_km":  5,  "max_km": 999}, # BRTC / local bus
}

WEATHER_MULT = {
    "Sunny":   1.00,
    "Regular": 1.10,
    "Rainy":   1.40,
    "Stormy":  1.65,
}

# ── Haversine distance between two lat/lng points (km) ───────────────────
def distance_km(loc1, loc2):
    lat1, lon1 = LOCATIONS[loc1]
    lat2, lon2 = LOCATIONS[loc2]
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    straight = R * 2 * math.asin(math.sqrt(a))
    return round(straight * 1.35, 1)   # 1.35x road factor for Dhaka traffic

# ── Generate dataset ──────────────────────────────────────────────────────
loc_list = list(LOCATIONS.keys())
rows = []

for _ in range(1200):
    from_loc = np.random.choice(loc_list)
    to_loc   = np.random.choice([l for l in loc_list if l != from_loc])
    vehicle  = np.random.choice(list(VEHICLES.keys()))
    weather  = np.random.choice(list(WEATHER_MULT.keys()), p=[0.40, 0.25, 0.25, 0.10])
    hour     = np.random.randint(0, 24)
    dow      = np.random.randint(0, 7)
    demand   = round(np.random.uniform(0.1, 1.0), 2)

    dist = distance_km(from_loc, to_loc)
    v    = VEHICLES[vehicle]

    # Skip unrealistic rickshaw long trips
    if vehicle == "Rickshaw" and dist > v["max_km"]:
        continue

    base  = v["base"] + dist * v["per_km"]
    w_m   = WEATHER_MULT[weather]

    # Rush hours
    if 8 <= hour <= 10 or 17 <= hour <= 20:
        base *= 1.30
    # Late night
    if 23 <= hour or hour <= 5:
        base *= 1.15
    # Friday lower demand
    if dow == 4:
        base *= 0.90
    # Bus is not affected much by demand/weather
    if vehicle == "Bus":
        w_m     = min(w_m, 1.10)
        demand *= 0.3

    price = base * w_m + demand * 25 + np.random.normal(0, 8)
    price = round(max(v["base"], price), 0)

    rows.append([from_loc, to_loc, vehicle, weather, hour, dow, demand, dist, price])

df = pd.DataFrame(rows, columns=["from_loc","to_loc","vehicle","weather","hour","dow","demand","distance_km","price"])
df.to_csv("rides.csv", index=False)

print(f"✅ Dataset saved: rides.csv  ({len(df)} rows)")
for v in VEHICLES:
    sub = df[df.vehicle == v]
    print(f"   {v:10s}  avg ৳{sub.price.mean():.0f}  |  range ৳{sub.price.min():.0f}–৳{sub.price.max():.0f}")

# ── Train model ───────────────────────────────────────────────────────────
X = df[["from_loc","to_loc","vehicle","weather","hour","dow","demand","distance_km"]]
y = df["price"]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["from_loc","to_loc","vehicle","weather"])
], remainder="passthrough")

model = Pipeline([
    ("pre", pre),
    ("reg", GradientBoostingRegressor(n_estimators=200, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"\n✅ Model trained  |  MAE: ৳{mae:.1f}")

joblib.dump(model,     "model.pkl")
joblib.dump(LOCATIONS, "locations.pkl")
joblib.dump(VEHICLES,  "vehicles.pkl")
print("✅ Saved: model.pkl  locations.pkl  vehicles.pkl")
print("\nNext → python api.py")
