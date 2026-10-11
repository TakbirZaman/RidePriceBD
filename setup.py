# setup.py — Run this ONCE
# python setup.py

import pandas as pd
import numpy as np
import joblib
import math
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

np.random.seed(42)

LOCATIONS = {
    "Jagannath University":         (23.7066, 90.4070),
    "Tikatuli":                     (23.7250, 90.4250),
    "Sayedabad Bus Terminal":       (23.7180, 90.4320),
    "Jatrabari":                    (23.7100, 90.4380),
    "Maniknagar":                   (23.7200, 90.4400),
    "Mugda":                        (23.7330, 90.4430),
    "Kamalapur":                    (23.7334, 90.4196),
    "Khilgaon Flyover":             (23.7470, 90.4280),
    "Shahjahanpur":                 (23.7490, 90.4200),
    "Rajarbagh":                    (23.7400, 90.4300),
    "Malibagh Railgate":            (23.7510, 90.4270),
    "Malibagh Chowdhury Para":      (23.7530, 90.4310),
    "Rampura Bridge":               (23.7630, 90.4320),
    "Abul Hotel":                   (23.7680, 90.4330),
    "Merul Badda":                  (23.7779, 90.4351),
    "Middle Badda":                 (23.7800, 90.4380),
    "Badda":                        (23.7850, 90.4430),
    "Notun Bazar":                  (23.7920, 90.4480),
    "Nadda":                        (23.8050, 90.4300),
    "Bashundhara / Jamuna FP":      (23.8130, 90.4224),
    "Kuril / Kuratoli (AIUB)":      (23.8200, 90.4250),
    "Khilkhet":                     (23.8330, 90.4150),
    "Airport":                      (23.8433, 90.4033),
    "Jashimuddin (Uttara)":         (23.8680, 90.3960),
    "Rajlakshmi":                   (23.8720, 90.3850),
    "Azampur":                      (23.8780, 90.3820),
    "House Building":               (23.8800, 90.3780),
    "Abdullahpur":                  (23.8900, 90.3750),
    "Ray Saheb Bazar":              (23.7120, 90.4080),
    "Naya Bazar":                   (23.7180, 90.4060),
    "Gulistan":                     (23.7260, 90.4167),
    "Paltan":                       (23.7330, 90.4167),
    "Press Club":                   (23.7310, 90.4100),
    "High Court":                   (23.7360, 90.4080),
    "Matsya Bhaban":                (23.7410, 90.4000),
    "Shahbagh":                     (23.7394, 90.3964),
    "Bangla Motor":                 (23.7470, 90.3930),
    "Karwan Bazar":                 (23.7530, 90.3930),
    "Farmgate":                     (23.7582, 90.3900),
    "Satrasta":                     (23.7650, 90.3900),
    "Nabisco":                      (23.7700, 90.3950),
    "Mohakhali Bus Terminal":       (23.7799, 90.4070),
    "Amtoli (Mohakhali)":           (23.7820, 90.4070),
    "Chairman Bari":                (23.7900, 90.4070),
    "Kakali (Banani)":              (23.7937, 90.4066),
    "Sainik Club / Banani Railgate":(23.7960, 90.4060),
    "Cantonment (Radisson)":        (23.8030, 90.4000),
    "MES":                          (23.8100, 90.3980),
    "Azimpur / Palashi":            (23.7313, 90.3870),
    "New Market / Nilkhet":         (23.7313, 90.3855),
    "City College / Science Lab":   (23.7400, 90.3820),
    "Kalabagan":                    (23.7430, 90.3780),
    "Dhanmondi 15/32":              (23.7461, 90.3742),
    "Dhanmondi 27":                 (23.7490, 90.3700),
    "Asad Gate":                    (23.7570, 90.3680),
    "College Gate":                 (23.7600, 90.3660),
    "Shishu Mela":                  (23.7640, 90.3640),
    "Shyamoli":                     (23.7678, 90.3620),
    "Kalyanpur":                    (23.7750, 90.3600),
    "Technical":                    (23.7800, 90.3620),
    "Darussalam":                   (23.7850, 90.3620),
    "Mazar Road":                   (23.7900, 90.3630),
    "Mirpur 1":                     (23.7940, 90.3660),
    "Mirpur 2":                     (23.8000, 90.3680),
    "Mirpur 10":                    (23.8073, 90.3680),
    "Mirpur 11":                    (23.8140, 90.3660),
    "Pallabi":                      (23.8200, 90.3650),
    "Mirpur 12":                    (23.8260, 90.3640),
    "Agargaon (IDB Bhaban)":        (23.7772, 90.3800),
    "Jahangir Gate":                (23.7780, 90.3780),
    "Kochukhet":                    (23.7950, 90.3950),
    "Mirpur 14":                    (23.8200, 90.3700),
    "Gulshan 1":                    (23.7814, 90.4143),
    "Gulshan 2":                    (23.7936, 90.4153),
    "Badda Link Road":              (23.7850, 90.4400),
    "Chandpur":                     (23.2333, 90.6552),
}

VEHICLES = {
    # Uber/BRTA-aligned (Dhaka 2025-26, fair street prices):
    # Bike = Uber Moto (base 30, 12/km, 0.5/min)
    # Car  = UberX (base 40, 18/km, 3/min)
    # CNG  = BRTA meter (40/2km + 12/km) + street reality -> ~15/km, 1/min
    # Bus  = BRTA city bus 2.6/km, min 10
    # Rickshaw = negotiated street rate, short-trip friendly
    "Rickshaw": {"base": 20,  "per_km": 22,  "per_min": 0,   "max_km": 6},
    "Bike":     {"base": 30,  "per_km": 12,  "per_min": 0.5, "max_km": 999},
    "Car":      {"base": 40,  "per_km": 18,  "per_min": 3,   "max_km": 999},
    "Bus":      {"base": 10,  "per_km": 2.6, "per_min": 0,   "max_km": 999},
    "CNG":      {"base": 40,  "per_km": 14,  "per_min": 1,   "max_km": 999},
}

WEATHER_MULT = {
    "Sunny":   1.00,
    "Regular": 1.05,
    "Rainy":   1.20,
    "Stormy":  1.40,
}

def distance_km(loc1, loc2):
    lat1, lon1 = LOCATIONS[loc1]
    lat2, lon2 = LOCATIONS[loc2]
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    # 1.25x road-detour factor (was 1.35x, overstated Dhaka road distance)
    return round(math.sqrt(a) * R * 2 * 1.25, 1)

loc_list = list(LOCATIONS.keys())
rows = []

for _ in range(1500):
    from_loc = np.random.choice(loc_list)
    to_loc   = np.random.choice([l for l in loc_list if l != from_loc])
    vehicle  = np.random.choice(list(VEHICLES.keys()))
    weather  = np.random.choice(list(WEATHER_MULT.keys()), p=[0.40, 0.25, 0.25, 0.10])
    hour     = np.random.randint(0, 24)
    dow      = np.random.randint(0, 7)
    demand   = round(np.random.uniform(0.1, 1.0), 2)

    dist = distance_km(from_loc, to_loc)
    v    = VEHICLES[vehicle]

    if vehicle == "Rickshaw" and dist > v["max_km"]:
        continue

    # Estimate travel time: Dhaka avg ~22km/h, ~12km/h in rush, ~25km/h at night
    is_rush  = 8 <= hour <= 10 or 17 <= hour <= 20
    is_night = hour >= 23 or hour <= 5
    speed    = 12 if is_rush else (25 if is_night else 22)
    est_mins = (dist / speed) * 60

    base  = v["base"] + dist * v["per_km"] + est_mins * v["per_min"]
    if vehicle == "Bus":
        # BRTA style: per-km fare with minimum, not base ON TOP of distance
        base = dist * v["per_km"]
    w_m   = WEATHER_MULT[weather]

    if is_rush:  base *= 1.20
    if is_night: base *= 1.10
    if dow == 4: base *= 0.95
    if vehicle == "Bus":
        w_m     = min(w_m, 1.05)
        demand *= 0.3

    price = base * w_m + demand * 15 + np.random.normal(0, 6)
    price = round(max(v["base"], price), 0)

    rows.append([from_loc, to_loc, vehicle, weather, hour, dow, demand, dist, price])

df = pd.DataFrame(rows, columns=["from_loc","to_loc","vehicle","weather","hour","dow","demand","distance_km","price"])
df.to_csv("rides.csv", index=False)

print(f"✅ Dataset saved: rides.csv  ({len(df)} rows, {len(LOCATIONS)} locations)")
for v in VEHICLES:
    sub = df[df.vehicle == v]
    if len(sub): print(f"   {v:10s}  avg ৳{sub.price.mean():.0f}  |  ৳{sub.price.min():.0f}–৳{sub.price.max():.0f}")

X = df[["from_loc","to_loc","vehicle","weather","hour","dow","demand","distance_km"]]
y = df["price"]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["from_loc","to_loc","vehicle","weather"])
], remainder="passthrough")

model = Pipeline([("pre", pre), ("reg", GradientBoostingRegressor(n_estimators=200, random_state=42))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"\n✅ Model trained  |  MAE: ৳{mae:.1f}")

joblib.dump(model,     "model.pkl")
joblib.dump(LOCATIONS, "locations.pkl")
joblib.dump(VEHICLES,  "vehicles.pkl")
print("✅ Saved: model.pkl  locations.pkl  vehicles.pkl")
print("\nNext → python api.py")
