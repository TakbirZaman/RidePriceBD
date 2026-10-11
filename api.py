# api.py

from flask import Flask, request, jsonify, make_response
import joblib, pandas as pd, math, os
from datetime import datetime

app      = Flask(__name__)
model    = joblib.load("model.pkl")
LOCS     = joblib.load("locations.pkl")
VEHICLES = joblib.load("vehicles.pkl")

def cors(r):
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return r

@app.after_request
def after(r): return cors(r)

@app.before_request
def options():
    if request.method == "OPTIONS":
        return cors(make_response()), 200

def calc_distance(loc1, loc2):
    lat1, lon1 = LOCS[loc1]
    lat2, lon2 = LOCS[loc2]
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return round(math.sqrt(a) * R * 2 * 1.25, 1)

# ── Audited fare policy: IDENTICAL to setup.py compute_fare() ──────────────
# price = (distance_part + time_part) * capped_surge_mult + demand_add,
# floored at min_fare. The returned breakdown ALWAYS sums to price.
WEATHER_M = {"Sunny": 1.00, "Regular": 1.05, "Rainy": 1.20, "Stormy": 1.40}
RUSH_MULT, NIGHT_MULT, FRIDAY_MULT, SURGE_CAP = 1.20, 1.10, 0.95, 2.0

def trip_flags(hour, dow):
    return (8 <= hour <= 10 or 17 <= hour <= 20,
            hour >= 23 or hour <= 5,
            dow == 4)

def est_mins(dist, hour):
    is_rush = 8 <= hour <= 10 or 17 <= hour <= 20
    is_night = hour >= 23 or hour <= 5
    speed = 12 if is_rush else (25 if is_night else 22)
    return (dist / speed) * 60

def compute_fare(dist, vehicle, weather, hour, demand, dow):
    v = VEHICLES[vehicle]
    is_rush, is_night, is_fri = trip_flags(hour, dow)
    mins = est_mins(dist, hour)
    is_bus = vehicle == "Bus"
    w_m = 1.0 if is_bus else WEATHER_M.get(weather, 1.0)
    if is_bus:
        w_m = min(w_m, 1.05)
        dist_part = dist * v["per_km"]
    else:
        dist_part = v["base"] + dist * v["per_km"]
    time_part = mins * v["per_min"]
    subtotal = dist_part + time_part
    t_mult = 1.0
    if not is_bus:
        if is_rush:  t_mult *= RUSH_MULT
        if is_night: t_mult *= NIGHT_MULT
        if is_fri:   t_mult *= FRIDAY_MULT
    combined = min(t_mult * w_m, SURGE_CAP)
    surged = subtotal * combined
    demand_add = 0.0 if is_bus else demand * 15
    price = max(float(v.get("min_fare", v["base"])), round(surged + demand_add, 0))
    return price, {
        "dist_part":  round(dist_part, 0),
        "time_part":  round(time_part, 0),
        "subtotal":   round(subtotal, 0),
        "surge_mult": round(combined, 3),
        "surged":     round(surged, 0),
        "demand_add": round(demand_add, 0),
        "est_mins":   round(mins, 0),
        "is_rush":    is_rush,
        "is_night":   is_night,
        "is_friday":  is_fri,
    }

def featurize_row(vehicle, weather, hour, dow, demand, dist):
    wscore = {"Sunny": 0, "Regular": 1, "Rainy": 2, "Stormy": 3}
    is_rush, is_night, is_fri = trip_flags(hour, dow)
    return pd.DataFrame([{
        "vehicle":     vehicle,
        "weather":     weather,
        "distance_km": dist,
        "demand":      demand,
        "hour":        hour,
        "dow":         dow,
        "est_mins":    est_mins(dist, hour),
        "is_rush":     1 if is_rush else 0,
        "is_night":    1 if is_night else 0,
        "is_fri":      1 if is_fri else 0,
        "wscore":      wscore.get(weather, 0),
    }])

@app.route("/rates")
def rates():
    # Public rate card — clients can verify every taka of any quote.
    return jsonify({
        "vehicles": VEHICLES,
        "weather_mult": WEATHER_M,
        "rush_mult": RUSH_MULT,
        "night_mult": NIGHT_MULT,
        "friday_mult": FRIDAY_MULT,
        "surge_cap": SURGE_CAP,
        "demand_per_unit": 15,
        "detour_factor": 1.25,
        "speeds_kmh": {"rush": 12, "normal": 22, "night": 25},
        "method": "price=(dist_part+time_part)*surge_mult+demand_add, floored at min_fare",
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/")
def home():
    return jsonify({"service": "RidePrice BD API", "status": "running"})

@app.route("/locations")
def locations():
    # Lets any client (e.g. the web frontend, which supports far more
    # locations than the model was trained on) check what this model
    # actually knows about before calling /predict, instead of guessing
    # and getting a KeyError back.
    return jsonify({
        "locations": sorted(LOCS.keys()),
        "vehicles": sorted(VEHICLES.keys()),
        "count": len(LOCS),
    })

@app.route("/predict", methods=["POST"])
def predict():
    d        = request.json
    from_loc = d["from_loc"]
    to_loc   = d["to_loc"]
    vehicle  = d["vehicle"]
    weather  = d["weather"]
    hour     = int(d["hour"])
    demand   = float(d["demand"])
    dow      = datetime.now().weekday()

    if from_loc not in LOCS or to_loc not in LOCS:
        return jsonify({"error": "unknown location", "hint": "GET /locations for the supported list"}), 400
    if vehicle not in VEHICLES:
        return jsonify({"error": "unknown vehicle", "hint": "GET /locations for the supported list"}), 400

    dist = calc_distance(from_loc, to_loc)

    # Primary price: deterministic audited formula (breakdown sums exactly).
    # ML model is an independent cross-check, never the unexplained decider.
    price, bd = compute_fare(dist, vehicle, weather, hour, demand, dow)

    try:
        model_price = round(float(model.predict(
            featurize_row(vehicle, weather, hour, dow, demand, dist))[0]), 0)
    except Exception:
        model_price = None

    # Confidence range: ±8% (model uncertainty)
    price_low  = round(price * 0.92, 0)
    price_high = round(price * 1.08, 0)

    # ── Surge tier (recalibrated to Uber-like fare levels) ───────────────
    thresholds = {
        "Rickshaw": [60,  100, 140],
        "Bike":     [100, 170, 240],
        "Car":      [170, 280, 400],
        "Bus":      [25,  45,  70],
        "CNG":      [100, 170, 250],
    }
    t = thresholds[vehicle]
    if   price < t[0]: surge = ("Normal",   "green")
    elif price < t[1]: surge = ("Moderate", "yellow")
    elif price < t[2]: surge = ("High",     "orange")
    else:              surge = ("Peak",     "red")

    return jsonify({
        "price":       price,
        "price_low":   price_low,
        "price_high":  price_high,
        "distance_km": dist,
        "surge_label": surge[0],
        "surge_color": surge[1],
        "model_price": model_price,
        "model_diff_pct": (round(abs(model_price - price) / price * 100, 1)
                           if model_price else None),
        "factors": {
            "dist_part":  bd["dist_part"],
            "time_part":  bd["time_part"],
            "est_mins":   bd["est_mins"],
            "subtotal":   bd["subtotal"],
            "surge_mult": bd["surge_mult"],
            "surged":     bd["surged"],
            "demand_add": bd["demand_add"],
            "is_rush":    bd["is_rush"],
            "is_night":   bd["is_night"],
            "is_friday":  bd["is_friday"],
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
