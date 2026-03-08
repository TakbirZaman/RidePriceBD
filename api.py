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
    return round(math.sqrt(a) * R * 2 * 1.35, 1)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/")
def home():
    return jsonify({"service": "RidePrice BD API", "status": "running"})

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
    dist     = calc_distance(from_loc, to_loc)

    row = pd.DataFrame([{
        "from_loc":    from_loc,
        "to_loc":      to_loc,
        "vehicle":     vehicle,
        "weather":     weather,
        "hour":        hour,
        "dow":         dow,
        "demand":      demand,
        "distance_km": dist,
    }])

    price = round(float(model.predict(row)[0]), 0)
    price = max(VEHICLES[vehicle]["base"], price)

    # ── Factor breakdown ──────────────────────────────────────────────
    v         = VEHICLES[vehicle]
    base_fare = v["base"] + dist * v["per_km"]

    WEATHER_M = {"Sunny":1.0,"Regular":1.1,"Rainy":1.4,"Stormy":1.65}
    is_rush   = 8 <= hour <= 10 or 17 <= hour <= 20
    is_night  = hour >= 23 or hour <= 5
    is_friday = dow == 4

    weather_add = round(base_fare * (WEATHER_M.get(weather, 1.0) - 1.0), 0)
    time_add    = round(base_fare * (0.30 if is_rush else (0.15 if is_night else 0.0)), 0)
    demand_add  = round(demand * 25, 0)
    friday_disc = round(base_fare * 0.10, 0) if is_friday else 0

    # Confidence range: ±8% (model uncertainty)
    price_low  = round(price * 0.92, 0)
    price_high = round(price * 1.08, 0)

    # ── Surge tier ────────────────────────────────────────────────────
    thresholds = {
        "Rickshaw": [80,  120, 160],
        "Bike":     [100, 160, 220],
        "Car":      [180, 300, 420],
        "Bus":      [40,  70,  100],
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
        "factors": {
            "base_fare":    round(base_fare, 0),
            "weather_add":  weather_add,
            "time_add":     time_add,
            "demand_add":   demand_add,
            "friday_disc":  friday_disc,
            "is_rush":      is_rush,
            "is_night":     is_night,
            "is_friday":    is_friday,
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
