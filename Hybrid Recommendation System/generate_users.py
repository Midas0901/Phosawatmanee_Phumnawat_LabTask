"""
Generate a synthetic user list (10,000 users) from dramas.csv.

Design principle: every distribution is derived from the real data where
possible, rather than invented.
  - gender          <- aggregated reviewer_gender_info Counters
  - location        <- aggregated reviewer_location_info Counters (normalized)
  - drama popularity<- real no_of_viewers column
  - genre / country <- real catalog distributions, weighted by viewership

Watch histories are sampled with a taste bias (~75% of titles match the
user's preferred genre and/or country) so the dataset contains a learnable
signal rather than uniform noise.
"""

import ast
import json
import re
from collections import Counter

import numpy as np
import pandas as pd

DRAMA_CSV = "data/dramas.csv"
OUT_CSV = "results/synthetic_users.csv"
OUT_JSONL = "results/synthetic_users.jsonl"
N_USERS = 10_000
MAX_HISTORY = 20
TASTE_RATIO = 0.75          # share of history matching user's stated taste
SEED = 461

RNG = np.random.default_rng(SEED)

# --- location normalization -------------------------------------------------
# The reviewer_location_info field is free-text and full of jokes
# ("planet pluto", "dramaland"). Map recognizable real places to countries;
# everything unmatched is dropped.
US_HINTS = {
    "usa", "us", "u.s.", "u.s.a.", "united states", "united states of america",
    "america", "california", "florida", "texas", "new york", "nyc", "colorado",
    "arkansas", "arizona", "washington", "oregon", "michigan", "virginia",
    "georgia", "illinois", "ohio", "seattle", "chicago", "los angeles",
    "san francisco", "san diego", "boston", "atlanta", "denver", "houston",
    "dallas", "phoenix", "portland", "las vegas", "miami", "orlando",
    "new jersey", "pennsylvania", "maryland", "minnesota", "wisconsin",
    "north carolina", "south carolina", "tennessee", "missouri", "indiana",
    "louisiana", "alabama", "kentucky", "oklahoma", "utah", "nevada",
    "new mexico", "kansas", "iowa", "hawaii", "alaska", "midwest", "east coast",
    "west coast",
}
UK_HINTS = {
    "uk", "u.k.", "united kingdom", "england", "scotland", "wales",
    "northern ireland", "london", "manchester", "birmingham", "sheffield",
    "essex", "kent", "britain", "great britain",
}
COUNTRY_ALIASES = {
    "brasil": "Brazil", "brazil": "Brazil",
    "india": "India", "bharat": "India", "mumbai": "India", "chennai": "India",
    "delhi": "India", "bangalore": "India", "kerala": "India",
    "philippines": "Philippines", "ph": "Philippines", "manila": "Philippines",
    "pilipinas": "Philippines",
    "indonesia": "Indonesia", "jakarta": "Indonesia",
    "malaysia": "Malaysia", "kuala lumpur": "Malaysia", "sarawak": "Malaysia",
    "singapore": "Singapore",
    "vietnam": "Vietnam", "viet nam": "Vietnam",
    "thailand": "Thailand", "bangkok": "Thailand",
    "south korea": "South Korea", "korea": "South Korea", "seoul": "South Korea",
    "incheon": "South Korea",
    "japan": "Japan", "tokyo": "Japan", "osaka": "Japan",
    "china": "China", "mainland china": "China", "beijing": "China",
    "shanghai": "China",
    "taiwan": "Taiwan", "taipei": "Taiwan",
    "hong kong": "Hong Kong", "hongkong": "Hong Kong",
    "canada": "Canada", "toronto": "Canada", "montreal": "Canada",
    "vancouver": "Canada", "ontario": "Canada", "quebec": "Canada",
    "australia": "Australia", "sydney": "Australia", "melbourne": "Australia",
    "canberra": "Australia", "brisbane": "Australia", "perth": "Australia",
    "new zealand": "New Zealand",
    "germany": "Germany", "deutschland": "Germany", "berlin": "Germany",
    "munich": "Germany", "hamburg": "Germany",
    "france": "France", "paris": "France", "lyon": "France",
    "italy": "Italy", "italia": "Italy", "rome": "Italy", "milan": "Italy",
    "spain": "Spain", "espana": "Spain", "madrid": "Spain",
    "barcelona": "Spain",
    "portugal": "Portugal", "lisbon": "Portugal",
    "netherlands": "Netherlands", "holland": "Netherlands",
    "amsterdam": "Netherlands",
    "belgium": "Belgium", "sweden": "Sweden", "norway": "Norway",
    "denmark": "Denmark", "finland": "Finland", "iceland": "Iceland",
    "poland": "Poland", "warsaw": "Poland",
    "czech republic": "Czech Republic", "czechia": "Czech Republic",
    "prague": "Czech Republic",
    "austria": "Austria", "vienna": "Austria",
    "switzerland": "Switzerland", "greece": "Greece", "athens": "Greece",
    "romania": "Romania", "romania ": "Romania", "bucharest": "Romania",
    "hungary": "Hungary", "budapest": "Hungary",
    "serbia": "Serbia", "belgrade": "Serbia", "croatia": "Croatia",
    "bulgaria": "Bulgaria", "slovakia": "Slovakia", "slovenia": "Slovenia",
    "ukraine": "Ukraine", "russia": "Russia", "moscow": "Russia",
    "turkey": "Turkey", "istanbul": "Turkey",
    "mexico": "Mexico", "argentina": "Argentina", "chile": "Chile",
    "colombia": "Colombia", "peru": "Peru", "venezuela": "Venezuela",
    "ecuador": "Ecuador", "uruguay": "Uruguay", "guatemala": "Guatemala",
    "puerto rico": "Puerto Rico", "dominican republic": "Dominican Republic",
    "jamaica": "Jamaica", "trinidad": "Trinidad and Tobago",
    "egypt": "Egypt", "cairo": "Egypt", "morocco": "Morocco",
    "algeria": "Algeria", "tunisia": "Tunisia",
    "south africa": "South Africa", "nigeria": "Nigeria", "kenya": "Kenya",
    "ghana": "Ghana", "ethiopia": "Ethiopia",
    "pakistan": "Pakistan", "karachi": "Pakistan", "lahore": "Pakistan",
    "bangladesh": "Bangladesh", "dhaka": "Bangladesh",
    "sri lanka": "Sri Lanka", "nepal": "Nepal", "bhutan": "Bhutan",
    "myanmar": "Myanmar", "cambodia": "Cambodia", "laos": "Laos",
    "brunei": "Brunei", "mongolia": "Mongolia",
    "saudi arabia": "Saudi Arabia", "uae": "United Arab Emirates",
    "dubai": "United Arab Emirates", "qatar": "Qatar", "kuwait": "Kuwait",
    "bahrain": "Bahrain", "oman": "Oman", "jordan": "Jordan",
    "lebanon": "Lebanon", "israel": "Israel", "iran": "Iran", "iraq": "Iraq",
    "ireland": "Ireland", "dublin": "Ireland",
}


def normalize_location(raw: str):
    """Map a free-text location string to a country, or None if unusable."""
    s = str(raw).strip().lower()
    if not s or len(s) > 60:
        return None
    s = re.sub(r"[^\w\s,./-]", "", s)          # strip emoji/symbols
    parts = [p.strip() for p in re.split(r"[,/|]| - ", s) if p.strip()]
    candidates = parts + [s]
    for part in candidates:
        if part in COUNTRY_ALIASES:
            return COUNTRY_ALIASES[part]
        if part in US_HINTS:
            return "United States"
        if part in UK_HINTS:
            return "United Kingdom"
    for part in candidates:                     # substring fallback
        for hint in US_HINTS:
            if len(hint) > 5 and hint in part:
                return "United States"
        for hint in UK_HINTS:
            if len(hint) > 5 and hint in part:
                return "United Kingdom"
        for alias, country in COUNTRY_ALIASES.items():
            if len(alias) > 5 and alias in part:
                return country
    return None


def parse_counter(cell):
    try:
        return eval(str(cell), {"Counter": Counter, "__builtins__": {}})
    except Exception:
        return Counter()


def weighted_pool(counter: Counter):
    """Counter -> (values array, normalized probability array)."""
    items = [(k, v) for k, v in counter.items() if v > 0]
    keys = np.array([k for k, _ in items], dtype=object)
    w = np.array([v for _, v in items], dtype=float)
    return keys, w / w.sum()


def main():
    df = pd.read_csv(DRAMA_CSV)
    df = df.dropna(subset=["genres", "country", "no_of_viewers"]).reset_index(drop=True)
    n_items = len(df)
    print(f"Catalog: {n_items} dramas ({df['name'].nunique()} unique titles)")

    # ---- empirical gender distribution ----
    gender_counts = Counter()
    for cell in df["reviewer_gender_info"]:
        gender_counts.update(parse_counter(cell))
    gender_counts = Counter({k.capitalize(): v for k, v in gender_counts.items()
                             if k.strip() in ("female", "male")})
    # Add a small non-binary/undisclosed share; the source data only has M/F.
    total = sum(gender_counts.values())
    gender_counts["Non-binary"] = int(total * 0.01)
    gender_counts["Prefer not to say"] = int(total * 0.02)
    gender_keys, gender_probs = weighted_pool(gender_counts)
    print("Gender distribution:",
          {k: round(p, 4) for k, p in zip(gender_keys, gender_probs)})

    # ---- empirical location distribution ----
    loc_raw = Counter()
    for cell in df["reviewer_location_info"]:
        loc_raw.update(parse_counter(cell))
    loc_counts = Counter()
    for raw, n in loc_raw.items():
        norm = normalize_location(raw)
        if norm:
            loc_counts[norm] += n
    matched = sum(loc_counts.values())
    unmatched = sum(loc_raw.values()) - matched
    print(f"Locations: {matched:,} mapped / {unmatched:,} dropped "
          f"({len(loc_counts)} countries)")
    print("Top 10 locations:", loc_counts.most_common(10))
    loc_keys, loc_probs = weighted_pool(loc_counts)

    # ---- catalog structures ----
    df["genre_list"] = df["genres"].astype(str).str.split(",").apply(
        lambda gs: [g.strip() for g in gs if g.strip()]
    )
    viewers = df["no_of_viewers"].to_numpy(dtype=float)
    pop_w = viewers / viewers.sum()          # real popularity skew

    # genre popularity weighted by viewership
    genre_counts = Counter()
    for gl, v in zip(df["genre_list"], viewers):
        for g in gl:
            genre_counts[g] += v
    genre_keys, genre_probs = weighted_pool(genre_counts)

    country_counts = Counter()
    for c, v in zip(df["country"], viewers):
        country_counts[c] += v
    country_keys, country_probs = weighted_pool(country_counts)
    print("Countries:", {k: round(p, 3) for k, p in zip(country_keys, country_probs)})

    # index lookups for fast subsetting
    genre_idx = {g: np.array([i for i, gl in enumerate(df["genre_list"]) if g in gl])
                 for g in genre_keys}
    country_idx = {c: np.flatnonzero((df["country"] == c).to_numpy())
                   for c in country_keys}
    titles = df["name"].to_numpy(dtype=object)
    all_idx = np.arange(n_items)

    # ---- generate users ----
    users = []
    for i in range(N_USERS):
        pref_genre = RNG.choice(genre_keys, p=genre_probs)
        pref_country = RNG.choice(country_keys, p=country_probs)

        # history length: skewed toward mid-range, capped at 20
        hist_len = int(np.clip(RNG.geometric(0.12), 1, MAX_HISTORY))

        taste_idx = np.union1d(genre_idx[pref_genre], country_idx[pref_country])
        n_taste = min(int(round(hist_len * TASTE_RATIO)), len(taste_idx))

        w = pop_w[taste_idx]
        picks = list(RNG.choice(taste_idx, size=n_taste, replace=False,
                                p=w / w.sum()))

        remaining = hist_len - len(picks)
        if remaining > 0:
            leftover = np.setdiff1d(all_idx, picks, assume_unique=False)
            w2 = pop_w[leftover]
            picks += list(RNG.choice(leftover, size=min(remaining, len(leftover)),
                                     replace=False, p=w2 / w2.sum()))

        picks = np.array(picks)
        RNG.shuffle(picks)                    # recency order, not taste-then-noise

        users.append({
            "user_id": f"U{i + 1:06d}",
            "gender": RNG.choice(gender_keys, p=gender_probs),
            "location": RNG.choice(loc_keys, p=loc_probs),
            "user_watch_history": [str(t) for t in titles[picks]],
            "preferred_genre": str(pref_genre),
            "preferred_country": str(pref_country),
        })

        if (i + 1) % 2500 == 0:
            print(f"  ...{i + 1:,} users")

    # ---- write outputs ----
    out = pd.DataFrame(users)
    csv_out = out.copy()
    csv_out["user_watch_history"] = csv_out["user_watch_history"].apply(json.dumps)
    csv_out.to_csv(OUT_CSV, index=False)

    with open(OUT_JSONL, "w") as f:
        for rec in users:
            f.write(json.dumps(rec) + "\n")

    # ---- validation ----
    lens = out["user_watch_history"].apply(len)
    print("\n--- VALIDATION ---")
    print(f"Users: {len(out):,}  |  unique ids: {out['user_id'].nunique():,}")
    print(f"History length: min={lens.min()} max={lens.max()} "
          f"mean={lens.mean():.1f} median={lens.median():.0f}")
    dupes_within = sum(len(h) != len(set(h)) for h in out["user_watch_history"])
    print(f"Users with duplicate titles in history: {dupes_within}")
    watched = Counter(t for h in out["user_watch_history"] for t in h)
    print(f"Distinct dramas watched: {len(watched):,} / {n_items:,} "
          f"({len(watched)/n_items:.1%} coverage)")
    print(f"Total interactions: {sum(watched.values()):,}")
    print("Top 5 watched:", watched.most_common(5))

    # signal check: does history actually reflect stated preference?
    hits = []
    title_to_row = {t: i for i, t in enumerate(titles)}
    for _, r in out.head(2000).iterrows():
        rows = [title_to_row[t] for t in r["user_watch_history"]]
        match = sum(
            r["preferred_genre"] in df["genre_list"].iloc[j]
            or df["country"].iloc[j] == r["preferred_country"]
            for j in rows
        )
        hits.append(match / len(rows))
    print(f"Mean share of history matching stated taste: {np.mean(hits):.1%} "
          f"(target ~{TASTE_RATIO:.0%}+)")
    print(f"\nWrote {OUT_CSV}\nWrote {OUT_JSONL}")


if __name__ == "__main__":
    main()