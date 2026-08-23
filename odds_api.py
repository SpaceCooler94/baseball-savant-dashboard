#!/usr/bin/env python3
# ============================================================================
# odds_api.py -- The Odds API (api.the-odds-api.com), MLB milestone player
# props, DraftKings + FanDuel. Documentation read directly before writing any
# of this (the-odds-api.com/liveapi/guides/v4 and .../betting-markets.html),
# not from memory -- market keys, response schema, and quota mechanics below
# are all confirmed against the current docs, not guessed.
#
# CONFIRMED FACTS THIS MODULE DEPENDS ON:
#  - Player props require ONE event at a time:
#    GET /v4/sports/baseball_mlb/events/{eventId}/odds
#        ?apiKey=...&bookmakers=draftkings,fanduel
#        &markets=batter_home_runs_alternate
#        &oddsFormat=american
#  - Milestone (X+) markets use the _alternate market keys. Exact MLB key
#    in use, from the betting-markets reference page:
#      batter_home_runs_alternate  -- Alternate batter home runs (Over/Under)
#    Returns MULTIPLE point thresholds (0.5, 1.5, 2.5, ...) as separate
#    Over/Under outcome pairs per player, all inside one market's outcomes
#    list -- not one call per threshold.
#    HR-only as of the credit crunch below -- batter_hits_alternate dropped
#    to halve per-event cost. Re-add to MARKETS (and MARKET_TO_STAT) if the
#    quota situation changes; parse_event_odds()/apply_odds_to_row() already
#    handle "hit" generically and need no other change to bring it back.
#  - Quota cost = [unique markets in the response] x [region-equivalent].
#    Specifying bookmakers=draftkings,fanduel (2 books) is charged as ONE
#    region (every group of <=10 named bookmakers = 1 region-equivalent), so
#    asking for exactly these two books costs the SAME as asking for one.
#    1 market x 1 region-equivalent = 1 credit per event now that hits is
#    dropped (was 2 credits/event with both markets) -- see refresh_odds.py
#    for the updated operational math this changes.
#  - Response schema (event-odds endpoint specifically): outcomes carry BOTH
#    "name" (Over/Under) and "description" (the player's name) -- description
#    only appears on markets that are player-scoped, confirmed in the docs'
#    own player-prop example. "point" is the threshold (0.5, 1.5, ...).
#    "last_update" sits on the MARKET, not the bookmaker, for this endpoint
#    (different from the plain /odds endpoint) -- not used here, but a real
#    difference worth knowing if this module is ever extended.
#  - Odds requested in American format directly (oddsFormat=american) rather
#    than requesting decimal and converting, because the docs explicitly warn
#    decimal-to-American conversion can introduce small rounding
#    discrepancies for some bookmakers. American-to-decimal (the direction
#    this module actually needs, for EV math) is exact, ordinary arithmetic,
#    not subject to that warning.
# ============================================================================

import re
import unicodedata

# HR-only: batter_hits_alternate dropped to cut per-event cost in half after
# running out of credits on the free tier (500/mo) -- see header note above.
MARKETS = "batter_home_runs_alternate"
BOOKMAKERS = "draftkings,pinnacle,fanduel"
MARKET_TO_STAT = {
    "batter_home_runs_alternate": "hr",
}


def _nv(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def american_to_decimal(price):
    """Exact, ordinary conversion -- not the direction the docs warn about
    rounding on. None-safe."""
    p = _nv(price)
    if p is None or p == 0:
        return None
    return 1 + (p / 100.0 if p > 0 else 100.0 / abs(p))


def norm_name(name):
    """Same normalization discipline already used everywhere else in this
    pipeline for cross-source name matching (retrospectives, lineup joins) --
    one function, not a second ad-hoc version for odds data."""
    n = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", n.lower()).strip()


def devig_two_way(price_over, price_under):
    """Multiplicative devig of a two-sided Over/Under market -> the BOOK'S
    OWN no-vig fair probabilities (fair_over, fair_under), or (None, None) if
    either price is missing. This is a DIAGNOSTIC number -- "what does the
    book itself think is fair, independent of its own margin" -- for
    comparing against the model's own no-vig probability apples-to-apples.
    It is NOT the edge calculation; see compute_edge() for that."""
    do, du = american_to_decimal(price_over), american_to_decimal(price_under)
    if do is None or du is None:
        return None, None
    io, iu = 1.0 / do, 1.0 / du
    total = io + iu
    if total <= 0:
        return None, None
    return round(io / total, 4), round(iu / total, 4)


def compute_edge(model_prob, book_price_american):
    """The actionable number: expected value per $1 staked at the book's
    ACTUAL (vigged) price, using the model's probability as the true rate.

    edge = model_prob * decimal_odds - 1

    This is the standard, correct EV formula -- NOT a probability-vs-
    probability comparison. Comparing my no-vig probability directly against
    a devigged book probability would answer a different, softer question
    ("do we agree on the true rate") and silently present it as if it were
    the same thing as "is this bet profitable at the price actually offered."
    Those are two different numbers (see devig_two_way for the first one);
    this function computes only the second, and only the second is called
    'edge' anywhere in this pipeline, to avoid exactly that confusion.

    Returns None if either input is unusable -- never a fabricated 0."""
    p = _nv(model_prob)
    dec = american_to_decimal(book_price_american)
    if p is None or dec is None:
        return None
    return round(p * dec - 1, 4)


def parse_event_odds(payload, markets=MARKET_TO_STAT):
    """One /events/{id}/odds response -> nested dict:
        {stat: {player_norm_name: {point: {book_key: {"over": price, "under": price}}}}}
    stat is 'hr' (via MARKET_TO_STAT), not the raw market key -- so
    callers never need to know The Odds API's key names past this function.

    Defensive throughout: a malformed or partial bookmaker/market/outcome is
    skipped, never raised past this function -- one bad entry in a 15-event
    pull must not take down the rest of the slate's odds."""
    out = {}
    if not isinstance(payload, dict):
        return out
    for bm in payload.get("bookmakers") or []:
        if not isinstance(bm, dict):
            continue
        book_key = bm.get("key")
        if not book_key:
            continue
        for mkt in bm.get("markets") or []:
            if not isinstance(mkt, dict):
                continue
            stat = markets.get(mkt.get("key"))
            if not stat:
                continue
            for oc in mkt.get("outcomes") or []:
                if not isinstance(oc, dict):
                    continue
                name = str(oc.get("name") or "").strip().lower()  # 'over'/'under'
                if name not in ("over", "under"):
                    continue
                player = norm_name(oc.get("description"))
                point = _nv(oc.get("point"))
                price = _nv(oc.get("price"))
                if not player or point is None or price is None:
                    continue
                stat_d = out.setdefault(stat, {})
                player_d = stat_d.setdefault(player, {})
                point_d = player_d.setdefault(point, {})
                book_d = point_d.setdefault(book_key, {})
                book_d[name] = price
    return out


def milestone_threshold_to_k(point):
    """DK/FD post a milestone line as 'Over 1.5' meaning >=2 -- the point is
    the HALF-INTEGER just below the count threshold, always. k = point + 0.5,
    rounded, defensively (0.5 -> 1, 1.5 -> 2, 2.5 -> 3, ...)."""
    p = _nv(point)
    if p is None:
        return None
    return int(round(p + 0.5))
