#!/usr/bin/env python3
# ============================================================================
# odds_api.py -- SportsGameOdds (api.sportsgameodds.com/v2), MLB HR player
# props. Replaces the earlier The Odds API integration entirely, after a
# real, live-verified structural difference made continuing untenable at
# the free tier: The Odds API charges per market+bookmaker (the whole
# reason Pinnacle/Novig additions had to be weighed against a 10-bookmaker
# region cap), while SportsGameOdds charges per EVENT regardless of how
# many bookmakers or markets come back in it -- confirmed directly from
# their pricing page, not a guess.
#
# CONFIRMED FACTS THIS MODULE DEPENDS ON (verified against a real, live,
# authenticated /v2/events pull on 2026-09-05, not docs alone -- docs and
# a live response disagreed on the one detail that mattered most, see
# below, so nothing here is trusted without having seen it in real data):
#
#  - Auth: X-API-Key header OR apiKey query param, either works.
#  - Endpoint: GET https://api.sportsgameodds.com/v2/events
#      ?leagueID=MLB&oddsAvailable=true&oddIDs=<comma list>&limit=N
#    Unlike The Odds API's one-call-per-event pattern, ONE call with a
#    high enough limit returns every game on the slate, each with its own
#    full odds set already embedded -- confirmed on a real 15-game-shaped
#    pull. This is a real, structural cost reduction, not just a nicer API.
#  - oddID format: {statID}-{statEntityID}-{periodID}-{betTypeID}-{sideID}.
#    HR market statID is "batting_homeRuns" (confirmed present in a real
#    response; the docs' own worked examples only ever showed
#    batting_hits/batting_singles, so this was verified against live data,
#    not assumed from the pattern). Use statEntityID="PLAYER_ID" as a
#    literal wildcard meaning "any player" -- confirmed to work, returns
#    every player's line, not a request for a player literally named that.
#  - Real oddID betType/side pairs seen for HR: "-ou-over" / "-ou-under"
#    (the one this module uses) and "-yn-yes" / "-yn-no" (a Yes/No framing
#    of the exact same market -- mathematically redundant with the 0.5
#    Over/Under line, not separately fetched).
#  - THE ONE THING THAT ACTUALLY CHANGES THE PARSING MODEL: there is no
#    fixed threshold. The docs' passing-yards example suggested one
#    canonical line per market; live HR data shows every bookmaker
#    reporting its OWN "overUnder" value independently under the exact
#    same oddID -- confirmed on a real pull where, for one active power
#    hitter, DraftKings and Pinnacle were quoting Over 0.5 while FanDuel
#    and BetMGM were quoting Over 1.5 and ESPN Bet / Hard Rock were
#    quoting Over 2.5, all at the same moment, same oddID. A book can and
#    does feature a different HR line depending on the hitter. This means
#    "DK's price at the 0.5 threshold" isn't a safe query -- the correct
#    read is "whatever threshold DK is actually quoting today," which is
#    what parse_event_odds() below does: it buckets each bookmaker under
#    ITS OWN reported point value, not a single assumed one.
#  - Player identity: SportsGameOdds' own playerID scheme
#    (FIRSTNAME_LASTNAME_N_MLB) has no relationship to the MLBAM numeric
#    hitterId this pipeline uses everywhere else, and no crosswalk exists.
#    Every event's own "players" object carries a real display "name"
#    field per playerID (e.g. "Zac Thornton" -- confirmedly NOT always a
#    literal firstName+lastName concatenation, e.g. "Zachary" vs "Zac"),
#    which is what this module joins on via the SAME norm_name()
#    discipline already used for cross-source name matching elsewhere in
#    this pipeline -- one normalization function, not a second ad-hoc one.
#  - Prices are American odds as strings ("+1900", "-137") -- cast to
#    float, same as the previous integration; nothing about the actual
#    number format changed.
#
# Pure math below (american_to_decimal, devig_two_way, compute_edge,
# milestone_threshold_to_k) is UNCHANGED from the prior integration --
# none of it depends on which API the prices came from, only on having a
# real American-odds price, which SportsGameOdds also provides in the
# same format. Only the fetch/parse layer is new.
# ============================================================================

import re
import unicodedata

BASE_URL = "https://api.sportsgameodds.com/v2/events"
STAT_ID = "batting_homeRuns"
# Wildcard statEntityID -- confirmed to mean "any player" against a real
# pull, not a literal player search.
ODD_IDS = f"{STAT_ID}-PLAYER_ID-game-ou-over,{STAT_ID}-PLAYER_ID-game-ou-under"


def _nv(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def american_to_decimal(price):
    p = _nv(price)
    if p is None or p == 0:
        return None
    return 1 + (p / 100.0 if p > 0 else 100.0 / abs(p))


def norm_name(name):
    n = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", n.lower()).strip()


def devig_two_way(price_over, price_under):
    """Multiplicative devig of a two-sided Over/Under market -> the BOOK'S
    OWN no-vig fair probabilities (fair_over, fair_under). Diagnostic only
    -- see compute_edge() for the actual EV number. Unchanged from the
    prior integration; this math doesn't care which API the prices came
    from."""
    do, du = american_to_decimal(price_over), american_to_decimal(price_under)
    if do is None or du is None:
        return None, None
    io, iu = 1.0 / do, 1.0 / du
    total = io + iu
    if total <= 0:
        return None, None
    return round(io / total, 4), round(iu / total, 4)


def compute_edge(model_prob, book_price_american):
    """edge = model_prob * decimal_odds - 1 -- expected value per $1 staked
    at the book's actual (vigged) price. Unchanged from the prior
    integration. Returns None on any unusable input, never a fabricated 0."""
    p = _nv(model_prob)
    dec = american_to_decimal(book_price_american)
    if p is None or dec is None:
        return None
    return round(p * dec - 1, 4)


def milestone_threshold_to_k(point):
    """Same convention as before: DK/FD/etc. post a milestone line as
    'Over 1.5' meaning >=2 -- point is always the half-integer just below
    the count threshold. k = point + 0.5, rounded defensively."""
    p = _nv(point)
    if p is None:
        return None
    return int(round(p + 0.5))


# Sharp/soft classification -- researched, not guessed. A live pull shows
# this feed can return 60+ distinct book keys; most are regional UK/EU/AU
# books that won't realistically carry US MLB player props at all, so this
# only classifies the ones actually relevant here, each with real sourcing
# rather than assumed from "offshore = recreational" intuition (which
# turned out wrong for several of these -- BetOnline, Bovada, and MyBookie
# are explicitly classified sharp by multiple current sources, not soft).
#
# SHARP: welcomes winners, sets/moves its own lines rather than copying,
# low margin. Pinnacle is the universal reference; Circa is the sharpest
# regulated US operator specifically; BetOnline/Bovada/MyBookie set their
# own numbers and don't limit winners the way retail books do.
SHARP_BOOKS = {"pinnacle", "circa", "betonline", "bovada", "mybookie"}

# EXCHANGE: no bookmaker at all -- bettors trade against each other, price
# is pure market consensus with no house margin to set. Functionally the
# sharpest category that exists, structurally distinct from a book that
# chooses to price sharp. Novig and Prophet Exchange are the two
# confirmed-available real-money sports exchanges here; Betfair Exchange
# and Matchbook also showed up in the raw book-key list from a real pull
# and are exchanges by the same definition.
EXCHANGE_BOOKS = {"novig", "prophetexchange", "betfairexchange", "matchbook"}

# PREDICTION_MARKET: not sports betting at all, but same "pure market
# price, no bookmaker" structure as an exchange -- Polymarket and Kalshi
# both showed up in a real pull's book-key list.
PREDICTION_MARKETS = {"polymarket", "kalshi"}

# SOFT: major regulated US retail sportsbooks. Cater to recreational
# volume, generally copy sharp lines with a delay and added margin, and
# are the ones known to limit consistently-winning accounts.
SOFT_BOOKS = {"draftkings", "fanduel", "betmgm", "caesars", "espnbet",
              "fanatics", "hardrockbet", "betrivers", "betparx", "ballybet",
              "fliff", "bet365"}

# PICKEM: fundamentally not a sportsbook -- fixed-multiplier pick'em/DFS
# platforms (PrizePicks, Underdog). Not comparable to sharp/soft at all;
# kept as its own bucket so it's never accidentally averaged into either.
PICKEM_BOOKS = {"underdog", "prizepicks"}


def book_category(book_key):
    """Returns 'sharp', 'exchange', 'prediction_market', 'soft', 'pickem',
    or None if the book isn't in any researched bucket -- 'unknown' (a
    real, literal source name this feed returns) and the ~35 regional
    international books seen in a real pull but not classified here both
    correctly return None rather than a guessed category."""
    if book_key in SHARP_BOOKS:
        return "sharp"
    if book_key in EXCHANGE_BOOKS:
        return "exchange"
    if book_key in PREDICTION_MARKETS:
        return "prediction_market"
    if book_key in SOFT_BOOKS:
        return "soft"
    if book_key in PICKEM_BOOKS:
        return "pickem"
    return None


def sharp_vs_soft_fair(books_at_threshold):
    """Given one threshold's {book_key: {"over": price, "under": price}}
    dict, returns (sharp_fair_over, soft_fair_over, sharp_n, soft_n) --
    the devigged consensus (mean of each book's own devig) on each side,
    using only books where BOTH over and under prices are present (devig
    needs both). Exchange and prediction-market books count toward the
    "sharp" side, matching their real structural sharpness even though
    they aren't technically bookmakers -- pickem and unclassified books
    count toward neither, not silently dropped, just excluded from both
    consensus numbers since they're not comparable this way.

    Returns None for either side that has zero qualifying books rather
    than fabricating a consensus from nothing."""
    sharp_fairs, soft_fairs = [], []
    for bk, prices in books_at_threshold.items():
        cat = book_category(bk)
        if cat not in ("sharp", "exchange", "prediction_market", "soft"):
            continue
        over, under = prices.get("over"), prices.get("under")
        if over is None or under is None:
            continue
        fair_over, _ = devig_two_way(over, under)
        if fair_over is None:
            continue
        if cat == "soft":
            soft_fairs.append(fair_over)
        else:
            sharp_fairs.append(fair_over)
    sharp_avg = round(sum(sharp_fairs) / len(sharp_fairs), 4) if sharp_fairs else None
    soft_avg = round(sum(soft_fairs) / len(soft_fairs), 4) if soft_fairs else None
    return sharp_avg, soft_avg, len(sharp_fairs), len(soft_fairs)


def parse_event_odds(event):
    """One SportsGameOdds /v2/events entry (a single game, with its own
    embedded 'odds' and 'players' objects) -> nested dict:
        {player_norm_name: {point: {book_key: {"over": price, "under": price}}}}
    Same output shape the rest of this pipeline (refresh_odds.py) already
    expects from the old integration, so nothing downstream of this
    function needs to change -- only how it gets built.

    Buckets each bookmaker under ITS OWN reported threshold (see module
    docstring) rather than assuming a shared one -- this is the one real
    behavioral difference from a naive port of the old parser, and the
    one confirmed necessary by live data, not decided in the abstract.

    Defensive throughout: a malformed or partial odds/player entry is
    skipped, never raised past this function."""
    out = {}
    if not isinstance(event, dict):
        return out
    players = event.get("players") or {}
    odds = event.get("odds") or {}
    for odd_id, entry in odds.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("statID") != STAT_ID:
            continue
        if entry.get("betTypeID") != "ou":
            continue
        side = entry.get("sideID")
        if side not in ("over", "under"):
            continue
        player_id = entry.get("playerID") or entry.get("statEntityID")
        player_info = players.get(player_id) or {}
        player_name = player_info.get("name")
        player = norm_name(player_name)
        if not player:
            continue
        by_book = entry.get("byBookmaker") or {}
        for book_key, book_data in by_book.items():
            if not isinstance(book_data, dict):
                continue
            if not book_data.get("available"):
                continue
            point = _nv(book_data.get("overUnder"))
            price = _nv(book_data.get("odds"))
            if point is None or price is None:
                continue
            stat_d = out.setdefault(player, {})
            point_d = stat_d.setdefault(point, {})
            book_d = point_d.setdefault(book_key, {})
            book_d[side] = price
    return out
