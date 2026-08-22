"""dry_run_replacement.py — harnais DRY-RUN pour valider grid_replacement.py
sur dellt320, avec la configuration RÉELLE de la grille active, un prix
courant SIMULÉ, et AUCUN appel à un endpoint d'écriture OKX.

--------------------------------------------------------------------------
GARANTIE DE SÉCURITÉ
--------------------------------------------------------------------------
DryRunAdapter (ci-dessous) enveloppe l'adaptateur OKX réel :

  - Lectures (get_instrument, get_fee_rates, get_account_config,
    get_balances, get_price_limit, get_klines, list_fills) : déléguées
    telles quelles à l'adaptateur réel -- endpoints GET uniquement, aucune
    conséquence sur le compte.
  - get_ticker : REMPLACÉ par un flux de prix simulés (SimulatedPriceFeed),
    fourni par --prices. Le prix réel du marché n'est jamais lu.
  - place_post_only_limit, place_market_buy, cancel_order : INTERCEPTÉS.
    Aucun de ces trois appels n'atteint jamais l'adaptateur réel. Chacun
    est simulé en mémoire (carnet d'ordres fictif) et journalisé dans
    dry_adapter.write_calls pour preuve, à la fin de l'exécution, qu'aucune
    écriture réelle n'a eu lieu.
  - place_market_buy met à jour un ledger de bookkeeping PUREMENT LOCAL
    (base += quantité, quote -= quantité * prix -- même formule que le
    FakeAdapter déjà validé par tests/test_grid_replacement.py), lu ensuite
    par get_balances() en plus de la valeur réelle. Aucune décision
    économique n'est prise ici (required_spot/spot_covered/q restent
    calculés par le moteur réel, inchangé) -- seule la conséquence
    comptable mécanique d'un fill déjà décidé par le moteur est rendue
    visible, pour permettre au scénario de remplacement d'aller jusqu'à
    ACTIVATED quand un achat patrimonial initial est nécessaire.
  - list_open_orders / get_order : combinent un INSTANTANÉ RÉEL, lu UNE
    SEULE FOIS en lecture seule au premier appel (les ordres de grille
    actuellement ouverts sur OKX), avec les ordres placés/annulés en
    mémoire par ce harnais depuis son démarrage.

Ce module ne touche jamais run_active_grid.py, ni grid_replacement.py --
comportement LIVE strictement inchangé. Il ne committe rien, ne pushe
rien : c'est un script d'exécution locale, pas un test pytest, pas une
modification de dépôt.

--------------------------------------------------------------------------
FICHIERS TOUCHÉS SUR DISQUE
--------------------------------------------------------------------------
Le fichier d'état RÉEL (active_grid_state_<inst_id>.json) n'est JAMAIS
ouvert en écriture par ce script -- seulement lu, une fois, pour récupérer
la configuration réelle de la grille active (P0/GLL/GUL/nu/nl/q/...).

Toute écriture de ce harnais (état "replacement"/"churn_protection",
identité persistée de la nouvelle grille, historique JSONL) va dans un
fichier SÉPARÉ : "<fichier_réel>.DRYRUN.json" (+ son
"..._history.jsonl"), recréé à zéro à chaque exécution sauf --resume.

--------------------------------------------------------------------------
UTILISATION SUR DELLT320
--------------------------------------------------------------------------
    OKX_API_KEY=... OKX_SECRET_KEY=... OKX_PASSPHRASE=... \\
        python3 dry_run_replacement.py --inst-id XRP-USDC \\
        --prices 1.02,1.02,1.15,1.35,1.55,1.60,1.60,1.60,1.60,1.60,1.60,1.60

Voir --help pour le détail des options. Aucune option --live : ce script
n'a pas de mode "réel", par construction (DryRunAdapter ne sait pas
écrire sur OKX).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace as dataclass_replace
from decimal import Decimal, InvalidOperation

import churn_protection
import grid_replacement
import market_reader
from grid_preflight import run_preflight
from grid_replacement import _atr_width_pct
from okx_spot_adapter import CancelResult, OkxApiError, OrderSnapshot, PlacementResult, Ticker
from run_active_grid import K_BUY, K_SELL, load_grid_state, state_file_path_for
from run_grid_trading import build_adapter


# ---------------------------------------------------------------------------
# Flux de prix simulé.
# ---------------------------------------------------------------------------


class SimulatedPriceFeed:
    """Fournit un prix simulé à chaque appel de get_ticker, dans l'ordre
    donné par --prices. Une fois la séquence épuisée, répète INDÉFINIMENT
    la dernière valeur -- ne lève jamais faute de "prochain prix".

    Ce comportement est délibéré, pas une simplification paresseuse : un
    seul cycle de run_replacement_check peut appeler get_ticker() PLUSIEURS
    fois (détection du dépassement, calcul des soldes après annulation,
    calcul du déplacement déjà réalisé) -- exactement comme en production,
    où ces appels successifs lisent le même prix de marché "instantané".
    Répéter la dernière valeur simule fidèlement ce cas ; il suffit de
    fournir la séquence des prix RÉELLEMENT distincts qui vous intéressent
    (ex: 2 prix dans la zone, puis 3 confirmations consécutives au-dessus
    de GUL, puis quelques prix de retour dans la nouvelle zone pour
    vérifier l'absence de cascade).
    """

    def __init__(self, prices: list[Decimal]):
        if not prices:
            raise ValueError("SimulatedPriceFeed exige au moins un prix")
        self._prices = prices
        self._index = 0
        self.calls = 0

    def next(self) -> Decimal:
        self.calls += 1
        value = self._prices[min(self._index, len(self._prices) - 1)]
        if self._index < len(self._prices) - 1:
            self._index += 1
        return value


def _with_state(order: OrderSnapshot, new_state: str) -> OrderSnapshot:
    return OrderSnapshot(
        order_id=order.order_id, client_order_id=order.client_order_id, inst_id=order.inst_id,
        side=order.side, price=order.price, quantity=order.quantity,
        accumulated_fill_quantity=order.accumulated_fill_quantity, state=new_state,
        order_type=order.order_type, created_at=order.created_at, updated_at=order.updated_at,
    )


# ---------------------------------------------------------------------------
# Adaptateur DRY-RUN.
# ---------------------------------------------------------------------------


class DryRunAdapter:
    def __init__(self, real_adapter, price_feed: SimulatedPriceFeed, log):
        self._real = real_adapter
        self._prices = price_feed
        self._log = log
        self._shadow_orders: dict[str, OrderSnapshot] = {}
        self._cancelled_ids: set[str] = set()
        self._real_open_snapshot: dict[str, OrderSnapshot] | None = None
        self.write_calls: list[str] = []
        self.last_price_seen: Decimal | None = None
        # Ledger de bookkeeping PUREMENT LOCAL, jamais écrit nulle part --
        # applique uniquement la conséquence comptable mécanique d'un fill
        # de market buy SIMULÉ (voir place_market_buy) : base += quantité,
        # quote -= quantité * prix. Aucune décision économique n'est prise
        # ici (required_spot, spot_covered, q restent calculés par le
        # moteur réel, inchangé) -- ce ledger ne fait que rendre visible,
        # pour get_balances(), l'effet d'un achat que le moteur a déjà
        # décidé et que ce harnais a déjà accepté de simuler. Nul tant
        # qu'aucun place_market_buy() n'a été simulé : get_balances()
        # reste alors un passthrough réel strict, comportement inchangé.
        self._shadow_base_delta = Decimal("0")
        self._shadow_quote_delta = Decimal("0")
        self.simulated_fills: list[tuple[Decimal, Decimal]] = []  # (quantité, coût)

    # ---- lectures déléguées telles quelles (GET uniquement) ----
    def get_instrument(self, inst_id):
        return self._real.get_instrument(inst_id)

    def get_fee_rates(self, inst_id):
        return self._real.get_fee_rates(inst_id)

    def get_account_config(self):
        return self._real.get_account_config()

    def get_balances(self, base_ccy, quote_ccy):
        real = self._real.get_balances(base_ccy, quote_ccy)
        if self._shadow_base_delta == 0 and self._shadow_quote_delta == 0:
            return real  # aucun fill simulé -- passthrough réel strict, inchangé
        adjusted = type(real)(
            real.base_available + self._shadow_base_delta,
            real.quote_available + self._shadow_quote_delta,
            real.base_total + self._shadow_base_delta,
            real.quote_total + self._shadow_quote_delta,
        )
        self._log(
            f"  [DRY-RUN] get_balances({base_ccy}/{quote_ccy}) -> réel {real.base_available}/"
            f"{real.quote_available} + delta simulé {self._shadow_base_delta}/"
            f"{self._shadow_quote_delta} = {adjusted.base_available}/{adjusted.quote_available} "
            f"(lecture réelle, ajustement local uniquement)"
        )
        return adjusted

    def get_price_limit(self, inst_id):
        return self._real.get_price_limit(inst_id)

    def get_klines(self, inst_id, bar="15m", limit=50):
        return self._real.get_klines(inst_id, bar=bar, limit=limit)

    def list_fills(self, inst_id, *, order_id=None):
        return self._real.list_fills(inst_id, order_id=order_id)

    # ---- prix courant : simulé, jamais lu sur le marché réel ----
    def get_ticker(self, inst_id):
        price = self._prices.next()
        self.last_price_seen = price
        self._log(f"  [DRY-RUN] get_ticker({inst_id}) -> prix SIMULÉ {price}")
        return Ticker(inst_id=inst_id, last=price, ts=0)

    # ---- carnet d'ordres : instantané réel figé + écritures en mémoire ----
    def _ensure_snapshot(self, inst_id):
        if self._real_open_snapshot is None:
            real_open = self._real.list_open_orders(inst_id)
            self._real_open_snapshot = {o.client_order_id: o for o in real_open}
            self._log(
                f"  [DRY-RUN] instantané réel figé (lecture seule) : "
                f"{len(self._real_open_snapshot)} ordre(s) ouvert(s) pour {inst_id}"
            )

    def list_open_orders(self, inst_id, client_order_id=None):
        self._ensure_snapshot(inst_id)
        combined = dict(self._real_open_snapshot)
        combined.update(self._shadow_orders)
        result = [
            o for cid, o in combined.items()
            if cid not in self._cancelled_ids and o.state in ("live", "partially_filled")
        ]
        if client_order_id is not None:
            result = [o for o in result if o.client_order_id == client_order_id]
        return tuple(result)

    def get_order(self, inst_id, *, order_id=None, client_order_id=None):
        if client_order_id is not None:
            if client_order_id in self._shadow_orders:
                order = self._shadow_orders[client_order_id]
                if client_order_id in self._cancelled_ids:
                    order = _with_state(order, "canceled")
                return order
            self._ensure_snapshot(inst_id)
            if client_order_id in self._real_open_snapshot:
                order = self._real_open_snapshot[client_order_id]
                if client_order_id in self._cancelled_ids:
                    order = _with_state(order, "canceled")
                return order
        # Ni simulé ni dans l'instantané réel figé -- délégation en lecture
        # seule (peut renvoyer un ordre FILLED historique légitime, ou lever
        # OkxApiError si réellement introuvable -- comportement identique
        # au réel dans les deux cas).
        return self._real.get_order(inst_id, order_id=order_id, client_order_id=client_order_id)

    # ---- écritures : JAMAIS transmises à OKX ----
    def cancel_order(self, inst_id, *, order_id=None, client_order_id=None):
        self.write_calls.append(f"cancel_order(inst_id={inst_id}, client_order_id={client_order_id})")
        self._log(f"  [DRY-RUN] SIMULÉ cancel_order({client_order_id}) -- AUCUN appel réel à OKX")
        if client_order_id is not None:
            self._cancelled_ids.add(client_order_id)
        return CancelResult(True, order_id, client_order_id, None, None)

    def place_post_only_limit(self, inst_id, side, price, quantity, client_order_id):
        self.write_calls.append(
            f"place_post_only_limit(side={side}, price={price}, qty={quantity}, id={client_order_id})"
        )
        self._log(
            f"  [DRY-RUN] SIMULÉ place_post_only_limit({side}, {price}, {quantity}, "
            f"{client_order_id}) -- AUCUN appel réel à OKX"
        )
        snapshot = OrderSnapshot(
            order_id=f"DRYRUN-{client_order_id[-10:]}", client_order_id=client_order_id,
            inst_id=inst_id, side=side, price=price, quantity=quantity,
            accumulated_fill_quantity=Decimal("0"), state="live",
            order_type="post_only", created_at=0, updated_at=0,
        )
        self._shadow_orders[client_order_id] = snapshot
        return PlacementResult(True, snapshot.order_id, client_order_id, None, None)

    def place_market_buy(self, inst_id, quantity, client_order_id):
        self.write_calls.append(f"place_market_buy(qty={quantity}, id={client_order_id})")
        price = self.last_price_seen if self.last_price_seen is not None else self._prices.next()
        cost = quantity * price
        self._log(
            f"  [DRY-RUN] SIMULÉ place_market_buy({quantity}, {client_order_id}) -- AUCUN appel "
            f"réel à OKX."
        )
        # Bookkeeping SIMULÉ : conséquence comptable mécanique d'un fill déjà
        # décidé par le moteur (required_spot/spot_covered), jamais une
        # décision économique prise ici. Même formule que le FakeAdapter déjà
        # validé par tests/test_grid_replacement.py (base += quantité, quote
        # -= quantité * prix, sans frais -- cohérence avec la référence
        # existante plutôt qu'une hypothèse de frais non vérifiée).
        self._shadow_base_delta += quantity
        self._shadow_quote_delta -= cost
        self.simulated_fills.append((quantity, cost))
        self._log(
            f"  [DRY-RUN] solde SIMULÉ mis à jour (bookkeeping seul, aucun échange réel) : "
            f"base +{quantity}, quote -{cost} (delta cumulé : {self._shadow_base_delta} / "
            f"{self._shadow_quote_delta})"
        )
        snapshot = OrderSnapshot(
            order_id=f"DRYRUN-{client_order_id[-10:]}", client_order_id=client_order_id,
            inst_id=inst_id, side="BUY", price=price, quantity=quantity,
            accumulated_fill_quantity=quantity, state="filled",
            order_type="market", created_at=0, updated_at=0,
        )
        self._shadow_orders[client_order_id] = snapshot
        return PlacementResult(True, snapshot.order_id, client_order_id, None, None)


# ---------------------------------------------------------------------------
# Affichage.
# ---------------------------------------------------------------------------


def _fmt_config_block(label: str, config, suffix: str = "") -> str:
    lines = [f"{label}:"]
    lines.append(f"  P0{suffix}  = {config.p0}")
    lines.append(f"  GLL{suffix} = {config.gll}")
    lines.append(f"  GUL{suffix} = {config.gul}")
    lines.append(f"  q{suffix}   = {config.q}")
    lines.append(f"  allocated_capital{suffix} = {config.allocated_capital}")
    return "\n".join(lines)


def _parse_prices(raw: str) -> list[Decimal]:
    try:
        return [Decimal(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]
    except InvalidOperation as error:
        raise SystemExit(f"--prices invalide : {error}")


def _dry_state_path(real_state_path: str) -> str:
    return f"{real_state_path}.DRYRUN.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dry-run du remplacement dynamique de grille (grid_replacement.py) "
        "-- prix simulé, aucun ordre réel envoyé à OKX.",
    )
    parser.add_argument("--inst-id", required=True, help="ex: XRP-USDC")
    parser.add_argument(
        "--prices", required=True,
        help="séquence de prix simulés, séparés par des virgules, dans l'ordre chronologique "
        "des appels get_ticker() (répète le dernier une fois épuisée). "
        "ex: 1.02,1.02,1.15,1.35,1.55,1.60,1.60,1.60,1.60,1.60,1.60,1.60",
    )
    parser.add_argument(
        "--max-cycles", type=int, default=25,
        help="nombre d'appels à run_replacement_check à effectuer (défaut: 25)",
    )
    parser.add_argument(
        "--state-file", default=None,
        help="chemin du fichier d'état RÉEL à lire (lecture seule). "
        "Défaut : active_grid_state_<inst_id>.json (comme run_active_grid.py).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="ne PAS repartir de zéro : reprend l'état de remplacement dry-run laissé par "
        "une exécution précédente de ce script (teste la reprise après redémarrage, "
        "scénario 13 -- réutilise le fichier .DRYRUN.json existant tel quel).",
    )
    args = parser.parse_args()

    def log(msg: str) -> None:
        print(msg)

    print("========== DRY-RUN — REMPLACEMENT DYNAMIQUE DE GRILLE ==========")
    print("AUCUN ordre réel ne sera envoyé, annulé ou modifié sur OKX.")
    print("Seuls des endpoints de LECTURE OKX sont appelés (config, soldes, ordres ouverts,")
    print("bougies, ticker RÉEL jamais consulté -- get_ticker est entièrement simulé).\n")

    real_state_path = args.state_file or state_file_path_for(args.inst_id)
    if not os.path.exists(real_state_path):
        print(f"❌ fichier d'état réel introuvable : {real_state_path}")
        return 1

    real_adapter = build_adapter("LIVE")

    try:
        real_config = load_grid_state(real_state_path)
    except Exception as error:
        print(f"❌ impossible de charger la configuration réelle depuis {real_state_path} : {error}")
        return 1
    if real_config is None:
        print(f"❌ {real_state_path} présent mais structurellement invalide -- abandon.")
        return 1

    dry_state_path = _dry_state_path(real_state_path)
    dry_history_prefix = dry_state_path.rsplit(".json", 1)[0]
    dry_history_path = f"{dry_history_prefix}_history.jsonl"

    if not args.resume:
        for path in (dry_state_path, dry_history_path):
            if os.path.exists(path):
                os.remove(path)
        print(f"État dry-run repartant de zéro ({dry_state_path} recréé au fil de l'exécution).\n")
    else:
        print(f"--resume : reprise de l'état dry-run existant ({dry_state_path}).\n")

    prices = _parse_prices(args.prices)
    price_feed = SimulatedPriceFeed(prices)
    dry_adapter = DryRunAdapter(real_adapter, price_feed, log)

    print(_fmt_config_block("OLD (grille active réelle, lue en lecture seule)", real_config))
    print()

    config = real_config
    history: list[tuple[int, str, str, bool]] = []  # (cycle, pre_status, post_status, materialize)
    replacement_completed_at: int | None = None
    observed_direction: str | None = None
    observed_confirmation_price: Decimal | None = None

    # Lecture seule, une fois -- sert à la fois à détecter la direction du
    # remplacement matérialisé (ci-dessous) et au rapport final.
    instrument = dry_adapter.get_instrument(args.inst_id)
    expected_p0_up = grid_replacement._round_down(real_config.gul, instrument.tick_size)
    expected_p0_down = grid_replacement._round_down(real_config.gll, instrument.tick_size)

    for cycle in range(1, args.max_cycles + 1):
        pre_state = grid_replacement.load_replacement_state(dry_state_path)
        print(
            f"--- cycle {cycle} --- avant: status={pre_state.lifecycle_status} "
            f"above_streak={pre_state.above_streak} below_streak={pre_state.below_streak}"
        )
        new_config, materialize = grid_replacement.run_replacement_check(
            dry_adapter, config, K_BUY, K_SELL, dry_state_path,
        )
        post_state = grid_replacement.load_replacement_state(dry_state_path)
        print(
            f"    après: status={post_state.lifecycle_status} "
            f"above_streak={post_state.above_streak} below_streak={post_state.below_streak} "
            f"materialize={materialize}"
        )
        history.append((cycle, pre_state.lifecycle_status, post_state.lifecycle_status, materialize))

        if new_config != config:
            print(f"    >>> REMPLACEMENT MATÉRIALISÉ au cycle {cycle} <<<")
            replacement_completed_at = cycle
            # Capturés ICI, au moment exact où le remplacement aboutit --
            # contrairement à une lecture de post_state.direction, ceci
            # fonctionne QUEL QUE SOIT le nombre de cycles internes qu'a
            # pris la transition. Depuis le dimensionnement Solution C
            # (q_new borné par les ressources réellement disponibles),
            # CANCELLING -> ACTIVATING -> ACTIVATED peut désormais aboutir
            # en un seul appel à run_replacement_check (plus d'achat
            # patrimonial intermédiaire nécessaire dans le cas général) --
            # post_state, relu APRÈS le retour de cet appel, refléterait
            # alors déjà l'état "ACTIVE" final (replacement réinitialisé
            # par _persist_new_active_config), jamais la direction/le prix
            # de confirmation intermédiaires. dry_adapter.last_price_seen,
            # lui, reste correct : le dernier get_ticker() de ce cycle est
            # celui de compute_new_config (déplacement_pct), et la
            # direction se déduit sans ambiguïté de new_config.p0 comparé
            # aux deux bornes arrondies (jamais par égalité brute, invalide
            # à cause du tick_size).
            observed_confirmation_price = dry_adapter.last_price_seen
            if new_config.p0 == expected_p0_up:
                observed_direction = "UP"
            elif new_config.p0 == expected_p0_down:
                observed_direction = "DOWN"
            config = new_config
        print()

    print("========== RÉSULTAT ==========\n")

    if replacement_completed_at is None:
        last_status = history[-1][2] if history else "ACTIVE"
        print("Aucun remplacement matérialisé sur les", args.max_cycles, "cycles fournis.")
        if last_status == "ACTIVE":
            print("(normal si --prices ne contient jamais 3 confirmations consécutives hors bornes)")
        else:
            print(
                f"Dépassement confirmé (état bloqué en {last_status}) mais la matérialisation n'a "
                f"pas abouti dans les {args.max_cycles} cycles fournis. Causes possibles :\n"
                f"  - annulation de l'ancienne génération pas encore vérifiée vide (CANCELLING) ;\n"
                f"  - placement des niveaux de grille encore en cours (K_BUY/K_SELL limitent le "
                f"nombre de niveaux posés par cycle) -- relancez avec --max-cycles plus élevé et/ou "
                f"--resume ;\n"
                f"  - si un achat patrimonial initial a été nécessaire (voir lignes [DRY-RUN] SIMULÉ "
                f"place_market_buy ci-dessus), son effet sur les soldes EST simulé par ce harnais "
                f"(bookkeeping local) -- si le blocage persiste malgré cela, le problème est "
                f"probablement réel (required_spot/spot_covered) et mérite investigation, pas un "
                f"relancement aveugle."
            )
        return 0

    direction = observed_direction or "?"
    borne_old = real_config.gul if direction == "UP" else real_config.gll
    expected_p0 = expected_p0_up if direction == "UP" else (expected_p0_down if direction == "DOWN" else None)
    prix_confirmation = observed_confirmation_price
    deplacement_pct = None
    if prix_confirmation is not None and borne_old:
        deplacement_pct = abs(float(prix_confirmation) - float(borne_old)) / float(borne_old)

    conditions = market_reader.read(dry_adapter, args.inst_id)
    atr_ref = _atr_width_pct(conditions.atr_norm_15m, conditions.di_ratio_15m)
    gul_pct_atr = atr_ref["gul_pct"] if direction == "UP" else atr_ref["gll_pct"]

    print(_fmt_config_block("NEW (grille de remplacement, activée par ce dry-run)", config, suffix="_new"))
    print(f"  direction        = {direction}")
    print(f"  déplacement_pct  = {deplacement_pct}")
    print(f"  gul_pct_ATR      = {gul_pct_atr}  (côté franchi -- référence ATR recalculée indépendamment, mêmes bougies)")

    # --- Vérification EXPLICITE de l'invariant (point 6 demandé) ---
    print("\n========== INVARIANT P0_new = borne franchie ==========")
    if direction == "UP":
        print(f"dépassement HAUT : P0_new == GUL_old ?")
        print(f"  GUL_old (brut, non arrondi)        = {real_config.gul}")
    elif direction == "DOWN":
        print(f"dépassement BAS : P0_new == GLL_old ?")
        print(f"  GLL_old (brut, non arrondi)         = {real_config.gll}")
    else:
        print("direction non observée -- invariant non vérifiable")
    if direction in ("UP", "DOWN"):
        tick_diff = abs(float(borne_old) - float(expected_p0))
        print(f"  round_down(borne_old, tick_size)   = {expected_p0}  (tick_size={instrument.tick_size})")
        print(f"  P0_new (config activée)            = {config.p0}")
        print(f"  P0_new == round_down(borne_old, tick_size) : {'OK' if config.p0 == expected_p0 else 'ÉCHEC'}")
        print(
            f"  (égalité vérifiée contre la borne ARRONDIE au tick -- l'égalité brute contre "
            f"{borne_old} ne peut structurellement pas tenir, écart de {tick_diff} < 1 tick, "
            f"c'est l'arrondi obligatoire de run_preflight, pas une approximation de ce harnais)"
        )
        print(f"  P0_new != prix_confirmation ({prix_confirmation}) : "
              f"{'OK' if config.p0 != prix_confirmation else 'ÉCHEC'} (jamais le prix courant)")

    # --- Niveaux min/max de la nouvelle grille (point 7 demandé) ---
    report = run_preflight(dry_adapter, config)
    print("\n========== NIVEAUX DE LA NOUVELLE GRILLE ==========")
    print(f"nombre de niveaux : {len(report.trellis)} (nl={config.nl} BUY, nu={config.nu} SELL, + P0)")
    print(f"niveau min = {min(report.trellis)}")
    print(f"niveau max = {max(report.trellis)}")

    if dry_adapter.simulated_fills:
        total_qty = sum(q for q, _ in dry_adapter.simulated_fills)
        total_cost = sum(c for _, c in dry_adapter.simulated_fills)
        print(
            f"\nAchat(s) patrimonial(aux) SIMULÉ(S) pendant ce dry-run : {len(dry_adapter.simulated_fills)} "
            f"(quantité totale {total_qty}, coût simulé total {total_cost}) -- bookkeeping local uniquement, "
            f"AUCUN ordre réel envoyé à OKX."
        )

    print("\n========== CHECKLIST DES 14 POINTS DEMANDÉS ==========\n")

    # Transition hors de ACTIVE (vers CANCELLING ou, en cas de cascade dans
    # le même cycle -- carnet déjà vide, preflight immédiatement valide --
    # directement vers ACTIVATING) : c'est le cycle de la 3e confirmation.
    n3_reached_cycle = next((c for c, pre, post, _ in history if pre == "ACTIVE" and post != "ACTIVE"), None)
    print(f"1.  détection du dépassement GUL/GLL : {'OK' if n3_reached_cycle else 'NON OBSERVÉ'}"
          + (f" (bascule ACTIVE->{next(post for c, pre, post, _ in history if c == n3_reached_cycle)} "
             f"au cycle {n3_reached_cycle})" if n3_reached_cycle else ""))
    print(f"2.  compteur N=3 : voir les lignes 'avant/après' ci-dessus (above_streak/below_streak) ; "
          f"REPLACEMENT_N={grid_replacement.REPLACEMENT_N}")
    no_early_replacement = n3_reached_cycle is None or all(
        post == "ACTIVE" for c, pre, post, _ in history if c < n3_reached_cycle
    )
    print(f"3.  absence de remplacement avant la 3e confirmation : "
          f"{'OK' if no_early_replacement else 'ÉCHEC — vérifier le détail ci-dessus'} "
          f"(tous les cycles avant le cycle {n3_reached_cycle} sont restés ACTIVE)"
          if n3_reached_cycle else "3.  absence de remplacement avant la 3e confirmation : N/A (aucun remplacement observé)")
    print(f"4.  P0_new = ancienne borne franchie (arrondie au tick), jamais le prix courant : "
          f"{'OK' if config.p0 == expected_p0 and config.p0 != prix_confirmation else 'À VÉRIFIER'}")
    print(f"5.  nouvelle borne = ATR + déplacement (côté franchi) : voir gul_pct_ATR/déplacement_pct ci-dessus, "
          f"et comparer à la largeur effective du côté franchi dans NEW ci-dessus")
    other_side_untouched = (
        abs(float(config.gll) - float(config.p0) * (1 - atr_ref["gll_pct"])) < 1e-6 if direction == "UP"
        else abs(float(config.gul) - float(config.p0) * (1 + atr_ref["gul_pct"])) < 1e-6
    )
    print(f"6.  élargissement uniquement du côté franchi (autre côté = ATR seule) : "
          f"{'OK' if other_side_untouched else 'À VÉRIFIER (arrondi tick_size possible)'}")
    print(f"7.  q recalculé (jamais copié) : ancien q={real_config.q} / nouveau q={config.q} -> "
          f"{'OK (différent)' if config.q != real_config.q else 'IDENTIQUE — à examiner'}")
    cancel_calls = [c for c in dry_adapter.write_calls if c.startswith("cancel_order")]
    print(f"8.  annulation logique de l'ancienne génération : {len(cancel_calls)} cancel_order() SIMULÉ(S) "
          f"— {'OK' if cancel_calls else 'AUCUN — ancienne grille avait-elle des ordres ouverts ?'}")
    place_calls = [c for c in dry_adapter.write_calls if c.startswith("place_post_only_limit")]
    print(f"9.  création de la nouvelle identité : {len(place_calls)} place_post_only_limit() SIMULÉ(S)")

    if os.path.exists(dry_state_path):
        with open(dry_state_path) as f:
            persisted = json.load(f)
        persisted_matches = persisted.get("p0") == str(config.p0) and persisted.get("gul") == str(config.gul)
        print(f"10. persistance dans le fichier d'état dry-run ({dry_state_path}) : "
              f"{'OK' if persisted_matches else 'ÉCHEC'} — p0 persisté = {persisted.get('p0')}")
    else:
        print("10. persistance : ÉCHEC — fichier dry-run absent")

    if os.path.exists(dry_history_path):
        with open(dry_history_path) as f:
            history_lines = [json.loads(line) for line in f if line.strip()]
        old_found = any(entry.get("p0") == str(real_config.p0) for entry in history_lines)
        print(f"11. historique JSONL ({dry_history_path}) : {len(history_lines)} entrée(s) — "
              f"{'OK, ancienne identité retrouvée' if old_found else 'ancienne identité NON retrouvée'}")
    else:
        print("11. historique JSONL : ÉCHEC — fichier absent")

    churn_state = churn_protection.load_churn_protection(dry_state_path)
    protected = churn_protection.protected_gis(churn_state)
    print(f"12. remise à zéro de churn_protection après remplacement : "
          f"{'OK (aucune cellule protégée)' if not protected else protected}")

    print("13. reprise après redémarrage : relancez ce script avec --resume (même --prices, "
          "ou une suite) — il repartira de l'état persisté ci-dessus au lieu de zéro.")

    post_replacement_cycles = [
        (c, pre, post, m) for c, pre, post, m in history if c > (replacement_completed_at or 0)
    ]
    no_cascade = all(post != "CANCELLING" for _, _, post, _ in post_replacement_cycles)
    print(f"14. absence de cascade immédiate après remplacement : "
          f"{'OK' if no_cascade or not post_replacement_cycles else 'CASCADE DÉTECTÉE — voir cycles ci-dessus'} "
          f"({len(post_replacement_cycles)} cycle(s) observé(s) après remplacement)")

    print(f"\nAppels d'écriture interceptés (aucun n'a atteint OKX) : {len(dry_adapter.write_calls)}")
    for call in dry_adapter.write_calls:
        print(f"  - {call}")

    print("\nAucun fichier réel modifié. Aucun commit. Aucun push.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
