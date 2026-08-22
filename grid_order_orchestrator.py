"""GridOrderOrchestrator — séquence annulation puis matérialisation.

L'orchestrateur reconstruit l'état économique des cellules, applique la
politique d'exposition, puis délègue le placement à GridActivationController.

L'identité d'une nouvelle occurrence est dérivée à partir de la dernière
occurrence remplie de la cellule :

    ROOT -> occurrence suivante -> occurrence suivante -> ...

GridActivationController reste responsable du sondage et du placement réel.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal

from cell_order_identity import derive_next_order_id, derive_root_order_id
from cell_state_reconstruction import reconstruct_cell_states
from churn_protection import (
    fetch_grid_fills, load_churn_protection, protected_gis, save_churn_protection,
    update_churn_protection,
)
from grid_activation_controller import GridActivationController, GridActivationResult
from grid_order_exposure_controller import ExposureDecision, select_exposed_orders
from grid_order_projection import BUY, SELL
from grid_preflight import PreflightReport
from okx_spot_adapter import OkxSpotAdapter


@dataclass(frozen=True)
class OrchestrationResult:
    """Résultat d'un cycle complet de réconciliation/exposition."""

    cancelled: tuple[str, ...]
    activation_result: GridActivationResult


@dataclass(frozen=True)
class ExposurePlan:
    """Plan de réconciliation PUR — reconstruction, sélection, dérivation
    d'identité — SANS AUCUNE EXÉCUTION (ni annulation, ni placement).

    Permet à un appelant d'inspecter `decision.to_cancel` et de décider
    comment agir AVANT toute écriture réelle — notamment pour refuser une
    activation initiale plutôt que d'annuler automatiquement des ordres
    déjà ouverts appartenant potentiellement à une autre configuration
    (cf. grid_trading_controller.py, garde-fou contre les ordres orphelins).
    """

    decision: "ExposureDecision"
    narrowed_report: PreflightReport
    order_ids: dict
    requested_to_materialize: tuple


def _filter_by_available_liquidity(
    instructions: tuple,
    base_available,
    quote_available,
) -> tuple:
    """Filtre les instructions non finançables IMMÉDIATEMENT, en respectant
    l'ordre de priorité déjà établi par select_exposed_orders() (BUY par
    distance d'index croissante, puis SELL, idem) — jamais une nouvelle
    priorité inventée ici.

    Règle volontairement simple (correctif #2) :
        BUY  : coût nécessaire        = price * quantity   (contre quote_available)
        SELL : quantité base requise  = quantity            (contre base_available)

    Le cumul est suivi séparément par côté (deux compteurs indépendants) --
    un BUY non finançable n'affecte jamais la disponibilité pour un SELL,
    et réciproquement (cf. CAS C/D du chantier). Chaque instruction
    consomme le solde restant SEULEMENT si elle passe le test ; une
    instruction rejetée ne consomme rien, permettant à l'instruction
    suivante de MÊME côté d'être tentée contre le solde encore intact.

    PURE : ne modifie jamais q, ni les prix, ni k_buy/k_sell, ni aucun
    état -- une simple restriction de la liste transmise à l'activation.
    """
    remaining_quote = Decimal(str(quote_available))
    remaining_base = Decimal(str(base_available))

    financable: list = []
    for instruction in instructions:
        side, price, quantity = instruction
        price_dec = Decimal(str(price))
        quantity_dec = Decimal(str(quantity))

        if side == BUY:
            cost = price_dec * quantity_dec
            if cost <= remaining_quote:
                remaining_quote -= cost
                financable.append(instruction)
            # sinon : non financé, ignoré -- ne consomme rien, ne bloque
            # jamais le côté SELL (compteurs indépendants)
        elif side == SELL:
            if quantity_dec <= remaining_base:
                remaining_base -= quantity_dec
                financable.append(instruction)
        else:
            raise ValueError(f"_filter_by_available_liquidity : side inconnu : {side!r}")

    return tuple(financable)


def _filter_by_price_limit(instructions: tuple, buy_lmt, sell_lmt) -> tuple:
    """Filtre les instructions hors de la bande de prix dynamique OKX
    (GET /api/v5/public/price-limit).

    Règle :
        BUY  : price <= buy_lmt
        SELL : price >= sell_lmt

    Un niveau rejeté est simplement absent du résultat -- jamais
    d'exception, jamais d'annulation d'un ordre existant, jamais de
    client_order_id dérivé pour ce niveau (ce filtre s'exécute avant
    la boucle de dérivation d'identité dans plan_exposure). Le niveau
    reste présent dans le treillis théorique et sera réévalué au cycle
    suivant, exactement comme un niveau non finançable (correctif #2).

    PURE : ne modifie jamais q, ni les prix, ni k_buy/k_sell, ni aucun
    état.
    """
    within_limit: list = []
    for instruction in instructions:
        side, price, quantity = instruction
        price_dec = Decimal(str(price))

        if side == BUY:
            if price_dec <= buy_lmt:
                within_limit.append(instruction)
            else:
                print(f"[PRICE_LIMIT_FILTER] BUY {price} > buy_lmt {buy_lmt} -- non matérialisé, retenté au cycle suivant")
        elif side == SELL:
            if price_dec >= sell_lmt:
                within_limit.append(instruction)
            else:
                print(f"[PRICE_LIMIT_FILTER] SELL {price} < sell_lmt {sell_lmt} -- non matérialisé, retenté au cycle suivant")
        else:
            raise ValueError(f"_filter_by_price_limit : side inconnu : {side!r}")

    return tuple(within_limit)


def plan_exposure(
    adapter,
    report: PreflightReport,
    k_buy: int,
    k_sell: int,
    *,
    churn_state_path: str | None = None,
) -> ExposurePlan:
    """Reconstruction + sélection + dérivation d'identité — PURE pour tout
    ce qui concerne la décision d'exposition elle-même.

    Aucun appel d'écriture d'ORDRE (ni cancel_order, ni
    place_post_only_limit) — seules des lectures (list_open_orders,
    list_fills, get_order via reconstruct_cell_states) et des calculs
    purs. Extrait de run_exposure_cycle (qui reste, lui, inchangé dans son
    comportement -- voir plus bas) pour permettre sa réutilisation sans
    jamais dupliquer la logique d'identité/ROOT.

    `churn_state_path` : chemin du fichier de persistance anti-churn
    (churn_protection.py). None (défaut) : mécanisme totalement désactivé,
    comportement strictement identique à avant son introduction (aucune
    lecture, aucune écriture, aucun filtre appliqué à
    select_exposed_orders) -- SEUL run() (run_active_grid.py) fournit ce
    chemin en usage réel. Si fourni, ce chemin est LU et RÉ-ÉCRIT ici (seule
    exception au caractère "pur" de cette fonction, au même titre que les
    lectures réseau ci-dessus) -- même fichier que save_grid_state, fusion
    stricte, jamais un remplacement des autres clés.
    """
    inst_id = report.config.inst_id
    open_orders = adapter.list_open_orders(inst_id)

    cell_states = reconstruct_cell_states(adapter, report, open_orders)

    if churn_state_path is not None:
        churn_state = load_churn_protection(churn_state_path)
        grid_fills = fetch_grid_fills(adapter, report)
        churn_state = update_churn_protection(churn_state, grid_fills, report.config.p0)
        save_churn_protection(churn_state, churn_state_path)
        churn_protected = protected_gis(churn_state)
    else:
        churn_protected = frozenset()

    decision = select_exposed_orders(report, cell_states, open_orders, k_buy, k_sell, churn_protected)
    requested_to_materialize = decision.to_materialize  # AVANT tout filtre -- observation pure, aucun filtre modifié

    # Correctif #2 : filtrage par liquidité IMMÉDIATEMENT DISPONIBLE
    # (report.balances.*_available, jamais *_total -- les fonds déjà
    # immobilisés dans des ordres ouverts ne sont pas disponibles à
    # nouveau). Ne modifie jamais decision.to_cancel : le filtrage ne
    # porte que sur ce qui serait matérialisé, jamais sur les annulations
    # déjà décidées par select_exposed_orders (aucun ordre existant n'est
    # jamais annulé pour une raison de liquidité, cf. CAS F).
    financable_to_materialize = _filter_by_available_liquidity(
        decision.to_materialize,
        report.balances.base_available,
        report.balances.quote_available,
    )

    # Filtre bande de prix dynamique OKX -- après liquidité, avant
    # dérivation d'identité. Un niveau filtré ici ne génère jamais de
    # client_order_id (la boucle ci-dessous n'opère que sur le résultat
    # déjà filtré) et reste dans le treillis théorique persisté, inchangé.
    #
    # isinstance(adapter, OkxSpotAdapter), PAS hasattr() : pour l'adaptateur
    # de production, get_price_limit() est OBLIGATOIRE -- appel
    # inconditionnel, jamais défensif. Si cette méthode disparaissait un
    # jour de OkxSpotAdapter, cet appel échouerait bruyamment
    # (AttributeError réel), jamais silencieusement masqué. Pour tout
    # autre adaptateur (simulacres de test, jamais des instances de
    # OkxSpotAdapter), cette contrainte dynamique d'OKX n'a simplement
    # aucun sens à s'appliquer -- ce n'est pas une absence accidentelle,
    # c'est une distinction architecturale explicite.
    if isinstance(adapter, OkxSpotAdapter):
        price_limit = adapter.get_price_limit(report.config.inst_id)
        financable_to_materialize = _filter_by_price_limit(
            financable_to_materialize,
            price_limit.buy_lmt,
            price_limit.sell_lmt,
        )

    decision = ExposureDecision(financable_to_materialize, decision.to_cancel)

    order_ids = {}

    for instruction in decision.to_materialize:
        side, price, quantity = instruction

        state = _find_cell_state(cell_states, price)

        if state is None:
            raise RuntimeError(
                f"Instruction sans cellule correspondante : "
                f"{instruction!r}"
            )

        if state.last_client_order_id is None:
            client_order_id = derive_root_order_id(
                report.config,
                report.instrument.tick_size,
                report.instrument.lot_size,
                side,
                price,
            )
        else:
            client_order_id = derive_next_order_id(
                report.config,
                report.instrument.tick_size,
                report.instrument.lot_size,
                side,
                price,
                state.last_client_order_id,
            )

        order_ids[instruction] = client_order_id

    narrowed_report = replace(
        report,
        orders=decision.to_materialize,
    )

    return ExposurePlan(decision, narrowed_report, order_ids, requested_to_materialize)


def run_exposure_cycle(
    adapter,
    report: PreflightReport,
    k_buy: int,
    k_sell: int,
    *,
    churn_state_path: str | None = None,
) -> OrchestrationResult:
    """Exécute un cycle : reconstruction, sélection, annulation, délégation.

    COMPORTEMENT STRICTEMENT INCHANGÉ par rapport à avant l'extraction de
    plan_exposure() -- cette fonction continue d'annuler `to_cancel`
    inconditionnellement (c'est son rôle légitime en régime de croisière,
    où to_cancel ne contient jamais que des ordres déjà réconciliés par
    des cycles précédents). Ne jamais utiliser cette fonction telle quelle
    à l'activation initiale -- voir plan_exposure() et
    grid_trading_controller.py.

    L'identité de chaque ordre matérialisé est déterminée à partir de la
    dernière occurrence remplie de la cellule.

    `churn_state_path` : transmis tel quel à plan_exposure() -- voir sa
    docstring (None par défaut = mécanisme anti-churn désactivé).
    """

    inst_id = report.config.inst_id

    plan = plan_exposure(adapter, report, k_buy, k_sell, churn_state_path=churn_state_path)

    cancelled: list[str] = []

    for client_order_id in plan.decision.to_cancel:
        adapter.cancel_order(
            inst_id,
            client_order_id=client_order_id,
        )
        cancelled.append(client_order_id)

    activation_result = GridActivationController().run(
        adapter,
        plan.narrowed_report,
        order_ids=plan.order_ids,
        require_spot_coverage=False,
    )

    return OrchestrationResult(
        tuple(cancelled),
        activation_result,
    )


def _find_cell_state(cell_states, gi: float):
    """Retourne la reconstruction correspondant exactement au niveau gi."""

    for state in cell_states:
        if abs(float(state.cell.gi) - float(gi)) < 1e-12:
            return state

    return None
