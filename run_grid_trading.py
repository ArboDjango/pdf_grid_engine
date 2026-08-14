"""Point d'entrée sécurisé — DRY_RUN / DEMO / LIVE.

Assembleur pur : ce script ne contient aucune logique économique. Il ne
recalcule jamais q, required_spot ou le treillis, ne modifie jamais
PreflightReport, et n'appelle jamais directement InitializationController,
GridActivationController, place_market_buy ou place_post_only_limit.

Pour DEMO/LIVE, l'unique appel métier est GridTradingController().run(...).
Pour DRY_RUN, l'unique appel est run_preflight(...) — strictement
équivalent à dry_run_xrp.py, qui reste inchangé et n'est jamais remplacé
par ce fichier.

Trois modes, mutuellement exclusifs, jamais de défaut implicite :

    DRY_RUN  — lecture seule. N'appelle jamais GridTradingController.
    DEMO     — chaîne complète, OkxSpotAdapter(demo=True) (environnement
               simulé OKX, x-simulated-trading: 1).
    LIVE     — chaîne complète, OkxSpotAdapter(demo=False). Nécessite
               --confirm-live ; sans ce flag, refus avant toute lecture
               d'environnement, construction d'adaptateur ou appel réseau.

Paramètres de PreflightConfig repris à l'identique de dry_run_xrp.py
(gul=P0*1.15, gll=P0*0.85, nu=nl=5, EQUIDISTANT, spacing_h_pct=
abs(maker_fee), alpha=0.95, operational_margin=0.50) — aucune valeur
modifiée.

Codes de sortie :
    0 = succès (ACTIVE en DEMO/LIVE, ou DRY_RUN terminé normalement)
    1 = refus de configuration / précondition (mode absent, LIVE sans
        confirmation, feeType != "1")
    2 = initialisation incomplète ou erreur (BUYING, PARTIALLY_FILLED,
        ERROR au niveau InitializationController, ou GridTradingState.ERROR)
    3 = activation PARTIAL
    4 = erreur technique (activation ERROR, variable d'environnement
        manquante, échec réseau/API OKX lors de la détermination de P0
        ou du maker fee, etc.)
"""

from __future__ import annotations

import argparse
import os
import sys
from decimal import Decimal

from grid_preflight import PreflightConfig, format_report, run_preflight
from grid_trading_controller import GridTradingController, GridTradingState
from initialization_controller import InitState
from grid_activation_controller import GridActivationState
from okx_spot_adapter import OkxApiError, OkxSpotAdapter
from trellis_calculator import GridGeometry


MODES = ("DRY_RUN", "DEMO", "LIVE")


class EntrypointRefusal(Exception):
    """Refus de configuration détecté avant toute action — code de sortie 1."""


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Point d'entrée DRY_RUN / DEMO / LIVE pour le grid trading OKX.",
    )
    parser.add_argument(
        "--mode", choices=MODES, required=True,
        help="Mode d'exécution. Aucun défaut : doit être fourni explicitement.",
    )
    parser.add_argument(
        "--confirm-live", action="store_true",
        help="Requis pour --mode LIVE. Sans effet dans les autres modes.",
    )
    parser.add_argument(
        "--inst-id", default="XRP-USDC",
        help="Paire à trader. Défaut : XRP-USDC (comportement de dry_run_xrp.py).",
    )
    return parser.parse_args(argv)


def check_live_confirmation(mode: str, confirm_live: bool) -> None:
    """Doit être appelée avant toute lecture d'environnement ou construction d'adaptateur."""
    if mode == "LIVE" and not confirm_live:
        raise EntrypointRefusal(
            "Mode LIVE requiert --confirm-live. Refus avant toute lecture "
            "d'environnement, construction d'adaptateur ou appel réseau."
        )


def build_adapter(mode: str) -> OkxSpotAdapter:
    """demo=True pour DEMO, demo=False pour DRY_RUN et LIVE.

    La cohérence mode/adaptateur est garantie par construction : le mode
    détermine directement l'argument demo=, jamais une vérification a
    posteriori sur un attribut interne de l'adaptateur.
    """
    return OkxSpotAdapter(
        api_key=os.environ["OKX_API_KEY"],
        secret_key=os.environ["OKX_SECRET_KEY"],
        passphrase=os.environ["OKX_PASSPHRASE"],
        demo=(mode == "DEMO"),
    )


def build_config(adapter, inst_id: str) -> PreflightConfig:
    """Paramètres repris à l'identique de dry_run_xrp.py — aucune valeur modifiée."""
    p0 = adapter.get_ticker(inst_id).last
    maker_fee_rate = abs(adapter.get_fee_rates(inst_id).maker)
    return PreflightConfig(
        inst_id=inst_id,
        gul=p0 * Decimal("1.15"),
        gll=p0 * Decimal("0.85"),
        nu=5,
        nl=5,
        p0=p0,
        geometry=GridGeometry.EQUIDISTANT,
        spacing_h_pct=maker_fee_rate,
        alpha=Decimal("0.95"),
        operational_margin=Decimal("0.50"),
    )


def determine_exit_code(mode: str, result) -> int:
    """Mapping résultat -> code de sortie. Aucune interprétation supplémentaire."""
    if mode == "DRY_RUN":
        return 0

    if result.state == GridTradingState.FEE_TYPE_REJECTED:
        return 1
    if result.state == GridTradingState.INITIALIZATION_INCOMPLETE:
        return 2
    if result.state == GridTradingState.ERROR:
        return 2
    if result.state == GridTradingState.ACTIVATED:
        if result.activation_result.state == GridActivationState.ACTIVE:
            return 0
        if result.activation_result.state == GridActivationState.PARTIAL:
            return 3
        if result.activation_result.state == GridActivationState.ERROR:
            return 4
    return 4  # état inattendu — toujours un échec technique, jamais un succès par défaut


def render_dry_run(inst_id: str, adapter, config: PreflightConfig, report) -> None:
    print(f"[MODE] DRY_RUN")
    print(f"[INSTRUMENT] {inst_id}")
    print(f"[P0] {config.p0}")
    print(f"[MAKER FEE] {config.spacing_h_pct}")
    print(format_report(report))


def render_live_or_demo(mode: str, inst_id: str, config: PreflightConfig, result) -> None:
    print(f"[MODE] {mode}")
    print(f"[INSTRUMENT] {inst_id}")
    print(f"[P0] {config.p0}")
    print(f"[MAKER FEE] {config.spacing_h_pct}")
    print(f"[FEE TYPE] {result.fee_type}")

    if result.initialization_result is not None:
        print(f"[INITIALISATION] {result.initialization_result.state.name}")
    else:
        print("[INITIALISATION] non atteinte")

    if result.activation_result is not None:
        print(f"[ACTIVATION] {result.activation_result.state.name}")
    else:
        print("[ACTIVATION] non atteinte")

    print(f"[RÉSULTAT FINAL] {result.state.name} — {result.detail}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    try:
        check_live_confirmation(args.mode, args.confirm_live)

        if args.mode == "DRY_RUN":
            adapter = build_adapter(args.mode)
            config = build_config(adapter, args.inst_id)
            report = run_preflight(adapter, config)
            render_dry_run(args.inst_id, adapter, config, report)
            return determine_exit_code(args.mode, None)

        adapter = build_adapter(args.mode)
        config = build_config(adapter, args.inst_id)
        result = GridTradingController().run(adapter, config)
        render_live_or_demo(args.mode, args.inst_id, config, result)
        return determine_exit_code(args.mode, result)

    except EntrypointRefusal as error:
        print(f"[REFUS] {error}")
        return 1
    except KeyError as error:
        print(f"[ERREUR] Variable d'environnement manquante : {error}")
        return 4
    except OkxApiError as error:
        print(f"[ERREUR] Appel OKX a échoué : {error}")
        return 4


if __name__ == "__main__":
    sys.exit(main())
