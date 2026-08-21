"""Tests pour market_reader.py.

Entièrement hors réseau : un adaptateur factice fournit des Candle
construites en mémoire. Les valeurs de référence (ATR/DI+/DI-) sont
recalculées indépendamment via la bibliothèque ta directement dans ce
fichier de tests -- jamais en réimportant la logique de market_reader.py
pour se comparer à elle-même.
"""

import math
from decimal import Decimal

import pandas as pd
import pytest
import ta

from market_reader import (
    BAR,
    CANDLES_REQUESTED,
    MarketConditions,
    MarketReaderError,
    read,
)
from okx_spot_adapter import Candle


# ---------------------------------------------------------------------------
# Jeu de bougies déterministe — mouvement de prix réel, pas plat, pour
# produire des valeurs ATR/DI non triviales (ni NaN, ni dégénérées à 0.0).
# ---------------------------------------------------------------------------

_CLOSES = [
    1.000, 1.002, 0.999, 1.003, 1.001, 1.005, 1.004, 1.008, 1.006, 1.010,
    1.009, 1.012, 1.011, 1.015, 1.013, 1.017, 1.016, 1.020, 1.018, 1.022,
    1.021, 1.025, 1.023, 1.027, 1.026, 1.030, 1.028, 1.032, 1.031, 1.035,
    1.033, 1.037, 1.036, 1.040, 1.038, 1.042, 1.041, 1.045, 1.043, 1.047,
    1.046, 1.050, 1.048, 1.052, 1.051, 1.055, 1.053, 1.057, 1.056, 1.060,
]
assert len(_CLOSES) == 50


def _reference_values(closes):
    """Recalcul indépendant, via ta directement -- jamais via market_reader.py."""
    close = pd.Series(closes)
    high = pd.Series([c + 0.003 for c in closes])
    low = pd.Series([c - 0.003 for c in closes])
    atr = float(ta.volatility.average_true_range(high, low, close, window=14).iloc[-1])
    dip = float(ta.trend.adx_pos(high, low, close, window=14).iloc[-1])
    dim = float(ta.trend.adx_neg(high, low, close, window=14).iloc[-1])
    return atr, dip, dim


def _make_candles(closes, *, chronological=True):
    """Construit des Candle à partir d'une liste de closes.

    chronological=True : ts croissants, ordre chronologique correct (le
    format attendu APRÈS normalisation par market_reader).
    chronological=False : ordre OKX réel -- la plus récente en premier
    (ts décroissants), pour tester la normalisation elle-même.
    """
    n = len(closes)
    candles = []
    for i, c in enumerate(closes):
        ts = 1_000_000 + i * 900_000  # 15 min = 900_000 ms, croissant avec i
        high = Decimal(str(c + 0.003))
        low = Decimal(str(c - 0.003))
        close = Decimal(str(c))
        candles.append(Candle(ts=ts, open=close, high=high, low=low, close=close, volume=Decimal("1000")))
    if not chronological:
        candles = list(reversed(candles))
    return tuple(candles)


class FakeAdapter:
    """Adaptateur factice, aucun réseau. Enregistre les paramètres reçus."""

    def __init__(self, candles):
        self._candles = candles
        self.calls = []

    def get_klines(self, inst_id, bar="15m", limit=50):
        self.calls.append({"inst_id": inst_id, "bar": bar, "limit": limit})
        return self._candles


# ---------------------------------------------------------------------------
# 1-4. Calculs corrects ATR / DI+ / DI- / di_ratio_15m
# ---------------------------------------------------------------------------

class TestIndicatorCalculations:
    def test_atr_norm_15m_matches_independent_reference(self):
        atr_ref, dip_ref, dim_ref = _reference_values(_CLOSES)
        adapter = FakeAdapter(_make_candles(_CLOSES))

        conditions = read(adapter, "XRP-USDC")

        expected_atr_norm = atr_ref / _CLOSES[-1]
        assert conditions.atr_norm_15m == pytest.approx(expected_atr_norm, rel=1e-9)

    def test_di_plus_contributes_correctly_to_ratio(self):
        """
        Vérifie que DI+ (pas DI-) est bien au numérateur : un jeu de
        bougies fortement haussier doit produire dip > dim, donc
        di_ratio_15m > 1.
        """
        strongly_bullish = [1.0 + i * 0.01 for i in range(50)]  # tendance haussière nette
        adapter = FakeAdapter(_make_candles(strongly_bullish))

        conditions = read(adapter, "XRP-USDC")

        assert conditions.di_ratio_15m > 1.0  # DI+ dominant, cohérent avec la hausse

    def test_di_minus_contributes_correctly_to_ratio(self):
        """Symétrique : tendance baissière nette -> di_ratio_15m < 1."""
        strongly_bearish = [2.0 - i * 0.01 for i in range(50)]
        adapter = FakeAdapter(_make_candles(strongly_bearish))

        conditions = read(adapter, "XRP-USDC")

        assert conditions.di_ratio_15m < 1.0  # DI- dominant, cohérent avec la baisse

    def test_di_ratio_15m_matches_independent_reference_exactly(self):
        atr_ref, dip_ref, dim_ref = _reference_values(_CLOSES)
        adapter = FakeAdapter(_make_candles(_CLOSES))

        conditions = read(adapter, "XRP-USDC")

        expected_ratio = dip_ref / max(dim_ref, 0.001)
        assert conditions.di_ratio_15m == pytest.approx(expected_ratio, rel=1e-9)


# ---------------------------------------------------------------------------
# 5. Normalisation de l'ordre inverse des bougies
# ---------------------------------------------------------------------------

class TestCandleOrderNormalization:
    def test_reverse_chronological_input_produces_same_result_as_chronological(self):
        """
        market_reader doit normaliser explicitement -- le résultat ne doit
        JAMAIS dépendre de l'ordre dans lequel l'adaptateur retourne les
        bougies.
        """
        adapter_chrono = FakeAdapter(_make_candles(_CLOSES, chronological=True))
        adapter_reverse = FakeAdapter(_make_candles(_CLOSES, chronological=False))

        result_chrono = read(adapter_chrono, "XRP-USDC")
        result_reverse = read(adapter_reverse, "XRP-USDC")

        assert result_chrono.atr_norm_15m == pytest.approx(result_reverse.atr_norm_15m, rel=1e-12)
        assert result_chrono.di_ratio_15m == pytest.approx(result_reverse.di_ratio_15m, rel=1e-12)

    def test_never_depends_implicitly_on_adapter_return_order(self):
        """
        Un adaptateur retournant un ordre ARBITRAIRE (ni chronologique ni
        antichronologique) doit tout de même produire le résultat correct
        -- preuve que le tri est explicite sur ts, pas une simple
        supposition sur l'ordre habituel d'OKX.
        """
        chrono_candles = list(_make_candles(_CLOSES, chronological=True))
        import random
        shuffled = chrono_candles[:]
        random.Random(42).shuffle(shuffled)
        adapter_shuffled = FakeAdapter(tuple(shuffled))
        adapter_chrono = FakeAdapter(tuple(chrono_candles))

        result_shuffled = read(adapter_shuffled, "XRP-USDC")
        result_chrono = read(adapter_chrono, "XRP-USDC")

        assert result_shuffled.atr_norm_15m == pytest.approx(result_chrono.atr_norm_15m, rel=1e-12)


# ---------------------------------------------------------------------------
# 6. Exactement 50 bougies demandées
# ---------------------------------------------------------------------------

class TestRequestParameters:
    def test_requests_exactly_50_candles(self):
        adapter = FakeAdapter(_make_candles(_CLOSES))

        read(adapter, "XRP-USDC")

        assert len(adapter.calls) == 1
        assert adapter.calls[0]["limit"] == 50
        assert adapter.calls[0]["limit"] == CANDLES_REQUESTED

    def test_requests_bar_15m(self):
        adapter = FakeAdapter(_make_candles(_CLOSES))

        read(adapter, "XRP-USDC")

        assert adapter.calls[0]["bar"] == "15m"
        assert adapter.calls[0]["bar"] == BAR

    def test_requests_correct_inst_id(self):
        adapter = FakeAdapter(_make_candles(_CLOSES))

        read(adapter, "XRP-USDC")

        assert adapter.calls[0]["inst_id"] == "XRP-USDC"


# ---------------------------------------------------------------------------
# 7. Données insuffisantes -> erreur
# ---------------------------------------------------------------------------

class TestInsufficientData:
    def test_fewer_than_50_candles_raises(self):
        adapter = FakeAdapter(_make_candles(_CLOSES[:49]))  # 49, un de moins

        with pytest.raises(MarketReaderError):
            read(adapter, "XRP-USDC")

    def test_empty_response_raises(self):
        adapter = FakeAdapter(())

        with pytest.raises(MarketReaderError):
            read(adapter, "XRP-USDC")

    def test_exactly_15_candles_would_produce_degenerate_zero_di_but_is_rejected(self):
        """
        Vérification directe de la découverte empirique documentée dans
        market_reader.py : à 15 bougies (minimum théorique window+1), DI+
        et DI- sont calculés à EXACTEMENT 0.0 par ta -- une valeur finie
        mais dégénérée. Ce test confirme que MIN_CANDLES_REQUIRED (fixé à
        50, pas 15) rejette bien ce cas AVANT que la valeur dégénérée ne
        soit produite.
        """
        adapter = FakeAdapter(_make_candles(_CLOSES[:15]))

        with pytest.raises(MarketReaderError):
            read(adapter, "XRP-USDC")


# ---------------------------------------------------------------------------
# 8. close <= 0 -> erreur
# ---------------------------------------------------------------------------

class TestInvalidClose:
    def test_zero_close_raises(self):
        closes = _CLOSES[:-1] + [0.0]
        adapter = FakeAdapter(_make_candles(closes))

        with pytest.raises(MarketReaderError):
            read(adapter, "XRP-USDC")

    def test_negative_close_raises(self):
        closes = _CLOSES[:-1] + [-1.0]
        adapter = FakeAdapter(_make_candles(closes))

        with pytest.raises(MarketReaderError):
            read(adapter, "XRP-USDC")

    def test_no_silent_fallback_to_arbitrary_value(self):
        """
        Vérification négative explicite : un close invalide ne doit
        jamais être silencieusement remplacé par une valeur arbitraire
        (contrairement au bot Gate, qui utilise un repli 0.01 -- non
        reproduit ici, une erreur explicite est préférée par défaut).
        """
        closes = _CLOSES[:-1] + [0.0]
        adapter = FakeAdapter(_make_candles(closes))

        with pytest.raises(MarketReaderError) as exc_info:
            read(adapter, "XRP-USDC")
        assert "invalide" in str(exc_info.value).lower() or "0" in str(exc_info.value)

    def test_last_close_nan_raises_market_reader_error_not_decimal_exception(self):
        """
        Régression ciblée : un dernier close NaN comparé directement via
        `<= 0` sur un Decimal lève decimal.InvalidOperation, pas une
        MarketReaderError -- c'est précisément pourquoi la vérification de
        finitude doit intervenir AVANT toute comparaison/arithmétique sur
        last_close. Ce test échouerait avec l'ancien ordre (InvalidOperation
        non interceptée par pytest.raises(MarketReaderError)).
        """
        candles = list(_make_candles(_CLOSES))
        c = candles[-1]
        candles[-1] = Candle(ts=c.ts, open=c.open, high=c.high, low=c.low, close=Decimal("NaN"), volume=c.volume)
        adapter = FakeAdapter(tuple(candles))

        with pytest.raises(MarketReaderError):
            read(adapter, "XRP-USDC")

    def test_last_close_infinity_raises_market_reader_error(self):
        """Symétrique : dernier close = Infinity -> MarketReaderError, pas une exception non gérée."""
        candles = list(_make_candles(_CLOSES))
        c = candles[-1]
        candles[-1] = Candle(ts=c.ts, open=c.open, high=c.high, low=c.low, close=Decimal("Infinity"), volume=c.volume)
        adapter = FakeAdapter(tuple(candles))

        with pytest.raises(MarketReaderError):
            read(adapter, "XRP-USDC")

    def test_last_close_negative_infinity_raises_market_reader_error(self):
        """Cas limite supplémentaire : -Infinity, techniquement <= 0 mais non fini -- doit être intercepté par la vérification de finitude, pas par la comparaison de positivité."""
        candles = list(_make_candles(_CLOSES))
        c = candles[-1]
        candles[-1] = Candle(ts=c.ts, open=c.open, high=c.high, low=c.low, close=Decimal("-Infinity"), volume=c.volume)
        adapter = FakeAdapter(tuple(candles))

        with pytest.raises(MarketReaderError):
            read(adapter, "XRP-USDC")


# ---------------------------------------------------------------------------
# 9. NaN / non-fini -> erreur
# ---------------------------------------------------------------------------

class TestNonFiniteValues:
    def test_nan_high_raises(self):
        candles = list(_make_candles(_CLOSES))
        c = candles[-1]
        candles[-1] = Candle(ts=c.ts, open=c.open, high=Decimal("NaN"), low=c.low, close=c.close, volume=c.volume)
        adapter = FakeAdapter(tuple(candles))

        with pytest.raises(MarketReaderError):
            read(adapter, "XRP-USDC")

    def test_infinite_low_raises(self):
        candles = list(_make_candles(_CLOSES))
        c = candles[0]
        candles[0] = Candle(ts=c.ts, open=c.open, high=c.high, low=Decimal("Infinity"), close=c.close, volume=c.volume)
        adapter = FakeAdapter(tuple(candles))

        with pytest.raises(MarketReaderError):
            read(adapter, "XRP-USDC")


# ---------------------------------------------------------------------------
# 10. MarketConditions contient exactement les deux champs attendus
# ---------------------------------------------------------------------------

class TestMarketConditionsShape:
    def test_exactly_two_fields(self):
        import dataclasses
        fields = {f.name for f in dataclasses.fields(MarketConditions)}
        assert fields == {"atr_norm_15m", "di_ratio_15m"}

    def test_is_frozen_dataclass(self):
        conditions = MarketConditions(atr_norm_15m=0.01, di_ratio_15m=1.5)
        with pytest.raises(Exception):
            conditions.atr_norm_15m = 0.02

    def test_never_contains_gul_gll_nu_nl_or_adx(self):
        """Vérification structurelle explicite, conforme à l'interdiction."""
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(MarketConditions)}
        forbidden = {"gul", "gll", "nu", "nl", "adx", "volume", "spread", "regime"}
        assert field_names.isdisjoint(forbidden)


# ---------------------------------------------------------------------------
# Frontière architecturale — MarketReader ne connaît aucun composant de décision
# ---------------------------------------------------------------------------

class TestGetKlinesRealAdapter:
    """
    Tests de la VRAIE méthode OkxSpotAdapter.get_klines (pas le FakeAdapter
    des tests ci-dessus) -- comble une lacune : aucun test existant
    n'exerçait le parsing réel des lignes de bougies OKX avant cet ajout.
    Transport simulé (RecordingTransport), aucun réseau réel, même patron
    que les autres tests existants de OkxSpotAdapter.
    """

    def test_parses_raw_okx_candle_rows_correctly(self):
        from datetime import datetime, timezone
        from okx_spot_adapter import Candle, OkxSpotAdapter

        class RecordingTransport:
            def __init__(self, response):
                self.response = response
                self.calls = []

            def __call__(self, method, path, params, body, headers):
                self.calls.append({"method": method, "path": path, "params": params})
                return self.response

        rows = [
            ["1700000000000", "1.000", "1.010", "0.995", "1.005", "1234.5"],
            ["1699999100000", "0.998", "1.002", "0.990", "1.000", "1100.0"],
        ]
        transport = RecordingTransport({"code": "0", "msg": "", "data": rows})
        adapter = OkxSpotAdapter("key", "secret", "pass", rest_transport=transport, now=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc))

        candles = adapter.get_klines("XRP-USDC", bar="15m", limit=50)

        assert len(candles) == 2
        assert candles[0] == Candle(ts=1700000000000, open=Decimal("1.000"), high=Decimal("1.010"), low=Decimal("0.995"), close=Decimal("1.005"), volume=Decimal("1234.5"))

    def test_uses_public_endpoint_no_signature(self):
        from datetime import datetime, timezone
        from okx_spot_adapter import OkxSpotAdapter

        class RecordingTransport:
            def __init__(self, response):
                self.response = response
                self.calls = []

            def __call__(self, method, path, params, body, headers):
                self.calls.append({"method": method, "path": path, "params": params, "headers": headers})
                return self.response

        transport = RecordingTransport({"code": "0", "msg": "", "data": []})
        adapter = OkxSpotAdapter("key", "secret", "pass", rest_transport=transport, now=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc))

        adapter.get_klines("XRP-USDC")

        call = transport.calls[0]
        assert call["method"] == "GET"
        assert call["path"] == "/api/v5/market/candles"
        assert call["params"] == {"instId": "XRP-USDC", "bar": "15m", "limit": "50"}
        # Endpoint public : aucune en-tête de signature (contrairement aux
        # endpoints privés déjà testés ailleurs, get_order/get_balances/...).
        assert "OK-ACCESS-SIGN" not in call["headers"]

    def test_okx_error_code_raises_okx_api_error(self):
        from datetime import datetime, timezone
        from okx_spot_adapter import OkxApiError, OkxSpotAdapter

        class FailingTransport:
            def __call__(self, method, path, params, body, headers):
                return {"code": "51001", "msg": "Instrument ID does not exist", "data": []}

        adapter = OkxSpotAdapter("key", "secret", "pass", rest_transport=FailingTransport(), now=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc))

        with pytest.raises(OkxApiError):
            adapter.get_klines("XRP-USDC")

    def test_incomplete_row_raises_okx_api_error(self):
        from datetime import datetime, timezone
        from okx_spot_adapter import OkxApiError, OkxSpotAdapter

        class RecordingTransport:
            def __call__(self, method, path, params, body, headers):
                return {"code": "0", "msg": "", "data": [["1700000000000", "1.0", "1.01"]]}  # incomplète, < 6 champs

        adapter = OkxSpotAdapter("key", "secret", "pass", rest_transport=RecordingTransport(), now=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc))

        with pytest.raises(OkxApiError):
            adapter.get_klines("XRP-USDC")


class TestArchitecturalBoundary:
    def test_module_never_imports_grid_components(self):
        import market_reader
        forbidden_modules = [
            "grid_preflight", "grid_activation_controller", "grid_order_orchestrator",
            "candidate_economic_validator", "grid_economic_validator", "cell",
            "grid_parameter_policy",
        ]
        for name in forbidden_modules:
            assert f"import {name}" not in open(market_reader.__file__).read()
            assert f"from {name}" not in open(market_reader.__file__).read()

    def test_only_production_import_is_okx_spot_adapter_and_ta(self):
        """Vérification positive : les seules dépendances externes de production sont ta/pandas/okx_spot_adapter."""
        import market_reader
        with open(market_reader.__file__) as f:
            content = f.read()
        import_lines = [l.strip() for l in content.splitlines() if l.strip().startswith(("import ", "from ")) and "__future__" not in l]
        allowed_prefixes = ("import math", "from dataclasses", "import pandas", "import ta", "from okx_spot_adapter")
        for line in import_lines:
            assert line.startswith(allowed_prefixes), f"Import inattendu : {line!r}"
