"""Frontière opérationnelle minimale pour OKX SPOT.

Cet adaptateur normalise les réponses OKX en ``Decimal`` et expose les
opérations REST / le flux privé ``orders`` nécessaires au moteur. Il ne
contient aucune règle de grid ni aucune interprétation de ``client_order_id``.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BUY = "BUY"
SELL = "SELL"


class OkxApiError(RuntimeError):
    """Réponse API OKX invalide ou erreur globale OKX.

    Attributes:
        data: Le contenu de response["data"] au moment de l'erreur, si la
            réponse OKX en portait un (ex. le tableau contenant
            sCode/sMsg par item, pour un rejet de POST /api/v5/trade/order
            dont l'enveloppe globale indique déjà code != "0"). None si
            l'erreur ne provient pas d'une réponse OKX exploitable (ex.
            erreur de parsing locale) -- jamais recalculé, jamais deviné,
            transporte exactement ce qu'OKX a retourné, ou rien.

            Rétrocompatible : ce paramètre est optionnel (défaut None) et
            n'affecte ni str(error) ni error.args -- tout code existant
            qui catch OkxApiError et n'utilise que le message continue de
            fonctionner à l'identique.
    """

    def __init__(self, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.data = data


class RestTransport(Protocol):
    def __call__(
        self,
        method: str,
        path: str,
        params: Mapping[str, str] | None,
        body: Mapping[str, str] | None,
        headers: Mapping[str, str],
    ) -> Mapping[str, Any]: ...


class WebSocketConnection(Protocol):
    async def send(self, message: str) -> None: ...
    async def recv(self) -> str: ...
    async def close(self) -> None: ...


WebSocketFactory = Callable[[str], Awaitable[WebSocketConnection]]


@dataclass(frozen=True)
class InstrumentRules:
    inst_id: str
    base_ccy: str
    quote_ccy: str
    tick_size: Decimal
    lot_size: Decimal
    min_size: Decimal
    min_notional: Decimal | None
    price_precision: int | None
    size_precision: int | None
    state: str


@dataclass(frozen=True)
class FeeRates:
    inst_id: str
    maker: Decimal
    taker: Decimal


@dataclass(frozen=True)
class Balances:
    base_available: Decimal
    quote_available: Decimal
    base_total: Decimal
    quote_total: Decimal


@dataclass(frozen=True)
class PlacementResult:
    accepted: bool
    order_id: str | None
    client_order_id: str
    rejection_code: str | None
    rejection_message: str | None


@dataclass(frozen=True)
class CancelResult:
    accepted: bool
    order_id: str | None
    client_order_id: str | None
    rejection_code: str | None
    rejection_message: str | None


@dataclass(frozen=True)
class OrderSnapshot:
    order_id: str
    client_order_id: str
    inst_id: str
    side: str
    price: Decimal
    quantity: Decimal
    accumulated_fill_quantity: Decimal
    state: str
    order_type: str
    created_at: int | None
    updated_at: int | None


@dataclass(frozen=True)
class Fill:
    trade_id: str
    order_id: str
    client_order_id: str
    inst_id: str
    side: str
    fill_price: Decimal
    fill_quantity: Decimal
    accumulated_fill_quantity: Decimal | None
    order_state: str
    filled_at: int | None
    fee: Decimal | None
    fee_currency: str | None
    rebate: Decimal | None
    rebate_currency: str | None


@dataclass(frozen=True)
class Ticker:
    inst_id: str
    last: Decimal
    ts: int


@dataclass(frozen=True)
class PriceLimit:
    """Bande de prix dynamique OKX -- distincte de InstrumentRules (statique)
    et de Balances. Recalculée en continu par OKX (index + prime moyenne
    sur 1 minute), jamais mise en cache au-delà d'un appel."""
    inst_id: str
    buy_lmt: Decimal
    sell_lmt: Decimal
    ts: int


@dataclass(frozen=True)
class Candle:
    """Une bougie OKX publique. Miroir fidèle de la ligne brute
    [ts, o, h, l, c, vol, ...] retournée par GET /api/v5/market/candles."""
    ts: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass(frozen=True)
class AccountConfig:
    fee_type: str


def _decimal(value: Any, *, optional: bool = False) -> Decimal | None:
    if value in (None, ""):
        if optional:
            return None
        raise OkxApiError("Valeur décimale OKX absente")
    try:
        return Decimal(str(value))
    except Exception as error:  # DecimalException subclasses vary by Python version.
        raise OkxApiError(f"Valeur décimale OKX invalide : {value!r}") from error


def _timestamp(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value))
    except ValueError as error:
        raise OkxApiError(f"Horodatage OKX invalide : {value!r}") from error


def _side(value: str) -> str:
    lowered = value.lower()
    if lowered == "buy":
        return BUY
    if lowered == "sell":
        return SELL
    raise OkxApiError(f"Côté OKX inconnu : {value!r}")


class OkxSpotAdapter:
    """Adaptateur OKX SPOT sans état métier ni logique économique."""

    REST_URL = "https://eea.okx.com"
    PRIVATE_WS_URL = "wss://wseea.okx.com:8443/ws/v5/private"

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        passphrase: str,
        *,
        demo: bool = False,
        rest_transport: RestTransport | None = None,
        websocket_factory: WebSocketFactory | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._passphrase = passphrase
        self._demo = demo
        self._transport = rest_transport or self._urllib_transport
        self._websocket_factory = websocket_factory or self._default_websocket_factory
        self._now = now or (lambda: datetime.now(timezone.utc))

    def get_instrument(self, inst_id: str) -> InstrumentRules:
        """Lit les caractéristiques de l'instrument. Endpoint public OKX : aucune signature envoyée."""
        response = self._transport("GET", "/api/v5/public/instruments", {"instType": "SPOT", "instId": inst_id}, None, {})
        if response.get("code") != "0":
            raise OkxApiError(f"Erreur OKX {response.get('code')}: {response.get('msg')}")
        data = response.get("data", [])
        if len(data) != 1:
            raise OkxApiError(f"Réponse OKX : un élément attendu, {len(data)} reçu")
        item = data[0]
        return InstrumentRules(
            inst_id=item["instId"], base_ccy=item["baseCcy"], quote_ccy=item["quoteCcy"],
            tick_size=_required_decimal(item, "tickSz"), lot_size=_required_decimal(item, "lotSz"),
            min_size=_required_decimal(item, "minSz"), min_notional=_decimal(item.get("minNotional"), optional=True),
            price_precision=_optional_int(item.get("pxPrecision")), size_precision=_optional_int(item.get("szPrecision")),
            state=item["state"],
        )

    def get_fee_rates(self, inst_id: str) -> FeeRates:
        item = self._one("GET", "/api/v5/account/trade-fee", {"instType": "SPOT", "instId": inst_id})
        group = item.get("feeGroup", [{}])[0]
        return FeeRates(inst_id=inst_id, maker=_required_decimal(group, "maker"), taker=_required_decimal(group, "taker"))

    def get_account_config(self) -> AccountConfig:
        """Précondition de lancement : feeType doit valoir "1" (frais Spot BUY
        payés en devise de cotation) avant tout achat MARKET. Lecture pure,
        aucun appel à set-fee-type n'est jamais effectué par ce module."""
        item = self._one("GET", "/api/v5/account/config")
        return AccountConfig(fee_type=item["feeType"])

    def get_balances(self, base_ccy: str, quote_ccy: str) -> Balances:
        item = self._one("GET", "/api/v5/account/balance", {"ccy": f"{base_ccy},{quote_ccy}"})
        details = {detail["ccy"]: detail for detail in item.get("details", [])}
        base = details.get(base_ccy, {})
        quote = details.get(quote_ccy, {})
        return Balances(
            base_available=_decimal(base.get("availBal", "0")) or Decimal("0"),
            quote_available=_decimal(quote.get("availBal", "0")) or Decimal("0"),
            base_total=_decimal(base.get("cashBal", "0")) or Decimal("0"),
            quote_total=_decimal(quote.get("cashBal", "0")) or Decimal("0"),
        )

    def get_ticker(self, inst_id: str) -> Ticker:
        """Lit le dernier prix négocié. Endpoint public OKX : aucune signature envoyée."""
        response = self._transport("GET", "/api/v5/market/ticker", {"instId": inst_id}, None, {})
        if response.get("code") != "0":
            raise OkxApiError(f"Erreur OKX {response.get('code')}: {response.get('msg')}")
        data = response.get("data", [])
        if len(data) != 1:
            raise OkxApiError(f"Réponse OKX : un élément attendu, {len(data)} reçu")
        item = data[0]
        return Ticker(inst_id=item.get("instId", inst_id), last=_required_decimal(item, "last"), ts=_required_timestamp(item, "ts"))

    def get_price_limit(self, inst_id: str) -> PriceLimit:
        """Lit la bande de prix dynamique OKX. Endpoint public : aucune signature envoyée.

        Distinct de InstrumentRules (statique) et de Balances -- recalculé
        en continu par OKX (index + prime, cf. documentation officielle),
        jamais mis en cache au-delà d'un appel."""
        response = self._transport("GET", "/api/v5/public/price-limit", {"instId": inst_id}, None, {})
        if response.get("code") != "0":
            raise OkxApiError(f"Erreur OKX {response.get('code')}: {response.get('msg')}")
        data = response.get("data", [])
        if len(data) != 1:
            raise OkxApiError(f"Réponse OKX : un élément attendu, {len(data)} reçu")
        item = data[0]
        return PriceLimit(
            inst_id=item.get("instId", inst_id),
            buy_lmt=_required_decimal(item, "buyLmt"),
            sell_lmt=_required_decimal(item, "sellLmt"),
            ts=_required_timestamp(item, "ts"),
        )

    def get_klines(self, inst_id: str, bar: str = "15m", limit: int = 50) -> tuple[Candle, ...]:
        """Lit des bougies publiques OKX. Endpoint public : aucune signature envoyée.

        Lecture ponctuelle — aucun stockage, aucun historique conservé par
        cette méthode au-delà de l'appel. L'ordre de retour d'OKX
        (antichronologique, la plus récente en premier) n'est jamais
        normalisé ici — c'est la responsabilité de l'appelant (voir
        market_reader.py), pour que cette méthode reste un miroir fidèle
        de la réponse brute, sans decision implicite sur l'ordre attendu.
        """
        response = self._transport("GET", "/api/v5/market/candles", {"instId": inst_id, "bar": bar, "limit": str(limit)}, None, {})
        if response.get("code") != "0":
            raise OkxApiError(f"Erreur OKX {response.get('code')}: {response.get('msg')}")
        rows = response.get("data", [])
        return tuple(_candle(row) for row in rows)

    def place_post_only_limit(self, inst_id: str, side: str, price: Decimal, quantity: Decimal, client_order_id: str) -> PlacementResult:
        payload = {"instId": inst_id, "tdMode": "cash", "side": _to_okx_side(side), "ordType": "post_only", "px": str(price), "sz": str(quantity), "clOrdId": client_order_id}
        item = self._one("POST", "/api/v5/trade/order", body=payload, allow_item_error=True)
        accepted = item.get("sCode") == "0"
        return PlacementResult(accepted, item.get("ordId") or None, item.get("clOrdId", client_order_id), None if accepted else item.get("sCode"), None if accepted else item.get("sMsg"))

    def place_market_buy(self, inst_id: str, quantity: Decimal, client_order_id: str) -> PlacementResult:
        """Achat marketable, quantité exprimée en actif de base (tgtCcy=base_ccy).

        Distinct des ordres de grille (jamais post_only) : réservé à
        l'initialisation patrimoniale, où l'exécution immédiate est requise
        pour débloquer le placement des SELL. La quantité passée doit
        provenir de report.required_spot (grid_preflight), jamais recalculée
        ici.
        """
        payload = {"instId": inst_id, "tdMode": "cash", "side": "buy", "ordType": "market", "sz": str(quantity), "tgtCcy": "base_ccy", "clOrdId": client_order_id}
        item = self._one("POST", "/api/v5/trade/order", body=payload, allow_item_error=True)
        accepted = item.get("sCode") == "0"
        return PlacementResult(accepted, item.get("ordId") or None, item.get("clOrdId", client_order_id), None if accepted else item.get("sCode"), None if accepted else item.get("sMsg"))

    def list_open_orders(self, inst_id: str, client_order_id: str | None = None) -> tuple[OrderSnapshot, ...]:
        items = self._data("GET", "/api/v5/trade/orders-pending", {"instType": "SPOT", "instId": inst_id})
        orders = tuple(_order(item) for item in items)
        return orders if client_order_id is None else tuple(order for order in orders if order.client_order_id == client_order_id)

    def get_order(self, inst_id: str, *, order_id: str | None = None, client_order_id: str | None = None) -> OrderSnapshot:
        if (order_id is None) == (client_order_id is None):
            raise ValueError("Fournir exactement order_id ou client_order_id")
        params = {"instId": inst_id}
        params["ordId" if order_id is not None else "clOrdId"] = order_id or client_order_id or ""
        return _order(self._one("GET", "/api/v5/trade/order", params))

    def cancel_order(self, inst_id: str, *, order_id: str | None = None, client_order_id: str | None = None) -> CancelResult:
        if (order_id is None) == (client_order_id is None):
            raise ValueError("Fournir exactement order_id ou client_order_id")
        body = {"instId": inst_id}
        body["ordId" if order_id is not None else "clOrdId"] = order_id or client_order_id or ""
        item = self._one("POST", "/api/v5/trade/cancel-order", body=body, allow_item_error=True)
        accepted = item.get("sCode") == "0"
        return CancelResult(accepted, item.get("ordId") or None, item.get("clOrdId") or None, None if accepted else item.get("sCode"), None if accepted else item.get("sMsg"))

    def list_fills(self, inst_id: str, *, order_id: str | None = None) -> tuple[Fill, ...]:
        params = {"instType": "SPOT", "instId": inst_id}
        if order_id is not None:
            params["ordId"] = order_id
        return tuple(_fill(item) for item in self._data("GET", "/api/v5/trade/fills", params))

    async def listen_orders(self, inst_id: str, on_update: Callable[[OrderSnapshot], Awaitable[None]], *, reconnect_attempts: int = 1) -> None:
        """Écoute le canal privé orders; après coupure, rattrape les ordres ouverts REST."""
        attempts = 0
        while True:
            connection: WebSocketConnection | None = None
            try:
                connection = await self._websocket_factory(self.PRIVATE_WS_URL)
                await connection.send(json.dumps({"op": "login", "args": [self._ws_login()]}))
                await connection.send(json.dumps({"op": "subscribe", "args": [{"channel": "orders", "instType": "SPOT", "instId": inst_id}]}))
                while True:
                    message = json.loads(await connection.recv())
                    for item in message.get("data", []):
                        await on_update(_order(item))
            except Exception:
                if attempts >= reconnect_attempts:
                    raise
                attempts += 1
                for order in self.list_open_orders(inst_id):
                    await on_update(order)
            finally:
                if connection is not None:
                    await connection.close()

    def _data(self, method: str, path: str, params: Mapping[str, str] | None = None, body: Mapping[str, str] | None = None) -> Sequence[Mapping[str, Any]]:
        payload = self._request(method, path, params, body)
        return payload.get("data", [])

    def _one(self, method: str, path: str, params: Mapping[str, str] | None = None, body: Mapping[str, str] | None = None, allow_item_error: bool = False) -> Mapping[str, Any]:
        data = self._data(method, path, params, body)
        if len(data) != 1:
            raise OkxApiError(f"Réponse OKX : un élément attendu, {len(data)} reçu")
        item = data[0]
        if not allow_item_error and item.get("sCode") not in (None, "0"):
            raise OkxApiError(f"Erreur OKX {item.get('sCode')}: {item.get('sMsg')}")
        return item

    def _request(self, method: str, path: str, params: Mapping[str, str] | None, body: Mapping[str, str] | None) -> Mapping[str, Any]:
        timestamp = self._now().isoformat(timespec="milliseconds").replace("+00:00", "Z")
        query = f"?{urlencode(params)}" if params else ""
        body_text = json.dumps(body, separators=(",", ":")) if body else ""
        signature = _sign(self._secret_key, f"{timestamp}{method}{path}{query}{body_text}")
        headers = {"OK-ACCESS-KEY": self._api_key, "OK-ACCESS-SIGN": signature, "OK-ACCESS-TIMESTAMP": timestamp, "OK-ACCESS-PASSPHRASE": self._passphrase, "Content-Type": "application/json"}
        if self._demo:
            headers["x-simulated-trading"] = "1"
        response = self._transport(method, path, params, body, headers)
        if response.get("code") != "0":
            raise OkxApiError(f"Erreur OKX {response.get('code')}: {response.get('msg')}", data=response.get("data"))
        return response

    def _ws_login(self) -> Mapping[str, str]:
        timestamp = str(int(self._now().timestamp()))
        return {"apiKey": self._api_key, "passphrase": self._passphrase, "timestamp": timestamp, "sign": _sign(self._secret_key, f"{timestamp}GET/users/self/verify")}

    def _urllib_transport(self, method: str, path: str, params: Mapping[str, str] | None, body: Mapping[str, str] | None, headers: Mapping[str, str]) -> Mapping[str, Any]:
        query = f"?{urlencode(params)}" if params else ""
        data = json.dumps(body, separators=(",", ":")).encode() if body else None
        final_headers = {"User-Agent": "pdf_grid_engine/1.0"}
        final_headers.update(headers)
        request = Request(f"{self.REST_URL}{path}{query}", data=data, headers=final_headers, method=method)
        try:
            with urlopen(request, timeout=15) as response:  # nosec B310 - fixed HTTPS endpoint
                return json.loads(response.read().decode())
        except HTTPError as error:
            # OKX répond parfois avec un statut HTTP d'erreur (400/401/403/...)
            # au lieu de HTTP 200 + code OKX en erreur. Dans les deux cas, le
            # résultat doit rester une enveloppe exploitable par les
            # vérifications déjà en place (_request, get_ticker,
            # get_instrument), jamais une HTTPError brute.
            return _okx_error_envelope(error)

    async def _default_websocket_factory(self, url: str) -> WebSocketConnection:
        try:
            import websockets
        except ImportError as error:
            raise RuntimeError("Le paquet optionnel websockets est requis pour le flux privé OKX") from error
        return await websockets.connect(url)  # type: ignore[no-any-return]


def _sign(secret: str, payload: str) -> str:
    return base64.b64encode(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest()).decode()


def _okx_error_envelope(error: HTTPError) -> Mapping[str, Any]:
    """Convertit une HTTPError (400/401/403/500/...) en enveloppe au format
    OKX ({"code", "msg", "data"}), exploitable telle quelle par les
    vérifications déjà en place (_request, get_ticker, get_instrument), qui
    font toutes `if response.get("code") != "0": raise OkxApiError(...)`.

    Si le corps de la réponse est un JSON OKX exploitable (contient "code"),
    il est retourné tel quel — le message d'erreur final reprend alors
    exactement le code/msg réel d'OKX. Sinon, une enveloppe de repli est
    construite avec le seul statut HTTP, sans jamais exposer les en-têtes
    de la requête (donc jamais de clé API, secret ou passphrase, qui n'y
    figurent jamais).
    """
    try:
        raw = error.read()
    except Exception:
        raw = b""
    parsed: Any = None
    if raw:
        try:
            parsed = json.loads(raw.decode())
        except (UnicodeDecodeError, json.JSONDecodeError):
            parsed = None
    if isinstance(parsed, Mapping) and "code" in parsed:
        return parsed
    return {"code": str(error.code), "msg": f"HTTP {error.code} sans corps JSON OKX exploitable", "data": []}


def _required_decimal(item: Mapping[str, Any], key: str) -> Decimal:
    value = _decimal(item.get(key))
    assert value is not None
    return value


def _required_timestamp(item: Mapping[str, Any], key: str) -> int:
    value = _timestamp(item.get(key))
    if value is None:
        raise OkxApiError(f"Horodatage OKX absent : {key!r}")
    return value


def _optional_int(value: Any) -> int | None:
    return None if value in (None, "") else int(str(value))


def _candle(row: Sequence[Any]) -> "Candle":
    """Parse une ligne brute [ts, o, h, l, c, vol, ...] de GET /api/v5/market/candles."""
    if len(row) < 6:
        raise OkxApiError(f"Ligne de bougie OKX incomplète : {row!r}")
    ts = _timestamp(row[0])
    if ts is None:
        raise OkxApiError(f"Horodatage de bougie OKX absent : {row!r}")
    return Candle(
        ts=ts,
        open=_decimal(row[1]),
        high=_decimal(row[2]),
        low=_decimal(row[3]),
        close=_decimal(row[4]),
        volume=_decimal(row[5]),
    )


def _to_okx_side(side: str) -> str:
    if side == BUY:
        return "buy"
    if side == SELL:
        return "sell"
    raise ValueError(f"Côté moteur inconnu : {side!r}")


def _order_price(item: Mapping[str, Any]) -> Decimal:
    """Prix de l'ordre. Pour un ordre LIMIT, px est toujours renseigné et
    prime. Pour un ordre MARKET, OKX renvoie px="" (aucun prix limite
    n'a de sens) et renseigne à la place avgPx (prix moyen d'exécution).
    Ce repli ne concerne que la lecture d'un ordre déjà passé (get_order,
    list_open_orders, listen_orders) — il n'affecte jamais la construction
    des ordres de grille (toujours LIMIT, px toujours fourni par
    place_post_only_limit) ni place_market_buy (qui n'envoie jamais px)."""
    px = item.get("px")
    if px not in (None, ""):
        return _required_decimal(item, "px")
    return _required_decimal(item, "avgPx")


def _order(item: Mapping[str, Any]) -> OrderSnapshot:
    return OrderSnapshot(item["ordId"], item.get("clOrdId", ""), item["instId"], _side(item["side"]), _order_price(item), _required_decimal(item, "sz"), _required_decimal(item, "accFillSz"), item["state"], item["ordType"], _timestamp(item.get("cTime")), _timestamp(item.get("uTime")))


def _fill(item: Mapping[str, Any]) -> Fill:
    return Fill(item["tradeId"], item["ordId"], item.get("clOrdId", ""), item["instId"], _side(item["side"]), _required_decimal(item, "fillPx"), _required_decimal(item, "fillSz"), _decimal(item.get("accFillSz"), optional=True), item.get("state", ""), _timestamp(item.get("fillTime")), _decimal(item.get("fee"), optional=True), item.get("feeCcy") or None, _decimal(item.get("rebate"), optional=True), item.get("rebateCcy") or None)
