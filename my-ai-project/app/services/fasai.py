"""Fasai (ฟ้าใส) agent logic: turn timing, session state, and tool handling."""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from google.genai import types

from app.config import settings
from app.services import address_book
from app.services.knowledge_base import get_active_knowledge_base_promotions
from app.services.site_content import get_site_content

logger = logging.getLogger(__name__)

PERSONA_NAME = "ฟ้าใส"
LISTEN_DELAY_S = 1.8
HELLO_AFTER_S = 2.0
HANGUP_AFTER_S = 1.0
HELLO_REPLY_TIMEOUT_S = 2.0
CALL_TIME_LIMIT_S = 600
CALL_TIME_WARNING_S = 585
CALL_ADDRESS_EXTENSION_S = 120
CALL_CLOSE_GRACE_S = 15
PCM_BYTES_PER_SECOND = 48000  # Gemini Live output: 24 kHz, 16-bit, mono
GREETING_TEXT = "ออลเดย์ออย สวัสดีค่ะ ฟ้าใสรับสายค่ะ สนใจงาดำออลเดย์หรือสอบถามสถานะออเดอร์คะ"
CLOSING_TEXT = "ขอบคุณที่ใช้บริการนะคะ สวัสดีค่ะ"
HANGUP_TEXT = "ฟ้าใสขออนุญาตวางสายค่ะ"
HOME_URL = "/"
FREE_TEXT_FIELDS = ("recipient_name", "house_no", "phone")
SHIPPING_FIELDS = (
    "recipient_name",
    "house_no",
    "subdistrict",
    "district",
    "province",
    "postcode",
    "phone",
)
MAX_LINE_QUANTITY = 10

WAIT_NOTICE_PROMPT = (
    "[ระบบ] ลูกค้าพูดจบแล้วแต่ยังประมวลผลคำตอบไม่ได้ "
    "ให้พูดเพียงประโยคเดียวว่า “รอซักครู่นะคะลูกค้า” แล้วหยุดพูด"
)
ASK_REPEAT_PROMPT = (
    "[ระบบ] ยังประมวลผลคำพูดของลูกค้าไม่ได้ "
    "ให้พูดเพียงประโยคเดียวว่า “รบกวนลูกค้าแจ้งอีกครั้งได้ไหมคะ” แล้วหยุดพูด"
)
UNANSWERED_HINT = (
    "[ระบบ] คำพูดของลูกค้าที่ยังไม่ได้ตอบ: “{text}” "
    "เมื่อลูกค้าพูดครั้งถัดไป ให้ตีความรวมกับข้อความนี้เป็นชุดเดียวกันก่อนตอบ"
)
GREETING_PROMPT = (
    "[ระบบ] มีสายเข้า ให้พูดประโยคต่อไปนี้ตรงตัวทุกคำ ห้ามเพิ่ม ลด หรือเปลี่ยนคำ "
    f"แล้วหยุดรอลูกค้า: “{GREETING_TEXT}”"
)
HELLO_PROMPT = (
    "[ระบบ] ลูกค้าเงียบไป ให้พูดตรงตัวทุกคำว่า "
    "“ฮัลโหลคะ ฮัลโหลคะ ลูกค้าอยู่ในสายไหมคะ” แล้วหยุดรอลูกค้า"
)
HANGUP_PROMPT = (
    "[ระบบ] ลูกค้ายังเงียบอยู่ ให้พูดตรงตัวทุกคำว่า "
    f"“{HANGUP_TEXT} {CLOSING_TEXT}” แล้วหยุดพูด"
)
CALL_TIME_LIMIT_PROMPT = (
    "[ระบบ] ใกล้ครบเวลาสนทนาแล้ว ให้แจ้งสั้น ๆ ว่าลูกค้าดูโปรหรือสั่งซื้อได้ที่หน้าร้าน "
    "แล้วเรียก end_call จากนั้นพูดประโยคปิดท้ายตามที่กำหนด ห้ามถามหรือเสนอขายต่อ"
)
CALL_ADDRESS_EXTENSION_PROMPT = (
    "[ระบบ] ใกล้ครบเวลาสนทนาแล้ว ข้อมูลสั่งซื้อและที่อยู่ครบ "
    "ให้ทวนที่อยู่ที่บันทึกไว้และถามยืนยันให้จบโดยเร็ว ห้ามเริ่มหัวข้อใหม่"
)


class TurnWatchdog:
    """Tracks an unanswered customer turn.

    Live VAD already waits LISTEN_DELAY_S of silence before the model answers.
    If the model has not started answering one further LISTEN_DELAY_S later we
    ask the customer to wait, and one window after that we ask them to repeat.
    """

    def __init__(self, delay_s: float = LISTEN_DELAY_S) -> None:
        self.delay_s = delay_s
        self._last_speech: float | None = None
        self._notice_sent = False
        self._repeat_sent = False
        self._filler_turn = False
        self._pending: list[str] = []

    @property
    def unanswered_text(self) -> str:
        return "".join(self._pending).strip()

    def on_user_speech(self, text: str, now: float) -> None:
        self._pending.append(text)
        self._last_speech = now
        self._notice_sent = False
        self._repeat_sent = False

    def on_model_output(self) -> None:
        if self._filler_turn:
            return
        self._last_speech = None
        self._pending.clear()

    def on_turn_complete(self) -> None:
        self._filler_turn = False

    def poll(self, now: float) -> str | None:
        if self._last_speech is None:
            return None
        elapsed = now - self._last_speech
        if not self._notice_sent and elapsed >= 2 * self.delay_s:
            self._notice_sent = True
            self._filler_turn = True
            return "wait_notice"
        if self._notice_sent and not self._repeat_sent and elapsed >= 3 * self.delay_s:
            self._repeat_sent = True
            self._filler_turn = True
            return "ask_repeat"
        return None


class SilenceGuard:
    """Detects a customer who has stopped answering after Fasai finished speaking.

    Silence is measured from the estimated end of Fasai's audio playback on the
    customer's device. After HELLO_AFTER_S we say hello; if the customer is still
    silent HANGUP_AFTER_S after that, we hang up.
    """

    def __init__(
        self,
        hello_after_s: float = HELLO_AFTER_S,
        hangup_after_s: float = HANGUP_AFTER_S,
        hello_reply_timeout_s: float = HELLO_REPLY_TIMEOUT_S,
    ) -> None:
        self.hello_after_s = hello_after_s
        self.hangup_after_s = hangup_after_s
        self.hello_reply_timeout_s = hello_reply_timeout_s
        self.stage = 0  # 0 listening, 1 hello sent, 2 hangup sent
        self._agent_busy = False
        self._playback_end = 0.0
        self._quiet_since: float | None = None
        self._hello_sent_at = 0.0

    def on_agent_audio(self, now: float, pcm_bytes: int) -> None:
        self._agent_busy = True
        self._quiet_since = None
        self._playback_end = max(self._playback_end, now) + pcm_bytes / PCM_BYTES_PER_SECOND

    def on_agent_turn_complete(self, now: float) -> None:
        self._agent_busy = False
        self._quiet_since = max(now, self._playback_end)

    def on_customer_activity(self) -> None:
        self.stage = 0
        self._quiet_since = None

    def poll(self, now: float) -> str | None:
        if self.stage >= 2 or self._agent_busy:
            return None
        if self._quiet_since is None:
            if self.stage == 1 and now - self._hello_sent_at >= self.hello_reply_timeout_s:
                self.stage = 2
                return "hangup"
            return None
        elapsed = now - self._quiet_since
        if self.stage == 0 and elapsed >= self.hello_after_s:
            self.stage = 1
            self._quiet_since = None
            self._hello_sent_at = now
            return "hello"
        if self.stage == 1 and elapsed >= self.hangup_after_s:
            self.stage = 2
            return "hangup"
        return None


@dataclass
class AgentSession:
    ticket_no: str | None = None
    cart: dict[str, int] = field(default_factory=dict)
    transcript: list[dict[str, str]] = field(default_factory=list)
    pending_redirect: str | None = None
    close_reason: str | None = None
    shipping: dict[str, str] = field(default_factory=dict)
    order_confirmed: bool = False
    callback_logged: bool = False
    sales_discount_thb: int = 0

    def add_transcript(self, role: str, text: str, at: str) -> None:
        if not text.strip():
            return
        if self.transcript and self.transcript[-1]["role"] == role:
            self.transcript[-1]["text"] += text
        else:
            self.transcript.append({"role": role, "text": text, "at": at})

    def cart_lines(self, products: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        lines = []
        for product_id, quantity in self.cart.items():
            product = products.get(product_id, {})
            line_total = _fasai_offer_price(product, quantity)
            if line_total is None:
                unit_price = product.get("sale_price") or product.get("regular_price") or 0
                line_total = unit_price * quantity
            if self.sales_discount_thb and _is_fasai_product(product) and quantity == 3:
                line_total -= self.sales_discount_thb
            lines.append(
                {
                    "product_id": product_id,
                    "name": product.get("name", product_id),
                    "quantity": quantity,
                    "unit_price": round(line_total / quantity, 2),
                    "line_total": line_total,
                }
            )
        return lines


def _is_fasai_product(product: dict[str, Any]) -> bool:
    name = "".join(str(product.get("name", "")).casefold().split())
    return "งาดำ" in name and ("คอลลาเจนไทพ์ทู" in name or "typeii" in name)


def _fasai_offer_price(product: dict[str, Any], quantity: int) -> int | None:
    if not _is_fasai_product(product):
        return None
    promotions = get_active_knowledge_base_promotions()["promotions"]
    for promotion in promotions:
        if (
            "งาดำ" in str(promotion.get("product", ""))
            and promotion.get("quantity") == quantity
        ):
            return int(promotion["price_thb"])
    return None


def published_products() -> dict[str, dict[str, Any]]:
    try:
        products = get_site_content().get("products", [])
    except Exception:
        logger.warning("Could not load the published catalog for Fasai.", exc_info=True)
        return {}
    return {p["product_id"]: p for p in products if p.get("active")}


def catalog_prompt() -> str:
    products = published_products()
    if not products:
        return "รายการสินค้าที่หยิบใส่ตะกร้าได้: ยังไม่มีสินค้าที่เผยแพร่ ห้ามเสนอขายหรือเรียก add_to_cart"
    lines = []
    for product in products.values():
        price = product.get("sale_price") or product.get("regular_price")
        lines.append(
            f"- รหัส {product['product_id']} | {product['name']} | ราคา {price} บาท | "
            f"คงเหลือ {product.get('stock_quantity', 0)} | รายละเอียด: {product.get('description', '')}"
        )
    return "รายการสินค้าที่หยิบใส่ตะกร้าได้ (ข้อมูลต้นทางเท่านั้น):\n" + "\n".join(lines)


def tool_config() -> list[types.Tool]:
    schema = types.Schema
    kind = types.Type
    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="add_to_cart",
                    description=(
                        "หยิบสินค้าใส่ตะกร้าเมื่อลูกค้ายืนยันซื้อแล้ว "
                        "กรณีลูกค้ายอมรับส่วนลดที่ได้รับอนุญาต ให้ส่ง discount_thb (ไม่เกิน 50) "
                        "พร้อมการสั่งโปร 3 กระปุกเท่านั้น"
                    ),
                    parameters=schema(
                        type=kind.OBJECT,
                        properties={
                            "product_id": schema(type=kind.STRING, description="รหัสสินค้าจากรายการ"),
                            "quantity": schema(type=kind.INTEGER, description="จำนวนที่ลูกค้าต้องการ"),
                            "discount_thb": schema(
                                type=kind.INTEGER,
                                description="ส่วนลดเพิ่มที่ลูกค้ายอมรับแล้ว ไม่เกิน 50 บาท ใช้กับโปร 3 กระปุกเท่านั้น",
                            ),
                        },
                        required=["product_id", "quantity"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="save_shipping_field",
                    description="บันทึกข้อมูลจัดส่งหนึ่งช่องที่ลูกค้าพูด เรียกทันทีเมื่อได้คำตอบแต่ละช่อง",
                    parameters=schema(
                        type=kind.OBJECT,
                        properties={
                            "field": schema(
                                type=kind.STRING,
                                enum=list(FREE_TEXT_FIELDS),
                                description="ชื่อช่อง",
                            ),
                            "value": schema(type=kind.STRING, description="ค่าที่ลูกค้าพูด"),
                        },
                        required=["field", "value"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="verify_address",
                    description=(
                        "ตรวจตำบล อำเภอ จังหวัด รหัสไปรษณีย์ที่ลูกค้าพูดกับฐานข้อมูลที่อยู่ "
                        "ถ้าตรงจะบันทึกให้ทันที ถ้าไม่ตรงจะคืนรายการตัวเลือกตามรหัสไปรษณีย์"
                    ),
                    parameters=schema(
                        type=kind.OBJECT,
                        properties={
                            "subdistrict": schema(type=kind.STRING, description="ตำบล/แขวง"),
                            "district": schema(type=kind.STRING, description="อำเภอ/เขต"),
                            "province": schema(type=kind.STRING, description="จังหวัด"),
                            "postcode": schema(type=kind.STRING, description="รหัสไปรษณีย์ 5 หลัก"),
                        },
                        required=["subdistrict", "district", "province", "postcode"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="lookup_postcode",
                    description="ค้นจังหวัด อำเภอ ตำบล จากรหัสไปรษณีย์ ใช้เมื่อ verify_address ไม่ตรง",
                    parameters=schema(
                        type=kind.OBJECT,
                        properties={"postcode": schema(type=kind.STRING, description="รหัสไปรษณีย์ 5 หลัก")},
                        required=["postcode"],
                    ),
                ),
                types.FunctionDeclaration(
                    name="end_call",
                    description=(
                        "ใช้เมื่อลูกค้าบอกว่าไม่ต้องการอะไรเพิ่มหรือขอวางสาย "
                        "เรียกก่อนพูดประโยคปิดท้าย ระบบจะตัดสายหลังพูดจบ"
                    ),
                ),
                types.FunctionDeclaration(
                    name="confirm_order",
                    description="ยืนยันคำสั่งซื้อหลังลูกค้ายืนยันว่าที่อยู่ถูกต้อง ระบบจะตัดสายหลังพูดจบ",
                ),
                types.FunctionDeclaration(
                    name="log_callback",
                    description="บันทึกเรื่องให้หน่วยงานที่เกี่ยวข้องติดต่อกลับ เมื่อข้อมูลไม่เพียงพอจะตอบ",
                    parameters=schema(
                        type=kind.OBJECT,
                        properties={
                            "reason": schema(type=kind.STRING, description="หัวข้อที่ลูกค้าสอบถาม"),
                            "details": schema(type=kind.STRING, description="รายละเอียดที่ลูกค้าแจ้ง"),
                        },
                        required=["reason"],
                    ),
                ),
            ]
        )
    ]


def run_tool(
    name: str,
    args: dict[str, Any],
    session: AgentSession,
    *,
    save_callback: Any,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Execute a model tool call. Returns (response for the model, event for the browser)."""
    products = published_products()
    if name == "add_to_cart":
        product_id = str(args.get("product_id", "")).strip()
        product = products.get(product_id)
        try:
            quantity = int(args.get("quantity", 1))
        except (TypeError, ValueError):
            quantity = 0
        if product is None:
            return {"ok": False, "error": "ไม่พบสินค้านี้ในรายการที่เผยแพร่"}, None
        if not 1 <= quantity <= MAX_LINE_QUANTITY:
            return {"ok": False, "error": f"จำนวนต้องอยู่ระหว่าง 1 ถึง {MAX_LINE_QUANTITY}"}, None
        if _is_fasai_product(product) and (quantity > 3 or session.cart.get(product_id, 0) + quantity > 3):
            return {"ok": False, "error": "โปรโมชันมีตัวเลือก 1, 2 หรือ 3 กระปุกต่อคำสั่งซื้อ"}, None
        discount = args.get("discount_thb", 0)
        if isinstance(discount, bool) or not isinstance(discount, int) or not 0 <= discount <= 50:
            return {"ok": False, "error": "ส่วนลดเพิ่มต้องเป็นจำนวนเต็มตั้งแต่ 0 ถึง 50 บาท"}, None
        if discount and (
            quantity != 3
            or not _is_fasai_product(product)
            or session.cart
            or session.sales_discount_thb
        ):
            return {"ok": False, "error": "ส่วนลดเพิ่มใช้ได้เฉพาะโปรงาดำ 3 กระปุกและใช้ได้ครั้งเดียว"}, None
        stock = int(product.get("stock_quantity", 0))
        already = session.cart.get(product_id, 0)
        if already + quantity > stock:
            return {"ok": False, "error": f"สินค้าคงเหลือ {stock} ชิ้น"}, None
        session.cart[product_id] = already + quantity
        session.sales_discount_thb = discount
        lines = session.cart_lines(products)
        return (
            {"ok": True, "cart": lines, "total": sum(line["line_total"] for line in lines)},
            {"type": "cart_add", "product_id": product_id, "quantity": quantity},
        )
    if name == "verify_address":
        result = address_book.verify_address(
            str(args.get("subdistrict", "")),
            str(args.get("district", "")),
            str(args.get("province", "")),
            str(args.get("postcode", "")),
        )
        if result["match"]:
            address = result["address"]
            session.shipping.update(address)
            events = [{"type": "shipping_field", "field": key, "value": value} for key, value in address.items()]
            return {"ok": True, "verified": True, "address": address, **_progress(session)}, events
        return {"ok": False, "verified": False, **{k: v for k, v in result.items() if k != "match"}}, (
            {"type": "address_candidates", **result} if result.get("found") else None
        )
    if name == "lookup_postcode":
        result = address_book.lookup_postcode(str(args.get("postcode", "")))
        return result, ({"type": "address_candidates", **result} if result.get("found") else None)
    if name == "save_shipping_field":
        field_name = str(args.get("field", "")).strip()
        if field_name not in FREE_TEXT_FIELDS:
            return {"ok": False, "error": "ช่องที่อยู่ต้องบันทึกผ่าน verify_address เท่านั้น"}, None
        value, error = normalize_shipping_value(field_name, str(args.get("value", "")))
        if error:
            return {"ok": False, "error": error}, None
        session.shipping[field_name] = value
        return (
            {"ok": True, "saved": {field_name: value}, **_progress(session)},
            {"type": "shipping_field", "field": field_name, "value": value},
        )
    if name == "confirm_order":
        missing = [name_ for name_ in SHIPPING_FIELDS if name_ not in session.shipping]
        if not session.cart:
            return {"ok": False, "error": "ตะกร้ายังว่างอยู่"}, None
        if missing:
            return {"ok": False, "error": "ข้อมูลจัดส่งยังไม่ครบ", "missing": missing}, None
        session.order_confirmed = True
        session.pending_redirect = HOME_URL
        session.close_reason = "order_completed"
        return {
            "ok": True,
            "message": (
                "บันทึกคำสั่งซื้อแล้ว ให้ยืนยันการบันทึกตามจริง ห้ามระบุวันหรือระยะเวลาจัดส่งที่ไม่มีข้อมูลยืนยัน "
                "แล้วพูดประโยคปิดท้ายตรงตัวเป็นประโยคสุดท้าย: "
                f"{CLOSING_TEXT} ระบบจะตัดสายเอง"
            ),
        }, {"type": "order_confirmed"}
    if name == "end_call":
        session.pending_redirect = HOME_URL
        session.close_reason = session.close_reason or "customer_ended"
        return {
            "ok": True,
            "message": f"ให้พูดประโยคปิดท้ายตรงตัวเป็นประโยคสุดท้าย: {CLOSING_TEXT} ระบบจะตัดสายเอง",
        }, None
    if name == "log_callback":
        reason = str(args.get("reason", "")).strip()
        if not reason or session.ticket_no is None:
            return {"ok": False, "error": "ยังไม่มีเลขตั๋วหรือไม่มีหัวข้อ"}, None
        save_callback(session.ticket_no, reason, str(args.get("details", "")))
        session.callback_logged = True
        return {"ok": True, "message": "บันทึกเรื่องให้เจ้าหน้าที่ติดต่อกลับแล้ว"}, None
    return {"ok": False, "error": "ไม่รู้จักเครื่องมือนี้"}, None


def live_input_config() -> types.RealtimeInputConfig:
    return types.RealtimeInputConfig(
        automatic_activity_detection=types.AutomaticActivityDetection(
            silence_duration_ms=int(LISTEN_DELAY_S * 1000),
        )
    )


def trial_price() -> int:
    return settings.FASAI_TRIAL_PRICE_THB


def _progress(session: AgentSession) -> dict[str, Any]:
    missing = [name_ for name_ in SHIPPING_FIELDS if name_ not in session.shipping]
    return {"missing": missing, "complete": not missing}


def normalize_shipping_value(field_name: str, raw: str) -> tuple[str, str | None]:
    if field_name not in SHIPPING_FIELDS:
        return "", "ไม่รู้จักช่องนี้"
    value = " ".join(raw.split())
    if field_name == "postcode":
        digits = re.sub(r"\D", "", value)
        if not re.fullmatch(r"\d{5}", digits):
            return "", "รหัสไปรษณีย์ต้องเป็นตัวเลข 5 หลัก"
        return digits, None
    if field_name == "phone":
        digits = re.sub(r"\D", "", value)
        if not re.fullmatch(r"0\d{9}", digits):
            return "", "เบอร์โทรต้องเป็นตัวเลข 10 หลักขึ้นต้นด้วย 0"
        return digits, None
    if len(value) < 2:
        return "", "ข้อมูลสั้นเกินไป"
    if len(value) > 200:
        return "", "ข้อมูลยาวเกินไป"
    return value, None


def build_instructions(base: str) -> str:
    return (
        base.replace("{{TRIAL_PRICE}}", str(trial_price()))
        .replace("{{GREETING}}", GREETING_TEXT)
        .replace("{{CLOSING}}", CLOSING_TEXT)
    )
