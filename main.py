"""小巴 2.0 - 《工程本体论》学习助手（智谱 GLM-4-Flash · 免费版 + Supabase 对话持久化）"""
import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger("xiaoba2")

# ── WhatsApp Business Cloud API 配置 ────────────────────────────────────────
WHATSAPP_VERIFY_TOKEN  = os.environ.get("WHATSAPP_VERIFY_TOKEN", "")
WHATSAPP_APP_SECRET    = os.environ.get("WHATSAPP_APP_SECRET", "")
WHATSAPP_ACCESS_TOKEN  = os.environ.get("WHATSAPP_ACCESS_TOKEN", "")
WHATSAPP_PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
WHATSAPP_API_VERSION   = os.environ.get("WHATSAPP_API_VERSION", "v21.0")
WA_ENABLED = bool(WHATSAPP_ACCESS_TOKEN and WHATSAPP_PHONE_NUMBER_ID)

# ── 微信公众号（订阅号·明文模式）配置 ────────────────────────────────────────
# 公众号后台「设置与开发 → 消息推送」里的 Token 要跟这个环境变量填一样的值
WECHAT_TOKEN = os.environ.get("WECHAT_TOKEN", "")
WX_ENABLED = bool(WECHAT_TOKEN)
# 微信要求 5 秒内必须返回被动回复，超时用户会看到空白。给模型调用留 4.3 秒预算，
# 查/建 Supabase 对话再单独留一点余量，差一点点就直接返回兜底文案，不等它慢慢超时。
WECHAT_REPLY_BUDGET_SECONDS = 4.3

# ── 智谱 GLM 配置 ────────────────────────────────────────────────────────────
# API Key 在 https://open.bigmodel.cn/usercenter/apikeys 免费申请
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
ZHIPU_MODEL    = os.environ.get("ZHIPU_MODEL", "glm-4-flash")  # glm-4-flash 完全免费

# ── Supabase 配置（对话持久化，复用 book-reader 的项目）─────────────────────
SUPABASE_URL         = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
SB_ENABLED = bool(SUPABASE_URL and SUPABASE_SERVICE_KEY)
SB_HEADERS = {
    "apikey": SUPABASE_SERVICE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    "Content-Type": "application/json",
}

# ── 知识库路径（可通过环境变量覆盖；默认用仓库自带的 kb/ 子集，部署到 Linux 也能跑）─
KB_DIR = Path(os.environ.get("XIAOBA_KB", str(Path(__file__).parent / "kb")))


def build_preamble() -> str:
    """在启动时把 INDEX.md + 术语表拼入 system prompt。"""
    index_path = KB_DIR / "INDEX.md"
    index_text = index_path.read_text(encoding="utf-8") if index_path.exists() else "（知识库未找到）"

    glossary_lines: list[str] = []
    page_path = KB_DIR / "page.json"
    if page_path.exists():
        page = json.loads(page_path.read_text(encoding="utf-8"))
        for t in page.get("glossary", []):
            defn = t.get("definition", "")[:100].rstrip()
            glossary_lines.append(
                f"- **{t['term']}**（{t.get('zhMeaning', '')}）：{defn}…"
            )

    glossary_block = "\n".join(glossary_lines)

    return f"""你是学习助手「小巴」，专门帮助读者深入理解《工程本体论》（张东普著）。

## 材料知识库

{index_text}

### 术语速查
{glossary_block}

## 知识来源协议
1. 先检查材料知识库；有依据时标记[材料依据]并给出章节名称。
2. 材料未覆盖时先标记[材料未覆盖]，再以[模型补充]回答。
3. 不得伪造材料出处，不得把模型补充说成材料原意。

## 三级递进讲解协议
第一层（首次回答）：给出精确定义、机制、条件与边界，不主动使用生活类比，不生成图片。
第二层：只有用户明确说没懂或要求通俗解释时，提供一个贴切的场景类比，类比后回到严谨定义并说明类比边界。
第三层：用户仍不懂或主动要求图表时才考虑，无图像工具时如实说明。

请用中文回答，回答清晰简洁，重点突出。"""


PREAMBLE = build_preamble()

app = FastAPI(title="小巴2.0 · 工程本体论学习助手（智谱免费版）")


# ── Supabase 存取层（通过 PostgREST，httpx 异步调用）────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def sb_create_conversation(
    title: str, wa_phone: Optional[str] = None, wx_openid: Optional[str] = None
) -> dict:
    payload = {"title": title}
    if wa_phone:
        payload["wa_phone"] = wa_phone
    if wx_openid:
        payload["wx_openid"] = wx_openid
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{SUPABASE_URL}/rest/v1/xiaoba_conversations",
            headers={**SB_HEADERS, "Prefer": "return=representation"},
            json=payload,
        )
        r.raise_for_status()
        return r.json()[0]


async def sb_find_conversation_by(column: str, value: str, timeout: Optional[float] = None) -> Optional[dict]:
    """按某个渠道标识列（wa_phone / wx_openid）查已绑定的固定对话。"""
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/xiaoba_conversations",
            headers=SB_HEADERS,
            params={column: f"eq.{value}", "select": "id,title", "limit": "1"},
        )
        r.raise_for_status()
        rows = r.json()
        return rows[0] if rows else None


async def sb_list_conversations() -> list[dict]:
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/xiaoba_conversations",
            headers=SB_HEADERS,
            params={"select": "id,title,created_at,updated_at", "order": "updated_at.desc"},
        )
        r.raise_for_status()
        return r.json()


async def sb_get_messages(conversation_id: str) -> list[dict]:
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/xiaoba_messages",
            headers=SB_HEADERS,
            params={
                "conversation_id": f"eq.{conversation_id}",
                "select": "role,content,created_at",
                "order": "created_at.asc",
            },
        )
        r.raise_for_status()
        return r.json()


async def sb_add_message(conversation_id: str, role: str, content: str) -> None:
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{SUPABASE_URL}/rest/v1/xiaoba_messages",
            headers=SB_HEADERS,
            json={"conversation_id": conversation_id, "role": role, "content": content},
        )
        r.raise_for_status()


async def sb_touch_conversation(conversation_id: str) -> None:
    async with httpx.AsyncClient() as client:
        r = await client.patch(
            f"{SUPABASE_URL}/rest/v1/xiaoba_conversations",
            headers=SB_HEADERS,
            params={"id": f"eq.{conversation_id}"},
            json={"updated_at": _now_iso()},
        )
        r.raise_for_status()


async def sb_delete_conversation(conversation_id: str) -> None:
    async with httpx.AsyncClient() as client:
        r = await client.delete(
            f"{SUPABASE_URL}/rest/v1/xiaoba_conversations",
            headers=SB_HEADERS,
            params={"id": f"eq.{conversation_id}"},
        )
        r.raise_for_status()


def _require_sb():
    if not SB_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="未配置 SUPABASE_URL / SUPABASE_SERVICE_KEY，对话历史功能不可用。",
        )


# ── 页面 ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse((Path(__file__).parent / "index.html").read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "kb": str(KB_DIR),
        "preamble_chars": len(PREAMBLE),
        "model": ZHIPU_MODEL,
        "api_key_set": bool(os.environ.get("ZHIPU_API_KEY")),
        "supabase_enabled": SB_ENABLED,
        "whatsapp_enabled": WA_ENABLED,
        "wechat_enabled": WX_ENABLED,
    }


# ── 对话管理接口 ─────────────────────────────────────────────────────────

@app.get("/api/conversations")
async def api_list_conversations():
    _require_sb()
    try:
        return await sb_list_conversations()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Supabase 读取失败: {e.response.text}")


@app.get("/api/conversations/{conversation_id}/messages")
async def api_get_messages(conversation_id: str):
    _require_sb()
    try:
        return await sb_get_messages(conversation_id)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Supabase 读取失败: {e.response.text}")


@app.delete("/api/conversations/{conversation_id}")
async def api_delete_conversation(conversation_id: str):
    _require_sb()
    try:
        await sb_delete_conversation(conversation_id)
        return {"ok": True}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Supabase 删除失败: {e.response.text}")


# ── 聊天接口 ─────────────────────────────────────────────────────────────

class Msg(BaseModel):
    role: str   # "user" | "assistant"
    text: str


class ChatReq(BaseModel):
    message: str
    history: List[Msg] = []
    conversation_id: Optional[str] = None


def generate_reply(message: str, history: List[dict]) -> str:
    """history: [{"role": "user"|"assistant", "text": "..."}]"""
    api_key = os.environ.get("ZHIPU_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="未设置 ZHIPU_API_KEY 环境变量。请到 https://open.bigmodel.cn/usercenter/apikeys 免费申请，"
                   "然后 $env:ZHIPU_API_KEY='你的key' 后重启服务。",
        )

    client = OpenAI(api_key=api_key, base_url=ZHIPU_BASE_URL)

    messages = [{"role": "system", "content": PREAMBLE}]
    for m in history:
        role = "user" if m.get("role") == "user" else "assistant"
        messages.append({"role": role, "content": m.get("text", "")})
    messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model=ZHIPU_MODEL,
            messages=messages,
            temperature=0.3,
            top_p=0.75,
            max_tokens=1200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"智谱 GLM 调用失败: {e}")


@app.post("/chat")
async def chat(req: ChatReq):
    reply = generate_reply(req.message, [m.model_dump() for m in req.history])

    conversation_id = req.conversation_id
    is_new = False
    title = None

    if SB_ENABLED:
        try:
            if not conversation_id:
                title = req.message[:24] + ("…" if len(req.message) > 24 else "")
                conv = await sb_create_conversation(title)
                conversation_id = conv["id"]
                is_new = True

            await sb_add_message(conversation_id, "user", req.message)
            await sb_add_message(conversation_id, "assistant", reply)
            await sb_touch_conversation(conversation_id)
        except httpx.HTTPStatusError as e:
            # 持久化失败不影响本次回答，只是不会被记录
            conversation_id = req.conversation_id
            is_new = False
            title = None

    return JSONResponse({
        "reply": reply,
        "conversation_id": conversation_id,
        "is_new": is_new,
        "title": title,
    })


# ── WhatsApp Business Cloud API 对接 ─────────────────────────────────────────
# Meta 文档: https://developers.facebook.com/docs/whatsapp/cloud-api/guides/set-up-webhooks

async def send_whatsapp_text(to: str, body: str) -> None:
    # WhatsApp 单条文本消息上限 4096 字符，超出的分段发送
    chunks = [body[i:i + 4000] for i in range(0, len(body), 4000)] or [""]
    url = f"https://graph.facebook.com/{WHATSAPP_API_VERSION}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"}
    async with httpx.AsyncClient() as client:
        for chunk in chunks:
            r = await client.post(
                url,
                headers=headers,
                json={
                    "messaging_product": "whatsapp",
                    "to": to,
                    "type": "text",
                    "text": {"body": chunk},
                },
            )
            if r.status_code >= 400:
                logger.error("WhatsApp 发送失败 %s: %s", r.status_code, r.text)


def _verify_whatsapp_signature(raw_body: bytes, signature_header: Optional[str]) -> bool:
    if not WHATSAPP_APP_SECRET:
        # 未配置 app secret 时不做签名校验（仅本地/测试阶段应该这样，生产环境务必配置）
        return True
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(WHATSAPP_APP_SECRET.encode(), raw_body, hashlib.sha256).hexdigest()
    got = signature_header.removeprefix("sha256=")
    return hmac.compare_digest(expected, got)


@app.get("/webhook/whatsapp")
async def whatsapp_verify(request: Request):
    """Meta 配置 webhook 时的一次性验证握手。"""
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge", "")
    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN and WHATSAPP_VERIFY_TOKEN:
        return PlainTextResponse(challenge)
    raise HTTPException(status_code=403, detail="verify_token 不匹配")


@app.post("/webhook/whatsapp")
async def whatsapp_incoming(request: Request):
    raw_body = await request.body()
    if not _verify_whatsapp_signature(raw_body, request.headers.get("x-hub-signature-256")):
        raise HTTPException(status_code=403, detail="签名校验失败")

    payload = json.loads(raw_body or b"{}")

    try:
        entry = (payload.get("entry") or [{}])[0]
        change = (entry.get("changes") or [{}])[0]
        value = change.get("value") or {}
        messages = value.get("messages") or []
    except (IndexError, AttributeError):
        messages = []

    # 没有 messages 字段的通常是已读/送达状态回调（statuses），直接确认收到即可
    if not messages:
        return JSONResponse({"ok": True})

    if not WA_ENABLED:
        logger.warning("收到 WhatsApp 消息，但未配置 WHATSAPP_ACCESS_TOKEN/WHATSAPP_PHONE_NUMBER_ID，无法回复")
        return JSONResponse({"ok": True})

    msg = messages[0]
    from_phone = msg.get("from", "")
    if msg.get("type") != "text":
        await send_whatsapp_text(from_phone, "小巴目前只支持文字消息，请直接发文字问题～")
        return JSONResponse({"ok": True})

    user_text = (msg.get("text") or {}).get("body", "").strip()
    if not user_text:
        return JSONResponse({"ok": True})

    # 下面任何一步失败都不能让这个 handler 抛异常——Meta 对非 2xx 响应会反复重试整个
    # webhook，一旦问题修复，之前失败的那几条消息会被重新投递并再触发一次回复，造成重复/错乱。
    # 所以这里统一 try/except，兜底回一句抱歉文案，始终对 Meta 返回 200。
    try:
        history: List[dict] = []
        conversation_id = None
        if SB_ENABLED:
            try:
                conv = await sb_find_conversation_by("wa_phone", from_phone)
                if conv:
                    conversation_id = conv["id"]
                    past = await sb_get_messages(conversation_id)
                    history = [{"role": r["role"], "text": r["content"]} for r in past[-20:]]
            except httpx.HTTPStatusError as e:
                logger.error("Supabase 查询失败: %s", e.response.text)

        reply = generate_reply(user_text, history)

        if SB_ENABLED:
            try:
                if not conversation_id:
                    title = f"WhatsApp {from_phone}"
                    conv = await sb_create_conversation(title, wa_phone=from_phone)
                    conversation_id = conv["id"]
                await sb_add_message(conversation_id, "user", user_text)
                await sb_add_message(conversation_id, "assistant", reply)
                await sb_touch_conversation(conversation_id)
            except httpx.HTTPStatusError as e:
                logger.error("Supabase 写入失败: %s", e.response.text)

        await send_whatsapp_text(from_phone, reply)
    except Exception:
        logger.exception("处理 WhatsApp 消息失败 (from=%s)", from_phone)
        try:
            await send_whatsapp_text(from_phone, "小巴这边刚才出了点小问题，请稍后再问一次～")
        except Exception:
            logger.exception("连兜底回复都发送失败 (from=%s)", from_phone)

    return JSONResponse({"ok": True})


# ── 微信公众号（订阅号·明文模式·被动回复）对接 ───────────────────────────────
# 文档: https://developers.weixin.qq.com/doc/offiaccount/Basic_Information/Access_Overview.html
# 关键约束：微信要求服务器在收到消息后 5 秒内返回被动回复 XML，超时用户会看到空白/报错，
# 而且不支持像 WhatsApp 那样"先 ack 再异步推送"（公众号的异步推送是客服消息接口，
# 需要认证服务号，个人订阅号用不了）。所以这条路径必须全程算时间预算，宁可提前兜底
# 返回一句"请稍后再问"，也不能让 generate_reply() 无限期跑下去。

def _wechat_xml_reply(to_user: str, from_user: str, content: str) -> str:
    # 公众号被动回复用 CDATA 包裹正文，模型回复里常见的 &/</> 等字符不需要额外转义
    return (
        f"<xml><ToUserName><![CDATA[{to_user}]]></ToUserName>"
        f"<FromUserName><![CDATA[{from_user}]]></FromUserName>"
        f"<CreateTime>{int(time.time())}</CreateTime>"
        f"<MsgType><![CDATA[text]]></MsgType>"
        f"<Content><![CDATA[{content}]]></Content></xml>"
    )


def _verify_wechat_signature(signature: Optional[str], timestamp: Optional[str], nonce: Optional[str]) -> bool:
    if not (WX_ENABLED and signature and timestamp and nonce):
        return False
    expected = hashlib.sha1("".join(sorted([WECHAT_TOKEN, timestamp, nonce])).encode()).hexdigest()
    return hmac.compare_digest(expected, signature)


@app.get("/webhook/wechat")
async def wechat_verify(request: Request):
    """公众号后台「消息推送 → 启用」时的一次性服务器验证。"""
    params = request.query_params
    if _verify_wechat_signature(params.get("signature"), params.get("timestamp"), params.get("nonce")):
        return PlainTextResponse(params.get("echostr", ""))
    raise HTTPException(status_code=403, detail="签名校验失败，检查 WECHAT_TOKEN 是否和公众号后台填的一致")


@app.post("/webhook/wechat")
async def wechat_incoming(request: Request):
    params = request.query_params
    if not _verify_wechat_signature(params.get("signature"), params.get("timestamp"), params.get("nonce")):
        raise HTTPException(status_code=403, detail="签名校验失败")

    deadline = time.monotonic() + WECHAT_REPLY_BUDGET_SECONDS
    raw_body = await request.body()

    try:
        root = ET.fromstring(raw_body)
    except ET.ParseError:
        return Response(content="success", media_type="text/plain")

    def _field(tag: str) -> str:
        el = root.find(tag)
        return el.text or "" if el is not None else ""

    from_user = _field("FromUserName")  # 用户的 openid
    to_user = _field("ToUserName")      # 公众号自己的原始 ID
    msg_type = _field("MsgType")

    if msg_type != "text" or not from_user:
        # 关注/取关/点击菜单等事件，或非文本消息：按微信规范直接返回空字符串表示不回复
        return Response(content="", media_type="text/plain")

    user_text = _field("Content").strip()
    if not user_text:
        return Response(content="", media_type="text/plain")

    history: List[dict] = []
    conversation_id: Optional[str] = None
    if SB_ENABLED:
        remaining = max(deadline - time.monotonic(), 0.3)
        try:
            conv = await asyncio.wait_for(
                sb_find_conversation_by("wx_openid", from_user, timeout=remaining), timeout=remaining
            )
            if conv:
                conversation_id = conv["id"]
                past = await asyncio.wait_for(sb_get_messages(conversation_id), timeout=remaining)
                history = [{"role": r["role"], "text": r["content"]} for r in past[-20:]]
        except (httpx.HTTPStatusError, asyncio.TimeoutError) as e:
            logger.warning("微信消息处理：查历史失败/超时，退化成无历史单轮回答 (%s)", e)

    remaining = deadline - time.monotonic()
    if remaining <= 0.5:
        # 查历史就已经把时间预算耗光了，没时间再等模型，直接认怂返回兜底文案
        reply = "问题有点多，小巴还在想，请隔几秒再发一次～"
    else:
        try:
            reply = await asyncio.wait_for(
                asyncio.to_thread(generate_reply, user_text, history), timeout=remaining
            )
        except asyncio.TimeoutError:
            logger.warning("微信消息处理：模型调用超过 %.1fs 预算，返回兜底文案", WECHAT_REPLY_BUDGET_SECONDS)
            reply = "小巴想得有点久，没来得及想完，请再问一次试试～"
        except HTTPException as e:
            logger.error("微信消息处理：模型调用失败 %s", e.detail)
            reply = "小巴这边刚才出了点小问题，请稍后再问一次～"

    if SB_ENABLED:
        async def _persist():
            try:
                cid = conversation_id
                if not cid:
                    conv = await sb_create_conversation(f"微信 {from_user}", wx_openid=from_user)
                    cid = conv["id"]
                await sb_add_message(cid, "user", user_text)
                await sb_add_message(cid, "assistant", reply)
                await sb_touch_conversation(cid)
            except httpx.HTTPStatusError as e:
                logger.error("Supabase 写入失败: %s", e.response.text)

        # 不等这步完成——已经没有时间预算了，被动回复必须先发出去；持久化失败不影响本次回复
        asyncio.create_task(_persist())

    return Response(content=_wechat_xml_reply(from_user, to_user, reply), media_type="application/xml")
