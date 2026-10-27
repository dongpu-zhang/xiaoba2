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
# 实测 generate_reply() 单轮常规问答在线上要 ~10 秒（GLM-4-Flash 生成几百字中文 +
# Render 美东到智谱国内 API 的跨境延迟），稳定超过微信被动回复 5 秒的硬限制，不是偶发
# 慢请求，调小这个数字也救不回来。所以只在没配 WECHAT_APPID/SECRET 时才会用到这个同步
# 兜底预算；配了之后走下面的异步客服消息推送，不受 5 秒限制。
WECHAT_REPLY_BUDGET_SECONDS = 4.3

# 客服消息接口（异步推送真实回答，绕开被动回复的 5 秒限制）需要 AppID+AppSecret 换
# access_token。公众号「设置与开发→基础信息」页能看到 AppID，AppSecret 在「开发密钥」
# 那一块（可能要点"重置"才能看到明文）。未认证个人订阅号这个接口的实际调用额度官方文档
# 写得含糊，配了之后如果日志里看到 errcode 45015/48001 这类权限错误，大概率是额度/权限
# 不够，这是已知风险，不是代码 bug。
WECHAT_APPID = os.environ.get("WECHAT_APPID", "")
WECHAT_APPSECRET = os.environ.get("WECHAT_APPSECRET", "")
WX_PUSH_ENABLED = bool(WECHAT_APPID and WECHAT_APPSECRET)
_wx_token_cache: dict = {"token": None, "expires_at": 0.0}

# ── Telegram Bot API 配置 ───────────────────────────────────────────────────
# @BotFather 创建 bot 拿 token。架构和 WhatsApp 完全一样（JSON webhook + 主动推送 API，
# 没有微信那种 5 秒被动回复限制），不需要 fire_and_forget 异步兜底这套机制。
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
# setWebhook 时可以带一个 secret_token，Telegram 之后每次回调都会把它放进这个 header，
# 用来验证请求确实来自 Telegram（不是必须，但配了更安全）
TELEGRAM_WEBHOOK_SECRET = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")
TG_ENABLED = bool(TELEGRAM_BOT_TOKEN)

# asyncio.create_task() 只持有任务的弱引用——如果没有别的地方强引用这个 task 对象，
# 事件循环的 gc 随时可能在它跑完之前把它回收掉，表现就是"偶尔/经常悄无声息地不执行"，
# 不会报错不会留日志。所有"fire and forget"后台任务都要用这个 helper 建，强引用存进
# 一个模块级 set，跑完/出错后通过 done_callback 自动摘掉，不会无限增长。
_background_tasks: set = set()


def fire_and_forget(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task

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
    title: str,
    wa_phone: Optional[str] = None,
    wx_openid: Optional[str] = None,
    tg_chat_id: Optional[str] = None,
) -> dict:
    payload = {"title": title}
    if wa_phone:
        payload["wa_phone"] = wa_phone
    if wx_openid:
        payload["wx_openid"] = wx_openid
    if tg_chat_id:
        payload["tg_chat_id"] = tg_chat_id
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
        "wechat_push_enabled": WX_PUSH_ENABLED,
        "telegram_enabled": TG_ENABLED,
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


def generate_reply(message: str, history: List[dict], max_tokens: int = 1200, brief: bool = False) -> str:
    """history: [{"role": "user"|"assistant", "text": "..."}]
    brief=True：微信渠道专用，回答长度本身就是实测把微信5秒被动回复预算挤爆的主因（见
    xiaoba2 memory 记录），所以不只是砍 max_tokens 硬截断（截断会在句子中间断掉，很难看），
    还在用户消息末尾加一句明确要求简短作答，让模型自己收着写、在合理长度内说完整句话。"""
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
    final_message = message
    if brief:
        final_message += "\n\n（提醒：这是微信渠道，请用不超过150字的中文简要回答，直接给结论，不用展开讲解或分点列举，如需深入可以让我再追问。）"
    messages.append({"role": "user", "content": final_message})

    try:
        response = client.chat.completions.create(
            model=ZHIPU_MODEL,
            messages=messages,
            temperature=0.3,
            top_p=0.75,
            max_tokens=max_tokens,
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


async def wx_get_access_token() -> Optional[str]:
    """客服消息接口要用的 access_token，7200 秒有效期，提前 200 秒刷新，内存缓存。"""
    now = time.time()
    if _wx_token_cache["token"] and _wx_token_cache["expires_at"] > now:
        return _wx_token_cache["token"]
    async with httpx.AsyncClient() as client:
        r = await client.get(
            "https://api.weixin.qq.com/cgi-bin/token",
            params={"grant_type": "client_credential", "appid": WECHAT_APPID, "secret": WECHAT_APPSECRET},
        )
        data = r.json()
    if "access_token" not in data:
        logger.error("获取微信 access_token 失败: %s", data)
        return None
    _wx_token_cache["token"] = data["access_token"]
    _wx_token_cache["expires_at"] = now + data.get("expires_in", 7200) - 200
    return _wx_token_cache["token"]


# 客服消息接口目前对这个未认证个人订阅号返回 errcode 48001（api unauthorized），
# WX_PUSH_ENABLED 平时是关着的（没配 WECHAT_APPID/SECRET）。这个 dict 记最近一次发送
# 结果，留着是为了以后账号升级成认证服务号、重新打开这条路时排查用，不含 openid/正文。
_wx_last_send_debug: dict = {}


async def wx_send_customer_text(openid: str, content: str) -> None:
    """客服消息接口：不受被动回复 5 秒限制，但只能在用户最近 48 小时内主动联系过才能推送。"""
    token = await wx_get_access_token()
    if not token:
        _wx_last_send_debug.update({"ts": time.time(), "step": "get_token", "ok": False})
        return
    # 微信客服消息单条长度限制 2048 字节，按中文 3 字节/字粗略切成安全长度分段发送
    chunks = [content[i:i + 600] for i in range(0, len(content), 600)] or [""]
    async with httpx.AsyncClient() as client:
        for chunk in chunks:
            r = await client.post(
                "https://api.weixin.qq.com/cgi-bin/message/custom/send",
                params={"access_token": token},
                json={"touser": openid, "msgtype": "text", "text": {"content": chunk}},
            )
            data = r.json()
            _wx_last_send_debug.update({
                "ts": time.time(),
                "step": "send",
                "ok": data.get("errcode", 0) == 0,
                "errcode": data.get("errcode"),
                "errmsg": data.get("errmsg"),
            })
            if data.get("errcode", 0) != 0:
                logger.error("微信客服消息发送失败 errcode=%s errmsg=%s", data.get("errcode"), data.get("errmsg"))


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

    if WX_PUSH_ENABLED:
        # 实测 generate_reply() 常规单轮要 ~10 秒，稳定超过被动回复 5 秒上限：被动回复只
        # 秒回一句"正在想"占位，真正的答案通过客服消息接口异步推送（不受 5 秒限制，
        # 唯一约束是只能推给 48 小时内联系过的用户，闲聊场景完全够用）。
        fire_and_forget(_wechat_async_reply(from_user, user_text))
        ack = "💭 小巴正在想，几秒后把答案发过来～"
        return Response(content=_wechat_xml_reply(from_user, to_user, ack), media_type="application/xml")

    # 没配 WECHAT_APPID/SECRET 时的旧同步路径：在 5 秒预算内硬等模型，大概率超时兜底
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
                asyncio.to_thread(generate_reply, user_text, history, 350, True), timeout=remaining
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
        fire_and_forget(_persist())

    return Response(content=_wechat_xml_reply(from_user, to_user, reply), media_type="application/xml")


async def _wechat_async_reply(from_user: str, user_text: str) -> None:
    """WX_PUSH_ENABLED 时的真实处理：不受 5 秒限制，失败了也只能记日志——被动回复那条
    占位消息已经发出去了，这里没有别的输出通道能补发一个更友好的错误提示给用户。"""
    history: List[dict] = []
    conversation_id: Optional[str] = None
    if SB_ENABLED:
        try:
            conv = await sb_find_conversation_by("wx_openid", from_user)
            if conv:
                conversation_id = conv["id"]
                past = await sb_get_messages(conversation_id)
                history = [{"role": r["role"], "text": r["content"]} for r in past[-20:]]
        except httpx.HTTPStatusError as e:
            logger.warning("微信异步处理：查历史失败，退化成无历史单轮回答 (%s)", e.response.text)

    try:
        reply = await asyncio.to_thread(generate_reply, user_text, history)
    except HTTPException as e:
        logger.error("微信异步处理：模型调用失败 %s", e.detail)
        reply = "小巴这边刚才出了点小问题，请稍后再问一次～"
    except Exception:
        logger.exception("微信异步处理：模型调用出现未预期异常 (openid=%s)", from_user)
        reply = "小巴这边刚才出了点小问题，请稍后再问一次～"

    if SB_ENABLED:
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

    await wx_send_customer_text(from_user, reply)


# ── Telegram Bot API 对接 ────────────────────────────────────────────────────
# 文档: https://core.telegram.org/bots/api#setwebhook
# 架构和 WhatsApp 一样：webhook 收到消息 → 同步生成回复 → 调用 sendMessage 主动推送，
# 没有微信那种 5 秒硬限制，也不需要 IP 白名单。部署好之后要手动调一次 setWebhook
# 把 Render 的 URL 告诉 Telegram（见本文件末尾的说明，或直接问我）。

async def send_telegram_text(chat_id: str, body: str) -> None:
    # Telegram 单条消息上限 4096 字符，超出的分段发送
    chunks = [body[i:i + 4000] for i in range(0, len(body), 4000)] or [""]
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        for chunk in chunks:
            r = await client.post(url, json={"chat_id": chat_id, "text": chunk})
            if r.status_code >= 400:
                logger.error("Telegram 发送失败 %s: %s", r.status_code, r.text)


def _verify_telegram_secret(request: Request) -> bool:
    if not TELEGRAM_WEBHOOK_SECRET:
        # 未配置 secret 时不做校验（仅本地/测试阶段应该这样，生产环境建议配置）
        return True
    return request.headers.get("x-telegram-bot-api-secret-token") == TELEGRAM_WEBHOOK_SECRET


@app.post("/webhook/telegram")
async def telegram_incoming(request: Request):
    if not _verify_telegram_secret(request):
        raise HTTPException(status_code=403, detail="secret_token 不匹配")

    update = json.loads((await request.body()) or b"{}")
    message = update.get("message") or {}
    chat = message.get("chat") or {}
    chat_id = str(chat.get("id", ""))

    # 没有 message 字段的是编辑消息/频道贴/按钮点击等其它更新类型，确认收到即可，不回复
    if not chat_id:
        return JSONResponse({"ok": True})

    if not TG_ENABLED:
        logger.warning("收到 Telegram 消息，但未配置 TELEGRAM_BOT_TOKEN，无法回复")
        return JSONResponse({"ok": True})

    if "text" not in message:
        await send_telegram_text(chat_id, "小巴目前只支持文字消息，请直接发文字问题～")
        return JSONResponse({"ok": True})

    user_text = message["text"].strip()
    if not user_text:
        return JSONResponse({"ok": True})

    # 和 WhatsApp 一样：这里任何一步失败都不能抛异常让 Telegram 重投整个 webhook，
    # 统一 try/except 兜底一句抱歉文案，始终对 Telegram 返回 200。
    try:
        history: List[dict] = []
        conversation_id = None
        if SB_ENABLED:
            try:
                conv = await sb_find_conversation_by("tg_chat_id", chat_id)
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
                    title = f"Telegram {chat_id}"
                    conv = await sb_create_conversation(title, tg_chat_id=chat_id)
                    conversation_id = conv["id"]
                await sb_add_message(conversation_id, "user", user_text)
                await sb_add_message(conversation_id, "assistant", reply)
                await sb_touch_conversation(conversation_id)
            except httpx.HTTPStatusError as e:
                logger.error("Supabase 写入失败: %s", e.response.text)

        await send_telegram_text(chat_id, reply)
    except Exception:
        logger.exception("处理 Telegram 消息失败 (chat_id=%s)", chat_id)
        try:
            await send_telegram_text(chat_id, "小巴这边刚才出了点小问题，请稍后再问一次～")
        except Exception:
            logger.exception("连兜底回复都发送失败 (chat_id=%s)", chat_id)

    return JSONResponse({"ok": True})
