"""小巴 2.0 - 《工程本体论》学习助手（智谱 GLM-4-Flash · 免费版 + Supabase 对话持久化）"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from openai import OpenAI
from pydantic import BaseModel

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


async def sb_create_conversation(title: str) -> dict:
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{SUPABASE_URL}/rest/v1/xiaoba_conversations",
            headers={**SB_HEADERS, "Prefer": "return=representation"},
            json={"title": title},
        )
        r.raise_for_status()
        return r.json()[0]


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


@app.post("/chat")
async def chat(req: ChatReq):
    api_key = os.environ.get("ZHIPU_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="未设置 ZHIPU_API_KEY 环境变量。请到 https://open.bigmodel.cn/usercenter/apikeys 免费申请，"
                   "然后 $env:ZHIPU_API_KEY='你的key' 后重启服务。",
        )

    client = OpenAI(api_key=api_key, base_url=ZHIPU_BASE_URL)

    messages = [{"role": "system", "content": PREAMBLE}]
    for m in req.history:
        role = "user" if m.role == "user" else "assistant"
        messages.append({"role": role, "content": m.text})
    messages.append({"role": "user", "content": req.message})

    try:
        response = client.chat.completions.create(
            model=ZHIPU_MODEL,
            messages=messages,
            temperature=0.3,
            top_p=0.75,
            max_tokens=1200,
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"智谱 GLM 调用失败: {e}")

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
