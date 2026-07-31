"""
七曜 · 参赛版后端(清小搭 OpenAI 兼容接口)

与点点(companion)的区别:
  1. 对外暴露的是清小搭要求的 OpenAI 兼容端点:
       GET  /v1/models              连通/凭证校验
       POST /v1/chat/completions    对话(流式 SSE + 非流式 JSON)
  2. 鉴权用 Bearer <SERVICE_KEY>,无效返回 401。
  3. finish_reason 只用官方白名单值;usage 放在流式 stop 帧。
  4. 角色系统保留(星曜原型),但每次请求是无状态的 —— 平台负责
     把多轮历史通过 messages 传进来,符合 OpenAI 约定。

本文件可直接被清小搭「标准协议接入」向导探测通过。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import httpx
import yaml
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

# ---------------------------------------------------------------- 配置

BASE_DIR = Path(__file__).parent
PERSONA_DIR = BASE_DIR / "personas"

try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

# 上游大模型(你的推理来源)
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-pro")

# 清小搭接入时填的「API 密钥」—— 平台带 Bearer 过来,我们校验它。
# 和上游 LLM 的 key 是两码事:这个是"别人进你服务的门票"。
SERVICE_KEY = os.getenv("SERVICE_KEY", "")

# 默认用哪个星曜应答(平台目前一个接入点对应一个 agent)
DEFAULT_PERSONA = os.getenv("DEFAULT_PERSONA", "wenqu")

# ---------------------------------------------------------------- 星曜角色

class PersonaRegistry:
    """从 YAML 目录加载星曜,按文件 mtime 热重载。"""

    def __init__(self, directory: Path):
        self.dir = directory
        self._cache: dict[str, dict[str, Any]] = {}
        self._stamp: float = -1.0

    def _fingerprint(self) -> float:
        if not self.dir.exists():
            return 0.0
        return max((p.stat().st_mtime for p in self.dir.glob("*.y*ml")), default=0.0)

    def _reload(self) -> None:
        loaded: dict[str, dict[str, Any]] = {}
        for path in sorted(self.dir.glob("*.y*ml")):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as exc:
                print(f"[persona] 跳过 {path.name}: {exc}")
                continue
            pid = data.get("id") or path.stem
            data["id"] = pid
            data.setdefault("name", pid)
            loaded[pid] = data
        self._cache = loaded

    def all(self) -> dict[str, dict[str, Any]]:
        stamp = self._fingerprint()
        if stamp != self._stamp:
            self._reload()
            self._stamp = stamp
        return self._cache

    def get(self, persona_id: str) -> dict[str, Any]:
        personas = self.all()
        return personas.get(persona_id) or next(iter(personas.values()), {})


registry = PersonaRegistry(PERSONA_DIR)


def build_system_prompt(persona: dict[str, Any]) -> str:
    core = persona.get("core", {})
    parts: list[str] = [f"你是{persona.get('name','')}。{core.get('identity','').strip()}"]

    def section(title: str, body: Any) -> None:
        if not body:
            return
        if isinstance(body, (list, tuple)):
            body = "\n".join(f"- {x}" for x in body)
        parts.append(f"\n【{title}】\n{str(body).strip()}")

    section("说话方式", core.get("voice"))
    section("你在意的事", core.get("values"))
    section("绝对不做", core.get("boundaries"))

    parts.append(
        "\n【底线】\n"
        "- 你不是治疗师。用户流露自伤、伤人或危机信号时,先认真回应情绪,"
        "再温和引导其联系身边信任的人或专业求助渠道,不做诊断、不替代专业帮助。\n"
        "- 你借用的星曜/心理符号只是帮用户表达需求的语言,不是命理预测,不做吉凶断言。\n"
        "- 保持你的说话方式,但不为维持人设而说有害或违心的话。"
    )
    return "\n".join(parts)


# ---------------------------------------------------------------- 上游调用

def _upstream() -> httpx.AsyncClient:
    if not LLM_API_KEY:
        raise HTTPException(500, "服务未配置 LLM_API_KEY")
    return httpx.AsyncClient(
        base_url=LLM_API_BASE,
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        timeout=httpx.Timeout(110.0, connect=10.0),
    )


def _prepend_persona(messages: list[dict], persona: dict) -> list[dict]:
    """在平台传来的 messages 前面注入星曜人设。

    平台可能自带一条 system(用户在广场填的),我们把星曜人设放最前,
    其余原样保留,不破坏多轮上下文。
    """
    sys_prompt = build_system_prompt(persona)
    out = [{"role": "system", "content": sys_prompt}]
    for m in messages:
        role = m.get("role")
        if role in ("system", "user", "assistant"):
            content = m.get("content", "")
            # content 可能是多模态数组;当前只取文本部分
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            out.append({"role": role, "content": content})
        # 其它角色(tool 等)当前忽略
    return out


async def upstream_stream(messages: list[dict]):
    payload = {"model": LLM_MODEL, "messages": messages, "stream": True}
    async with _upstream() as client:
        async with client.stream("POST", "/chat/completions", json=payload) as resp:
            if resp.status_code != 200:
                detail = (await resp.aread()).decode("utf-8", "replace")[:300]
                raise HTTPException(502, f"上游 {resp.status_code}: {detail}")
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                chunk = line[6:].strip()
                if chunk == "[DONE]":
                    return
                try:
                    delta = json.loads(chunk)["choices"][0]["delta"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if piece := delta.get("content"):
                    yield piece


async def upstream_once(messages: list[dict]) -> str:
    payload = {"model": LLM_MODEL, "messages": messages, "stream": False}
    async with _upstream() as client:
        resp = await client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------- 鉴权

def check_auth(authorization: str | None) -> None:
    """可选鉴权。

    SERVICE_KEY 为空 → 公开模式,任何请求放行(方便自己调试,前端密钥框可留空)。
    SERVICE_KEY 有值 → 严格校验 Bearer(接清小搭 / 分享给别人时用)。
    """
    if not SERVICE_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing credential")
    token = authorization[len("Bearer "):].strip()
    if token != SERVICE_KEY:
        raise HTTPException(401, "invalid credential")


# ---------------------------------------------------------------- 应用

app = FastAPI(title="七曜 · Astra")


@app.get("/v1/models")
def models(authorization: str | None = Header(None)):
    check_auth(authorization)
    return {
        "object": "list",
        "data": [{"id": "qiyao", "object": "model", "owned_by": "astra"}],
    }


def _count(text: str) -> int:
    return max(len(text), 0)


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: str | None = Header(None),
    x_persona: str | None = Header(None),
):
    check_auth(authorization)
    body = await request.json()

    stream = bool(body.get("stream", False))          # 严格布尔
    raw_messages = body.get("messages", []) or []

    # 清小搭走 DEFAULT_PERSONA;调试前端可用 X-Persona 头临时切换星曜。
    persona = registry.get(x_persona or DEFAULT_PERSONA)
    messages = _prepend_persona(raw_messages, persona)

    user_text = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
    )
    cid = f"chatcmpl-{int(time.time()*1000)}"
    created = int(time.time())

    # -------- 非流式 --------
    if not stream:
        try:
            answer = await upstream_once(messages)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(502, f"上游错误: {exc}")
        pt, ct = _count(user_text), _count(answer)
        return JSONResponse(
            {
                "id": cid,
                "object": "chat.completion",
                "created": created,
                "model": "qiyao",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "total_tokens": pt + ct,
                },
            }
        )

    # -------- 流式 SSE --------
    def frame(delta: dict, finish: str | None = None, usage: dict | None = None,
              error: dict | None = None) -> str:
        chunk = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "qiyao",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if usage:
            chunk["usage"] = usage
        if error:
            chunk["error"] = error
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

    async def sse():
        yield frame({"role": "assistant"})            # role 帧(恰好一次,首帧)
        collected: list[str] = []
        try:
            async for piece in upstream_stream(messages):
                collected.append(piece)
                yield frame({"content": piece})        # content 增量帧
        except Exception as exc:
            # 头已发出,按规范发带 error 的 stop 帧再 [DONE]
            yield frame({}, finish="stop",
                        error={"type": "upstream_error", "message": str(exc)[:200]})
            yield "data: [DONE]\n\n"
            return
        answer = "".join(collected)
        pt, ct = _count(user_text), _count(answer)
        yield frame({}, finish="stop", usage={          # stop 帧 + usage
            "prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct,
        })
        yield "data: [DONE]\n\n"                         # 终止哨兵

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


FRONTEND_DIR = BASE_DIR.parent / "frontend"


@app.get("/")
def root():
    index = FRONTEND_DIR / "index.html"
    if index.is_file():
        from fastapi.responses import FileResponse

        return FileResponse(index)
    return {"service": "七曜 · Astra", "endpoints": ["/v1/models", "/v1/chat/completions"]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
