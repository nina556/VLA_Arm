"""LLM intent router for the Unoarm web console (SiliconFlow / OpenAI-compatible).

Classifies each chat message as either:
- ``action`` — start an ACT rollout (policy ignores language; instruction is a label)
- ``chat`` — reply in Chinese, do not move the robot

No task whitelist: the LLM decides freely whether the user wants the robot to act.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "THUDM/GLM-4-9B-0414"
DEFAULT_TIMEOUT_S = 20.0

_SYSTEM_PROMPT = """你是 Unoarm 双臂机器人的语音助手路由器。判断用户输入是闲聊还是要机器人执行动作。

你的职责：
1. 如果用户闲聊、问候、提问、或输入与机器人动作无关，输出 chat，reply 用中文自然回复。
2. 如果用户要求机器人做任何动作（抓取、放置、移动、拿东西等），输出 action。
3. 不要因为措辞不标准而拒绝；只要意图是让机器人动手，就输出 action。
4. 输出 action 时，instruction 用一句简短中文或英文概括用户意图（仅作日志标签，不必匹配固定白名单）。

输出格式必须是严格 JSON 对象，不要 markdown，不要解释，不要额外文字：
- 聊天：{"route":"chat","reply":"中文回复"}
- 动作：{"route":"action","instruction":"用户意图摘要"}

示例：
- 用户说 "hello"：{"route":"chat","reply":"你好！需要我控制机械臂做事时直接说就行。"}
- 用户说 "把圆柱放到目标圈上"：{"route":"action","instruction":"Pick up the cylinder and place it on the target circle"}
- 用户说 "帮我抓一下那个东西放过去"：{"route":"action","instruction":"Grasp and place the object"}
"""


def _parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from raw model text, tolerating markdown fences."""
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.removeprefix("```json").removeprefix("```").strip()
        if raw.endswith("```"):
            raw = raw[: -len("```")].strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start : end + 1]
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object, got {type(obj).__name__}")
    return obj


def _clean_chat_reply(text: str) -> str:
    text = text.strip()
    prefixes = (
        "不调用工具。",
        "不调用工具:",
        "不调用工具：",
        "无需调用工具。",
        "无需调用工具:",
        "无需调用工具：",
    )
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


@dataclass
class ActionDecision:
    """LLM decided this is an action request."""

    instruction: str


@dataclass
class ChatReply:
    """LLM decided this is chit-chat and produced a text reply."""

    text: str


class ChatRouter:
    """Classify user input into action vs. chit-chat via an OpenAI-compatible LLM."""

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        verbose: bool = True,
    ) -> None:
        from openai import OpenAI

        if not str(api_key or "").strip():
            raise ValueError("LLM API key is empty; set it in 设置 or SILICONFLOW_API_KEY")

        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
            max_retries=0,
        )
        self.timeout_s = timeout_s
        self.verbose = verbose
        self.history: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT}
        ]

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[router] {message}", flush=True)

    def route(self, user_input: str) -> ActionDecision | ChatReply:
        """Send user_input to the LLM and return either an action or a chat reply."""
        self._log(f"收到用户输入: {user_input!r}")
        self.history.append({"role": "user", "content": user_input})
        self._log(
            f"开始请求 LLM: model={self.model}, timeout={self.timeout_s:.1f}s, "
            f"messages={len(self.history)}"
        )
        request_started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                temperature=0.1,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            elapsed_s = time.perf_counter() - request_started
            self._log(f"LLM 请求失败: {type(e).__name__}: {e} (耗时 {elapsed_s:.2f}s)")
            raise
        elapsed_s = time.perf_counter() - request_started
        self._log(f"LLM 已返回 (耗时 {elapsed_s:.2f}s)")

        choice = response.choices[0] if response.choices else None
        msg = choice.message if choice is not None else None
        if msg is None:
            reply = "我这边没有拿到有效判断结果，所以先不执行机器人动作。你可以再说一次。"
            self.history.append({"role": "assistant", "content": reply})
            self._log("LLM 返回空 message，已按聊天回复处理")
            return ChatReply(reply)

        raw_text = msg.content or ""
        raw_preview = raw_text.replace("\n", " ")[:160] or "<empty>"
        self._log(f"LLM 原始返回: {raw_preview}")
        self.history.append({"role": "assistant", "content": raw_text})

        try:
            route_obj = _parse_json_object(raw_text)
        except (json.JSONDecodeError, ValueError) as e:
            reply = _clean_chat_reply(raw_text) or "我没解析清楚这个请求，所以先不执行机器人动作。"
            self._log(f"JSON 路由解析失败: {type(e).__name__}: {e}，已按 CHAT 处理")
            return ChatReply(reply)

        route = str(route_obj.get("route", "chat")).strip().lower()
        if route == "action":
            instruction = str(route_obj.get("instruction", "")).strip() or user_input.strip()
            self._log(f"路由结果: ACTION -> {instruction!r}")
            return ActionDecision(instruction=instruction)

        reply = _clean_chat_reply(str(route_obj.get("reply", "")).strip())
        if not reply:
            reply = "我可以陪你聊天；需要机械臂做事时直接说就行。"
        preview = reply.replace("\n", " ")[:120]
        self._log(f"路由结果: CHAT -> {preview}")
        return ChatReply(text=reply)
