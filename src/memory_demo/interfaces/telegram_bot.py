from __future__ import annotations

import json
import time
from urllib import parse, request


def split_telegram_message(text: str, limit: int = 3_900) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        boundary = remaining.rfind("\n", 0, limit)
        if boundary < limit // 2:
            boundary = limit
        chunks.append(remaining[:boundary])
        remaining = remaining[boundary:].lstrip("\n")
    return chunks


class TelegramBot:
    def __init__(self, token: str, query_engine):
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN is not configured")
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.query_engine = query_engine

    def _call(self, method: str, params: dict, timeout: float = 45.0):
        body = parse.urlencode(params).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url}/{method}", data=body, method="POST"
        )
        with request.urlopen(http_request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not payload.get("ok"):
            raise RuntimeError(payload)
        return payload["result"]

    def send_message(self, chat_id: int, text: str) -> None:
        for chunk in split_telegram_message(text):
            self._call("sendMessage", {"chat_id": chat_id, "text": chunk})

    def run(self) -> None:
        offset = 0
        while True:
            try:
                updates = self._call(
                    "getUpdates", {"timeout": 30, "offset": offset}, timeout=40.0
                )
                for update in updates:
                    offset = max(offset, int(update["update_id"]) + 1)
                    message = update.get("message") or update.get("edited_message")
                    if not message or not message.get("text"):
                        continue
                    chat_id = int(message["chat"]["id"])
                    text = str(message["text"]).strip()
                    if text in {"/start", "/help"}:
                        self.send_message(
                            chat_id,
                            "请直接发送剧情问题。回答只使用当前数据库中的证据。",
                        )
                        continue
                    try:
                        result = self.query_engine.query(text)
                        self.send_message(chat_id, result["answer"])
                    except Exception as exc:
                        self.send_message(chat_id, f"查询失败：{exc}")
            except KeyboardInterrupt:
                return
            except Exception:
                time.sleep(2)

