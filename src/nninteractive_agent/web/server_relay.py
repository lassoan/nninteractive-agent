"""A local relay between Claude Code and an Anthropic-compatible LLM server (LM Studio, vLLM, ...).

Claude Code sends models it does not know "system" messages in the middle of the conversation (the
environment description, for example), and sends them otherwise only if the server answers HTTP 400
or 422. The chat templates of many local models (Qwen's, for example) raise "System message must be
at the beginning" instead, which LM Studio reports as HTTP 500, so the request fails again on every
retry. The relay moves those messages into the system prompt before passing the request on, and
streams the server's answer back unchanged.
"""

from __future__ import annotations

import http.server
import json
import logging
import threading
import urllib.error
import urllib.request

log = logging.getLogger(__name__)

_HOP_HEADERS = ("host", "content-length", "connection", "transfer-encoding", "keep-alive")
_relays: dict = {}  # target url -> server
_lock = threading.Lock()


def move_system_messages(request: dict) -> dict:
    """``request`` (a Messages API request body) with its "system" messages moved to the end of its
    system prompt."""
    messages = request.get("messages") or []
    moved = [m for m in messages if m.get("role") == "system"]
    if not moved:
        return request
    system = request.get("system") or []
    system = [{"type": "text", "text": system}] if isinstance(system, str) else list(system)
    for message in moved:
        content = message.get("content")
        if isinstance(content, str):
            system.append({"type": "text", "text": content})
        else:
            system += [{"type": "text", "text": b["text"]} for b in content or [] if b.get("type") == "text"]
    return {**request, "system": system, "messages": [m for m in messages if m.get("role") != "system"]}


def _handler_class(target: str):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # the target is reached directly

    class Handler(http.server.BaseHTTPRequestHandler):
        def _relay(self):
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length) if length else None
            if body and self.path.split("?")[0].endswith("/messages"):
                try:
                    body = json.dumps(move_system_messages(json.loads(body))).encode()
                except (ValueError, AttributeError, KeyError, TypeError):
                    pass  # not a request we understand: pass it on as it is
            headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP_HEADERS}
            request = urllib.request.Request(target + self.path, data=body, method=self.command, headers=headers)
            try:
                response = opener.open(request, timeout=3600)
            except urllib.error.HTTPError as e:
                response = e
            except OSError as e:
                self.send_error(502, f"relay: {target} is not reachable ({e})")
                return
            with response:
                self.send_response(response.status if hasattr(response, "status") else response.code)
                for k, v in response.headers.items():
                    if k.lower() not in _HOP_HEADERS:
                        self.send_header(k, v)
                self.send_header("Connection", "close")
                self.end_headers()
                while chunk := response.read1(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()

        do_GET = do_POST = do_PUT = do_DELETE = _relay

        def log_message(self, format, *args):  # noqa: A002 - the base class's name
            log.debug("relay: " + format, *args)

    return Handler


def relay_url(target: str) -> str:
    """The URL of a relay on this machine to ``target`` (started on first use)."""
    target = target.rstrip("/")
    with _lock:
        server = _relays.get(target)
        if server is None:
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _handler_class(target))
            server.daemon_threads = True
            threading.Thread(target=server.serve_forever, name="llm-relay", daemon=True).start()
            _relays[target] = server
    return f"http://127.0.0.1:{server.server_address[1]}"
