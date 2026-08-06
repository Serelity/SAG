"""Private loopback HTTP server for the SAG annotation workbench."""

import hmac
import json
from http import cookies
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from ragflow_style_pipeline.sag_annotation_workbench import (
    AnnotationStoreError,
    new_session_token,
)

_ASSET_DIR = Path(__file__).with_name("sag_annotation_assets")
_MAX_REQUEST_BYTES = 1024 * 1024


class _LoopbackServer(HTTPServer):
    allow_reuse_address = False


class WorkbenchHandler(BaseHTTPRequestHandler):
    server_version = "SAGAnnotationWorkbench/1"
    sys_version = ""

    def log_message(self, _format, *_args):
        return

    def _security_headers(self, content_type):
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'none'; base-uri 'none'; "
            "form-action 'none'; frame-ancestors 'none'",
        )

    def _json(self, status, value):
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self._security_headers("application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _host_valid(self):
        return self.headers.get("Host", "") == self.server.expected_host

    def _cookie_token(self):
        jar = cookies.SimpleCookie()
        try:
            jar.load(self.headers.get("Cookie", ""))
        except cookies.CookieError:
            return ""
        morsel = jar.get("SAGWorkbench")
        return morsel.value if morsel else ""

    def _authorized(self):
        return self._host_valid() and hmac.compare_digest(
            self._cookie_token(), self.server.session_token
        )

    def _origin_valid(self):
        return self.headers.get("Origin", "") == "http://" + self.server.expected_host

    def _authorize_initial_request(self, parsed):
        query = parse_qs(parsed.query, keep_blank_values=True)
        supplied = query.get("token", [""])[0]
        if (
            parsed.path != "/" or not self._host_valid()
            or not self.server.bootstrap_token
            or not hmac.compare_digest(supplied, self.server.bootstrap_token)
        ):
            return False
        self.server.bootstrap_token = ""
        self.send_response(303)
        self._security_headers("text/plain; charset=utf-8")
        self.send_header("Location", "/")
        self.send_header(
            "Set-Cookie",
            "SAGWorkbench=" + self.server.session_token
            + "; Path=/; HttpOnly; SameSite=Strict",
        )
        self.send_header("Content-Length", "0")
        self.end_headers()
        return True

    def _asset(self, name, content_type):
        path = (_ASSET_DIR / name).resolve()
        if path.parent != _ASSET_DIR.resolve() or not path.is_file():
            self._json(404, {"error": "not_found"})
            return
        body = path.read_bytes()
        self.send_response(200)
        self._security_headers(content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlsplit(self.path)
        if not self._authorized():
            if self._authorize_initial_request(parsed):
                return
            self._json(403, {"error": "forbidden"})
            return
        if parsed.query:
            self._json(400, {"error": "unexpected_query"})
        elif parsed.path == "/":
            self._asset("index.html", "text/html; charset=utf-8")
        elif parsed.path == "/app.js":
            self._asset("app.js", "text/javascript; charset=utf-8")
        elif parsed.path == "/style.css":
            self._asset("style.css", "text/css; charset=utf-8")
        elif parsed.path == "/api/summary":
            self._json(200, self.server.store.summary())
        elif parsed.path.startswith("/api/records/"):
            try:
                index = int(parsed.path.rsplit("/", 1)[-1])
                self._json(200, self.server.store.record(index))
            except (ValueError, AnnotationStoreError) as exc:
                code = str(exc) if isinstance(exc, AnnotationStoreError) else "record_index_invalid"
                self._json(400, {"error": code})
        else:
            self._json(404, {"error": "not_found"})

    def do_POST(self):
        parsed = urlsplit(self.path)
        if not self._authorized() or not self._origin_valid():
            self._json(403, {"error": "forbidden"})
            return
        if parsed.query or not parsed.path.startswith("/api/records/"):
            self._json(404, {"error": "not_found"})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(415, {"error": "json_content_type_required"})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            length = -1
        if not 0 <= length <= _MAX_REQUEST_BYTES:
            self._json(413, {"error": "request_size_invalid"})
            return
        try:
            body = json.loads(self.rfile.read(length))
            index = int(parsed.path.rsplit("/", 1)[-1])
            if not isinstance(body, dict):
                raise ValueError
            result = self.server.store.save(
                index, body.get("revision"), body.get("payload")
            )
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid_json"})
            return
        except AnnotationStoreError as exc:
            code = str(exc)
            status = 409 if code in {
                "annotation_revision_conflict", "annotation_file_changed_externally"
            } else 400
            self._json(status, {"error": code})
            return
        except ValueError:
            self._json(400, {"error": "request_invalid"})
            return
        self._json(200 if result["saved"] else 422, result)


def create_workbench_server(store, port=0, session_token="", bootstrap_token=""):
    server = _LoopbackServer(("127.0.0.1", int(port)), WorkbenchHandler)
    server.store = store
    server.session_token = session_token or new_session_token()
    server.bootstrap_token = bootstrap_token or new_session_token()
    server.expected_host = f"127.0.0.1:{server.server_port}"
    return server
