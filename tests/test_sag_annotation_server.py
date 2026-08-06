import http.client
import json
import threading
import unittest

from ragflow_style_pipeline.sag_annotation_server import create_workbench_server


class _Store:
    def __init__(self):
        self.saved = None

    def summary(self):
        return {
            "records": 2,
            "status_counts": {"in_progress": 2, "completed": 0},
            "revision": "sha256:" + "a" * 64,
            "annotator": "annotator-a",
        }

    def record(self, index):
        if index != 0:
            raise ValueError("bad index")
        return {
            "index": 0,
            "records": 2,
            "subset": "production",
            "clean_fields": {"case_content_clean": "脱敏文本"},
            "metadata": {},
            "issues": [],
            "declared_intents": [],
            "direct_emotions": [],
            "satisfaction": {"label": "unknown", "target": "", "evidence": ""},
            "urgency": {"level": "normal", "evidence": ""},
            "annotation": {"status": "in_progress", "notes": ""},
            "revision": "sha256:" + "a" * 64,
        }

    def save(self, index, revision, payload):
        self.saved = (index, revision, payload)
        return {
            "saved": True,
            "validation": {"errors": [], "warnings": []},
            "revision": "sha256:" + "b" * 64,
            "status_counts": {"in_progress": 1, "completed": 1},
        }


class TestAnnotationServer(unittest.TestCase):
    def setUp(self):
        self.store = _Store()
        self.server = create_workbench_server(
            self.store,
            port=0,
            session_token="private-session-token",
            bootstrap_token="private-bootstrap-token",
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host = self.server.expected_host

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        request_headers = {"Host": self.host, **(headers or {})}
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        data = response.read()
        values = {key.lower(): value for key, value in response.getheaders()}
        connection.close()
        return response.status, values, data

    def test_bootstrap_sets_http_only_cookie_and_protects_api(self):
        status, _headers, body = self._request("GET", "/api/summary")
        self.assertEqual(status, 403)
        self.assertNotIn(b"private-bootstrap-token", body)
        status, headers, _body = self._request(
            "GET", "/?token=private-bootstrap-token"
        )
        self.assertEqual(status, 303)
        self.assertEqual(headers["location"], "/")
        self.assertIn("HttpOnly", headers["set-cookie"])
        self.assertIn("SameSite=Strict", headers["set-cookie"])
        cookie = headers["set-cookie"].split(";", 1)[0]
        self.assertEqual(cookie, "SAGWorkbench=private-session-token")
        second_status, _second_headers, _second_body = self._request(
            "GET", "/?token=private-bootstrap-token"
        )
        self.assertEqual(second_status, 403)
        status, headers, body = self._request(
            "GET", "/api/summary", headers={"Cookie": cookie}
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["cache-control"], "no-store, max-age=0")
        self.assertIn("default-src 'none'", headers["content-security-policy"])
        self.assertEqual(json.loads(body)["records"], 2)

    def test_serves_chinese_annotation_guide_and_labels(self):
        status, headers, _body = self._request(
            "GET", "/?token=private-bootstrap-token"
        )
        self.assertEqual(status, 303)
        cookie = headers["set-cookie"].split(";", 1)[0]
        status, _headers, html = self._request(
            "GET", "/", headers={"Cookie": cookie}
        )
        self.assertEqual(status, 200)
        decoded_html = html.decode("utf-8")
        self.assertIn("第一次使用？点这里看填写步骤", decoded_html)
        self.assertIn("事实与诉求单元", decoded_html)
        status, _headers, javascript = self._request(
            "GET", "/app.js", headers={"Cookie": cookie}
        )
        self.assertEqual(status, 200)
        decoded_javascript = javascript.decode("utf-8")
        self.assertIn('value: "problem", label: "问题/故障', decoded_javascript)
        self.assertIn("未保存，请按下面提示修改", decoded_javascript)

    def test_rejects_wrong_host_and_cross_origin_post(self):
        status, _headers, _body = self._request(
            "GET", "/?token=private-bootstrap-token", headers={"Host": "localhost:1"}
        )
        self.assertEqual(status, 403)
        cookie = "SAGWorkbench=private-session-token"
        body = json.dumps({"revision": "r", "payload": {}})
        status, _headers, _body = self._request(
            "POST", "/api/records/0", body=body,
            headers={
                "Cookie": cookie,
                "Origin": "http://evil.invalid",
                "Content-Type": "application/json",
            },
        )
        self.assertEqual(status, 403)
        self.assertIsNone(self.store.saved)

    def test_same_origin_post_delegates_to_store(self):
        cookie = "SAGWorkbench=private-session-token"
        body = json.dumps({"revision": "r", "payload": {"safe": True}})
        status, _headers, response = self._request(
            "POST", "/api/records/0", body=body,
            headers={
                "Cookie": cookie,
                "Origin": "http://" + self.host,
                "Content-Type": "application/json",
            },
        )
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(response)["saved"])
        self.assertEqual(self.store.saved, (0, "r", {"safe": True}))


if __name__ == "__main__":
    unittest.main()
