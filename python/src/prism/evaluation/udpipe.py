"""Minimal client for reproducible UDPipe REST benchmark predictions."""

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_UDPIPE_REST_ENDPOINT = "https://lindat.mff.cuni.cz/services/udpipe/api/process"


@dataclass(frozen=True, slots=True, kw_only=True)
class UdpipeRestClient:
    endpoint: str = DEFAULT_UDPIPE_REST_ENDPOINT
    timeout_seconds: float = 120.0

    def tag_gold_tokenized_conllu(
        self,
        *,
        model: str,
        conllu: str,
    ) -> str:
        if not model or model.strip() != model:
            raise ValueError("UDPipe model name must be non-empty and trimmed.")
        if not conllu.strip():
            raise ValueError("UDPipe input must contain CoNLL-U data.")

        request = Request(
            self.endpoint,
            data=urlencode(
                {
                    "data": conllu,
                    "input": "conllu",
                    "model": model,
                    "output": "conllu",
                    "tagger": "",
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "Prism-UDPipe-benchmark/1",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as error:
            raise RuntimeError(f"UDPipe REST request failed: {error}") from error

        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, str) or not result.strip():
            raise RuntimeError("UDPipe REST response does not contain CoNLL-U output.")
        return result
