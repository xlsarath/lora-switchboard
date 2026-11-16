"""Fake vLLM server used by the test-suite.

The implementation lives in the package (``lora_switchboard.fake_vllm``) so the
same fake also powers ``lora-switchboard fake-vllm`` and the docker-compose
demo. Run this file directly for a quick local server::

    python tests/fake_vllm.py --port 8001
"""

from lora_switchboard.fake_vllm import DEFAULT_BASE_MODEL, FakeState, create_fake_vllm

__all__ = ["DEFAULT_BASE_MODEL", "FakeState", "create_fake_vllm"]

if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--token-delay", type=float, default=0.02)
    args = parser.parse_args()
    uvicorn.run(create_fake_vllm(token_delay_s=args.token_delay), host=args.host, port=args.port)
