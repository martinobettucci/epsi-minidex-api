import base64
import os
import random
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, Deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------
# Configuration
# ----------------------------
IMAGES_DIR = os.getenv("POKEMON_IMAGES_DIR", "./pokemon")
IMAGE_PATTERN = os.getenv("POKEMON_IMAGE_PATTERN", "image_{:02d}.png")
MIN_INDEX = int(os.getenv("POKEMON_MIN_INDEX", "0"))
MAX_INDEX = int(os.getenv("POKEMON_MAX_INDEX", "5"))
BEARER_TOKEN = os.getenv("POKEMON_BEARER_TOKEN", "").strip()
RATE_LIMIT_PER_MIN = int(os.getenv("POKEMON_RATE_LIMIT_PER_MIN", "60"))
WINDOW_SECONDS = 60

RARITY_BUCKETS = [
    ("F", 24), ("E", 20), ("D", 16), ("C", 12), ("B", 10), ("A", 8), ("S", 5), ("S+", 1)
]

POKEMON_NAMES = [
    "Voltadraco", "Aquapyre", "Floraclaw", "Terravault", "Cryosting", "Luminox",
    "Pyroquill", "Nébulo", "Ferromite", "Galefang", "Noctyx", "Solamar",
]

app = FastAPI(title="Pokémon Image Generator API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

_requests_window: Dict[str, Deque[datetime]] = defaultdict(deque)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _new_id() -> str:
    u = uuid.uuid4()
    b32 = base64.b32encode(u.bytes).decode("ascii").rstrip("=")
    return f"pkm_{b32}"


def _pick_name() -> str:
    if random.random() < 0.7:
        return random.choice(POKEMON_NAMES)
    syl_a = ["Vo", "Aqua", "Flora", "Terra", "Cryo", "Lumi", "Pyro", "Nébu", "Ferro", "Gale", "Noct", "Sola"]
    syl_b = ["ta", "py", "ra", "va", "sto", "nox", "quil", "lo", "mite", "fang", "yx", "mar"]
    return random.choice(syl_a) + random.choice(syl_b)


def _pick_rarity() -> str:
    labels, weights = zip(*RARITY_BUCKETS)
    return random.choices(labels, weights=weights, k=1)[0]


def _choose_image_path() -> str:
    idx = random.randint(MIN_INDEX, MAX_INDEX)
    filename = IMAGE_PATTERN.format(idx)
    path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image introuvable: {path}")
    return path


def _auth_check(req: Request) -> None:
    if BEARER_TOKEN:
        auth = req.headers.get("authorization") or req.headers.get("Authorization")
        if not auth or not auth.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Authorization bearer token requis.")
        token = auth.split(" ", 1)[1].strip()
        if token != BEARER_TOKEN:
            raise HTTPException(status_code=401, detail="Jeton d'accès invalide.")


def _rate_limit_check(client_id: str) -> None:
    if RATE_LIMIT_PER_MIN <= 0:
        return
    now = datetime.now(timezone.utc)
    window = _requests_window[client_id]
    while window and (now - window[0]).total_seconds() > WINDOW_SECONDS:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail="Trop de requêtes, réessayez plus tard.")
    window.append(now)


def _error_response(code: int, internal_code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content={
            "error": {
                "code": internal_code,
                "message": message,
                "timestamp": _now_utc_iso(),
            }
        },
    )


@app.get("/v1/generate")
async def generate(req: Request):
    try:
        _auth_check(req)
        client_ip = req.client.host if req.client else "anonymous"
        _rate_limit_check(client_ip)
        img_path = _choose_image_path()
        with open(img_path, "rb") as f:
            raw = f.read()
        img_b64 = base64.b64encode(raw).decode("ascii")
        payload = {
            "imageBase64": img_b64,
            "metadata": {
                "id": _new_id(),
                "name": _pick_name(),
                "rarity": _pick_rarity(),
            },
            "generatedAt": _now_utc_iso(),
        }
        return JSONResponse(status_code=200, content=payload)

    except HTTPException as http_exc:
        if http_exc.status_code == 401:
            return _error_response(401, "UNAUTHORIZED", str(http_exc.detail))
        if http_exc.status_code == 429:
            return _error_response(429, "RATE_LIMITED", str(http_exc.detail))
        return _error_response(http_exc.status_code, "HTTP_ERROR", str(http_exc.detail))

    except FileNotFoundError as e:
        return _error_response(500, "GENERATION_FAILED", f"{e}")

    except Exception:
        return _error_response(500, "GENERATION_FAILED", "Une erreur est survenue lors de la génération du Pokémon.")


@app.get("/health")
async def health():
    return {"status": "ok", "time": _now_utc_iso()}


if __name__ == "__main__":
    import uvicorn

    port = 22222
    uvicorn.run(
        "static-server:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        ssl_keyfile="certs/key.pem",   # path to your private key
        ssl_certfile="certs/cert.pem"  # path to your public certificate
    )