import base64
import os
import random
import uuid
import threading
import time
import logging
from io import BytesIO
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, Deque, Optional

import json
import re
import requests

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

GEN_BACKEND = os.getenv("POKEMON_GENERATION_BACKEND", "files").lower().strip()

BEARER_TOKEN = os.getenv("POKEMON_BEARER_TOKEN", "").strip()
RATE_LIMIT_PER_MIN = int(os.getenv("POKEMON_RATE_LIMIT_PER_MIN", "60"))
WINDOW_SECONDS = 60

SDXL_MODEL_ID = os.getenv("SDXL_TURBO_MODEL", "stabilityai/sdxl-turbo")
SDXL_WIDTH = int(os.getenv("SDXL_WIDTH", "512"))
SDXL_HEIGHT = int(os.getenv("SDXL_HEIGHT", "512"))
SDXL_STEPS = int(os.getenv("SDXL_STEPS", "1"))

RARITY_BUCKETS = [
    ("F", 24), ("E", 20), ("D", 16), ("C", 12), ("B", 10), ("A", 8), ("S", 5), ("S+", 1)
]

POKEMON_NAMES = [
    "Voltadraco", "Aquapyre", "Floraclaw", "Terravault", "Cryosting", "Luminox",
    "Pyroquill", "Nébulo", "Ferromite", "Galefang", "Noctyx", "Solamar",
]

# --- Backend de nom (OpenAI-compatible / Ollama) ---
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://192.168.0.37:11434/v1").rstrip("/")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "llama3.2:1b")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy")
NAME_BACKEND = os.getenv("POKEMON_NAME_BACKEND", "local").lower().strip()  # "remote" | "local"
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "5.0"))  # secondes

# --- Logging runtime flags ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()  # DEBUG/INFO/WARN/ERROR
LOG_JSON = os.getenv("LOG_JSON", "1") not in ("0", "false", "False")
LOG_REQUEST_BODY = os.getenv("LOG_REQUEST_BODY", "0") not in ("0", "false", "False")
LOG_REMOTE_CONTENT = os.getenv("LOG_REMOTE_CONTENT", "0") not in ("0", "false", "False")
LOG_IMAGE_B64 = os.getenv("LOG_IMAGE_B64", "0") not in ("0", "false", "False")  # dangereux
LOG_SAMPLE_IMAGE_BYTES = int(os.getenv("LOG_SAMPLE_IMAGE_BYTES", "0"))  # 0 = off, sinon N premiers bytes

# ----------------------------
# Logging setup
# ----------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            # merge keys without overriding base fields
            for k, v in record.extra.items():
                if k not in payload:
                    payload[k] = v
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

_logger = logging.getLogger("pokemon")
_handler = logging.StreamHandler()
_handler.setLevel(LOG_LEVEL)
if LOG_JSON:
    _handler.setFormatter(JsonFormatter())
else:
    _handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ"
    ))
_logger.setLevel(LOG_LEVEL)
if not _logger.handlers:
    _logger.addHandler(_handler)
_logger.propagate = False

def log_info(msg: str, **extra):
    _logger.info(msg, extra={"extra": extra} if extra else None)

def log_debug(msg: str, **extra):
    _logger.debug(msg, extra={"extra": extra} if extra else None)

def log_warning(msg: str, **extra):
    _logger.warning(msg, extra={"extra": extra} if extra else None)

def log_error(msg: str, **extra):
    _logger.error(msg, extra={"extra": extra} if extra else None)

# ----------------------------
# App
# ----------------------------
app = FastAPI(title="Pokémon Image Generator API", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

_requests_window: Dict[str, Deque[datetime]] = defaultdict(deque)
_PRELOADED_B64: Optional[list[str]] = None

# ----------------------------
# SDXL globals
# ----------------------------
_pipe = None
_device = "cpu"
_dtype = None
_sdxl_lock = threading.Lock()
_sdxl_warmed_up = False


# ----------------------------
# Helpers
# ----------------------------

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _new_id() -> str:
    b32 = base64.b32encode(uuid.uuid4().bytes).decode("ascii").rstrip("=")
    return f"pkm_{b32}"


def _extract_name(text: str) -> str:
    """
    Tente d'extraire un nom depuis du JSON strict ou depuis du texte libre.
    Renvoie un fallback lisible en cas d'échec.
    """
    raw_len = len(text or "")
    log_debug("extract_name.input", raw_len=raw_len, preview=(text[:80] if text else ""))
    text = (text or "").strip()
    if not text:
        fallback = random.choice(POKEMON_NAMES)
        log_warning("extract_name.empty_text_fallback", fallback=fallback)
        return fallback
    
    # Nettoyer les accolades en trop (ex: }})
    while text.endswith("}}"):
        text = text[:-1]
    
    # Tentative JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "name" in obj and isinstance(obj["name"], str):
            nm = obj["name"].strip()
            if nm and nm.lower() != "name":  # ✅ Éviter le littéral "name"
                log_info("extract_name.json_ok", name=nm)
                return nm
    except Exception as e:
        log_debug("extract_name.json_parse_failed", error=str(e))
    
    # ✅ Extraction regex AVANT nettoyage brutal
    match = re.search(r'"name"\s*:\s*"([^"]+)"', text)
    if match:
        nm = match.group(1).strip()
        if nm and nm.lower() != "name":
            log_info("extract_name.regex_ok", name=nm)
            return nm
    
    # Nettoyage et prise du premier token valide qui n'est pas "name"
    cleaned = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ0-9\- ']", " ", text)
    cleaned = cleaned.strip().strip("\"'`""'")
    parts = [p for p in cleaned.split() if p.lower() != "name"]  # ✅ Filtrer "name"
    if parts:
        log_info("extract_name.cleaned_ok", token=parts[0], cleaned_preview=cleaned[:80])
        return parts[0]
    
    fallback = random.choice(POKEMON_NAMES)
    log_warning("extract_name.cleaned_empty_fallback", fallback=fallback)
    return fallback
    

def _pick_name_remote() -> str:
    """
    Appelle un endpoint OpenAI-compatible (ex: Ollama) au format JSON simple.
    Attends une réponse structurée, mais tolère aussi du texte libre.
    """
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "temperature": 1.05,
        "max_tokens": 30,  # ✅ Augmenté de 8 à 30
        "messages": [
            {
                "role": "system",
                "content": (
                    "Tu crées un unique nom original de créature façon 'Pokémon', "
                    "lisible en français ou pseudo-latin, sans reprendre un nom existant. "
                    "Réponds strictement au format JSON: {\"name\":\"<nom>\"} avec le nom direct exclusivement. "
                    "Exemple: {\"name\":\"GrandFork\"} ou {\"name\":\"Pyromatrix\"} ou encore {\"name\":\"Ectoplasmon\"}. "
                ),
            },
            {"role": "user", "content": "Génère un seul nom de créature."},
        ],
    }
    t0 = time.perf_counter()
    try:
        log_info("remote_name.request",
                 url=url, model=OPENAI_MODEL, timeout=OPENAI_TIMEOUT,
                 payload_size=len(json.dumps(payload, ensure_ascii=False)))
        resp = requests.post(url, headers=headers, json=payload, timeout=OPENAI_TIMEOUT)
        dt = time.perf_counter() - t0
        log_info("remote_name.response_meta",
                 status=resp.status_code, elapsed_ms=int(dt * 1000), content_length=len(resp.text))
        if resp.status_code != 200:
            snippet = resp.text[:300]
            log_warning("remote_name.bad_status", status=resp.status_code, snippet=snippet)
            raise RuntimeError(f"HTTP {resp.status_code}: {snippet}")
        data = resp.json()
        print(data)
        # Compatibilité avec OpenAI v1/chat/completions
        content = None
        if isinstance(data, dict):
            choices = data.get("choices")
            if choices and isinstance(choices, list):
                first = choices[0] if choices else {}
                if isinstance(first, dict):
                    msg = first.get("message")
                    if isinstance(msg, dict):
                        content = msg.get("content")
                    else:
                        content = first.get("text") or first.get("message")
        if content is None:
            content = resp.text
        if LOG_REMOTE_CONTENT:
            log_debug("remote_name.content", content_preview=content[:200])
        name = _extract_name(content)
        log_info("remote_name.ok", name=name)
        return name
    except Exception as e:
        dt = time.perf_counter() - t0
        log_error("remote_name.error", elapsed_ms=int(dt * 1000), error=str(e))
        raise RuntimeError(f"Remote name generation failed: {e}")


def _pick_name() -> str:
    """
    Orchestrateur: si NAME_BACKEND == "remote", tente l'appel distant
    et retombe sur la génération locale en cas d'échec.
    """
    log_debug("pick_name.start", backend=NAME_BACKEND)
    if NAME_BACKEND == "remote":
        try:
            name = _pick_name_remote()
            log_info("pick_name.remote_success", name=name)
            return name
        except Exception as e:
            log_warning("pick_name.remote_failed_fallback_local", error=str(e))

    # Génération locale (logique originale)
    if random.random() < 0.7:
        name = random.choice(POKEMON_NAMES)
    else:
        a = ["Vo", "Aqua", "Flora", "Terra", "Cryo", "Lumi", "Pyro", "Nébu", "Ferro", "Gale", "Noct", "Sola"]
        b = ["ta", "py", "ra", "va", "sto", "nox", "quil", "lo", "mite", "fang", "yx", "mar"]
        name = random.choice(a) + random.choice(b)
    log_info("pick_name.local", name=name)
    return name


def _pick_rarity() -> str:
    labels, weights = zip(*RARITY_BUCKETS)
    rarity = random.choices(labels, weights=weights, k=1)[0]
    log_debug("rarity.pick", rarity=rarity)
    return rarity


def _choose_image_path() -> str:
    idx = random.randint(MIN_INDEX, MAX_INDEX)
    filename = IMAGE_PATTERN.format(idx)
    path = os.path.join(IMAGES_DIR, filename)
    exists = os.path.exists(path)
    log_debug("image.choose_path", idx=idx, path=path, exists=exists)
    if not exists:
        raise FileNotFoundError(f"Image introuvable: {path}")
    return path


def _auth_check(req: Request) -> None:
    if BEARER_TOKEN:
        auth = req.headers.get("authorization") or req.headers.get("Authorization")
        auth_present = bool(auth)
        log_debug("auth.check", header_present=auth_present)
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
    # purge
    purged = 0
    while window and (now - window[0]).total_seconds() > WINDOW_SECONDS:
        window.popleft()
        purged += 1
    log_debug("ratelimit.window", client_id=client_id, purge_count=purged, current=len(window))
    if len(window) >= RATE_LIMIT_PER_MIN:
        log_warning("ratelimit.blocked", client_id=client_id, size=len(window))
        raise HTTPException(status_code=429, detail="Trop de requêtes, réessayez plus tard.")
    window.append(now)


def _error_response(code: int, internal_code: str, message: str) -> JSONResponse:
    log_error("error.response", status=code, code=internal_code, message=message)
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


# ----------------------------
# Image preload (files backend)
# ----------------------------

def _preload_images_base64():
    global _PRELOADED_B64
    if _PRELOADED_B64 is not None:
        log_debug("preload.skip_already_loaded", count=len(_PRELOADED_B64))
        return
    imgs = []
    for idx in range(MIN_INDEX, MAX_INDEX + 1):
        filename = IMAGE_PATTERN.format(idx)
        path = os.path.join(IMAGES_DIR, filename)
        if os.path.exists(path):
            with open(path, "rb") as f:
                imgs.append(base64.b64encode(f.read()).decode("ascii"))
    if not imgs:
        log_error("preload.no_images", dir=IMAGES_DIR, pattern=IMAGE_PATTERN)
        raise FileNotFoundError(f"Aucune image trouvée dans {IMAGES_DIR}")
    _PRELOADED_B64 = imgs
    log_info("preload.done", count=len(_PRELOADED_B64))


def _get_file_image_b64() -> str:
    img = random.choice(_PRELOADED_B64)
    if LOG_IMAGE_B64:
        log_debug("image.pick_base64", length=len(img))
    elif LOG_SAMPLE_IMAGE_BYTES > 0:
        try:
            raw = base64.b64decode(img)
            sample = raw[:LOG_SAMPLE_IMAGE_BYTES]
            log_debug("image.pick_sample", sample_len=len(sample))
        except Exception as e:
            log_warning("image.sample_decode_failed", error=str(e))
    return img


# ----------------------------
# SDXL Turbo backend (singleton)
# ----------------------------

def _ensure_sdxl_turbo():
    global _pipe, _device, _dtype
    if _pipe is not None:
        return _pipe

    with _sdxl_lock:
        if _pipe is not None:
            return _pipe
        try:
            import torch
            from diffusers import AutoPipelineForText2Image

            _device = "cuda" if torch.cuda.is_available() else (
                "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
            )
            _dtype = torch.bfloat16 if _device in ("cuda", "mps") else torch.float32

            kwargs = {"torch_dtype": _dtype, "use_safetensors": True}

            t0 = time.perf_counter()
            pipe = AutoPipelineForText2Image.from_pretrained(SDXL_MODEL_ID, **kwargs)
            pipe = pipe.to(_device)
            dt = time.perf_counter() - t0
            log_info("sdxl.loaded", model=SDXL_MODEL_ID, device=_device, dtype=str(_dtype), elapsed_ms=int(dt*1000))

            try:
                import torch as _t  # reuse symbol
                _t.backends.cudnn.benchmark = True
                log_debug("sdxl.cudnn_benchmark", enabled=True)
            except Exception as e:
                log_debug("sdxl.cudnn_benchmark_skip", reason=str(e))
            try:
                pipe.enable_attention_slicing()
                log_debug("sdxl.attention_slicing", enabled=True)
            except Exception as e:
                log_debug("sdxl.attention_slicing_skip", reason=str(e))

            _pipe = pipe
            return _pipe
        except Exception as e:
            log_error("sdxl.init_failed", error=str(e))
            raise RuntimeError(f"Initialisation SDXL Turbo échouée: {e}")


def _generate_with_sdxl(name: str) -> bytes:
    pipe = _ensure_sdxl_turbo()
    try:
        import torch
        global _sdxl_warmed_up

        if not _sdxl_warmed_up:
            with _sdxl_lock:
                if not _sdxl_warmed_up:
                    t0 = time.perf_counter()
                    _ = pipe(
                        prompt="warmup",
                        negative_prompt="",
                        width=SDXL_WIDTH,
                        height=SDXL_HEIGHT,
                        num_inference_steps=max(1, SDXL_STEPS),
                        guidance_scale=0.0,
                    ).images[0]
                    dt = time.perf_counter() - t0
                    log_info("sdxl.warmup_done", elapsed_ms=int(dt * 1000))
                    _sdxl_warmed_up = True

        seed = random.randint(0, 2**31 - 1)
        generator = torch.Generator(device=_device).manual_seed(seed)
        prompt = (
            f"pokemon-like highly coloured creature named {name}, clean solid single color background, high detail, studio lighting, cute, toyetic"
        )
        negative = "text, watermark, logo, nsfw, deformed, extra limbs, low quality, blurry"

        t0 = time.perf_counter()
        image = pipe(
            prompt=prompt,
            negative_prompt=negative,
            width=SDXL_WIDTH,
            height=SDXL_HEIGHT,
            num_inference_steps=max(1, SDXL_STEPS),
            guidance_scale=0.0,
            generator=generator,
        ).images[0]
        dt = time.perf_counter() - t0
        log_info("sdxl.generated", name=name, seed=seed, elapsed_ms=int(dt * 1000),
                 width=SDXL_WIDTH, height=SDXL_HEIGHT, steps=max(1, SDXL_STEPS))

        buf = BytesIO()
        image.save(buf, format="PNG")
        png = buf.getvalue()
        log_debug("sdxl.png_ready", size=len(png))
        return png
    except Exception as e:
        log_error("sdxl.generate_failed", name=name, error=str(e))
        raise RuntimeError(f"Génération SDXL échouée: {e}")


# ----------------------------
# Middleware de logging requêtes
# ----------------------------
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    req_id = _new_id()
    request.state.req_id = req_id
    t0 = time.perf_counter()

    # Inputs
    client = request.client.host if request.client else "unknown"
    headers = dict(request.headers)
    # Ne pas logguer les tokens
    headers.pop("authorization", None)
    headers.pop("Authorization", None)

    body_preview = ""
    if LOG_REQUEST_BODY:
        try:
            body_bytes = await request.body()
            body_preview = body_bytes[:400].decode("utf-8", errors="ignore")
        except Exception:
            body_preview = "<unreadable>"

    log_info("http.request",
             req_id=req_id, method=request.method, path=request.url.path,
             client_ip=client, query=str(request.url.query),
             headers_preview=str({k: headers[k] for k in list(headers)[:8]}),
             body_preview=body_preview)

    try:
        response = await call_next(request)
        dt = time.perf_counter() - t0
        log_info("http.response",
                 req_id=req_id, status=response.status_code,
                 elapsed_ms=int(dt * 1000))
        return response
    except Exception as e:
        dt = time.perf_counter() - t0
        log_error("http.exception", req_id=req_id, elapsed_ms=int(dt * 1000), error=str(e))
        raise


# ----------------------------
# Routes
# ----------------------------
@app.on_event("startup")
async def startup():
    config_snapshot = {
        "backend_image": GEN_BACKEND,
        "backend_name": NAME_BACKEND,
        "images_dir": IMAGES_DIR,
        "image_pattern": IMAGE_PATTERN,
        "min_index": MIN_INDEX,
        "max_index": MAX_INDEX,
        "sdxl_model": SDXL_MODEL_ID,
        "sdxl_wxh": f"{SDXL_WIDTH}x{SDXL_HEIGHT}",
        "sdxl_steps": SDXL_STEPS,
        "openai_base_url": OPENAI_BASE_URL,
        "openai_model": OPENAI_MODEL,
        "rate_limit_per_min": RATE_LIMIT_PER_MIN,
        "log_level": LOG_LEVEL,
        "log_json": LOG_JSON,
    }
    log_info("startup.config", **config_snapshot)

    try:
        if GEN_BACKEND == "files":
            _preload_images_base64()
        elif GEN_BACKEND == "sdxl":
            _ensure_sdxl_turbo()
        log_info("startup.ready")
    except Exception as e:
        log_error("startup.failed", error=str(e))
        raise


@app.get("/v1/generate")
async def generate(req: Request):
    req_id = getattr(req.state, "req_id", _new_id())
    try:
        _auth_check(req)
        client_ip = req.client.host if req.client else "anonymous"
        _rate_limit_check(client_ip)

        name = _pick_name()
        rarity = _pick_rarity()

        source = GEN_BACKEND
        t0 = time.perf_counter()
        if GEN_BACKEND == "sdxl":
            try:
                png_bytes = _generate_with_sdxl(name)
            except Exception as e:
                log_warning("generate.sdxl_failed_fallback_file", error=str(e))
                img_path = _choose_image_path()
                with open(img_path, "rb") as f:
                    png_bytes = f.read()
                source = "files-fallback"
            img_b64 = base64.b64encode(png_bytes).decode("ascii")
        else:
            img_b64 = _get_file_image_b64()
        dt = time.perf_counter() - t0

        payload = {
            "imageBase64": img_b64 if LOG_IMAGE_B64 else ("<redacted>" if not LOG_IMAGE_B64 else img_b64),
            "metadata": {
                "id": _new_id(),
                "name": name,
                "rarity": rarity,
            },
            "generatedAt": _now_utc_iso(),
        }

        log_info("generate.success",
                 req_id=req_id, name=name, rarity=rarity,
                 image_backend=source, elapsed_ms=int(dt * 1000),
                 image_b64_len=(len(img_b64) if not LOG_IMAGE_B64 else -1))

        # On renvoie l'image en clair malgré redaction logs
        payload["imageBase64"] = img_b64
        return JSONResponse(status_code=200, content=payload)

    except HTTPException as http_exc:
        log_warning("generate.http_exception", status=http_exc.status_code, detail=str(http_exc.detail))
        if http_exc.status_code == 401:
            return _error_response(401, "UNAUTHORIZED", str(http_exc.detail))
        if http_exc.status_code == 429:
            return _error_response(429, "RATE_LIMITED", str(http_exc.detail))
        return _error_response(http_exc.status_code, "HTTP_ERROR", str(http_exc.detail))

    except FileNotFoundError as e:
        return _error_response(500, "GENERATION_FAILED", f"{e}")

    except Exception as e:
        log_error("generate.unexpected_error", error=str(e))
        return _error_response(500, "GENERATION_FAILED", "Une erreur est survenue lors de la génération du Pokémon.")


@app.get("/health")
async def health():
    status = {"status": "ok", "time": _now_utc_iso(), "backend": GEN_BACKEND}
    log_debug("health.check", **status)
    return status


if __name__ == "__main__":
    import uvicorn

    port = 22222
    log_info("uvicorn.boot", host="0.0.0.0", port=port, reload=False)
    uvicorn.run(
        "gen-server:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # keep stable and fast
        ssl_keyfile="certs/key.pem",
        ssl_certfile="certs/cert.pem"
    )
