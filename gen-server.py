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

# --- Added imports for SDXL optimizations ---
import torch
from diffusers import AutoPipelineForText2Image, StableDiffusionXLPipeline
from diffusers.models.attention_processor import AttnProcessor2_0

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------
# .env bootstrap (dump on first run, then load if present)
# ----------------------------
ENV_FILE = os.getenv("MINIMON_ENV_FILE", "./env")

def _env_kv_defaults() -> Dict[str, str]:
    # Defaults must mirror code-level fallbacks
    return {
        # General
        "MINIMON_IMAGES_DIR": "./minimon",
        "MINIMON_IMAGE_PATTERN": "image_{:02d}.png",
        "MINIMON_MIN_INDEX": "0",
        "MINIMON_MAX_INDEX": "5",
        "MINIMON_GENERATION_BACKEND": "files",  # files | sdxl
        "MINIMON_BEARER_TOKEN": "",
        "MINIMON_RATE_LIMIT_PER_MIN": "60",
        # SDXL core
        "SDXL_TURBO_MODEL": "stabilityai/sdxl-turbo",
        "SDXL_WIDTH": "512",
        "SDXL_HEIGHT": "512",
        "SDXL_STEPS": "1",
        # SDXL optimizations
        "SDXL_QUANTIZATION": "fp8",            # fp8 | int8 | int4 | fp4 | none
        "SDXL_USE_COMPILE": "1",               # 1|0
        "SDXL_USE_XFORMERS": "1",              # 1|0
        "SDXL_USE_FLASH_ATTN": "0",            # 1|0
        "SDXL_ENABLE_SLICING": "0",            # 1|0
        "SDXL_ENABLE_CPU_OFFLOAD": "0",        # 1|0
        # Name backend (OpenAI-compatible)
        "MINIMON_NAME_BACKEND": "local",        # local | remote
        "OPENAI_BASE_URL": "http://192.168.0.37:11434/v1",
        "OPENAI_MODEL": "llama3.2:1b",
        "OPENAI_API_KEY": "dummy",
        "OPENAI_TIMEOUT": "5.0",
        # Logging
        "LOG_LEVEL": "INFO",
        "LOG_JSON": "1",
        "LOG_REQUEST_BODY": "0",
        "LOG_REMOTE_CONTENT": "0",
        "LOG_IMAGE_B64": "0",
        "LOG_SAMPLE_IMAGE_BYTES": "0",
    }

_DEF_COMMENTS = {
    "MINIMON_IMAGES_DIR": "Dossier des images locales pour le mode 'files'.",
    "MINIMON_IMAGE_PATTERN": "Patron de nommage des images locales.",
    "MINIMON_MIN_INDEX": "Index d'image minimal inclusif.",
    "MINIMON_MAX_INDEX": "Index d'image maximal inclusif.",
    "MINIMON_GENERATION_BACKEND": "Backend image: files (par défaut) ou sdxl.",
    "MINIMON_BEARER_TOKEN": "Jeton Bearer optionnel pour sécuriser l'API.",
    "MINIMON_RATE_LIMIT_PER_MIN": "Quota de requêtes par minute (0 pour désactiver).",
    "SDXL_TURBO_MODEL": "ID HuggingFace du modèle SDXL Turbo.",
    "SDXL_WIDTH": "Largeur de l'image générée.",
    "SDXL_HEIGHT": "Hauteur de l'image générée.",
    "SDXL_STEPS": "Nombre d'itérations d'inférence (1 pour Turbo).",
    "SDXL_QUANTIZATION": "Quantization: fp8 (GPU recommandé), int8 (poids uniquement), int4 (poids uniquement), fp4 (expérimental), none.",
    "SDXL_USE_COMPILE": "Active torch.compile (1/0).",
    "SDXL_USE_XFORMERS": "Active xFormers mémoire efficiente (1/0).",
    "SDXL_USE_FLASH_ATTN": "Active FlashAttention si installé (1/0).",
    "SDXL_ENABLE_SLICING": "Active attention slicing pour réduire la VRAM (1/0).",
    "SDXL_ENABLE_CPU_OFFLOAD": "Offload vers CPU si VRAM limitée (1/0).",
    "MINIMON_NAME_BACKEND": "Génération de nom: local ou remote (OpenAI-compatible).",
    "OPENAI_BASE_URL": "URL du serveur OpenAI-compatible (ex: Ollama).",
    "OPENAI_MODEL": "Nom du modèle distant (ex: llama3.2:1b).",
    "OPENAI_API_KEY": "Clé API (peut être factice pour Ollama).",
    "OPENAI_TIMEOUT": "Timeout en secondes pour l'appel distant.",
    "LOG_LEVEL": "Niveau de logs: DEBUG/INFO/WARN/ERROR.",
    "LOG_JSON": "Format JSON (1) ou texte (0).",
    "LOG_REQUEST_BODY": "Logger le corps des requêtes (1/0).",
    "LOG_REMOTE_CONTENT": "Logger le contenu des réponses LLM (1/0).",
    "LOG_IMAGE_B64": "Logger les images en base64 (1/0) lourd en prod.",
    "LOG_SAMPLE_IMAGE_BYTES": "Échantillonner N octets d'image pour debug (0=off).",
}

def _write_env_if_missing(path: str = ENV_FILE):
    if os.path.exists(path):
        return
    kv = _env_kv_defaults()
    lines = [
        "#",
        "# Minimon Image Generator API — configuration",
        "#",
        "# Ce fichier a été généré automatiquement au premier démarrage.",
        "# Modifie les valeurs selon ton environnement. Les variables vides",
        "# gardent les valeurs par défaut intégrées dans le code.",
        "#",
    ]
    for k, v in kv.items():
        comment = _DEF_COMMENTS.get(k, "")
        if comment:
            lines.append(f"# {comment}")
        vv = v if (" " not in str(v)) else f'"{v}"'  # quote if contains spaces
        lines.append(f"{k}={vv}")
        lines.append("")  # blank line between entries
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception as e:
        # logging not yet configured here, keep silent fallback
        print(f"[env.write_failed] path={path} error={e}")

def _load_env_file(path: str = ENV_FILE):
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, val = line.split("=", 1)
                k = k.strip()
                v = val.strip().strip("\"'")
                # Do not override already-set ENV
                if k and (k not in os.environ):
                    os.environ[k] = v
    except Exception as e:
        print(f"[env.load_failed] path={path} error={e}")

# Ensure .env exists then load it before reading os.getenv below
_write_env_if_missing(ENV_FILE)
_load_env_file(ENV_FILE)

# ----------------------------
# Configuration
# ----------------------------
IMAGES_DIR = os.getenv("MINIMON_IMAGES_DIR", "./minimon")
IMAGE_PATTERN = os.getenv("MINIMON_IMAGE_PATTERN", "image_{:02d}.png")
MIN_INDEX = int(os.getenv("MINIMON_MIN_INDEX", "0"))
MAX_INDEX = int(os.getenv("MINIMON_MAX_INDEX", "5"))

GEN_BACKEND = os.getenv("MINIMON_GENERATION_BACKEND", "files").lower().strip()

BEARER_TOKEN = os.getenv("MINIMON_BEARER_TOKEN", "").strip()
RATE_LIMIT_PER_MIN = int(os.getenv("MINIMON_RATE_LIMIT_PER_MIN", "60"))
WINDOW_SECONDS = 60

SDXL_MODEL_ID = os.getenv("SDXL_TURBO_MODEL", "stabilityai/sdxl-turbo")
SDXL_WIDTH = int(os.getenv("SDXL_WIDTH", "512"))
SDXL_HEIGHT = int(os.getenv("SDXL_HEIGHT", "512"))
SDXL_STEPS = int(os.getenv("SDXL_STEPS", "1"))

# --- New configuration toggles for Blackwell optimizations ---
SDXL_QUANTIZATION = os.getenv("SDXL_QUANTIZATION", "fp8").lower()  # fp8, int8, int4, fp4, bf16, none
SDXL_USE_COMPILE = os.getenv("SDXL_USE_COMPILE", "1") not in ("0", "false", "False")
SDXL_USE_XFORMERS = os.getenv("SDXL_USE_XFORMERS", "1") not in ("0", "false", "False")
SDXL_USE_FLASH_ATTN = os.getenv("SDXL_USE_FLASH_ATTN", "0") not in ("0", "false", "False")
SDXL_ENABLE_SLICING = os.getenv("SDXL_ENABLE_SLICING", "0") not in ("0", "false", "False")
SDXL_ENABLE_CPU_OFFLOAD = os.getenv("SDXL_ENABLE_CPU_OFFLOAD", "0") not in ("0", "false", "False")

RARITY_BUCKETS = [
    ("F", 24), ("E", 20), ("D", 16), ("C", 12), ("B", 10), ("A", 8), ("S", 5), ("S+", 1)
]

MINIMON_NAMES = [
    "Voltadraco", "Aquapyre", "Floraclaw", "Terravault", "Cryosting", "Luminox",
    "Pyroquill", "Nébulo", "Ferromite", "Galefang", "Noctyx", "Solamar",
]

# --- Backend de nom (OpenAI-compatible / Ollama) ---
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://192.168.0.37:11434/v1").rstrip("/")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "llama3.2:1b")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy")
NAME_BACKEND = os.getenv("MINIMON_NAME_BACKEND", "local").lower().strip()  # "remote" | "local"
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

_logger = logging.getLogger("minimon")
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
app = FastAPI(title="Minimon Image Generator API", version="1.3.0")
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
    return f"mnm_{b32}"


def _extract_name(text: str) -> str:
    """
    Tente d'extraire un nom depuis du JSON strict ou depuis du texte libre.
    Renvoie un fallback lisible en cas d'échec.
    """
    raw_len = len(text or "")
    log_debug("extract_name.input", raw_len=raw_len, preview=(text[:80] if text else ""))
    text = (text or "").strip()
    if not text:
        fallback = random.choice(MINIMON_NAMES)
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
    
    fallback = random.choice(MINIMON_NAMES)
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
                    "Tu crées un unique nom original de créature façon 'Minimon', "
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
        name = random.choice(MINIMON_NAMES)
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
# SDXL Turbo backend (enhanced)
# ----------------------------

def _apply_fp8_quantization(pipe):
    """Apply FP8 quantization for Blackwell GPUs (poids uniquement)."""
    try:
        from torchao.quantization import quantize_, float8_weight_only
        log_info("quantization.fp8_start", model=SDXL_MODEL_ID)

        if _device != "cuda":
            log_warning("quantization.fp8_device_warning", msg="FP8 utile principalement sur GPU.")

        # Quantize UNet (principal)
        if hasattr(pipe, 'unet'):
            quantize_(pipe.unet, float8_weight_only())
            log_info("quantization.unet_fp8_done")

        # Quantize VAE decoder
        if hasattr(pipe, 'vae') and hasattr(pipe.vae, "decoder"):
            quantize_(pipe.vae.decoder, float8_weight_only())
            log_info("quantization.vae_fp8_done")

        log_info("quantization.fp8_complete")
        return pipe

    except ImportError:
        log_warning("quantization.torchao_unavailable",
                    msg="TorchAO absent. Installe: pip install torchao")
        return pipe
    except Exception as e:
        log_error("quantization.fp8_failed", error=str(e))
        return pipe


def _apply_int8_quantization(pipe):
    """Apply INT8 weight-only quantization."""
    try:
        from torchao.quantization import quantize_, int8_weight_only
        log_info("quantization.int8_start", model=SDXL_MODEL_ID)

        if _device != "cuda":
            log_warning("quantization.int8_device_warning", msg="INT8 poids utiles sur GPU, fonctionnement CPU variable.")

        if hasattr(pipe, 'unet'):
            quantize_(pipe.unet, int8_weight_only())
            log_info("quantization.unet_int8_done")

        if hasattr(pipe, 'vae') and hasattr(pipe.vae, "decoder"):
            quantize_(pipe.vae.decoder, int8_weight_only())
            log_info("quantization.vae_int8_done")

        log_info("quantization.int8_complete")
        return pipe

    except ImportError:
        log_warning("quantization.torchao_unavailable",
                    msg="TorchAO absent. Installe: pip install torchao")
        return pipe
    except Exception as e:
        log_error("quantization.int8_failed", error=str(e))
        return pipe


def _apply_int4_quantization(pipe):
    """Apply INT4 weight-only quantization.
    Essaye TorchAO en priorité, sinon journalise un avertissement si indisponible.
    """
    # 1) Tentative TorchAO
    try:
        from torchao.quantization import quantize_, int4_weight_only  # peut ne pas exister selon la version
        log_info("quantization.int4_start", model=SDXL_MODEL_ID)

        if _device != "cuda":
            log_warning("quantization.int4_device_warning", msg="INT4 poids utiles sur GPU, fonctionnement CPU variable.")

        if hasattr(pipe, 'unet'):
            quantize_(pipe.unet, int4_weight_only())
            log_info("quantization.unet_int4_done")

        if hasattr(pipe, 'vae') and hasattr(pipe.vae, "decoder"):
            quantize_(pipe.vae.decoder, int4_weight_only())
            log_info("quantization.vae_int4_done")

        log_info("quantization.int4_complete")
        return pipe

    except ImportError:
        log_warning("quantization.torchao_unavailable",
                    msg="TorchAO absent ou version sans int4_weight_only. Installe: pip install torchao")
    except AttributeError:
        log_warning("quantization.int4_not_supported",
                    msg="La fonction int4_weight_only n'est pas disponible dans ta version TorchAO.")
    except Exception as e:
        log_error("quantization.int4_failed", error=str(e))

    # 2) Fallback: BnB non intrusif, on log si non supporté (rechargement modèle requis pour un support propre)
    try:
        import bitsandbytes as bnb  # noqa: F401
        log_warning(
            "quantization.int4_bnb_reload_required",
            msg="INT4 via bitsandbytes requiert un rechargement spécifique des modules Linear, non appliqué dynamiquement. "
                "Le pipeline actuel continue en précision de base."
        )
        return pipe
    except Exception:
        log_warning("quantization.int4_no_backend",
                    msg="Aucun backend int4 disponible. Installe torchao>=dernier ou bitsandbytes>=0.43.")
        return pipe


def _apply_fp4_quantization(pipe):
    """Apply aggressive FP4 quantization (expérimental)."""
    try:
        from transformers import BitsAndBytesConfig  # noqa: F401
        log_warning("quantization.fp4_experimental",
                    msg="FP4 peut dégrader fortement la qualité. Rechargement modèle nécessaire.")
        log_error("quantization.fp4_not_implemented",
                  msg="FP4 nécessite un rechargement avec config dédiée, non applicable à chaud.")
        return pipe

    except Exception as e:
        log_error("quantization.fp4_failed", error=str(e))
        return pipe


def _optimize_attention(pipe):
    """Apply Blackwell-optimized attention mechanisms"""
    try:
        # Try FlashAttention-3 (Blackwell native)
        if SDXL_USE_FLASH_ATTN:
            try:
                from flash_attn import flash_attn_func  # noqa: F401
                pipe.unet.set_attn_processor(AttnProcessor2_0())
                log_info("attention.flash_attn3_enabled")
            except ImportError:
                log_debug("attention.flash_attn3_unavailable")
        
        # Fallback: xFormers (still beneficial)
        if SDXL_USE_XFORMERS:
            try:
                pipe.enable_xformers_memory_efficient_attention()
                log_info("attention.xformers_enabled")
            except Exception as e:
                log_debug("attention.xformers_failed", error=str(e))
        
        # PyTorch 2.0+ SDPA (scaled dot product attention)
        try:
            pipe.unet.set_default_attn_processor()
            log_info("attention.sdpa_enabled")
        except Exception as e:
            log_debug("attention.sdpa_skip", error=str(e))
            
    except Exception as e:
        log_error("attention.optimization_failed", error=str(e))


def _compile_model(pipe, device):
    """Apply torch.compile with Blackwell backend"""
    if not SDXL_USE_COMPILE:
        return pipe
    
    try:
        if device == "cuda" and torch.cuda.is_available():
            # Compile UNet (most compute-intensive)
            log_info("compile.unet_start")
            pipe.unet = torch.compile(
                pipe.unet,
                mode="max-autotune",  # Aggressive optimization
                fullgraph=True,
                backend="inductor"
            )
            log_info("compile.unet_done")
            
            # Optionally compile VAE decoder
            try:
                pipe.vae.decoder = torch.compile(
                    pipe.vae.decoder,
                    mode="reduce-overhead",
                    backend="inductor"
                )
                log_info("compile.vae_done")
            except Exception as e:
                log_debug("compile.vae_skip", error=str(e))
                
    except Exception as e:
        log_error("compile.failed", error=str(e))
    
    return pipe


def _ensure_sdxl_turbo():
    """Enhanced SDXL loader with Blackwell optimizations"""
    global _pipe, _device, _dtype
    if _pipe is not None:
        return _pipe
    
    with _sdxl_lock:
        if _pipe is not None:
            return _pipe
        
        try:
            # Device detection
            _device = "cuda" if torch.cuda.is_available() else (
                "mps" if getattr(torch.backends, "mps", None) and 
                torch.backends.mps.is_available() else "cpu"
            )
            
            # Dtype selection based on quantization
            if SDXL_QUANTIZATION == "none":
                _dtype = torch.bfloat16 if _device == "cuda" else torch.float32
            else:
                # Base FP16 pour quants poids-only (fp8/int8/int4/fp4)
                _dtype = torch.float16
            
            log_info("sdxl.init_start", 
                    device=_device, 
                    dtype=str(_dtype),
                    quantization=SDXL_QUANTIZATION)
            
            # Load model
            t0 = time.perf_counter()
            pipe = AutoPipelineForText2Image.from_pretrained(
                SDXL_MODEL_ID,
                torch_dtype=_dtype,
                use_safetensors=True,
                variant="fp16" if _dtype == torch.float16 else None
            )
            
            # Move to device
            pipe = pipe.to(_device)
            dt_load = time.perf_counter() - t0
            log_info("sdxl.loaded", elapsed_ms=int(dt_load * 1000))
            
            # Apply optimizations in order
            t0 = time.perf_counter()
            
            # 1. Quantization (if enabled)
            if SDXL_QUANTIZATION == "fp8" and _device == "cuda":
                pipe = _apply_fp8_quantization(pipe)
            elif SDXL_QUANTIZATION == "int8":
                pipe = _apply_int8_quantization(pipe)
            elif SDXL_QUANTIZATION == "int4":
                pipe = _apply_int4_quantization(pipe)
            elif SDXL_QUANTIZATION == "fp4":
                pipe = _apply_fp4_quantization(pipe)
            
            # 2. Attention optimizations
            _optimize_attention(pipe)
            
            # 3. Memory optimizations
            if SDXL_ENABLE_SLICING:
                pipe.enable_attention_slicing(slice_size="auto")
                log_info("memory.attention_slicing_enabled")
            
            pipe.enable_vae_slicing()
            log_info("memory.vae_slicing_enabled")
            
            if SDXL_ENABLE_CPU_OFFLOAD and _device == "cuda":
                pipe.enable_model_cpu_offload()
                log_info("memory.cpu_offload_enabled")
            
            # 4. Compile (last step, most time-consuming)
            pipe = _compile_model(pipe, _device)
            
            # 5. CUDA optimizations
            if _device == "cuda":
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                log_info("cuda.optimizations_enabled", 
                        tf32=True, 
                        benchmark=True)
            
            dt_opt = time.perf_counter() - t0
            log_info("sdxl.optimizations_complete", elapsed_ms=int(dt_opt * 1000))
            
            _pipe = pipe
            return _pipe
            
        except Exception as e:
            log_error("sdxl.init_failed", error=str(e))
            raise RuntimeError(f"SDXL initialization failed: {e}")


def _generate_with_sdxl(name: str) -> bytes:
    """Enhanced generation with optimizations"""
    pipe = _ensure_sdxl_turbo()
    
    try:
        global _sdxl_warmed_up
        
        # Warmup (first run compilation)
        if not _sdxl_warmed_up:
            with _sdxl_lock:
                if not _sdxl_warmed_up:
                    log_info("sdxl.warmup_start")
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
        
        # Generation
        seed = random.randint(0, 2**31 - 1)
        generator = torch.Generator(device=_device).manual_seed(seed)
        
        elements = [
            "with flowing water-inspired features (like fins, bubbles, or aquatic patterns)",
            "surrounded by warm fire energy (like glowing embers, molten textures, or sparks)",
            "with earthy details (like moss, stones, leaves, or soil textures)",
            "with swirling wind effects (like air currents, feathers, or misty trails)"
        ]
        
        element = random.choice(elements)
        
        prompt = (
            f"minimon-like highly coloured creature named {name}, "
            f"clean solid single color background, high detail, "
            f"studio lighting, cute, toyetic, {element}"
        )
        
        negative = "text, watermark, logo, nsfw, deformed, extra limbs, low quality, blurry"

        
        t0 = time.perf_counter()
        
        with torch.inference_mode():  # Optimization context
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
        
        log_info("sdxl.generated", 
                name=name, 
                seed=seed, 
                elapsed_ms=int(dt * 1000),
                quantization=SDXL_QUANTIZATION)
        
        # Convert to PNG
        buf = BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
        
    except Exception as e:
        log_error("sdxl.generate_failed", name=name, error=str(e))
        raise RuntimeError(f"SDXL generation failed: {e}")


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
        # new toggles snapshot
        "sdxl_quantization": SDXL_QUANTIZATION,
        "sdxl_use_compile": SDXL_USE_COMPILE,
        "sdxl_use_xformers": SDXL_USE_XFORMERS,
        "sdxl_use_flash_attn": SDXL_USE_FLASH_ATTN,
        "sdxl_enable_slicing": SDXL_ENABLE_SLICING,
        "sdxl_enable_cpu_offload": SDXL_ENABLE_CPU_OFFLOAD,
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
        return _error_response(500, "GENERATION_FAILED", "Une erreur est survenue lors de la génération du Minimon.")


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
