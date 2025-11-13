# 🧬 Minimon Image Generator API

**Minimon Image Generator API** est un microservice FastAPI capable de générer aléatoirement des créatures de type *Minimon*, avec image et métadonnées (nom, rareté, horodatage).

Il peut fonctionner avec :

* un backend **local** en mode fichiers statiques ;
* un backend **SDXL Turbo** intégrant de nombreuses optimisations Blackwell (FP8, int8/int4, FlashAttention, xFormers, torch.compile, slicing, CPU offload) ;
* un backend **OpenAI-compatible** (Ollama, LM Studio, vLLM…) pour générer dynamiquement les noms.

La version **1.3.0** apporte une refonte complète du pipeline SDXL, le support avancé de quantization et un nouveau système de certification cryptographique des scores.

---

## 🚀 Fonctionnalités principales

* **Génération d’images** via :

  * backend statique `files`
  * backend dynamique `sdxl`
* **Génération de noms** :

  * locale (suffixes/fragments internes)
  * distante via API OpenAI-compatible (`/chat/completions`)
* Système de rareté pondérée (F → S+)
* Authentification optionnelle par bearer token
* Rate limiting par IP
* Logs JSON ou texte
* Endpoint `/health`
* **Optimisations SDXL Blackwell-ready** :
  FP8, int8, int4, FP4 (expérimental), FlashAttention, xFormers, torch.compile, slicing attention/VAE, CPU offload
* Nouveau système **/v1/certify-score** :
  signature cryptographique, ledger append-only JSONL, compatibilité RSA/EC/Ed25519/Ed448.

---

## ⚙️ Installation

### 1. Dépendances API

```bash
uv add fastapi uvicorn requests cryptography
# ou
pip install fastapi uvicorn requests cryptography
```

### 2. Backend SDXL

```bash
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv add diffusers pillow
```

Options d’optimisation :

```bash
pip install torchao          # FP8 / int8 / int4
pip install flash-attn       # FlashAttention (CUDA dépendant)
pip install xformers         # attention mémoire efficiente
```

---

## 📁 Arborescence minimale

```
project/
│
├── gen-server.py
├── minimon/
│   ├── image_00.png
│   ├── image_01.png
│   └── ...
└── certs/
    ├── cert.pem
    ├── key.pem
    ├── app_signing_key.pem
    └── app_signing_cert.pem
```

---

## 🧩 Variables d’environnement

Les variables commencent toutes par `MINIMON_` dans le code.

### Backend images

| Variable                     | Description                | Défaut             |
| ---------------------------- | -------------------------- | ------------------ |
| `MINIMON_IMAGES_DIR`         | Dossier des images locales | `./minimon`        |
| `MINIMON_IMAGE_PATTERN`      | Modèle de nom              | `image_{:02d}.png` |
| `MINIMON_MIN_INDEX` / `MAX`  | Index min/max              | `0 → 5`            |
| `MINIMON_GENERATION_BACKEND` | `files` ou `sdxl`          | `files`            |

### Authentification et rate limit

| Variable                     | Description                | Défaut   |
| ---------------------------- | -------------------------- | -------- |
| `MINIMON_BEARER_TOKEN`       | Jeton Bearer facultatif    | *(vide)* |
| `MINIMON_RATE_LIMIT_PER_MIN` | Requêtes par minute par IP | `60`     |

### SDXL Turbo + optimisations Blackwell

| Variable                  | Description                          | Défaut                   |
| ------------------------- | ------------------------------------ | ------------------------ |
| `SDXL_TURBO_MODEL`        | Modèle SDXL Turbo                    | `stabilityai/sdxl-turbo` |
| `SDXL_WIDTH` / `HEIGHT`   | Résolution                           | `512`                    |
| `SDXL_STEPS`              | Itérations (Turbo = 1)               | `1`                      |
| `SDXL_QUANTIZATION`       | `fp8`, `int8`, `int4`, `fp4`, `none` | `fp8`                    |
| `SDXL_USE_COMPILE`        | Active torch.compile                 | `1`                      |
| `SDXL_USE_XFORMERS`       | Active xFormers                      | `1`                      |
| `SDXL_USE_FLASH_ATTN`     | Active FlashAttention                | `0`                      |
| `SDXL_ENABLE_SLICING`     | Attention slicing                    | `0`                      |
| `SDXL_ENABLE_CPU_OFFLOAD` | Offload GPU→CPU                      | `0`                      |

### Backend de noms OpenAI-compatible

| Variable               | Description         | Défaut                         |
| ---------------------- | ------------------- | ------------------------------ |
| `MINIMON_NAME_BACKEND` | `local` ou `remote` | `local`                        |
| `OPENAI_BASE_URL`      | URL API             | `http://192.168.0.37:11434/v1` |
| `OPENAI_MODEL`         | Modèle à invoquer   | `llama3.2:1b`                  |
| `OPENAI_API_KEY`       | Clé API             | `dummy`                        |
| `OPENAI_TIMEOUT`       | Timeout secondes    | `5.0`                          |

### Logging

| Variable                 | Description                | Défaut |
| ------------------------ | -------------------------- | ------ |
| `LOG_LEVEL`              | DEBUG/INFO/WARN/ERROR      | INFO   |
| `LOG_JSON`               | JSON=1, texte=0            | 1      |
| `LOG_REQUEST_BODY`       | Log du corps requête       | 0      |
| `LOG_REMOTE_CONTENT`     | Log du contenu LLM         | 0      |
| `LOG_IMAGE_B64`          | Log images base64          | 0      |
| `LOG_SAMPLE_IMAGE_BYTES` | Échantillon d’octets image | 0      |

### Signature cryptographique `/v1/certify-score`

| Variable                    | Description                       | Défaut                        |
| --------------------------- | --------------------------------- | ----------------------------- |
| `MINIMON_SIGNING_KEY_PATH`  | Clé privée PEM                    | `certs/app_signing_key.pem`   |
| `MINIMON_SIGNING_CERT_PATH` | Certificat X.509 PEM (facultatif) | `certs/app_signing_cert.pem`  |
| `MINIMON_SCORE_LEDGER_PATH` | Ledger JSONL append-only          | `data/certified_scores.jsonl` |

---

## 🧠 Fonctionnement interne

### Génération du nom

* Si `MINIMON_NAME_BACKEND=remote`, appel à `/chat/completions`
* Le prompt exige un JSON strict `{ "name": "…" }`
* Le parseur gère :

  * JSON correct
  * JSON incomplet ou mal formé
  * texte libre avec extraction regex
  * filtrage du cas pathologique `"name": "name"`

### Génération de l’image

#### Mode `files`

Retourne une image PNG encodée en base64 parmi les images préchargées.

#### Mode `sdxl`

* warmup automatique la première fois
* optimisations :

  * FlashAttention 3 (si dispo)
  * xFormers
  * SDPA PyTorch
  * quantization FP8 / int8 / int4 / FP4
  * slicing attention/VAE
  * torch.compile
  * CPU offload si demandé

La **rareté** pilote le prompt SDXL (textures, matériaux, effets, nombre de steps…).

---

## 🔌 Endpoints

### `GET /v1/generate`

Génère un Minimon complet : image, nom, rareté, horodatage.

Réponse :

```json
{
  "imageBase64": "...",
  "metadata": {
    "id": "mnm_XXXX",
    "name": "Floraclaw",
    "rarity": "B"
  },
  "generatedAt": "2025-11-12T20:41:32Z"
}
```

Codes d’erreur :

* `401 UNAUTHORIZED`
* `429 RATE_LIMITED`
* `500 GENERATION_FAILED`

---

### `POST /v1/certify-score`

Signature canonique d’un score utilisateur.

Payload attendu :

```json
{
  "score": 123.4,
  "subject": "player42",
  "nonce": "optionnel"
}
```

Contrainte : aucun champ extra n’est autorisé (schema strict).

Résultat :

```json
{
  "signed": {
    "payload": {
      "id": "mnm_XXXX",
      "score": 123.4,
      "nonce": "...",
      "issuedAt": "...",
      "subject": "player42"
    },
    "canonicalB64": "...",
    "signatureB64": "...",
    "algorithm": "ES256",
    "signatureFormat": "DER",
    "certificateFingerprintSHA256": "..."
  },
  "generatedAt": "..."
}
```

Un enregistrement append-only est stocké dans :
`MINIMON_SCORE_LEDGER_PATH` (JSONL).

---

### `GET /health`

Renvoie :

```json
{
  "status": "ok",
  "time": "2025-11-12T20:42:11Z",
  "backend": "files"
}
```

---

## 🧪 Exemples

```bash
curl https://localhost:22222/v1/generate -k
```

Avec token :

```bash
curl -H "Authorization: Bearer mytoken" \
     https://localhost:22222/v1/generate -k
```

Test backend de nom :

```bash
curl -s http://192.168.0.37:11434/v1/chat/completions \
  -H "Authorization: Bearer dummy" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "max_tokens": 30,
    "messages": [
      { "role": "system", "content": "Réponds avec {\"name\":\"...\"}" },
      { "role": "user", "content": "Nom !" }
    ]
  }'
```

---

## 🖥️ Exécution

Développement :

```bash
python gen-server.py
```

Production (exemple backend noms distant) :

```bash
export MINIMON_NAME_BACKEND=remote
export OPENAI_BASE_URL=http://192.168.0.37:11434/v1
export OPENAI_MODEL=llama3.2:1b
export OPENAI_API_KEY=dummy
python gen-server.py
```

SDXL optimisé :

```bash
export MINIMON_GENERATION_BACKEND=sdxl
export SDXL_QUANTIZATION=fp8
export SDXL_USE_COMPILE=1
export SDXL_USE_FLASH_ATTN=1
export SDXL_USE_XFORMERS=1
python gen-server.py
```

---

## 📦 Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gen-server.py .
COPY minimon/ ./minimon/
COPY certs/ ./certs/

ENV MINIMON_GENERATION_BACKEND=files

EXPOSE 22222
CMD ["python", "gen-server.py"]
```

GPU :

```bash
docker run --gpus all -p 22222:22222 \
  -e MINIMON_GENERATION_BACKEND=sdxl \
  -e SDXL_QUANTIZATION=fp8 \
  pokemon-api
```

---

## 🐛 Dépannage

### Nom généré incorrect (toujours "name")

Probable JSON incomplet. Augmenter `max_tokens` ou vérifier la disponibilité du modèle.

### Images SDXL lentes ou corrompues

* réduire la résolution à 256×256
* vérifier VRAM (`nvidia-smi`)
* activer slicing ou CPU offload
* installer `flash-attn` ou `xformers`

### Quantization FP8 inactive

Installer `torchao`.

---

## 🧭 Licence

MIT.

---

## 🤝 Contribution

Idées : nouveaux modèles, cache Redis, i18n des prompts, interface web, métriques Prometheus.

---

## 📌 Informations version

**Version API** : 1.3.0
**Dernière mise à jour** : 12 novembre 2025
