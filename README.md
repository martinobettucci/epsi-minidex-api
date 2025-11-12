# 🧬 Pokémon Image Generator API

**Pokémon Image Generator API** est un microservice FastAPI capable de générer des créatures de type "Pokémon" avec des images et des métadonnées (nom, rareté, horodatage).

Il peut fonctionner soit avec un backend local (fichiers d'images préchargées), soit avec un modèle de génération SDXL Turbo, et depuis la version 1.2.0, il peut appeler un serveur **OpenAI-compatible** (comme **Ollama**) pour créer dynamiquement des noms de créatures.

---

## 🚀 Fonctionnalités principales

- **Génération d'images** à partir :
  - de fichiers statiques (`files`)
  - ou d'un modèle SDXL (`sdxl`)
- **Génération de noms** :
  - locale (générateur aléatoire embarqué)
  - ou distante via API OpenAI-compatible (Ollama, LM Studio, vLLM, etc.)
- Attribution de rareté pondérée (F → S+)
- Système d'authentification par Bearer token (optionnel)
- Rate limiting configurable
- Logs structurés (JSON ou texte)
- Endpoint de santé intégré (`/health`)

---

## ⚙️ Installation

### 1. Dépendances

Installe les dépendances Python nécessaires :

```bash
# Installation via uv (recommandé)
uv add fastapi uvicorn requests

# Ou via pip
pip install fastapi uvicorn requests
```

Si tu veux utiliser le backend SDXL :

```bash
# Avec uv
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv add diffusers pillow

# Ou avec pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install diffusers pillow
```

### 2. Arborescence minimale

```
project/
│
├── gen-server.py
├── pokemon/
│   ├── image_00.png
│   ├── image_01.png
│   └── ...
└── certs/
    ├── cert.pem
    └── key.pem
```

Les images sont utilisées uniquement dans le mode `files`.

---

## 🧩 Variables d'environnement

Toutes les variables sont optionnelles (des valeurs par défaut sont intégrées dans le script).

### Configuration générale

| Variable | Description | Défaut |
|----------|-------------|--------|
| `POKEMON_IMAGES_DIR` | Dossier contenant les images locales | `./pokemon` |
| `POKEMON_IMAGE_PATTERN` | Nom de fichier des images | `image_{:02d}.png` |
| `POKEMON_MIN_INDEX` / `POKEMON_MAX_INDEX` | Plage d'images locales disponibles | `0` → `5` |
| `POKEMON_GENERATION_BACKEND` | Backend image : `files` ou `sdxl` | `files` |
| `POKEMON_BEARER_TOKEN` | Jeton Bearer pour sécuriser les appels | *(vide)* |
| `POKEMON_RATE_LIMIT_PER_MIN` | Limite de requêtes par minute | `60` |

### Configuration SDXL

| Variable | Description | Défaut |
|----------|-------------|--------|
| `SDXL_TURBO_MODEL` | Nom du modèle SDXL | `stabilityai/sdxl-turbo` |
| `SDXL_WIDTH`, `SDXL_HEIGHT` | Taille des images générées | `512` |
| `SDXL_STEPS` | Nombre d'itérations d'inférence | `1` |

### Configuration génération de noms (OpenAI-compatible)

| Variable | Description | Défaut |
|----------|-------------|--------|
| `POKEMON_NAME_BACKEND` | `remote` pour API OpenAI-compatible ou `local` | `local` |
| `OPENAI_BASE_URL` | URL du serveur OpenAI-compatible | `http://192.168.0.37:11434/v1` |
| `OPENAI_MODEL` | Nom du modèle à appeler | `llama3.2:1b` |
| `OPENAI_API_KEY` | API key utilisée (même factice pour Ollama) | `dummy` |
| `OPENAI_TIMEOUT` | Timeout de la requête API (secondes) | `5.0` |

### Configuration logging

| Variable | Description | Défaut |
|----------|-------------|--------|
| `LOG_LEVEL` | Niveau de log : DEBUG/INFO/WARN/ERROR | `INFO` |
| `LOG_JSON` | Logs en format JSON (`1`) ou texte (`0`) | `1` |
| `LOG_REQUEST_BODY` | Logger le corps des requêtes | `0` |
| `LOG_REMOTE_CONTENT` | Logger le contenu des réponses API | `0` |
| `LOG_IMAGE_B64` | Logger les images base64 (attention à la taille!) | `0` |

---

## 🧠 Fonctionnement

### Backend de noms

- Si `POKEMON_NAME_BACKEND=remote`, le serveur contacte un endpoint `/chat/completions` OpenAI-compatible
- Le prompt demande au modèle un nom de créature original et attend une réponse JSON :
  ```json
  {"name":"Aquaclaw"}
  ```
- Le système tolère les JSON mal formés (accolades en trop, troncature) grâce à des regex de secours
- Si le modèle échoue ou que l'API ne répond pas, le service retombe automatiquement sur le générateur local

### Backend d'images

- **`files`** : choisit aléatoirement une image préchargée encodée en base64
- **`sdxl`** : utilise `diffusers` pour générer l'image via le modèle SDXL Turbo
  - Warmup automatique au premier appel
  - Optimisations CUDA (attention slicing, cudnn benchmark)
  - Fallback sur `files` si la génération échoue

---

## 🔌 Endpoints

### `GET /v1/generate`

Génère un Pokémon (nom + image + rareté).

#### Exemple de réponse

```json
{
  "imageBase64": "iVBORw0KGgoAAAANSUhEUgA...",
  "metadata": {
    "id": "pkm_MFXJZ23A4FGH",
    "name": "Khra'gzathon",
    "rarity": "B"
  },
  "generatedAt": "2025-11-12T20:41:32Z"
}
```

#### Authentification

Si `POKEMON_BEARER_TOKEN` est défini :

```bash
Authorization: Bearer <votre_token>
```

#### Codes d'erreur

- `401 UNAUTHORIZED` : Token manquant ou invalide
- `429 RATE_LIMITED` : Trop de requêtes
- `500 GENERATION_FAILED` : Erreur lors de la génération

---

### `GET /health`

Renvoie un simple statut de disponibilité.

#### Exemple

```json
{
  "status": "ok",
  "time": "2025-11-12T20:42:11Z",
  "backend": "files"
}
```

---

## 🧪 Exemples d'utilisation

### Requête locale simple

```bash
curl https://localhost:22222/v1/generate -k
```

### Requête avec token

```bash
curl -H "Authorization: Bearer mytoken" \
     https://localhost:22222/v1/generate -k
```

### Test direct de l'API Ollama

```bash
curl -s http://192.168.0.37:11434/v1/chat/completions \
  -H "Authorization: Bearer dummy" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "max_tokens": 30,
    "messages": [
      {
        "role": "system",
        "content": "Tu crées un nom de créature Pokémon. Réponds au format JSON: {\"name\":\"<nom>\"}"
      },
      {
        "role": "user",
        "content": "Génère un seul nom."
      }
    ]
  }'
```

---

## 🖥️ Exécution

### Mode développement

```bash
python gen-server.py
```

Le serveur démarre sur : `https://0.0.0.0:22222`

### Mode production avec génération de noms distante

```bash
export POKEMON_NAME_BACKEND=remote
export OPENAI_BASE_URL=http://192.168.0.37:11434/v1
export OPENAI_MODEL=llama3.2:1b
export OPENAI_API_KEY=dummy
export POKEMON_BEARER_TOKEN=mon_token_secret
export LOG_LEVEL=INFO
export LOG_JSON=1

python gen-server.py
```

### Mode avec SDXL

```bash
export POKEMON_GENERATION_BACKEND=sdxl
export SDXL_WIDTH=512
export SDXL_HEIGHT=512
export SDXL_STEPS=1

python gen-server.py
```

---

## 📊 Logs structurés

Le serveur génère des logs structurés pour faciliter le monitoring :

```json
{
  "ts": "2025-11-12T16:35:27Z",
  "level": "INFO",
  "logger": "pokemon",
  "message": "remote_name.ok",
  "extra": {
    "name": "Khra'gzathon"
  }
}
```

### Événements clés

- `startup.config` : Configuration au démarrage
- `http.request` / `http.response` : Requêtes HTTP
- `remote_name.request` / `remote_name.ok` : Appels API de génération de noms
- `extract_name.*` : Extraction des noms depuis les réponses
- `generate.success` : Génération réussie
- `sdxl.*` : Événements du pipeline SDXL

---

## 🔒 Sécurité et bonnes pratiques

- ✅ Utiliser un certificat valide pour `certs/key.pem` et `certs/cert.pem` (ou désactiver SSL pour test)
- ✅ Activer `POKEMON_BEARER_TOKEN` en production
- ✅ Déployer derrière un reverse proxy (Nginx, Caddy, Traefik)
- ✅ Configurer des limites de rate limiting adaptées (`POKEMON_RATE_LIMIT_PER_MIN`)
- ✅ Éviter de logger les images base64 en production (`LOG_IMAGE_B64=0`)
- ✅ Monitorer les logs structurés avec un agrégateur (ELK, Loki, etc.)

---

## 🐛 Dépannage

### Le nom généré est toujours "name"

**Cause** : Le modèle LLM renvoie un JSON incomplet ou mal formé.

**Solution** :
1. Augmenter `max_tokens` dans le payload (déjà fait : 30)
2. Vérifier que le modèle fonctionne correctement :
   ```bash
   curl http://192.168.0.37:11434/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"llama3.2:1b","messages":[{"role":"user","content":"test"}]}'
   ```

### Images SDXL corrompues

**Cause** : Manque de VRAM ou mauvaise configuration CUDA.

**Solution** :
- Réduire `SDXL_WIDTH` et `SDXL_HEIGHT` (essayer 256x256)
- Activer `attention_slicing` (déjà activé dans le code)
- Vérifier `nvidia-smi` pour la disponibilité GPU

### Rate limit trop agressif

**Solution** : Augmenter `POKEMON_RATE_LIMIT_PER_MIN` ou le désactiver (`0`)

---

## 📦 Déploiement Docker (exemple)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gen-server.py .
COPY pokemon/ ./pokemon/
COPY certs/ ./certs/

ENV POKEMON_GENERATION_BACKEND=files
ENV POKEMON_NAME_BACKEND=remote
ENV OPENAI_BASE_URL=http://ollama:11434/v1

EXPOSE 22222
CMD ["python", "gen-server.py"]
```

```bash
docker build -t pokemon-api .
docker run -p 22222:22222 -e OPENAI_BASE_URL=http://host.docker.internal:11434/v1 pokemon-api
```

---

## 🧭 Licence

Projet distribué sous licence MIT.

Créé pour démontrer une architecture légère de génération de contenu via API compatibles OpenAI et backends locaux.

---

## 🤝 Contribution

Les PRs sont bienvenues ! Zones d'amélioration :

- Support d'autres modèles de génération (Flux, Stable Diffusion 3)
- Cache Redis pour les noms générés
- Support multi-langues pour les prompts
- Interface web de test
- Métriques Prometheus

---

**Auteur** : Votre nom  
**Version** : 1.2.0  
**Dernière mise à jour** : 12 novembre 2025