# 🧬 Pokémon Image Generator API

**Pokémon Image Generator API** est un microservice FastAPI capable de générer des créatures de type "Pokémon" avec des images et des métadonnées (nom, rareté, horodatage).

Il peut fonctionner soit avec un backend local (fichiers d'images préchargées), soit avec un modèle de génération SDXL Turbo. Depuis la version **1.2.0**, il peut également appeler un serveur **OpenAI-compatible** (par exemple **Ollama**) pour créer dynamiquement des noms de créatures. La variante **1.2.0+ Blackwell** ajoute des optimisations avancées pour SDXL, notamment quantization FP8, FlashAttention, xFormers, torch.compile, slicing et CPU offload.

---

## 🚀 Fonctionnalités principales

* **Génération d'images** à partir :

  * de fichiers statiques (`files`)
  * d'un modèle SDXL (`sdxl`)
* **Génération de noms** :

  * locale (générateur aléatoire embarqué)
  * distante via API OpenAI-compatible (Ollama, LM Studio, vLLM, etc.)
* Attribution de rareté pondérée (F → S+)
* Système d'authentification par Bearer token, optionnel
* Rate limiting configurable
* Logs structurés (JSON ou texte)
* Endpoint de santé (`/health`)
* **Optimisations GPU Blackwell-ready** pour SDXL : FP8, FlashAttention, xFormers, `torch.compile`, attention et VAE slicing, CPU offload

---

## ⚙️ Installation

### 1. Dépendances

Installe les dépendances Python nécessaires au noyau API :

```bash
# Installation via uv (recommandé)
uv add fastapi uvicorn requests

# Ou via pip
pip install fastapi uvicorn requests
```

Si tu veux utiliser le backend SDXL de base :

```bash
# Avec uv
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv add diffusers pillow

# Ou avec pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install diffusers pillow
```

Pour activer les **optimisations Blackwell** selon les besoins, installe les modules optionnels suivants, uniquement si tu choisis de les utiliser :

```bash
# Quantization FP8 via torchao
pip install torchao

# FlashAttention 3 ou 2.x selon ta pile CUDA
pip install flash-attn

# xFormers (attention mémoire efficiente)
pip install xformers
```

> Remarque : `flash-attn` et `xformers` dépendent de versions spécifiques de CUDA et de PyTorch. Utilise des roues précompilées adaptées à ta plateforme. Sans ces paquets, le serveur démarre, les optimisations manquantes sont simplement ignorées.

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

Toutes les variables sont optionnelles. Des valeurs par défaut existent dans le code.

### Configuration générale

| Variable                                  | Description                       | Défaut             |
| ----------------------------------------- | --------------------------------- | ------------------ |
| `POKEMON_IMAGES_DIR`                      | Dossier des images locales        | `./pokemon`        |
| `POKEMON_IMAGE_PATTERN`                   | Modèle de nom des images          | `image_{:02d}.png` |
| `POKEMON_MIN_INDEX` / `POKEMON_MAX_INDEX` | Plage des index d'images          | `0` → `5`          |
| `POKEMON_GENERATION_BACKEND`              | Backend image : `files` ou `sdxl` | `files`            |
| `POKEMON_BEARER_TOKEN`                    | Token Bearer pour sécuriser l'API | *(vide)*           |
| `POKEMON_RATE_LIMIT_PER_MIN`              | Limite de requêtes par minute     | `60`               |

### Configuration SDXL

| Variable                    | Description                     | Défaut                   |
| --------------------------- | ------------------------------- | ------------------------ |
| `SDXL_TURBO_MODEL`          | ID du modèle SDXL               | `stabilityai/sdxl-turbo` |
| `SDXL_WIDTH`, `SDXL_HEIGHT` | Taille des images générées      | `512`                    |
| `SDXL_STEPS`                | Nombre d'itérations d'inférence | `1`                      |

#### Optimisations SDXL Blackwell

| Variable                  | Description                                                                                                                                                     | Défaut |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| `SDXL_QUANTIZATION`       | `fp8`, `fp4`, `none`, contrôle la quantization poids UNet et VAE. `fp4` est expérimental et non implémenté dans le runtime courant, journalisé pour information | `fp8`  |
| `SDXL_USE_COMPILE`        | Active `torch.compile` sur UNet et VAE decoder                                                                                                                  | `1`    |
| `SDXL_USE_XFORMERS`       | Active l'attention mémoire efficiente xFormers si disponible                                                                                                    | `1`    |
| `SDXL_USE_FLASH_ATTN`     | Active FlashAttention si installé                                                                                                                               | `1`    |
| `SDXL_ENABLE_SLICING`     | Active `enable_attention_slicing("auto")`                                                                                                                       | `0`    |
| `SDXL_ENABLE_CPU_OFFLOAD` | Active `enable_model_cpu_offload()` sur GPU VRAM limitée                                                                                                        | `0`    |

> Notes d'exécution :
>
> 1. `SDXL_QUANTIZATION=fp8` nécessite `torchao`. Si non installé, le serveur journalise un avertissement et continue sans quantization.
> 2. `SDXL_USE_FLASH_ATTN=1` nécessite `flash-attn`. En absence de paquet, fallback xFormers puis SDPA PyTorch.
> 3. `SDXL_USE_COMPILE=1` requiert PyTorch 2.3+ idéalement. Si la compilation échoue, un log est émis et l'exécution continue.
> 4. Les options de slicing et CPU offload sont utiles pour réduire la pression VRAM avec un léger coût en latence.

### Configuration génération de noms (OpenAI-compatible)

| Variable               | Description                                     | Défaut                         |
| ---------------------- | ----------------------------------------------- | ------------------------------ |
| `POKEMON_NAME_BACKEND` | `remote` pour API compatible OpenAI ou `local`  | `local`                        |
| `OPENAI_BASE_URL`      | URL du serveur OpenAI-compatible                | `http://192.168.0.37:11434/v1` |
| `OPENAI_MODEL`         | Modèle à appeler                                | `llama3.2:1b`                  |
| `OPENAI_API_KEY`       | Clé API utilisée, y compris factice pour Ollama | `dummy`                        |
| `OPENAI_TIMEOUT`       | Timeout appel API en secondes                   | `5.0`                          |

### Configuration logging

| Variable                 | Description                                            | Défaut |
| ------------------------ | ------------------------------------------------------ | ------ |
| `LOG_LEVEL`              | Niveau de log, DEBUG/INFO/WARN/ERROR                   | `INFO` |
| `LOG_JSON`               | Logs JSON `1` ou texte `0`                             | `1`    |
| `LOG_REQUEST_BODY`       | Active le logging du corps des requêtes                | `0`    |
| `LOG_REMOTE_CONTENT`     | Active le logging du contenu renvoyé par l'API de noms | `0`    |
| `LOG_IMAGE_B64`          | Active le logging des images en base64, volumineux     | `0`    |
| `LOG_SAMPLE_IMAGE_BYTES` | Échantillonne N octets de l'image pour debug           | `0`    |

---

## 🧠 Fonctionnement

### Backend de noms

* Si `POKEMON_NAME_BACKEND=remote`, le serveur contacte l'endpoint `/chat/completions` OpenAI-compatible
* Le prompt demande un nom de créature original et attend une réponse JSON stricte :

  ```json
  {"name":"Aquaclaw"}
  ```
* Le parseur tolère les JSON mal formés, troncatures et doublons d'accolades grâce à des regex et nettoyages de secours
* En cas d'échec distant, le service retombe sur la génération locale

### Backend d'images

* **`files`** : sélection aléatoire d'une image préchargée encodée en base64
* **`sdxl`** : génération via `diffusers` et pipeline SDXL Turbo

  * Warmup automatique au premier appel
  * Optimisations CUDA activables, y compris FlashAttention, xFormers, SDPA
  * Quantization FP8 possible quand `torchao` est présent
  * `torch.compile` pour optimiser UNet et VAE decoder
  * Slicing et CPU offload pour profils mémoire contraints
  * Fallback automatique sur `files` si la génération échoue

---

## 🔌 Endpoints

### `GET /v1/generate`

Génère un Pokémon, comprenant nom, image et rareté.

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

* `401 UNAUTHORIZED` : token manquant ou invalide
* `429 RATE_LIMITED` : quota de requêtes dépassé
* `500 GENERATION_FAILED` : erreur lors de la génération

---

### `GET /health`

Renvoie un statut de disponibilité simple.

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

Le serveur écoute sur `https://0.0.0.0:22222`.

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

### Mode avec SDXL, optimisations Blackwell activées

```bash
export POKEMON_GENERATION_BACKEND=sdxl
export SDXL_WIDTH=512
export SDXL_HEIGHT=512
export SDXL_STEPS=1

# Optimisations
export SDXL_QUANTIZATION=fp8
export SDXL_USE_COMPILE=1
export SDXL_USE_FLASH_ATTN=1
export SDXL_USE_XFORMERS=1
export SDXL_ENABLE_SLICING=1
export SDXL_ENABLE_CPU_OFFLOAD=0

python gen-server.py
```

> Si la quantization FP8 échoue faute de `torchao`, un avertissement est loggé et l'exécution continue sans quantization.

---

## 📊 Logs structurés

Exemple de log JSON :

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

* `startup.config` : configuration au démarrage
* `http.request` et `http.response` : trafic HTTP
* `remote_name.request` et `remote_name.ok` : appels de génération de noms
* `extract_name.*` : pipeline d'extraction du nom
* `sdxl.*` : cycle de vie du pipeline SDXL, y compris warmup, compilation, quantization, attention
* `generate.success` : génération réussie

---

## 🔒 Sécurité et bonnes pratiques

* Utiliser un certificat valide pour `certs/key.pem` et `certs/cert.pem` en production, ou désactiver SSL pour des tests locaux
* Activer `POKEMON_BEARER_TOKEN` en production
* Placer le service derrière un reverse proxy, par exemple Nginx, Caddy, Traefik
* Régler une politique de rate limiting adaptée via `POKEMON_RATE_LIMIT_PER_MIN`
* Désactiver le logging des images base64 en production, `LOG_IMAGE_B64=0`
* Agréger et superviser les logs avec ELK, Loki ou équivalent

---

## 🐛 Dépannage

### Le nom généré est toujours "name"

Cause plausible, JSON incomplet ou mal formé renvoyé par le LLM.

Correctifs :

1. Augmenter `max_tokens` dans le payload côté LLM
2. Vérifier la santé du modèle :

   ```bash
   curl http://192.168.0.37:11434/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"llama3.2:1b","messages":[{"role":"user","content":"test"}]}'
   ```

### Images SDXL corrompues ou génération lente

Causes possibles, VRAM insuffisante, configuration CUDA sous-optimale, absence des modules d'optimisation.

Correctifs :

* Réduire `SDXL_WIDTH` et `SDXL_HEIGHT` à 256x256 pour valider le flux
* Activer `SDXL_ENABLE_SLICING=1` et éventuellement `SDXL_ENABLE_CPU_OFFLOAD=1`
* Vérifier `nvidia-smi` et la disponibilité GPU
* Installer et activer `flash-attn` ou `xformers` selon la plateforme
* Laisser `SDXL_USE_COMPILE=1` si la compilation aboutit, sinon le runtime continue sans compilation

### La quantization FP8 ne semble pas active

Cause probable, `torchao` non installé ou GPU non compatible.

Correctifs :

* Installer `torchao` puis relancer le serveur
* Basculer `SDXL_QUANTIZATION=none` si tu veux désactiver proprement

### Erreurs autour de FlashAttention

Si `flash-attn` n'est pas présent ou incompatible, les logs affichent `attention.flash_attn3_unavailable` et la pile bascule sur xFormers ou SDPA. Aucun changement côté API n'est requis.

### Rate limit trop agressif

Augmenter `POKEMON_RATE_LIMIT_PER_MIN` ou le désactiver avec `0`.

---

## 📦 Déploiement Docker

Exemple de `Dockerfile` minimal côté CPU et fichiers statiques :

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

Pour une image GPU avec SDXL et optimisations, ajoute les paquets nécessaires dans `requirements.txt` ou via `pip install` puis exécute le conteneur avec `--gpus all`.

```bash
docker build -t pokemon-api .
docker run --gpus all -p 22222:22222 \
  -e POKEMON_GENERATION_BACKEND=sdxl \
  -e SDXL_QUANTIZATION=fp8 \
  -e SDXL_USE_COMPILE=1 \
  pokemon-api
```

---

## 🧭 Licence

Projet distribué sous licence MIT.

Créé pour démontrer une architecture légère de génération de contenu avec API compatibles OpenAI et backends locaux.

---

## 🤝 Contribution

Les contributions sont bienvenues. Pistes utiles :

* Support d'autres modèles de génération, par exemple Flux, Stable Diffusion 3
* Cache Redis pour les noms générés
* Internationalisation des prompts
* Interface web de test
* Exposition de métriques Prometheus

---

**Auteur** : Votre nom
**Version** : 1.2.0
**Dernière mise à jour** : 12 novembre 2025
