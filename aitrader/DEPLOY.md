# Procedimiento de despliegue — Render

## Dónde desplegar
**Render.com** (plan free o starter). Netlify NO sirve: es una app FastAPI con servidor persistente; Netlify solo aloja estáticos y serverless functions.

## Pasos

### 1. Conectar repositorio
- Push este repo (`aitrader/`) a GitHub/GitLab.
- En Render → **New Web Service** → conectar el repo.

### 2. Configuración (o usa `render.yaml` incluido)
| Campo | Valor |
|---|---|
| **Runtime** | Python 3.11 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn app:app --host 0.0.0.0 --port $PORT` |
| **Health Check** | `/api/status` |

### 3. Variables de entorno
| Variable | Valor | Nota |
|---|---|---|
| `PORT` | (auto) | Render la inyecta |
| `HOST` | `0.0.0.0` | bindear a todas las interfaces |
| `PUBLIC_DEMO` | `0` | 1 = solo lectura público |
| `GMGN_API_KEY` | (tu key) | opcional: si no, arranca en Mock |
| `GMGN_PRIVATE_KEY` | (tu PEM) | solo para LIVE real |

### 4. Acceso
- URL: `https://<tu-servicio>.onrender.com`
- La app sirve el frontend (`static/index.html`) y la API en el mismo origen.

## Notas de producción
- La app **solo debe usarse en local o con túnel con autenticación**. Exponerla públicamente comparte tu IP/key.
- Para demo pública real, usar `PUBLIC_DEMO=1` + túnel (cloudflared/ngrok) con rate-limit.
- `outputs/` se crea automáticamente en el primer arranque.
