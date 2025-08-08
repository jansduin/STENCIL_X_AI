# STENCIL_X_AI

Aplicación multiplataforma (Web, iOS, Android) para generar stencils profesionales de tatuaje con dos modos: Simple (pipeline OpenCV) y AI (modelo de IA), con suscripción mensual.

## Características
- Generación de stencil en dos modos:
  - Simple: contornos nítidos, grosor controlado, XDoG/Canny, skeletonize opcional, exportación PNG con DPI.
  - Sketch: boceto tipo “lápiz” (color dodge, blur gaussiano, denoise), sin rellenos, pensado para retrato/realismo.
- Vista previa en caliente (UI) con parámetros ajustables: intensidad, grosor, suavizado, método de bordes, skeletonize, blur (sigma), denoise (h), tamaño y DPI.
- Seguridad: CORS parametrizable, validación estricta de descargas, rutas confinadas.
- Preparado para suscripción mensual (Stripe/Apple/Google) en fases posteriores.

## Estructura
```
backend/
  app/
    main.py            # FastAPI + CORS + rutas UI y API
    routes/stencil.py  # Endpoints /generate, /preview_simple, /download, /styles
    services/stencil_engine.py  # Pipelines OpenCV/XDoG/sketch, guardado PNG con DPI
  requirements.txt     # Dependencias Python
frontend/              # (WIP) base RN para iOS/Android
```

## Requisitos
- Python 3.10+
- macOS/Linux/WSL2 recomendado

## Instalación backend (desarrollo)
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

UI mínima: `http://localhost:8000/ui` (redirige desde `/`).

## Uso rápido (UI)
1. Subir imagen.
2. Modo Simple → Método:
   - auto/canny/xdog: stencils binarios de contorno.
   - sketch: boceto continuo (outline_only se desactiva automáticamente).
3. Ajustar sliders (Blur sigma y Denoise h en sketch).
4. Generar y descargar PNG (fondo transparente opcional, DPI y tamaño de página).

## Testing
```bash
cd backend
source .venv/bin/activate
pytest -q
```

## Seguridad
- Sanitización de nombre de archivo y confinamiento de rutas en descargas.
- CORS configurable por `ALLOWED_ORIGINS`.

## Roadmap (alto nivel)
- Integración HED/DexiNed y fusión regional con XDoG.
- Aislamiento de piel/rostro y herramientas manuales (borrar/pintar).
- Super-resolución x2 previo a vectorización (Douglas-Peucker) y suavizado de paths.
- Suscripciones (Stripe web, Apple/Google mobile) y gating de API.
- Exportación/impresión avanzada.

## Licencia
Pendiente de definir por el propietario del repositorio.
