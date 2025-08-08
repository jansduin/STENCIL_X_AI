"""
UI Routes - Minimal Web Interface for Validation
Serves a simple HTML UI to validate stencil generation flows.

Author: Stencil AI Team
Date: 2025-08-07
Dependencies: FastAPI, HTMLResponse
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse


router = APIRouter(tags=["ui"])


@router.get("/ui", response_class=HTMLResponse)
async def stencil_ui() -> HTMLResponse:
    """Serve a minimal UI for generating stencils.

    Returns:
        HTMLResponse: Inlined HTML page.
    """
    html = """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stencil Generator</title>
  <style>
    :root {
      --black-intense: #0a0a0a;
      --black-soft: #111213;
      --gray-dark: #1e1f22;
      --gray-soft: #9aa0a6;
      --green-soft: #6fcf97;
      --red-dark: #c0392b;
    }
    body {
      margin: 0; padding: 0; background: var(--black-intense); color: #e5e7eb; font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica Neue, Arial, "Apple Color Emoji", "Segoe UI Emoji";
    }
    .wrap { max-width: 980px; margin: 0 auto; padding: 24px; }
    .panel { background: var(--black-soft); border: 1px solid #222; border-radius: 12px; padding: 20px; }
    h1 { margin: 0 0 12px; font-size: 20px; font-weight: 600; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .field { display: flex; flex-direction: column; gap: 8px; }
    label { font-size: 12px; color: var(--gray-soft); }
    input[type="file"], select, input[type="range"] { background: var(--gray-dark); color: #e5e7eb; border: 1px solid #2a2d31; border-radius: 8px; padding: 10px; }
    .btn { background: var(--green-soft); color: #0a0a0a; border: none; padding: 12px 16px; border-radius: 10px; font-weight: 600; cursor: pointer; }
    .btn:disabled { opacity: .6; cursor: not-allowed; }
    .muted { color: var(--gray-soft); font-size: 12px; }
    .error { color: var(--red-dark); font-size: 12px; }
    .preview {
      /* Fondo tipo damero claro para mejor visibilidad del stencil */
      background-color: #fafafa;
      background-image:
        linear-gradient(45deg, #ececec 25%, transparent 25%),
        linear-gradient(-45deg, #ececec 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #ececec 75%),
        linear-gradient(-45deg, transparent 75%, #ececec 75%);
      background-size: 20px 20px;
      background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
      border: 1px solid #222; border-radius: 12px; padding: 16px; display: grid; place-items: center; min-height: 280px;
    }
    img { max-width: 100%; border-radius: 8px; image-rendering: -webkit-optimize-contrast; image-rendering: crisp-edges; }
  </style>
  <script>
    let aborter = null;

    async function loadStyles() {
      const sel = document.getElementById('style');
      sel.innerHTML = '<option value="">Cargando estilos...</option>';
      try {
        const resp = await fetch('/api/v1/stencils/styles');
        if (!resp.ok) throw new Error('Styles fetch failed');
        const data = await resp.json();
        sel.innerHTML = '';
        for (const s of data.styles) {
          const opt = document.createElement('option');
          opt.value = s; opt.textContent = s.replaceAll('_', ' ');
          if (s === data.default_style) opt.selected = true;
          sel.appendChild(opt);
        }
        // Preselect portrait realism if present for faster validation cycles
        if (data.styles.includes('portrait_realism')) {
          sel.value = 'portrait_realism';
        }
      } catch (e) {
        sel.innerHTML = '<option value="">Error cargando estilos</option>';
      }
    }

    function debounce(fn, ms) {
      let t; return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
    }

    async function generateStencil(ev) {
      ev.preventDefault();
      const form = document.getElementById('form');
      const file = document.getElementById('image').files[0];
      const style = document.getElementById('style').value || '';
      const edgeMethod = document.getElementById('edge_method')?.value || 'auto';
      const skeleton = document.getElementById('skeletonize')?.checked ? '1' : '0';
      const intensity = document.getElementById('intensity').value || '0.5';
      const thickness = document.getElementById('thickness').value || '2';
      const smooth = document.getElementById('smooth').value || '1';
      const size = document.getElementById('size').value || '1024';
      const pageSize = document.getElementById('page_size')?.value || 'none';
      const stencilWidth = document.getElementById('stencil_width_cm')?.value || '';
      const dpi = document.getElementById('dpi')?.value || '300';
      const sketchSigma = document.getElementById('sketch_sigma')?.value || '';
      const denoiseH = document.getElementById('denoise_h')?.value || '';
      const mode = document.querySelector('input[name="mode"]:checked')?.value || 'ai';
      const errorBox = document.getElementById('error');
      const result = document.getElementById('result');
      const btn = document.getElementById('submit');
      const dl = document.getElementById('download');
      const img = document.getElementById('preview');
      errorBox.textContent = ''; result.textContent=''; img.src=''; dl.href='';
      if (!file) { errorBox.textContent = 'Selecciona una imagen.'; return; }
      btn.disabled = true; btn.textContent = 'Generando...';
      try {
        const fd = new FormData();
        fd.append('image', file, file.name);
        if (style) fd.append('style', style);
        if (edgeMethod) fd.append('edge_method', edgeMethod);
        if (skeleton) fd.append('skeletonize', skeleton);
        fd.append('intensity', intensity);
        fd.append('user_id', 'ui');
        fd.append('mode', mode);
        fd.append('line_thickness', thickness);
        fd.append('smooth_skin', smooth);
        fd.append('target_size', size);
        if (pageSize) fd.append('page_size', pageSize);
        if (stencilWidth) fd.append('stencil_width_cm', stencilWidth);
        if (dpi) fd.append('dpi', dpi);
        if (sketchSigma) fd.append('sketch_sigma', sketchSigma);
        if (denoiseH) fd.append('denoise_h', denoiseH);
        fd.append('outline_only', edgeMethod === 'sketch' ? '0' : '1');
        const resp = await fetch('/api/v1/stencils/generate', { method: 'POST', body: fd });
        if (!resp.ok) throw new Error('Falló la generación: ' + resp.status);
        const data = await resp.json();
        result.textContent = 'OK (' + (data.options?.style || style) + ', modo ' + mode + ')';
        const url = data.download_url;
        if (url) {
          const abs = url.startsWith('http') ? url : (location.origin + url);
          dl.href = abs; dl.style.display='inline-block';
          img.src = abs;
        }
      } catch (e) {
        errorBox.textContent = e.message || String(e);
      } finally {
        btn.disabled = false; btn.textContent = 'Generar stencil';
      }
    }

    const previewLive = debounce(async function preview() {
      const file = document.getElementById('image').files[0];
      const style = document.getElementById('style').value || '';
      const edgeMethod = document.getElementById('edge_method')?.value || 'auto';
      const skeleton = document.getElementById('skeletonize')?.checked ? '1' : '0';
      const intensity = document.getElementById('intensity').value || '0.5';
      const thickness = document.getElementById('thickness').value || '2';
      const smooth = document.getElementById('smooth').value || '1';
      const mode = document.querySelector('input[name="mode"]:checked')?.value || 'ai';
      const img = document.getElementById('preview');
      const errorBox = document.getElementById('error');
      const size = '768';
      const outlineOnly = edgeMethod === 'sketch' ? '0' : '1';
      const sketchSigma = document.getElementById('sketch_sigma')?.value || '';
      const denoiseH = document.getElementById('denoise_h')?.value || '';
      if (!file || mode !== 'simple') return;
      errorBox.textContent = '';
      try {
        if (aborter) aborter.abort();
        aborter = new AbortController();
        const fd = new FormData();
        fd.append('image', file, file.name);
        if (style) fd.append('style', style);
        if (edgeMethod) fd.append('edge_method', edgeMethod);
        if (skeleton) fd.append('skeletonize', skeleton);
        fd.append('intensity', intensity);
        fd.append('line_thickness', thickness);
        fd.append('smooth_skin', smooth);
        fd.append('target_size', size);
        fd.append('outline_only', outlineOnly);
        // Fondo blanco para preview para máximo contraste
        fd.append('transparent_bg', '0');
        if (sketchSigma) fd.append('sketch_sigma', sketchSigma);
        if (denoiseH) fd.append('denoise_h', denoiseH);
        const resp = await fetch('/api/v1/stencils/preview_simple', { method:'POST', body: fd, signal: aborter.signal });
        if (!resp.ok) return;
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        img.src = url;
      } catch (e) {
        // ignore aborted
      }
    }, 250);

    window.addEventListener('DOMContentLoaded', () => {
      loadStyles();
      document.getElementById('form').addEventListener('submit', generateStencil);
      // Live preview listeners
      ['image','style','edge_method','skeletonize','sketch_sigma','denoise_h','intensity','thickness','smooth'].forEach(id => {
        const el = document.getElementById(id);
        el?.addEventListener('change', previewLive);
        el?.addEventListener('input', previewLive);
      });
      document.querySelectorAll('input[name="mode"]').forEach(el => el.addEventListener('change', previewLive));
    });
  </script>
  </head>
  <body>
    <div class="wrap">
      <div class="panel">
        <h1>Stencil Generator</h1>
        <form id="form">
          <div class="row">
            <div class="field">
              <label>Imagen</label>
              <input type="file" id="image" accept="image/*" />
              <span class="muted">Formatos: PNG/JPG. Se enviará al backend.</span>
            </div>
            <div class="field">
              <label>Modo</label>
              <div>
                <label class="muted"><input type="radio" name="mode" value="simple" checked /> Simple</label>
                &nbsp;&nbsp;
                <label class="muted"><input type="radio" name="mode" value="ai" /> IA</label>
              </div>
            </div>
          </div>
          <div class="row" style="margin-top:12px;">
            <div class="field">
              <label>Estilo</label>
              <select id="style"></select>
            </div>
            <div class="field">
              <label>Intensidad: <span id="ival">0.5</span></label>
              <input type="range" id="intensity" min="0" max="1" step="0.1" value="0.5" oninput="document.getElementById('ival').textContent=this.value" />
            </div>
          </div>
          <div class="row" style="margin-top:12px;">
            <div class="field">
              <label class="muted">Método de bordes (Simple)</label>
              <select id="edge_method">
                <option value="auto" selected>auto</option>
                <option value="canny">canny</option>
                <option value="xdog">xdog</option>
                <option value="sketch">sketch</option>
              </select>
            </div>
            <div class="field">
              <label class="muted">Contorno fino (skeletonize)</label>
              <label class="muted"><input type="checkbox" id="skeletonize" /> Activar</label>
            </div>
          </div>
          <div class="row" style="margin-top:12px;">
            <div class="field">
              <label>Grosor de línea: <span id="tval">2</span> px</label>
              <input type="range" id="thickness" min="1" max="5" step="1" value="2" oninput="document.getElementById('tval').textContent=this.value" />
            </div>
            <div class="field">
              <label>Suavizado de piel: <span id="sval">1</span></label>
              <input type="range" id="smooth" min="0" max="2" step="1" value="1" oninput="document.getElementById('sval').textContent=this.value" />
            </div>
          </div>
          <div class="row" style="margin-top:12px;">
            <div class="field">
              <label>Blur (sigma) [solo sketch]: <span id="bval">8.0</span></label>
              <input type="range" id="sketch_sigma" min="1" max="20" step="0.5" value="8.0" oninput="document.getElementById('bval').textContent=this.value" />
            </div>
            <div class="field">
              <label>Denoise (h) [solo sketch]: <span id="dval">7</span></label>
              <input type="range" id="denoise_h" min="0" max="15" step="1" value="7" oninput="document.getElementById('dval').textContent=this.value" />
            </div>
          </div>
          <div class="row" style="margin-top:12px;">
            <div class="field">
              <label>Resolución de salida</label>
              <select id="size">
                <option value="512">512</option>
                <option value="1024" selected>1024</option>
                <option value="2048">2048</option>
              </select>
            </div>
            <div class="field">
              <label class="muted">Exportación / Impresión</label>
              <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
                <select id="page_size">
                  <option value="none" selected>Sin página</option>
                  <option value="a4">A4</option>
                  <option value="letter">Letter</option>
                </select>
                <input type="number" id="stencil_width_cm" min="1" max="40" step="0.5" placeholder="Ancho (cm)" style="width:130px;" />
                <select id="dpi">
                  <option value="300" selected>300 DPI</option>
                  <option value="600">600 DPI</option>
                </select>
              </div>
            </div>
          </div>
          <div style="margin-top:16px; display:flex; gap:12px; align-items:center;">
            <button id="submit" class="btn" type="submit">Generar stencil</button>
            <a id="download" class="muted" target="_blank" rel="noopener" style="display:none;">Descargar PNG</a>
            <span id="result" class="muted"></span>
            <span id="error" class="error"></span>
          </div>
        </form>
      </div>

      <div class="preview" style="margin-top:16px;">
        <img id="preview" alt="Resultado" />
      </div>
    </div>
  </body>
  </html>
    """
    return HTMLResponse(content=html)


