import os
import io
import time
import base64
import logging
from typing import Optional, List, Dict, Any

# Disabilita oneDNN / MKLDNN e PIR per evitare il noto bug di PaddlePaddle 3.x su CPU:
# "ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"

# Configurazione thread per PaddlePaddle compilato con OpenBLAS (evita crash di OpenBLAS multi-thread)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PaddleOCR-Studio")

app = FastAPI(title="PaddleOCR Studio API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache to avoid re-initializing models for repeated requests with same settings
OCR_MODEL_CACHE: Dict[str, Any] = {}

SUPPORTED_LANGUAGES = [
    {"code": "it", "name": "Italiano", "flag": "🇮🇹"},
    {"code": "en", "name": "English", "flag": "🇬🇧"},
    {"code": "fr", "name": "Français", "flag": "🇫🇷"},
    {"code": "german", "name": "Deutsch", "flag": "🇩🇪"},
    {"code": "es", "name": "Español", "flag": "🇪🇸"},
    {"code": "pt", "name": "Português", "flag": "🇵🇹"},
    {"code": "ch", "name": "Chinese (Simplified)", "flag": "🇨🇳"},
    {"code": "chinese_cht", "name": "Chinese (Traditional)", "flag": "🇹🇼"},
    {"code": "japan", "name": "Japanese", "flag": "🇯🇵"},
    {"code": "korean", "name": "Korean", "flag": "🇰🇷"},
    {"code": "ru", "name": "Russian", "flag": "🇷🇺"},
    {"code": "ar", "name": "Arabic", "flag": "🇸🇦"},
    {"code": "latin", "name": "Latin / Multilingual", "flag": "🌐"},
]

def check_gpu_support() -> bool:
    try:
        import paddle
        return paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
    except Exception:
        return False

def get_ocr_engine(lang: str = "it", use_angle_cls: bool = True, use_gpu: Optional[bool] = None):
    from paddleocr import PaddleOCR
    import logging
    # Disabilita messaggi di debug/info invasivi di PaddleOCR senza usare il parametro rimosso show_log
    logging.getLogger('ppocr').setLevel(logging.ERROR)
    
    try:
        import paddle
        paddle.set_flags({'FLAGS_use_mkldnn': False, 'FLAGS_enable_pir_api': False})
    except Exception:
        pass

    if use_gpu is None:
        use_gpu = check_gpu_support()

    cache_key = f"{lang}_{use_angle_cls}_{use_gpu}"
    if cache_key not in OCR_MODEL_CACHE:
        logger.info(f"Caricamento modello PaddleOCR (lang={lang}, angle_cls={use_angle_cls}, gpu={use_gpu})...")
        device_str = "gpu" if use_gpu else "cpu"
        engine = None
        last_err = None

        # Tentativi di inizializzazione compatibili sia con PaddleOCR 2.x che 3.x / PaddleX
        # Nota: disabilitiamo use_doc_unwarping (UVDoc) e use_doc_orientation_classify per evitare
        # l'enorme consumo di RAM e calcolo del raddrizzamento 3D inutile su PDF piani che manda in crash la CPU.
        attempts = [
            # 1. PaddleOCR 3.x leggero e rapido (solo textline_ori + OCR, senza il pesante modello 3D UVDoc)
            lambda: PaddleOCR(
                lang=lang,
                use_textline_orientation=use_angle_cls,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                device=device_str,
                enable_mkldnn=False
            ),
            lambda: PaddleOCR(
                lang=lang,
                use_textline_orientation=use_angle_cls,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                device=device_str
            ),
            lambda: PaddleOCR(
                lang=lang,
                use_textline_orientation=use_angle_cls,
                enable_mkldnn=False
            ),
            # 2. PaddleOCR 2.x standard (use_angle_cls + use_gpu)
            lambda: PaddleOCR(lang=lang, use_angle_cls=use_angle_cls, use_gpu=use_gpu, enable_mkldnn=False),
            lambda: PaddleOCR(lang=lang, use_angle_cls=use_angle_cls, enable_mkldnn=False),
            lambda: PaddleOCR(lang=lang, enable_mkldnn=False),
            # 3. Senza enable_mkldnn
            lambda: PaddleOCR(lang=lang, use_textline_orientation=use_angle_cls, device=device_str),
            lambda: PaddleOCR(lang=lang, use_textline_orientation=use_angle_cls),
            lambda: PaddleOCR(lang=lang, use_angle_cls=use_angle_cls, use_gpu=use_gpu),
            lambda: PaddleOCR(lang=lang, use_angle_cls=use_angle_cls),
            lambda: PaddleOCR(lang=lang),
            lambda: PaddleOCR()
        ]

        for init_attempt in attempts:
            try:
                engine = init_attempt()
                if engine is not None:
                    break
            except Exception as err:
                last_err = err
                logger.warning(f"Inizializzazione PaddleOCR fallita con tentativo ({err}), provo alternativa...")

        if engine is None:
            raise RuntimeError(f"Impossibile inizializzare il motore PaddleOCR: {last_err}")

        OCR_MODEL_CACHE[cache_key] = engine
        logger.info("Modello PaddleOCR caricato con successo.")
    return OCR_MODEL_CACHE[cache_key]

def image_to_base64(img: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    buffered = io.BytesIO()
    # Convert RGBA to RGB if saving as JPEG
    if format.upper() == "JPEG" and img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.save(buffered, format=format, quality=quality)
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_b64}"

def extract_images_from_pdf(pdf_bytes: bytes, max_pages: int = 20) -> List[Image.Image]:
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(pdf_bytes)
    total_pages = len(pdf)
    pages_to_render = min(total_pages, max_pages)
    
    images = []
    for page_idx in range(pages_to_render):
        page = pdf[page_idx]
        # Render at 150 DPI for fast and crisp OCR
        bitmap = page.render(scale=2.0)
        pil_image = bitmap.to_pil()
        images.append(pil_image)
    return images

def normalize_ocr_result(raw_result) -> List[Dict[str, Any]]:
    """
    Normalizza l'output OCR in una lista omogenea di riquadri:
    [
        {"box": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], "text": "...", "confidence": 0.95},
        ...
    ]
    Compatibile sia con PaddleOCR 2.x (liste annidate) che con PaddleOCR 3.x / PaddleX
    (generatori, dict con dt_polys, rec_text/rec_texts, rec_score/rec_scores).
    """
    extracted_lines = []
    if raw_result is None:
        return extracted_lines

    # Se raw_result è un generatore o iteratore, convertiamolo in lista
    if not isinstance(raw_result, list):
        try:
            raw_result = list(raw_result)
        except Exception:
            raw_result = [raw_result]

    if len(raw_result) == 0:
        return extracted_lines

    first_item = raw_result[0]

    # CASO 1: PaddleOCR 2.x standard -> raw_result = [ [ [box, (text, conf)], ... ] ]
    if isinstance(first_item, list):
        for entry in first_item:
            try:
                if entry is None or len(entry) < 2:
                    continue
                box = entry[0]
                text_info = entry[1]
                text = str(text_info[0]) if isinstance(text_info, (list, tuple)) else str(text_info)
                conf = float(text_info[1]) if isinstance(text_info, (list, tuple)) and len(text_info) > 1 else 1.0
                
                box_pts = [[round(float(pt[0]), 1), round(float(pt[1]), 1)] for pt in box]
                extracted_lines.append({
                    "box": box_pts,
                    "text": text,
                    "confidence": round(conf, 4)
                })
            except Exception:
                continue
        return extracted_lines

    # CASO 2: PaddleOCR 3.x / PaddleX -> raw_result è una lista di oggetti Result o dizionari
    for res_obj in raw_result:
        res_dict = None
        if hasattr(res_obj, "json") and isinstance(res_obj.json, dict):
            res_dict = res_obj.json
        elif hasattr(res_obj, "res") and isinstance(res_obj.res, dict):
            res_dict = res_obj.res
        elif isinstance(res_obj, dict):
            res_dict = res_obj

        if not res_dict:
            continue

        boxes = res_dict.get("dt_polys") or res_dict.get("dt_boxes") or []
        texts = res_dict.get("rec_text") or res_dict.get("rec_texts") or []
        scores = res_dict.get("rec_score") or res_dict.get("rec_scores") or []

        if isinstance(texts, str):
            texts = [texts]
        if isinstance(scores, (int, float)):
            scores = [scores]

        for i in range(len(boxes)):
            try:
                box = boxes[i]
                text = str(texts[i]) if i < len(texts) else ""
                conf = float(scores[i]) if i < len(scores) else 1.0
                
                if hasattr(box, "tolist"):
                    box = box.tolist()
                box_pts = [[round(float(pt[0]), 1), round(float(pt[1]), 1)] for pt in box]
                extracted_lines.append({
                    "box": box_pts,
                    "text": text,
                    "confidence": round(conf, 4)
                })
            except Exception:
                continue

    return extracted_lines

@app.get("/api/languages")
def get_languages():
    return {"languages": SUPPORTED_LANGUAGES}

@app.get("/api/system")
def get_system_info():
    gpu_available = check_gpu_support()
    device_name = "CPU"
    if gpu_available:
        try:
            import paddle
            device_name = f"NVIDIA GPU ({paddle.device.cuda.get_device_name(0)})"
        except Exception:
            device_name = "NVIDIA CUDA GPU"

    return {
        "gpu_available": gpu_available,
        "device": device_name,
        "status": "ready"
    }

@app.post("/api/ocr")
async def run_ocr(
    file: UploadFile = File(...),
    lang: str = Form("it"),
    use_angle_cls: bool = Form(True),
    min_confidence: float = Form(0.0),
    use_gpu: Optional[bool] = Form(None)
):
    start_time = time.time()
    contents = await file.read()
    filename = file.filename or "uploaded_file"
    file_ext = os.path.splitext(filename)[1].lower()

    # Determine input type
    is_pdf = file_ext == ".pdf" or file.content_type == "application/pdf"

    pages_images: List[Image.Image] = []
    if is_pdf:
        try:
            pages_images = extract_images_from_pdf(contents)
        except Exception as e:
            logger.error(f"Errore lettura PDF: {e}")
            raise HTTPException(status_code=400, detail=f"Errore nella lettura del PDF: {str(e)}")
    else:
        try:
            img = Image.open(io.BytesIO(contents))
            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")
            pages_images = [img]
        except Exception as e:
            logger.error(f"Errore apertura immagine: {e}")
            raise HTTPException(status_code=400, detail=f"File non valido o immagine corrotta: {str(e)}")

    if not pages_images:
        raise HTTPException(status_code=400, detail="Nessuna pagina o immagine trovata nel documento.")

    try:
        ocr_engine = get_ocr_engine(lang=lang, use_angle_cls=use_angle_cls, use_gpu=use_gpu)
    except Exception as e:
        logger.error(f"Errore inizializzazione motore OCR: {e}")
        raise HTTPException(status_code=500, detail=f"Errore caricamento motore OCR: {str(e)}")

    pages_result = []
    total_boxes = 0
    all_confidences = []

    try:
        total_pages_count = len(pages_images)
        logger.info(f"Avvio elaborazione OCR su {total_pages_count} pagina/e...")

        for page_idx, pil_img in enumerate(pages_images):
            img_np = np.array(pil_img)
            w, h = pil_img.size
            logger.info(f"Elaborazione pagina {page_idx + 1}/{total_pages_count} ({w}x{h})...")

            # Invocazione flessibile dell'inferenza (compatibile sia 2.x che 3.x/PaddleX)
            raw_result = None
            try:
                raw_result = ocr_engine.ocr(img_np, cls=use_angle_cls)
            except (TypeError, ValueError, Exception):
                try:
                    raw_result = ocr_engine.ocr(img_np)
                except Exception as call_err:
                    logger.error(f"Errore chiamata inferenza ocr(): {call_err}")
                    raise call_err

            normalized_lines = normalize_ocr_result(raw_result)

            page_lines = []
            line_counter = 0

            for item in normalized_lines:
                box = item["box"]
                text = item["text"]
                confidence = float(item["confidence"])

                if confidence >= min_confidence:
                    total_boxes += 1
                    all_confidences.append(confidence)
                    page_lines.append({
                        "id": line_counter,
                        "text": text,
                        "confidence": round(confidence, 4),
                        "box": box
                    })
                    line_counter += 1

            # Ordina le righe dall'alto in basso, da sinistra a destra
            page_lines.sort(key=lambda item: (item["box"][0][1], item["box"][0][0]) if item["box"] else (0, 0))

            full_text = "\n".join([line["text"] for line in page_lines])
            logger.info(f"Pagina {page_idx + 1}/{total_pages_count} completata: {len(page_lines)} righe rilevate.")

            pages_result.append({
                "page_number": page_idx + 1,
                "width": w,
                "height": h,
                "image_data": image_to_base64(pil_img, format="JPEG", quality=85),
                "lines": page_lines,
                "full_text": full_text
            })
    except Exception as e:
        logger.error(f"Errore inferenza OCR: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Errore elaborazione OCR: {str(e)}")

    elapsed_ms = round((time.time() - start_time) * 1000, 1)
    avg_conf = round(sum(all_confidences) / len(all_confidences), 4) if all_confidences else 0.0

    return {
        "success": True,
        "filename": filename,
        "total_pages": len(pages_result),
        "total_boxes": total_boxes,
        "avg_confidence": avg_conf,
        "inference_time_ms": elapsed_ms,
        "device": "GPU" if check_gpu_support() else "CPU",
        "pages": pages_result
    }

def generate_structured_pdf(ocr_result: dict, mode: str = "searchable") -> bytes:
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader

    buffer = io.BytesIO()
    pages = ocr_result.get("pages", [])
    if not pages:
        c = canvas.Canvas(buffer)
        c.save()
        return buffer.getvalue()

    first_page = pages[0]
    c = canvas.Canvas(buffer, pagesize=(first_page.get("width", 595), first_page.get("height", 842)))

    for page in pages:
        pw = float(page.get("width", 595))
        ph = float(page.get("height", 842))
        c.setPageSize((pw, ph))

        img_b64 = page.get("image_data")
        if mode == "searchable" and img_b64:
            try:
                if "," in img_b64:
                    img_b64 = img_b64.split(",", 1)[1]
                img_bytes = base64.b64decode(img_b64)
                img_reader = ImageReader(io.BytesIO(img_bytes))
                c.drawImage(img_reader, 0, 0, width=pw, height=ph)
            except Exception as e:
                logger.warning(f"Impossibile renderizzare immagine di sfondo PDF: {e}")

        lines = page.get("lines", [])
        for line in lines:
            text = line.get("text", "")
            if not text:
                continue
            box = line.get("box", [])
            if len(box) >= 4:
                xs = [pt[0] for pt in box]
                ys = [pt[1] for pt in box]
                x_min = float(min(xs))
                x_max = float(max(xs))
                y_min = float(min(ys))
                y_max = float(max(ys))
                box_w = max(1.0, x_max - x_min)
                box_h = max(1.0, y_max - y_min)

                y_pdf = ph - y_max
                font_size = max(6.0, min(box_h * 0.75, 100.0))
                baseline_y = y_pdf + (box_h * 0.18)

                textobject = c.beginText()
                textobject.setTextOrigin(x_min, baseline_y)
                textobject.setFont("Helvetica", font_size)

                string_w = c.stringWidth(text, "Helvetica", font_size)
                if string_w > 0 and len(text) > 1 and box_w > string_w:
                    char_space = (box_w - string_w) / (len(text) - 1)
                    textobject.setCharSpace(min(char_space, font_size * 0.4))

                if mode == "searchable":
                    # Rendering mode 3: testo invisibile ma completamente selezionabile e ricercabile
                    textobject._code.append("3 Tr")
                else:
                    textobject.setFillColorRGB(0.08, 0.08, 0.08)

                textobject.textLine(text)
                c.drawText(textobject)

        c.showPage()

    c.save()
    buffer.seek(0)
    return buffer.getvalue()

from fastapi.responses import Response

@app.post("/api/export/pdf")
async def export_pdf(payload: Dict[str, Any]):
    mode = payload.get("mode", "searchable")
    ocr_data = payload.get("ocr_data", {})
    if not ocr_data:
        raise HTTPException(status_code=400, detail="Nessun dato OCR fornito per l'esportazione.")

    pdf_bytes = generate_structured_pdf(ocr_data, mode=mode)
    filename = f"paddleocr_{mode}_{int(time.time())}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


# Mount static folder for frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
def index_root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>PaddleOCR Studio Frontend in preparazione...</h1>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
