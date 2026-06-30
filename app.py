"""
api.py
FastAPI REST API para la aplicación de zapatos similares.

Endpoints:
  GET  /shoes             – lista todos los zapatos (con metadata)
  GET  /shoes/{id}        – detalle de un zapato
  GET  /similar/{id}      – top-K zapatos más similares al dado (por DB2 ID)
  POST /similar/upload    – sube una imagen y devuelve los zapatos más similares
  GET  /image/{id}        – sirve la imagen de un zapato (base64 o archivo)
  GET  /brands            – lista de marcas únicas en la BD

Arranque:
  uvicorn api:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import json
import os
import io
from typing import Optional

import ibm_db
import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from transformers import CLIPModel, CLIPProcessor

load_dotenv()

# ── Candidatos de texto (igual que en el notebook) ───────────────────────────
BRAND_CANDIDATES = {
    'Nike':         ['a photo of a Nike shoe', 'Nike sneaker with swoosh logo',
                     'Nike Air Force athletic footwear'],
    'Pirma':        ['a photo of a Pirma shoe', 'Pirma Mexican sports sneaker',
                     'Pirma athletic shoe made in Mexico', 'tenis Pirma mexicanos'],
    'Puma':         ['a photo of a Puma shoe', 'Puma sneaker with leaping cat logo',
                     'Puma athletic footwear'],
    'Charly':       ['a photo of a Charly shoe', 'Charly Mexican sport shoe',
                     'Charly tenis deportivos Mexico', 'Charly athletic sneaker'],
    'Reebok':       ['a photo of a Reebok shoe', 'Reebok Classic sneaker',
                     'Reebok athletic footwear'],
    'Vans':         ['a photo of a Vans shoe', 'Vans Old Skool skate shoe',
                     'Vans checkerboard canvas sneaker'],
    'Converse':     ['a photo of a Converse shoe', 'Converse Chuck Taylor All Star',
                     'Converse canvas high top sneaker'],
    'Under Armour': ['a photo of an Under Armour shoe', 'Under Armour running sneaker',
                     'Under Armour athletic footwear UA logo'],
    'Fila':         ['a photo of a Fila shoe', 'Fila Disruptor sneaker with F logo',
                     'Fila retro athletic footwear'],
    'New Balance':  ['a photo of a New Balance shoe', 'New Balance 574 sneaker with N logo',
                     'New Balance running shoe'],
    'Adidas':       ['a photo of an Adidas shoe', 'Adidas sneaker with three stripes',
                     'Adidas Stan Smith athletic footwear'],
    'Huarache':     ['a photo of a huarache sandal', 'Mexican woven leather huarache',
                     'huarache sandal with ankle strap and woven sole',
                     'traditional Mexican leather woven sandal',
                     'huarache flat sandal with braided leather'],
}

COLOR_CANDIDATES = {
    'white':  ['a white shoe', 'all white sneaker', 'white athletic shoe'],
    'black':  ['a black shoe', 'all black sneaker', 'black athletic shoe'],
    'red':    ['a red shoe', 'red sneaker', 'red athletic footwear'],
    'blue':   ['a blue shoe', 'blue sneaker', 'blue athletic footwear'],
    'green':  ['a green shoe', 'green sneaker'],
    'gray':   ['a gray shoe', 'grey sneaker', 'gray athletic shoe'],
    'yellow': ['a yellow shoe', 'yellow sneaker'],
    'pink':   ['a pink shoe', 'pink sneaker'],
    'brown':  ['a brown shoe', 'brown leather shoe'],
    'multi':  ['a multicolor shoe', 'colorful sneaker'],
}

CATEGORY_CANDIDATES = {
    'running':    ['a running shoe', 'athletic running footwear', 'marathon sneaker'],
    'casual':     ['a casual shoe', 'everyday sneaker', 'lifestyle footwear'],
    'basketball': ['a basketball shoe', 'high top basketball sneaker'],
    'training':   ['a training shoe', 'cross-training athletic shoe'],
    'skate':      ['a skate shoe', 'skateboarding footwear'],
    'sandal':     ['a sandal', 'open-toe footwear', 'huarache sandal'],
}

# ── Modelos Pydantic ──────────────────────────────────────────────────────────
class ShoeRecord(BaseModel):
    id:             int
    image_name:     str
    image_path:     str
    brand:          Optional[str] = None
    brand_score:    Optional[float] = None
    brand_top3:     Optional[str] = None
    color:          Optional[str] = None
    color_score:    Optional[float] = None
    category:       Optional[str] = None
    category_score: Optional[float] = None

class SimilarResult(BaseModel):
    rank:        int
    id:          int
    image_name:  str
    image_path:  str
    brand:       Optional[str] = None
    color:       Optional[str] = None
    category:    Optional[str] = None
    similarity:  float
    image_b64:   Optional[str] = None  # thumbnail en base64 (si se pide)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Shoes Similarity API",
    description="API REST para buscar zapatos similares usando CLIP fine-tuned + IBM DB2",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # en producción restringir al dominio del frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singleton: modelo CLIP ────────────────────────────────────────────────────
_model: Optional[CLIPModel] = None
_processor: Optional[CLIPProcessor] = None
_device: str = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = os.getenv("MODEL_DIR", "clip_shoes_ft_v2")


def get_model():
    global _model, _processor
    if _model is None:
        if os.path.exists(MODEL_DIR):
            _model = CLIPModel.from_pretrained(MODEL_DIR).to(_device)
            _processor = CLIPProcessor.from_pretrained(MODEL_DIR)
        else:
            _model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).to(_device)
            _processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
        _model.eval()
    return _model, _processor


# ── DB2 helpers ───────────────────────────────────────────────────────────────
def _db2_conn():
    conn_str = (
        f"DATABASE={os.getenv('DB2_DATABASE')};"
        f"HOSTNAME={os.getenv('DB2_HOSTNAME')};"
        f"PORT={os.getenv('DB2_PORT')};"
        f"PROTOCOL=TCPIP;"
        f"UID={os.getenv('DB2_UID')};"
        f"PWD={os.getenv('DB2_PWD')};"
        f"Security=SSL;"
    )
    try:
        return ibm_db.pconnect(conn_str, "", "")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"DB2 connection error: {e}")


def _row_to_shoe(stmt) -> dict:
    return {
        "id":             ibm_db.result(stmt, 0),
        "image_name":     ibm_db.result(stmt, 1),
        "image_path":     ibm_db.result(stmt, 2),
        "brand":          ibm_db.result(stmt, 3),
        "brand_score":    float(ibm_db.result(stmt, 4)) if ibm_db.result(stmt, 4) else None,
        "brand_top3":     ibm_db.result(stmt, 5),
        "color":          ibm_db.result(stmt, 6),
        "color_score":    float(ibm_db.result(stmt, 7)) if ibm_db.result(stmt, 7) else None,
        "category":       ibm_db.result(stmt, 8),
        "category_score": float(ibm_db.result(stmt, 9)) if ibm_db.result(stmt, 9) else None,
    }


# ── Embedding helpers ─────────────────────────────────────────────────────────
def _embed_image(img: Image.Image) -> np.ndarray:
    model, processor = get_model()
    inputs = processor(images=img, return_tensors="pt").to(_device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        vec = features.pooler_output
        vec = vec / (vec.norm(dim=-1, keepdim=True) + 1e-8)
    return vec.cpu().numpy().flatten().astype(np.float32)


def _embed_texts(texts: list) -> np.ndarray:
    model, processor = get_model()
    inputs = processor(
        text=texts, return_tensors="pt",
        padding=True, truncation=True, max_length=77
    ).to(_device)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        vecs = features.pooler_output
        vecs = vecs / (vecs.norm(dim=-1, keepdim=True) + 1e-8)
    return vecs.cpu().numpy().astype(np.float32)


def _classify(image_vec: np.ndarray, candidates: dict, top_n: int = 1):
    all_texts, ranges = [], {}
    ptr = 0
    for lbl, prompts in candidates.items():
        all_texts.extend(prompts)
        ranges[lbl] = (ptr, ptr + len(prompts))
        ptr += len(prompts)
    tvecs = _embed_texts(all_texts)
    sims = tvecs @ image_vec
    scores = {lbl: float(sims[s:e].mean()) for lbl, (s, e) in ranges.items()}
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


def _predict_metadata(image_vec: np.ndarray) -> dict:
    brand_res = _classify(image_vec, BRAND_CANDIDATES, top_n=3)
    color_res = _classify(image_vec, COLOR_CANDIDATES, top_n=1)
    cat_res   = _classify(image_vec, CATEGORY_CANDIDATES, top_n=1)
    return {
        "brand":          brand_res[0][0],
        "brand_score":    round(brand_res[0][1], 4),
        "brand_top3":     " | ".join(f"{l}({s:.3f})" for l, s in brand_res),
        "color":          color_res[0][0],
        "color_score":    round(color_res[0][1], 4),
        "category":       cat_res[0][0],
        "category_score": round(cat_res[0][1], 4),
    }


def _image_to_b64(image_path: str, max_size: int = 224) -> Optional[str]:
    """Convierte imagen a base64 para enviar al frontend."""
    try:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


def _run_similarity(q_vec: np.ndarray, top_k: int,
                    brand_filter, color_filter, category_filter,
                    include_images: bool) -> list[dict]:
    """Calcula similitudes contra DB2 y devuelve los top_k resultados."""
    conn = _db2_conn()

    where_parts = []
    if brand_filter:
        where_parts.append(f"BRAND = '{brand_filter}'")
    if color_filter:
        where_parts.append(f"COLOR = '{color_filter}'")
    if category_filter:
        where_parts.append(f"CATEGORY = '{category_filter}'")

    where_sql = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
    query = (
        "SELECT ID, IMAGE_NAME, IMAGE_PATH, "
        "BRAND, BRAND_SCORE, BRAND_TOP3, COLOR, COLOR_SCORE, CATEGORY, CATEGORY_SCORE, "
        "CLIP_EMBEDDING "
        f"FROM SHOE_EMBEDDINGS_FT{where_sql}"
    )

    try:
        stmt = ibm_db.exec_immediate(conn, query)
        rows = []
        while ibm_db.fetch_row(stmt):
            row = _row_to_shoe(stmt)
            row["embedding"] = np.array(
                json.loads(ibm_db.result(stmt, 10)), dtype=np.float32
            )
            rows.append(row)
    except Exception as e:
        ibm_db.close(conn)
        raise HTTPException(status_code=500, detail=f"DB2 query error: {e}")

    ibm_db.close(conn)

    if not rows:
        return []

    matrix = np.stack([r["embedding"] for r in rows])
    scores = matrix @ q_vec
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_idx, 1):
        r = rows[idx]
        result = {
            "rank":       rank,
            "id":         r["id"],
            "image_name": r["image_name"],
            "image_path": r["image_path"],
            "brand":      r["brand"],
            "color":      r["color"],
            "category":   r["category"],
            "similarity": round(float(scores[idx]), 4),
        }
        if include_images:
            result["image_b64"] = _image_to_b64(r["image_path"])
        results.append(result)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "service": "Shoes Similarity API v1"}


@app.get("/brands", tags=["metadata"])
def list_brands():
    """Devuelve las marcas únicas almacenadas en la base de datos."""
    conn = _db2_conn()
    try:
        stmt = ibm_db.exec_immediate(
            conn, "SELECT DISTINCT BRAND FROM SHOE_EMBEDDINGS_FT ORDER BY BRAND"
        )
        brands = []
        while ibm_db.fetch_row(stmt):
            brands.append(ibm_db.result(stmt, 0))
    except Exception as e:
        ibm_db.close(conn)
        raise HTTPException(status_code=500, detail=str(e))
    ibm_db.close(conn)
    return {"brands": brands}


@app.get("/shoes", tags=["shoes"], response_model=list[ShoeRecord])
def list_shoes(
    brand:    Optional[str] = Query(None, description="Filtrar por marca"),
    color:    Optional[str] = Query(None, description="Filtrar por color"),
    category: Optional[str] = Query(None, description="Filtrar por categoría"),
    limit:    int           = Query(100, ge=1, le=500),
    offset:   int           = Query(0,   ge=0),
):
    """Lista zapatos con paginación y filtros opcionales."""
    where_parts = []
    if brand:
        where_parts.append(f"BRAND = '{brand}'")
    if color:
        where_parts.append(f"COLOR = '{color}'")
    if category:
        where_parts.append(f"CATEGORY = '{category}'")

    where_sql = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
    query = (
        "SELECT ID, IMAGE_NAME, IMAGE_PATH, "
        "BRAND, BRAND_SCORE, BRAND_TOP3, COLOR, COLOR_SCORE, CATEGORY, CATEGORY_SCORE "
        f"FROM SHOE_EMBEDDINGS_FT{where_sql} "
        f"ORDER BY ID OFFSET {offset} ROWS FETCH NEXT {limit} ROWS ONLY"
    )

    conn = _db2_conn()
    try:
        stmt = ibm_db.exec_immediate(conn, query)
        shoes = []
        while ibm_db.fetch_row(stmt):
            shoes.append(ShoeRecord(**_row_to_shoe(stmt)))
    except Exception as e:
        ibm_db.close(conn)
        raise HTTPException(status_code=500, detail=str(e))

    ibm_db.close(conn)
    return shoes


@app.get("/shoes/{shoe_id}", tags=["shoes"], response_model=ShoeRecord)
def get_shoe(shoe_id: int):
    """Detalle de un zapato por ID."""
    conn = _db2_conn()
    query = (
        "SELECT ID, IMAGE_NAME, IMAGE_PATH, "
        "BRAND, BRAND_SCORE, BRAND_TOP3, COLOR, COLOR_SCORE, CATEGORY, CATEGORY_SCORE "
        f"FROM SHOE_EMBEDDINGS_FT WHERE ID = {shoe_id}"
    )
    try:
        stmt = ibm_db.exec_immediate(conn, query)
        if not ibm_db.fetch_row(stmt):
            ibm_db.close(conn)
            raise HTTPException(status_code=404, detail="Shoe not found")
        shoe = ShoeRecord(**_row_to_shoe(stmt))
    except HTTPException:
        raise
    except Exception as e:
        ibm_db.close(conn)
        raise HTTPException(status_code=500, detail=str(e))

    ibm_db.close(conn)
    return shoe


@app.get("/image/{shoe_id}", tags=["shoes"])
def get_shoe_image(shoe_id: int):
    """Devuelve la imagen en base64 para un zapato dado."""
    conn = _db2_conn()
    try:
        stmt = ibm_db.exec_immediate(
            conn,
            f"SELECT IMAGE_PATH FROM SHOE_EMBEDDINGS_FT WHERE ID = {shoe_id}"
        )
        if not ibm_db.fetch_row(stmt):
            ibm_db.close(conn)
            raise HTTPException(status_code=404, detail="Shoe not found")
        path = ibm_db.result(stmt, 0)
    except HTTPException:
        raise
    except Exception as e:
        ibm_db.close(conn)
        raise HTTPException(status_code=500, detail=str(e))

    ibm_db.close(conn)

    b64 = _image_to_b64(path)
    if b64 is None:
        raise HTTPException(status_code=404, detail="Image file not found")

    return {"id": shoe_id, "image_path": path, "image_b64": b64}


@app.get("/similar/{shoe_id}", tags=["similarity"])
def similar_by_id(
    shoe_id:        int,
    top_k:          int           = Query(5,    ge=1, le=50),
    brand_filter:   Optional[str] = Query(None, description="Filtrar resultados por marca"),
    color_filter:   Optional[str] = Query(None, description="Filtrar resultados por color"),
    category_filter:Optional[str] = Query(None, description="Filtrar resultados por categoría"),
    include_images: bool          = Query(False, description="Incluir thumbnail en base64"),
):
    """
    Encuentra los `top_k` zapatos más similares al zapato con `shoe_id`.
    El embedding se lee directamente de DB2 (sin re-inferir el modelo).
    """
    conn = _db2_conn()
    try:
        stmt = ibm_db.exec_immediate(
            conn,
            f"SELECT CLIP_EMBEDDING FROM SHOE_EMBEDDINGS_FT WHERE ID = {shoe_id}"
        )
        if not ibm_db.fetch_row(stmt):
            ibm_db.close(conn)
            raise HTTPException(status_code=404, detail="Shoe not found")
        emb_json = ibm_db.result(stmt, 0)
    except HTTPException:
        raise
    except Exception as e:
        ibm_db.close(conn)
        raise HTTPException(status_code=500, detail=str(e))

    ibm_db.close(conn)

    q_vec = np.array(json.loads(emb_json), dtype=np.float32)
    results = _run_similarity(
        q_vec, top_k, brand_filter, color_filter, category_filter, include_images
    )
    return {"query_id": shoe_id, "results": results}


@app.post("/similar/upload", tags=["similarity"])
async def similar_by_upload(
    file:            UploadFile     = File(..., description="Imagen de zapato a buscar"),
    top_k:           int            = Query(5,    ge=1, le=50),
    brand_filter:    Optional[str]  = Query(None),
    color_filter:    Optional[str]  = Query(None),
    category_filter: Optional[str]  = Query(None),
    include_images:  bool           = Query(False),
):
    """
    Recibe una imagen (upload), predice su marca/color/categoría con el
    modelo fine-tuned y devuelve los `top_k` zapatos más similares en DB2.
    """
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    q_vec  = _embed_image(img)
    q_meta = _predict_metadata(q_vec)

    results = _run_similarity(
        q_vec, top_k, brand_filter, color_filter, category_filter, include_images
    )
    return {
        "query_metadata": q_meta,
        "results":        results,
    }


# ── Arranque rápido ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

# Made with Bob
