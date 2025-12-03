import uvicorn
import asyncio
import numpy as np
import cv2
import io
from enum import Enum
from typing import List, Optional, Annotated

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# 프로젝트 모듈 임포트
from image_processor import ImageProcessor
from ocr_engine import OCREngine
from nlp_processor import NLLBTranslator  # 번역기 임포트

# --- 애플리케이션 및 모듈 초기화 ---
app = FastAPI(
    title="로컬 풀스택 OCR 문서 스캐너",
    description="Tesseract와 OpenCV, FastAPI를 사용하여 로컬에서 문서를 처리하는 API",
    version="1.0.0"
)

# 핵심 모듈 인스턴스 생성
processor = ImageProcessor()
ocr = OCREngine()

# AI 번역기 인스턴스 (비동기 로딩)
translator = None


@app.on_event("startup")
async def startup_event():
    """
    서버 시작 시 AI 모델을 비동기적으로 로드합니다.
    """
    global translator
    print("서버 시작... AI 번역 모델 로딩을 시작합니다. (시간이 걸릴 수 있음)")
    try:
        # nlp_processor.py의 NLLBTranslator 클래스 인스턴스 생성
        translator = await asyncio.to_thread(NLLBTranslator)
        # nlp_processor.py가 성공 로그를 출력할 것입니다.
    except Exception as e:
        print(f"AI 번역 모델 로딩 실패: {e}")
        print("번역 기능을 제외하고 서버를 시작합니다.")
        translator = None


# --- API 응답 모델 정의 (Pydantic) ---

class ProcessResponse(BaseModel):
    """
    문서 처리 API의 표준 응답 모델
    """
    original_filename: str
    content_type: str
    raw_text: str
    processed_text: Optional[str] = None
    translation: Optional[str] = None
    applied_language: str
    detected_language: Optional[str] = None
    deskew_angle: float
    message: str


class TargetLanguage(str, Enum):
    """
    번역 대상 언어 목록 (Flores-200 코드 기준)
    """
    KOREAN = "kor_Hang"
    ENGLISH = "eng_Latn"
    JAPANESE = "jpn_Jpan"
    CHINESE_SIMPLIFIED = "zho_Hans"
    SPANISH = "spa_Latn"
    FRENCH = "fra_Latn"
    GERMAN = "deu_Latn"
    NONE = "none"  # 번역 안 함


# --- API 엔드포인트 정의 ---

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """
    기본 HTML 프론트엔드 페이지를 제공합니다.
    (이 부분이 작동하여 UI가 보이고 있습니다)
    """
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>index.html 파일을 찾을 수 없습니다.</h1>", status_code=404)


# [가장 중요한 부분]
# "Not Found" 오류는 이 엔드포인트가 누락되었기 때문입니다.
@app.post("/api/v1/process/document", response_model=ProcessResponse)
async def process_document(
        file: Annotated[UploadFile, File()],
        ocr_language: Annotated[str, Form()] = 'kor+eng',
        target_language: Annotated[TargetLanguage, Form()] = TargetLanguage.NONE,
        auto_deskew: Annotated[bool, Form()] = True,
        optimize_resolution: Annotated[bool, Form()] = True
):
    """
    업로드된 문서 이미지(file)를 받아 OCR 및 번역(선택 사항)을 수행합니다.
    """
    try:
        # 1. 파일 읽기
        contents = await file.read()

        # 2. 이미지 처리 (OpenCV)
        nparr = np.frombuffer(contents, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_cv is None:
            raise HTTPException(status_code=400, detail="이미지 파일을 디코딩할 수 없습니다.")

        # 3. 이미지 전처리 (image_processor.py)
        if optimize_resolution:
            img_processed = processor.optimize_resolution(img_cv)
        else:
            img_processed = img_cv

        deskew_angle = 0.0
        if auto_deskew:
            img_processed, deskew_angle = processor.deskew(img_processed)

        # 4. OCR 텍스트 추출 (ocr_engine.py)
        raw_text = ocr.extract_text(img_processed, lang=ocr_language)

        # 5. (선택적) AI 번역 (nlp_processor.py)
        translated_text = None
        if translator is not None and target_language != TargetLanguage.NONE:
            print(f"번역 수행: {target_language.value}로 번역합니다.")
            translated_text = await asyncio.to_thread(
                translator.translate,
                raw_text,
                src_lang="kor_Hang",  # 참고: 현재 소스 언어를 'kor_Hang'로 고정
                tgt_lang=target_language.value
            )

        # 6. 응답 생성
        return ProcessResponse(
            original_filename=file.filename or "unknown",
            content_type=file.content_type or "unknown",
            raw_text=raw_text,
            translation=translated_text,
            applied_language=ocr_language,
            deskew_angle=deskew_angle,
            message="문서 처리가 성공적으로 완료되었습니다."
        )

    except Exception as e:
        print(f"처리 중 오류 발생: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "message": f"파일 처리 중 오류가 발생했습니다: {str(e)}"
            }
        )


# --- 서버 실행 (개발용) ---
if __name__ == "__main__":
    print("개발용 서버를 http://127.0.0.1:8000 에서 시작합니다.")
    print("HTML UI는 http://127.0.0.1:8000 에서 확인하세요.")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
