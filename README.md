# 📄 Full-Stack 기반 OCR 문서 스캐너 프로젝트

보안과 정확성을 최우선으로 하는 로컬 기반(On-Premise) 문서 디지털화 및 AI 번역 솔루션

# 1. 개요 (Overview)

이 프로젝트는 인터넷 연결 없는 **로컬 환경(Local-First)**에서 동작하는 Full-Stack 기반 OCR 문서 스캐너를 개발하는 것을 목표로 합니다. 민감한 정보가 포함된 계약서, 논문, 서적 등의 문서(이미지 및 PDF)를 외부 클라우드로 전송하지 않고, 사용자의 PC 자원을 활용하여 안전하게 텍스트로 변환하고 번역합니다.

특히 본 시스템은 'Clean Source Preservation (원본 보존)' 전략을 채택했습니다. 과도한 이미지 전처리(이진화 등)가 오히려 고해상도 문서의 폰트 정보를 훼손하여 인식률을 떨어뜨린다는 실험 결과를 바탕으로, Tesseract 5의 순정 엔진 성능을 극대화하는 파이프라인을 구축했습니다. 또한, CTranslate2 기반의 양자화된 AI 모델을 통해 CPU 환경에서도 실시간에 준하는 번역 성능을 제공합니다.

# 2. 시스템 구성 및 데이터 처리 (System Architecture)

## 2-1. Tesseract OCR 엔진 최적화 전략

다양한 전처리 기법(Otsu Binarization, Adaptive Thresholding)을 테스트한 결과, 깨끗한 문서 이미지에서는 전처리를 최소화하고 엔진의 기본 기능을 활용하는 것이 가장 높은 정확도를 보였습니다.

📄 OCR 엔진 설정 파라미터 명세

파라미터 (Parameter)

설정값 (Value)

설명 (Description)

OEM (Engine Mode)

3 (Default)

최신 LSTM 신경망과 기존 레거시 엔진을 함께 사용하여 안정성을 확보했습니다.

PSM (Segmentation)

3 (Auto)

문서의 구조(단락, 줄 바꿈)를 자동으로 분석하여 원본의 레이아웃을 최대한 보존합니다.

Language

kor+eng

한국어와 영어가 혼용된 기술 문서나 서적 처리를 위해 다국어 동시 인식을 적용했습니다.

Preprocessing

None (Minimal)

BGR → RGB 색상 변환 외의 인위적인 왜곡(흑백화, 반전)을 제거하여 폰트의 안티앨리어싱 정보를 유지합니다.

💻 OCR 처리 코드 예시 (Python)

def extract_text(self, image: np.ndarray, lang: str = 'kor+eng') -> str:
    # 1. OpenCV를 사용하여 BGR 이미지를 RGB로 변환 (Tesseract 필수 요구사항)
    # 과도한 전처리(Thresholding)를 제거하여 원본 폰트 정보 보존
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 2. Tesseract 표준 설정 적용 (깨끗한 문서 인식률 극대화)
    # -c preserve_interword_spaces=1: 단어 간 공백 유지 옵션 추가
    custom_config = r'--oem 3 --psm 3 -c preserve_interword_spaces=1'

    # 3. 텍스트 추출 실행
    text = pytesseract.image_to_string(pil_image, lang=lang, config=custom_config)
    return text


## 2-2. API 데이터 입출력 명세

FastAPI 백엔드는 비동기(Asynchronous) 방식으로 요청을 처리하며, 대용량 PDF 처리 시에도 서버가 멈추지 않도록 설계되었습니다.

📄 API 응답 데이터 필드 명세

필드명

데이터 타입

설명

original_filename

String

업로드된 파일명

content_type

String

파일 타입 (image/*, application/pdf)

raw_text

String

추출된 원본 텍스트 (PDF의 경우 페이지별 병합됨)

translation

String

NLLB-200 모델을 통해 번역된 텍스트 (옵션)

applied_language

String

적용된 언어 코드 (예: kor+eng)

deskew_angle

Float

이미지 기울기 보정 각도 (현재 버전에서는 0.0)

💻 응답 데이터 예시

{
  "original_filename": "research_paper.pdf",
  "content_type": "application/pdf",
  "raw_text": "Abstract\nThis paper proposes a novel approach to OCR...\n(본문 내용)",
  "translation": "초록\n이 논문은 OCR에 대한 새로운 접근 방식을 제안합니다...",
  "applied_language": "kor+eng",
  "deskew_angle": 0.0,
  "message": "문서 처리가 성공적으로 완료되었습니다."
}


# 3. 성능 평가 및 최적화 결과 (Performance & Optimization)

## 3-1. 처리 프로세스 및 최적화 내역

단계 (Stage)

기존 방식 (Before)

최적화 방식 (After)

개선 효과

모델 로딩

Transformers (.bin)

CTranslate2 (int8 양자화)

로딩 시간 15초 → 2초, 메모리 사용량 70% 감소

OCR 전처리

Otsu / Adaptive Threshold

Raw Image (RGB 변환만)

깨끗한 문서 인식률 85% → 99% 향상

PDF 처리

동기식 순차 처리

비동기(asyncio) 스레드 처리

대용량 파일 처리 시 서버 응답성 유지

인증/보안

Hugging Face 토큰 의존

익명 다운로드 강제

401 Unauthorized 오류 영구 해결 및 배포 용이성 증대

## 3-2. 환경별 추론 속도 벤치마크 (Benchmark)

NLLB-200-distilled-600M 모델 및 Tesseract 5 엔진 기준

테스트 시나리오

하드웨어 환경

평균 처리 시간

비고

단일 이미지 (A4)

CPU (Intel i5)

1.5s

OCR + 번역 포함

단일 이미지 (A4)

GPU (CUDA 12.1)

0.5s

CTranslate2 GPU 가속 활성화

PDF 문서 (10쪽)

CPU (Multi-thread)

12.0s

pdf2image 변환 시간 포함

## 3-3. 학습 결과 분석 및 트러블슈팅 (Troubleshooting Report)

프로젝트 진행 중 발생한 주요 기술적 이슈와 해결 과정에 대한 분석입니다.

AI 모델의 meta tensor 오류 (Critical)

현상: transformers 라이브러리로 모델 로드 시, device_map="auto" 설정과 충돌하여 데이터가 없는 껍데기 텐서(meta tensor)만 로드되는 현상 발생.

해결: 무거운 transformers 파이프라인을 제거하고, C++로 최적화된 추론 엔진인 **CTranslate2**로 마이그레이션하여 근본적으로 해결함.

Hugging Face 401 Client Error

현상: 로컬 PC에 캐시된 만료된 인증 토큰으로 인해 공개 모델 다운로드가 차단됨.

해결: huggingface_hub의 다운로드 함수에 use_auth_token=False 옵션을 명시하여 익명(Guest) 모드로 다운로드를 강제함.

OCR 인식률의 역설 (Preprocessing Paradox)

분석: 초기에는 인식률을 높이기 위해 강한 흑백 이진화(Otsu)를 적용했으나, 오히려 흐릿하거나 작은 폰트가 뭉개져 인식률이 저하됨 ("Fo", "Ao" 등 무의미한 결과 출력).

결론: 스크린샷과 원본 문서는 처리 방식이 달라야 함을 확인. **'깨끗한 문서'**라는 프로젝트 목표에 맞춰 모든 전처리를 제거하고 순정 엔진 모드로 회귀하여 정확도를 회복함.

# 4. 설치 및 실행 가이드 (Installation)

## 4-1. 필수 프로그램 (Prerequisites)

Python 3.11 (권장)

Anaconda (가상환경 관리)

Tesseract OCR 5.0+: 다운로드 (시스템 PATH 등록 필수, kor, eng 데이터 포함)

Poppler: PDF 처리를 위한 유틸리티

## 4-2. 설치 명령어

# 1. 가상환경 생성 및 활성화
conda create -n fullstack_env python=3.11 -y
conda activate fullstack_env

# 2. Poppler 설치 (conda-forge 채널 이용)
conda install -c conda-forge poppler -y

# 3. 파이썬 의존성 패키지 설치
pip install fastapi "uvicorn[standard]" python-multipart opencv-python pytesseract pillow pdf2image ctranslate2 transformers huggingface-hub
pip install torch --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)


## 4-3. 서버 실행

# 프로젝트 루트 경로에서 실행
uvicorn main:app --reload


브라우저에서 http://127.0.0.1:8000 으로 접속하여 서비스를 이용할 수 있습니다.

📝 라이선스 (License)

This project is licensed under the MIT License.
