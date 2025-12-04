📄 Full-Stack 기반 OCR 문서 스캐너 프로젝트

보안과 정확성을 최우선으로 하는 로컬 기반의 문서 디지털화 및 AI 번역 솔루션

1. 개요

이 프로젝트는 인터넷 연결 없는 로컬 환경(Local-First)에서 동작하는 Full-Stack 기반 OCR 문서 스캐너를 개발하는 것을 목표로 합니다. 사용자는 민감한 정보가 포함된 문서 이미지나 PDF 파일을 외부 클라우드로 전송하지 않고, 자신의 PC에서 안전하게 텍스트로 추출하고 번역할 수 있습니다.

특히 이 시스템은 Tesseract 5의 LSTM 엔진과 CTranslate2 기반의 경량화된 AI 번역 모델을 탑재하여, 하드웨어 리소스를 효율적으로 사용하면서도 높은 정확도를 제공합니다. 불필요한 이미지 왜곡을 방지하기 위해 과도한 전처리를 배제하고, 원본 문서의 가독성을 최대한 살리는 'Clean Source' 전략을 채택하여 책, 논문, 공문서 등의 인식률을 극대화했습니다.

2. 시스템 구성 및 데이터 처리

2-1. Tesseract OCR 엔진 설정 및 처리 전략

본 프로젝트는 다양한 형태의 문서를 처리하기 위해 Tesseract OCR 엔진의 설정을 최적화하였습니다. 스크린샷과 같은 저화질 이미지보다는 고품질 문서 스캔본에 초점을 맞추어 기본 엔진 설정을 채택했습니다.

📄 OCR 엔진 설정 파라미터 명세

파라미터

설정값

설명

OEM (OCR Engine Mode)

3 (Default)

최신 LSTM 엔진과 레거시 엔진을 함께 사용하여 안정성을 확보합니다.

PSM (Page Segmentation Mode)

3 (Auto)

문서의 레이아웃(단락, 줄 바꿈 등)을 자동으로 분석하여 원본 형식을 유지합니다.

Language

kor+eng

한국어와 영어가 혼용된 문서에서 두 언어를 동시에 감지하고 추출합니다.

Preprocessing

None

과도한 이진화(Binarization)를 제거하고 그레이스케일 변환만 수행하여 폰트 정보를 보존합니다.

💻 OCR 처리 코드 예시 (Python)

def extract_text(self, image: np.ndarray, lang: str = 'kor+eng') -> str:
    # 1. OpenCV를 사용하여 BGR 이미지를 RGB로 변환 (필수)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 2. Tesseract 표준 설정 적용 (깨끗한 문서 인식률 극대화)
    custom_config = r'--oem 3 --psm 3'

    # 3. 텍스트 추출 실행
    text = pytesseract.image_to_string(pil_image, lang=lang, config=custom_config)
    return text


2-2. API 데이터 입출력 명세 (JSON Response)

FastAPI 백엔드는 클라이언트(프론트엔드)의 요청을 비동기로 처리하며, OCR 결과와 AI 번역 결과, 그리고 문서의 메타데이터를 포함한 구조화된 JSON 데이터를 반환합니다.

📄 API 응답 데이터 필드 명세

필드명

데이터 타입

설명

original_filename

String

업로드된 원본 파일의 이름

content_type

String

파일의 MIME 타입 (예: image/png, application/pdf)

raw_text

String

OCR 엔진을 통해 추출된 원본 텍스트

translation

String

CTranslate2 모델을 통해 번역된 텍스트 (옵션)

applied_language

String

OCR 수행 시 적용된 언어 코드 (예: kor+eng)

deskew_angle

Float

이미지 기울기 보정 각도 (전처리 비활성화 시 0.0)

💻 응답 데이터 예시 (1개 요청)

{
  "original_filename": "paper_scan_01.pdf",
  "content_type": "application/pdf",
  "raw_text": "This is a sample document for OCR testing.\n이것은 OCR 테스트를 위한 샘플 문서입니다.",
  "translation": "이것은 OCR 테스트를 위한 표본 문서입니다.\nThis is a sample document for OCR testing.",
  "applied_language": "kor+eng",
  "deskew_angle": 0.0,
  "message": "문서 처리가 성공적으로 완료되었습니다."
}


3. 시스템 성능 평가 및 최적화 결과 (Performance Evaluation)

본 프로젝트는 모델의 직접 학습(Fine-tuning)보다는, 사전 학습된 모델(Pre-trained Models)의 추론 속도 최적화 및 파이프라인 효율성에 중점을 두었습니다.

3-1. 텍스트 추출 및 번역 프로세스 샘플

입력된 이미지가 전처리, OCR, 그리고 AI 번역을 거쳐 최종 결과물로 변환되는 과정입니다.

단계 (Stage)

처리 내용 (Content)

결과 데이터 (Output Sample)

Input

원본 이미지 (Clean Text)

(이미지 파일: image_clean.png)

Step 1

OCR (Tesseract)

"This is a sample text for OCR performance testing."

Step 2

Tokenization (NLLB)

[' This', ' is', ' a', ' sample', ' text', ...]

Step 3

Translation (CTranslate2)

"이것은 OCR 성능 테스트를 위한 샘플 텍스트입니다."

3-2. 환경별 추론 성능 벤치마크

다양한 입력 조건과 하드웨어 가속 여부에 따른 처리 속도(Latency) 및 리소스 사용량을 측정한 결과입니다.

테스트 시나리오

엔진 설정 (OEM/PSM)

가속 모드

평균 처리 시간 (Time)

메모리 사용량 (RAM)

단일 이미지 (A4, Clean)

OEM 3 / PSM 3

CPU (Standard)

1.2s

450 MB

단일 이미지 (A4, Clean)

OEM 3 / PSM 3

GPU (CUDA)

0.4s

1.8 GB (VRAM)

PDF 문서 (10 Pages)

OEM 3 / PSM 3

CPU (Multi-thread)

12.5s

850 MB

복잡한 스크린샷

OEM 1 / PSM 6

CPU (Standard)

2.1s

510 MB

3-3. 성능 분석 및 최적화 리포트

📊 주요 관찰점 (Key Observations)

1. 추론 속도의 획기적 개선 (CTranslate2 적용)

기존 Hugging Face Transformers 파이프라인 사용 시, 번역 모델 로딩에만 약 15초 이상 소요되었으며, meta tensor 메모리 복사 오류가 빈번히 발생했습니다.

**CTranslate2(C++ 기반 추론 엔진)**로 교체하고 모델을 int8로 양자화(Quantization)한 결과, 모델 로딩 시간은 3초 이내로 단축되었으며, 추론 속도는 약 300% 향상되었습니다.

2. OCR 인식률과 전처리의 역설 (The Preprocessing Paradox)

초기에는 Otsu Binarization 및 Adaptive Thresholding과 같은 강력한 이미지 전처리를 적용했으나, 오히려 깨끗한 원본 문서의 폰트 외곽선(Anti-aliasing)을 훼손하여 인식률이 저하되는 현상이 관찰되었습니다.

이에 따라 'Clean Source' 전략으로 선회하여, 과도한 전처리를 모두 제거하고 **Tesseract 기본 엔진(OEM 3)**에 원본 이미지를 그대로 전달하는 방식이 **가장 높은 정확도(98% 이상)**를 보임을 확인했습니다.

3. 대용량 PDF 처리 안정성

pdf2image를 통해 PDF를 이미지로 변환할 때, 메모리 스파이크를 방지하기 위해 비동기(Async) 처리 방식을 도입했습니다.

그 결과, 50페이지 이상의 대용량 논문 파일 처리 시에도 메모리 누수 없이 안정적인 처리가 가능했습니다.

4. 하드웨어 가속의 효율성

CUDA가 지원되는 환경에서는 GPU 가속이 자동으로 활성화되어 대량의 텍스트 번역 시 CPU 대비 약 4배 빠른 성능을 보였습니다. CPU 환경에서도 int8 양자화 덕분에 실시간성에 준하는 응답 속도를 확보했습니다.

4. 설치 및 실행 가이드

4-1. 필수 프로그램

Python 3.11 이상

Tesseract OCR 5.0 이상 (시스템 PATH 등록 필수)

Poppler (PDF 처리용)

4-2. 실행 방법

# 1. 가상환경 생성 및 활성화
conda create -n fullstack_env python=3.11 -y
conda activate fullstack_env

# 2. 의존성 패키지 설치
pip install -r requirements.txt
conda install -c conda-forge poppler -y

# 3. 서버 실행
uvicorn main:app --reload


📝 라이선스

This project is licensed under the MIT License.
