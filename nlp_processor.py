import torch
import ctranslate2
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download  # 명시적 다운로드를 위해 import
import warnings

# 경고 메시지 비활성화
warnings.filterwarnings("ignore", category=UserWarning)


class NLLBTranslator:
    """
    [최종 수정] 'meta tensor', '401', 'Repo Not Found' 오류를 모두 수정한 버전입니다.
    - ctranslate2 (빠른 추론 엔진) 사용
    - OpenNMT (올바른 모델 주소) 사용
    - use_auth_token=False (익명 다운로드 강제) 사용
    """

    def __init__(self,
                 model_id: str = "facebook/nllb-200-distilled-600M",
                 # [수정] 검증된 OpenNMT의 저장소 사용
                 ct2_model_id: str = "OpenNMT/nllb-200-distilled-600M-ct2"):
        """
        모델과 토크나이저를 초기화하고 메모리에 로드합니다.
        """
        print(f"'{model_id}' 토크나이저 로딩을 시작합니다...")

        try:
            # 1. 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            print("토크나이저 로딩 완료.")

            # 2. 기기 설정
            if torch.cuda.is_available():
                self.device = "cuda"
                compute_type = "int8_float16"
                print("CUDA 감지됨. ctranslate2 모델(int8)을 GPU로 로드합니다.")
            else:
                self.device = "cpu"
                compute_type = "int8"
                print("경고: CUDA를 사용할 수 없습니다. CPU로 모델을 로드합니다.")

            # 3. [수정] Hugging Face Hub에서 모델 파일을 명시적으로 다운로드
            print(f"'{ct2_model_id}' 모델 파일 다운로드를 시작합니다 (Hugging Face Hub)...")
            local_model_path = snapshot_download(
                repo_id=ct2_model_id,
                # [수정] 401 인증 오류를 피하기 위해 익명 다운로드 강제
                use_auth_token=False
            )
            print(f"모델 다운로드 완료. 로컬 경로: {local_model_path}")

            # 4. 다운로드한 '로컬 경로'에서 ctranslate2 모델을 로드
            print(f"ctranslate2 모델 로딩을 시작합니다...")
            self.model_ct2 = ctranslate2.Translator(
                local_model_path,
                device=self.device,
                compute_type=compute_type
            )
            # [수정] 성공 메시지
            print("AI 번역 모델 로딩이 완료되었습니다.")

        except Exception as e:
            print(f"모델 또는 토크나이저 로딩 중 치명적인 오류 발생: {e}")
            raise e

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        주어진 텍스트를 소스 언어에서 타겟 언어로 번역합니다.
        """
        if not text or text.strip() == "":
            return "(번역할 텍스트가 없습니다)"

        try:
            self.tokenizer.src_lang = src_lang
            source_tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))

            target_prefix = [tgt_lang]

            results = self.model_ct2.translate_batch(
                [source_tokens],
                target_prefix=[target_prefix]
            )

            target_tokens = results[0].hypotheses[0]

            if target_tokens[0] == tgt_lang:
                target_tokens = target_tokens[1:]

            translated_text = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(target_tokens))

            return translated_text
        except Exception as e:
            print(f"ctranslate2 번역 중 오류 발생: {e}")
            return f"번역 실패: {e}"


# --- 메인 실행 부분 (모듈 테스트용) ---
if __name__ == "__main__":
    """
    이 파일을 직접 실행하면 (python nlp_processor.py) 번역기 모듈을 테스트합니다.
    """

    examples = [
        {"text": "Hello, how are you today?", "src": "eng_Latn", "tgt": "kor_Hang", "tgt_name": "Korean"},
        {"text": "이것은 로컬에서 실행되는 OCR 및 번역 테스트입니다.", "src": "kor_Hang", "tgt": "eng_Latn", "tgt_name": "English"},
        {"text": "Tesseract is an optical character recognition engine for various operating systems.",
         "src": "eng_Latn", "tgt": "jpn_Jpan", "tgt_name": "Japanese"}
    ]

    print("NLLB (ctranslate2) 번역기 모듈 테스트를 시작합니다...")

    try:
        translator = NLLBTranslator()
    except Exception as e:
        print(f"테스트용 번역기 생성 실패. 인터넷 연결 또는 모델 ID를 확인하세요. 오류: {e}")
        exit()

    print("\n--- 번역 테스트 시작 ---")
    for example in examples:
        source_text = example["text"]
        source_lang_code = example["src"]
        target_lang_code = example["tgt"]
        target_lang_name = example["tgt_name"]

        print("-" * 30)
        print(f"원본 ({source_lang_code}):\n{source_text}")

        translated_text = translator.translate(source_text, source_lang_code, target_lang_code)

        print(f"\n번역 결과 ({target_lang_name} - {target_lang_code}):\n{translated_text}")
    print("\n--- 번역 테스트 종료 ---")
