# import torch
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
#
#
# class NLPProcessor:
#     """
#     Hugging Face NLLB 모델을 사용하여 고품질 다국어 번역을 수행하는 클래스.
#     4비트 양자화를 적용하여 고사양 소비자 GPU에서도 대형 모델을 실행할 수 있도록 최적화되었습니다.
#     """
#
#     def __init__(self, model_id: str = "facebook/nllb-200-3.3B"):
#         """
#         모델과 토크나이저를 초기화하고 메모리에 로드합니다.
#         최초 실행 시 Hugging Face Hub에서 모델을 다운로드하므로 시간이 걸릴 수 있습니다.
#         """
#         # GPU 사용 가능 여부를 확인하고, 가능하면 GPU(cuda)를, 아니면 CPU를 사용합니다.
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         print(f"'{model_id}' 모델 로딩을 시작합니다. VRAM 최적화를 위해 4비트 양자화를 적용합니다. ('{self.device}' 사용)")
#
#         # 4비트 양자화 설정
#         quantization_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_compute_dtype=torch.bfloat16
#         )
#
#         # 양자화 설정을 적용하여 모델과 토크나이저 로드
#         self.tokenizer = AutoTokenizer.from_pretrained(model_id)
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(
#             model_id,
#             quantization_config=quantization_config,
#             device_map=self.device  # 'auto' 대신 명시적으로 장치 지정
#         )
#         print("번역 모델 로딩이 완료되었습니다.")
#
#         # --- 요약 모델 로드 추가 ---
#         # (이전 4단계 코드에서 이 부분이 누락되었을 수 있으니 추가합니다.)
#         summarization_model_name = "sshleifer/distilbart-cnn-12-6"
#         self.summarizer = pipeline(
#             "summarization",
#             model=summarization_model_name,
#             device=self.device
#         )
#         print("요약 모델 로딩이 완료되었습니다.")
#
#     def summarize(self, text: str) -> str:
#         """
#         입력된 텍스트를 요약합니다.
#         :param text: 요약할 원본 텍스트.
#         :return: 요약된 텍스트.
#         """
#         if not text.strip():
#             return ""
#         # pipeline을 사용하여 요약을 수행하고, 결과에서 요약 텍스트만 추출합니다.
#         summary_list = self.summarizer(text, max_length=150, min_length=30, do_sample=False)
#         return summary_list['summary_text']
#
#     def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
#         """
#         주어진 텍스트를 소스 언어에서 타겟 언어로 번역합니다.
#
#         Args:
#             text (str): 번역할 텍스트.
#             src_lang (str): 소스 언어의 Flores-200 코드 (예: 'eng_Latn').
#             tgt_lang (str): 타겟 언어의 Flores-200 코드 (예: 'kor_Hang').
#
#         Returns:
#             str: 번역된 텍스트.
#         """
#         if not text.strip():
#             return ""
#
#         try:
#             # 토크나이저에 소스 언어 설정
#             self.tokenizer.src_lang = src_lang
#
#             # 텍스트를 토큰화하고 모델이 있는 디바이스(GPU)로 보냄
#             inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
#
#             # 번역된 토큰 ID 생성
#             # forced_bos_token_id를 통해 타겟 언어를 강제 지정
#             translated_tokens = self.model.generate(
#                 **inputs,
#                 forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang],
#                 max_length=512
#             )
#
#             # 토큰을 다시 텍스트로 디코딩
#             result = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
#             return result  # 리스트가 아닌 첫 번째 텍스트 문자열을 반환
#
#         except Exception as e:
#             print(f"번역 중 오류 발생: {e}")
#             return "번역에 실패했습니다."
