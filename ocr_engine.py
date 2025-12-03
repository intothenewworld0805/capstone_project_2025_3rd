import pytesseract
import numpy as np
from PIL import Image
import cv2  # BGR -> RGB 변환을 위해 필수


class OCREngine:
    """
    [띄어쓰기 개선] Tesseract의 띄어쓰기 인식률을 높이기 위해
    설정 옵션(preserve_interword_spaces)을 추가한 버전입니다.
    """

    def __init__(self):
        """
        Tesseract 실행 파일 경로를 지정할 수 있습니다.
        """
        pass  # 시스템 PATH에 Tesseract가 설정되어 있다고 가정

    def extract_text(self, image: np.ndarray, lang: str = 'kor+eng') -> str:
        """
        :param image: 2단계에서 전처리된 이미지 (numpy.ndarray).
        :param lang: Tesseract가 인식할 언어 코드.
        :return: 이미지에서 텍스트 문자열.
        """

        # [수정] 띄어쓰기 인식률 개선을 위한 설정 추가
        # --oem 3: Default engine mode (가장 안정적)
        # --psm 3: Automatic page segmentation (문서 구조 분석)
        # -c preserve_interword_spaces=1: Tesseract가 단어 사이의 공백을 무시하지 않고 '강제로 보존'하도록 명령
        custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'

        try:
            # OpenCV(BGR) 이미지를 PIL(RGB) 이미지로 변환
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Tesseract 실행
            text = pytesseract.image_to_string(
                pil_image,
                lang=lang,
                config=custom_config
            )
            return text
        except pytesseract.TesseractNotFoundError:
            print("=" * 80)
            print("오류: Tesseract가 설치되지 않았거나 시스템 PATH에 등록되지 않았습니다.")
            print("=" * 80)
            raise SystemExit("Tesseract가 없습니다. 서버를 중지합니다.")
        except Exception as e:
            print(f"OCR 처리 중 예상치 못한 오류가 발생했습니다: {e}")
            raise e
