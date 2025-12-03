import cv2
import numpy as np

class ImageProcessor:
    """
    [궁극의 리셋] 깨끗한 원본 이미지를 스캔하는 목표에 맞게,
    이미지를 훼손할 수 있는 모든 복잡한 전처리를 비활성화합니다.
    (UI 체크박스와 상관없이 원본 이미지를 반환합니다)
    """

    def deskew(self, image: np.ndarray) -> (np.ndarray, float):
        """
        [비활성화] 자동 기울기 보정.
        깨끗한 스캔본은 이 기능이 필요 없으며, 스크린샷에는 오히려 방해가 됩니다.
        """
        print("Deskew 기능 비활성화 (pass-through).")
        return image, 0.0 # 원본 이미지와 0도 각도를 반환

    def optimize_resolution(self, image: np.ndarray) -> np.ndarray:
        """
        [비활성화] 저해상도 최적화.
        고해상도 스캔본은 이 기능이 필요 없습니다.
        """
        print("Optimize Resolution 기능 비활성화 (pass-through).")
        return image # 원본 이미지 반환
