# 참고문헌 초안

> 제3장 방법론 본문에 실제 반영한 인용 기준 참고문헌 목록

---

## 본문 반영 참고문헌 목록

### 1. LBOX, 2024

- LBOX. (2024). *kbl* [Data set]. Hugging Face. [https://huggingface.co/datasets/lbox/kbl](https://huggingface.co/datasets/lbox/kbl)

메모:
- KBL 관련 공식 논문보다 데이터셋 카드가 현재로서는 가장 직접적인 출처로 보임
- 본문에서는 `LBOX, 2024` 형태로 우선 인용 가능

본문 예시:
- `평가 데이터로는 LBOX에서 공개한 KBL(Korean Bar Leaderboard) 데이터셋을 활용하였다 (LBOX, 2024).`

### 2. Kudo, 2005

- Kudo, T. (2005). *MeCab: Yet Another Part-of-Speech and Morphological Analyzer*. [https://taku910.github.io/mecab/](https://taku910.github.io/mecab/)
- Kudo, T. (2005). *MeCab: Yet Another Part-of-Speech and Morphological Analyzer*. CiNii Research record. [https://cir.nii.ac.jp/crid/1572543025344508032?lang=en](https://cir.nii.ac.jp/crid/1572543025344508032?lang=en)

메모:
- MeCab은 공식 웹 문서와 서지 레코드가 함께 유통되는 경우가 많음
- 논문 본문에서는 `Kudo, 2005`로 정리 가능

본문 예시:
- `형태소 분석기로는 Mecab을 채택하였다 (Kudo, 2005).`

### 3. Zipf, 1949

- Zipf, G. K. (1949). *Human Behavior and the Principle of Least Effort: An Introduction to Human Ecology*. Addison-Wesley.

메모:
- 고전 인용으로 안정적

본문 예시:
- `Zipf의 법칙에 따르면 자연어 텍스트에서 단어의 출현 빈도는 그 순위에 반비례하는 경향을 보인다 (Zipf, 1949).`

### 4. Shannon, 1948

- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal, 27*(4), 623-656. [https://doi.org/10.1002/j.1538-7305.1948.tb00917.x](https://doi.org/10.1002/j.1538-7305.1948.tb00917.x)

메모:
- 필요하면 27(3), 379-423까지 함께 병기 가능
- 현재 방법론 설명에는 한 항목만으로도 충분

본문 예시:
- `엔트로피는 Shannon의 정보이론에 기반한 분포 균등성 지표로 해석할 수 있다 (Shannon, 1948).`

### 5. Satopaa et al., 2011

- Satopää, V., Albrecht, J., Irwin, D., & Raghavan, B. (2011). Finding a “Kneedle” in a haystack: Detecting knee points in system behavior. In *2011 31st International Conference on Distributed Computing Systems Workshops* (pp. 166-171). IEEE. [https://doi.org/10.1109/ICDCSW.2011.20](https://doi.org/10.1109/ICDCSW.2011.20)

메모:
- Elbow point 자동 탐지 근거로 사용 가능

본문 예시:
- `Elbow point는 Kneedle 알고리즘을 활용하여 자동 탐지하였다 (Satopää et al., 2011).`

---

## 참고문헌 목록 형태 예시

- LBOX. (2024). *kbl* [Data set]. Hugging Face. [https://huggingface.co/datasets/lbox/kbl](https://huggingface.co/datasets/lbox/kbl)
- Kudo, T. (2005). *MeCab: Yet Another Part-of-Speech and Morphological Analyzer*. [https://taku910.github.io/mecab/](https://taku910.github.io/mecab/)
- Zipf, G. K. (1949). *Human Behavior and the Principle of Least Effort: An Introduction to Human Ecology*. Addison-Wesley.
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal, 27*(4), 623-656. [https://doi.org/10.1002/j.1538-7305.1948.tb00917.x](https://doi.org/10.1002/j.1538-7305.1948.tb00917.x)
- Satopää, V., Albrecht, J., Irwin, D., & Raghavan, B. (2011). Finding a “Kneedle” in a haystack: Detecting knee points in system behavior. In *2011 31st International Conference on Distributed Computing Systems Workshops* (pp. 166-171). IEEE. [https://doi.org/10.1109/ICDCSW.2011.20](https://doi.org/10.1109/ICDCSW.2011.20)

---

## 제3장에 넣은 본문 인용 예시

- `평가 데이터로는 LBOX에서 공개한 KBL(Korean Bar Leaderboard) 데이터셋을 활용하였다 (LBOX, 2024).`
- `형태소 분석기로는 Mecab을 채택하였다 (Kudo, 2005).`
- `Zipf의 법칙에 따르면 자연어 텍스트에서 단어의 출현 빈도는 그 순위에 반비례하는 경향을 보인다 (Zipf, 1949).`
- `엔트로피는 Shannon의 정보이론에 기반한 분포 균등성 지표로 해석할 수 있다 (Shannon, 1948).`
- `Elbow point는 Kneedle 알고리즘을 활용하여 자동 탐지하였다 (Satopää et al., 2011).`

---

## 추가 메모

- 현재 제3장 본문에는 `LBOX, 2024`, `Kudo, 2005`, `Zipf, 1949`, `Shannon, 1948`, `Satopää et al., 2011`만 반영함
- KBL은 향후 공식 논문 또는 워크숍 발표 자료가 확인되면 해당 인용으로 교체 가능
- MeCab은 학술논문보다 공식 프로젝트 문서 인용이 함께 쓰이는 경우가 많으므로, 투고 양식에 맞춰 정리 필요
- n-gram 길이 설정 근거는 외부 인용 없이 연구 설계 선택으로 서술해도 무방
