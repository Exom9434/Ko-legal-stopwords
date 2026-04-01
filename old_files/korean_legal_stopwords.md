# Korean Legal Stopwords List

## Overview
This document provides a comprehensive list of stopwords for Korean legal text processing. Stopwords are high-frequency words that carry little semantic meaning for text analysis tasks like topic modeling, information retrieval, and text classification.

## Important Notes for Korean Legal NLP

### Unique Challenges
1. **Agglutinative Language**: Korean is an agglutinative language where particles (조사) attach to nouns
2. **No Standard Dictionary**: Unlike English, Korean lacks a well-established standardized stopword dictionary
3. **Morphological Analysis Required**: Proper stopword removal often requires morphological analysis (형태소 분석) using tools like KoNLPy
4. **Legal Terminology**: Legal documents contain domain-specific terms that should NOT be removed

### Recommendations
- Use morphological analyzer (Mecab, Okt, Komoran, etc.) for proper tokenization
- Apply stopword removal AFTER morphological analysis
- Customize this list based on your specific corpus and use case

---

## 1. General Korean Stopwords

### 1.1 Particles (조사)

#### Case Particles (격조사)
```
이, 가, 께서, 에서        # Subject particles
을, 를                    # Object particles
의                        # Possessive particle
에, 에게, 한테, 께         # Location/Direction particles
에서                      # From (location)
로, 으로                  # Instrumental/Direction
와, 과, 하고, 랑, 이랑     # Conjunction particles
보다                      # Comparison particle
```

#### Auxiliary Particles (보조사)
```
은, 는                    # Topic particles
도                        # Also, too
만                        # Only
부터, 까지                # From, until
마저, 조차                # Even
밖에                      # Only (with negation)
```

### 1.2 Pronouns (대명사)

#### Personal Pronouns
```
나, 내, 저, 제
너, 네
그, 그녀
우리, 저희
너희, 당신
그들
```

#### Demonstrative Pronouns
```
이, 그, 저
이것, 그것, 저것
여기, 거기, 저기
이곳, 그곳, 저곳
```

### 1.3 Conjunctive Adverbs (접속부사)
```
그리고, 및
그러나, 하지만
그런데, 근데
그래서, 따라서
또한, 또
그렇지만
그러므로
그런즉
즉, 곧
또는, 혹은
```

### 1.4 Common Adverbs (부사)
```
매우, 너무, 정말, 참
아주, 무척, 몹시
조금, 좀, 약간
거의, 대략
아마, 어쩌면
항상, 늘, 언제나
가끔, 때때로
```

### 1.5 Auxiliary Verbs & Endings (보조동사/어미)
```
이다, 아니다
되다, 안되다
있다, 없다
하다
수, 것
```

### 1.6 Common Functional Words
```
등, 등등
때문, 까닭
경우
바
대로
듯
지
```

---

## 2. Legal Domain-Specific Considerations

### 2.1 Words to KEEP (Do NOT Remove)
These are important legal terms that should NOT be treated as stopwords:

#### Core Legal Terms (법률 핵심 용어)
```
법, 법률, 법령, 규정, 조례
조, 항, 호, 목                # Article, paragraph, item
권리, 의무
당사자, 피고, 원고
판사, 검사, 변호사
법원, 대법원
소송, 재판, 판결
증거, 증인
계약, 합의
위반, 처벌, 형벌
```

#### Legal Actions & Procedures (법률 행위/절차)
```
제소, 고소, 고발
항소, 상고
공소, 기소
선고, 판시
인정, 부인
주장, 입증
```

#### Rights & Obligations (권리/의무)
```
소유권, 점유권
채권, 채무
손해배상
```

### 2.2 Common Legal Boilerplate (Remove with Caution)

#### Formal Legal Expressions
These phrases appear frequently but may or may not be stopwords depending on your analysis goals:

```
다만
단, 다만
제외하고
경우에
있어서
관하여
대하여
의하여
따라
기타
외에
~에 관한
~에 대한
~에 의한
```

#### Legal Sentence Patterns
```
~할 수 있다
~하여야 한다
~하지 아니한다
~으로 한다
~으로 한다
~을 말한다
~라 함은
```

### 2.3 Time & Number Related (Context-Dependent)
```
년, 월, 일
제, 호
이상, 이하, 미만
초과
전, 후
내
```

---

## 3. Complete Consolidated List

### 3.1 High-Confidence Stopwords
Words that can almost always be safely removed from legal text:

```
그
그녀
나
너
저
저희
너희
이
그것
저것
이것
여기
거기
저기
매우
너무
정말
아주
조금
좀
거의
아마
항상
가끔
```

### 3.2 Medium-Confidence Stopwords
Consider document context before removing:

```
등
등등
때문
까닭
경우
바
대로
및
또한
또
```

### 3.3 Particles (Always Remove After Morphological Analysis)
```
# Subject markers
이, 가, 께서, 에서

# Object markers
을, 를

# Topic markers
은, 는

# Possessive
의

# Location/Direction
에, 에게, 에서, 로, 으로, 부터, 까지

# Conjunction
와, 과, 하고

# Auxiliary
도, 만, 마저, 조차, 밖에, 보다
```

---

## 4. Korean Legal Stopwords JSON Format

```json
{
  "particles": [
    "이", "가", "께서", "을", "를", "의", "에", "에게", "에서", "로", "으로",
    "와", "과", "하고", "은", "는", "도", "만", "부터", "까지", "보다",
    "마저", "조차", "밖에", "한테", "께", "랑", "이랑"
  ],
  "pronouns": [
    "나", "내", "저", "제", "너", "네", "그", "그녀", "우리", "저희", 
    "너희", "당신", "그들", "이", "그", "저", "이것", "그것", "저것",
    "여기", "거기", "저기"
  ],
  "conjunctive_adverbs": [
    "그리고", "및", "그러나", "하지만", "그런데", "그래서", "따라서",
    "또한", "또", "그렇지만", "그러므로", "즉", "곧", "또는", "혹은"
  ],
  "common_adverbs": [
    "매우", "너무", "정말", "참", "아주", "무척", "몹시", "조금", "좀",
    "약간", "거의", "대략", "아마", "항상", "늘", "가끔"
  ],
  "auxiliary_words": [
    "이다", "아니다", "되다", "있다", "없다", "하다", "수", "것"
  ],
  "functional_words": [
    "등", "등등", "때문", "까닭", "경우", "바", "대로", "듯", "지"
  ],
  "legal_boilerplate": [
    "다만", "단", "제외하고", "있어서", "관하여", "대하여", "의하여",
    "따라", "기타", "외에"
  ]
}
```

---

## 5. Implementation Examples

### 5.1 Using KoNLPy (Okt)
```python
from konlpy.tag import Okt

# Initialize
okt = Okt()

# Your legal text
text = "제1조 이 법은 법률 관계를 규정한다."

# Morphological analysis
morphs = okt.pos(text)

# Stopwords
stopwords = ['이', '가', '를', '을', '의', '에', '와', '과']

# Filter stopwords
filtered = [(word, pos) for word, pos in morphs if word not in stopwords]
```

### 5.2 Using Part-of-Speech Tags
```python
from konlpy.tag import Mecab

mecab = Mecab()

text = "피고는 원고에게 손해배상을 하여야 한다."
morphs = mecab.pos(text)

# Remove particles (JKS, JKO, JKB, etc.)
filtered = [(word, pos) for word, pos in morphs 
            if not pos.startswith('J')]  # J = Josa (particle)
```

---

## 6. Best Practices for Korean Legal Text

### Do's
1. ✅ Use morphological analyzer before applying stopwords
2. ✅ Keep domain-specific legal terminology
3. ✅ Customize stopword list based on your corpus
4. ✅ Test stopword removal impact on your specific task
5. ✅ Consider using POS tags instead of word lists

### Don'ts
1. ❌ Don't blindly remove all high-frequency words
2. ❌ Don't remove legal term indicators (제, 조, 항, 호)
3. ❌ Don't use English stopword removal strategies directly
4. ❌ Don't skip morphological analysis for Korean text
5. ❌ Don't forget to handle spacing inconsistencies

---

## 7. Additional Resources

### Korean NLP Libraries
- **KoNLPy**: Korean NLP library with multiple analyzers
- **Mecab-ko**: Fast Korean morphological analyzer
- **Kiwi**: Neural network-based Korean analyzer

### Existing Stopword Lists
- stopwords-iso (GitHub)
- ranks.nl Korean stopwords
- NLTK Korean extension

### Legal Corpus Resources
- Korean legislation database (law.go.kr)
- Court decisions database
- Legal document corpora from KLRI

---

## 8. Frequency-Based Approach

Consider augmenting this list with corpus-specific high-frequency words:

```python
from collections import Counter

def get_high_frequency_words(corpus, threshold=0.5):
    """
    Get words that appear in more than threshold% of documents
    These might be good stopword candidates
    """
    # Count document frequency
    doc_freq = Counter()
    for doc in corpus:
        unique_words = set(doc.split())
        doc_freq.update(unique_words)
    
    # Find high-frequency words
    total_docs = len(corpus)
    stopword_candidates = [
        word for word, freq in doc_freq.items()
        if freq / total_docs > threshold
    ]
    
    return stopword_candidates
```

---

## Version History
- **v1.0** (2026-03): Initial compilation for Korean legal text processing

## License
This stopword list is provided for research and commercial use.

## Contributing
Feel free to customize this list based on your specific legal domain (civil law, criminal law, corporate law, etc.)
