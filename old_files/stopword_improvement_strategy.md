# Korean Legal Stopword List Improvement Strategy

## 📊 Current Performance Analysis

### Benchmark Results Summary
```
Rank | Method                      | Count | Accuracy | F1-Score | Time
-----|----------------------------|-------|----------|----------|------
1    | General Stopwords          | 600   | 0.8762   | 0.8741   | 3.56s
2    | Baseline (No Stopwords)    | 0     | 0.8730   | 0.8708   | 3.74s
3    | Exp E (Intersection)       | 62    | 0.8699   | 0.8676   | 3.48s ⭐ Fastest
4    | Hybrid (General + E)       | 657   | 0.8678   | 0.8657   | 3.57s
5    | Exp A (Zipf)              | 187   | 0.8604   | 0.8586   | 3.17s
6    | Exp C (MP-VP)             | 221   | 0.8562   | 0.8544   | 3.18s
7    | Exp B (Entropy)           | 519   | 0.8395   | 0.8370   | 3.01s ❌ Worst
```

## 🔍 Problem Diagnosis

### Issue 1: **Over-Removal (Too Aggressive)**
Your corpus-specific methods are removing **discriminative legal terms**.

**Evidence:**
- Exp B (Entropy, 519 words) has WORST performance (0.8395)
- More stopwords ≠ Better performance
- General stopwords (600) > Your methods despite similar count

**Why This Happens:**
```
High-frequency legal terms get flagged as stopwords because:
1. Zipf: "피고", "원고", "법원" appear frequently → marked as stopwords
2. Entropy: Legal boilerplate has low entropy → removed
3. MP-VP: Stable legal phrases → marked as non-discriminative

But these ARE discriminative for case classification!
Example: "피고" frequency differs between 민사 vs 형사 cases
```

### Issue 2: **Domain Mismatch**
General stopwords work better because they're **purely grammatical**.

**General stopwords include:**
- 이, 가, 은, 는, 을, 를 (particles)
- 그, 이, 저 (pronouns)
- 매우, 너무, 조금 (adverbs)

**Your methods incorrectly flag:**
- 법원, 피고, 원고 (legal entities - SHOULD KEEP)
- 증거, 판결, 소송 (legal processes - SHOULD KEEP)
- High-frequency but discriminative terms

### Issue 3: **Insufficient Domain Knowledge Integration**
Your methods are **purely statistical** without legal domain constraints.

---

## 🎯 Improvement Strategies

### Strategy 1: **Add Domain-Aware Filtering (추천!)**

#### Step 1.1: Create Legal Protected Terms List
```python
# These should NEVER be removed
LEGAL_PROTECTED_TERMS = {
    # Parties
    '피고', '원고', '피의자', '피해자', '증인', '검사', '변호사',
    
    # Institutions  
    '법원', '대법원', '고등법원', '지방법원', '검찰', '경찰',
    
    # Legal Documents/Processes
    '판결', '결정', '명령', '소송', '재판', '항소', '상고', '고소', '고발',
    '증거', '증명', '입증', '주장', '인정', '기각', '인용',
    
    # Legal Basis
    '법', '법률', '법령', '규정', '조례', '조', '항', '호', '목',
    '헌법', '민법', '형법', '상법',
    
    # Rights/Obligations
    '권리', '의무', '책임', '손해', '배상', '보상',
    
    # Legal Outcomes
    '유죄', '무죄', '벌금', '징역', '집행유예', '선고', '처벌',
    
    # Case Types (CRITICAL for classification!)
    '민사', '형사', '행정', '가사', '특허', '조세',
    
    # Time/Procedure
    '기간', '기한', '통지', '송달', '공시', '공고'
}

# These high-frequency but low-discriminative terms are OK to remove
LEGAL_BOILERPLATE_OK_TO_REMOVE = {
    '것', '등', '및', '또는', '또', '다만', '다시', '이미', '이때',
    '경우', '때', '바', '대로', '따라', '의하여', '관하여', '대하여',
    '있어서', '대한', '관한', '의한', '위한'
}
```

#### Step 1.2: Apply Domain Filter
```python
def filter_with_domain_knowledge(candidate_stopwords, protected_terms):
    """
    Remove legally important terms from stopword candidates
    """
    filtered = set(candidate_stopwords) - set(protected_terms)
    return filtered

# Apply to your results
set_a_filtered = filter_with_domain_knowledge(set_a, LEGAL_PROTECTED_TERMS)
set_b_filtered = filter_with_domain_knowledge(set_b, LEGAL_PROTECTED_TERMS)
set_c_filtered = filter_with_domain_knowledge(set_c, LEGAL_PROTECTED_TERMS)
```

---

### Strategy 2: **Use Class-Conditional Analysis**

Instead of corpus-wide statistics, analyze **within vs between class variance**.

#### Concept:
```
Good stopword: Similar frequency across ALL case types
Bad stopword: Different frequency between case types (discriminative!)

Example:
- "것" appears 5% in 민사, 5% in 형사, 5% in 행정 → GOOD STOPWORD
- "피고" appears 8% in 민사, 2% in 형사 → BAD STOPWORD (discriminative!)
```

#### Implementation:
```python
def calculate_class_variance(word, class_labels, corpus):
    """
    Calculate frequency variance across classes
    High variance = discriminative = KEEP
    Low variance = non-discriminative = REMOVE
    """
    from collections import defaultdict
    
    class_freqs = defaultdict(list)
    
    for doc, label in zip(corpus, class_labels):
        word_count = doc.split().count(word)
        doc_len = len(doc.split())
        freq = word_count / doc_len if doc_len > 0 else 0
        class_freqs[label].append(freq)
    
    # Calculate mean frequency per class
    class_means = {cls: np.mean(freqs) for cls, freqs in class_freqs.items()}
    
    # Variance across classes
    variance = np.var(list(class_means.values()))
    
    return variance

# Filter: Keep high-variance (discriminative) words
def get_class_aware_stopwords(candidates, corpus, labels, variance_threshold=0.001):
    filtered_stopwords = []
    
    for word in candidates:
        var = calculate_class_variance(word, labels, corpus)
        if var < variance_threshold:  # Low variance = non-discriminative
            filtered_stopwords.append(word)
    
    return set(filtered_stopwords)
```

---

### Strategy 3: **Hybrid Approach with Tuned Threshold**

Your Exp E (intersection, 62 words) performed well (0.8699) and was FASTEST!

#### Analysis:
- Small, high-confidence list
- Likely contains mostly particles/pronouns
- Minimal information loss

#### Improvement:
```python
# Start with intersection (conservative)
core_stopwords = set_a.intersection(set_b).intersection(set_c)

# Add safe additions from general stopwords
general_stopwords_safe = {
    # Pure particles
    '이', '가', '은', '는', '을', '를', '의', '에', '에서', '로', '부터', '까지',
    '와', '과', '도', '만', '조차',
    
    # Pure pronouns  
    '그', '이', '저', '이것', '그것', '저것',
    
    # Safe adverbs (non-legal)
    '매우', '너무', '조금', '아주', '정말'
}

# Combine
improved_stopwords = core_stopwords.union(general_stopwords_safe)

# Remove any protected terms that slipped in
improved_stopwords = improved_stopwords - LEGAL_PROTECTED_TERMS
```

---

### Strategy 4: **Adjust Your Metrics**

#### Problem with Current Metrics:

**Zipf (Frequency):**
- Flags "피고", "법원" because they're frequent
- But these are DISCRIMINATIVE!

**Entropy:**
- Flags stable legal phrases
- But stability ≠ non-informative in legal domain

**MP-VP:**
- Mean position variance might miss class-specific patterns

#### Better Metric: **TF-IDF with Class Weighting**

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def get_class_weighted_tfidf_stopwords(corpus, labels, percentile=5):
    """
    Find words with consistently LOW TF-IDF across ALL classes
    """
    from collections import defaultdict
    
    class_corpus = defaultdict(list)
    for doc, label in zip(corpus, labels):
        class_corpus[label].append(doc)
    
    # Calculate TF-IDF per class
    class_tfidf_scores = {}
    
    for cls, docs in class_corpus.items():
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names_out()
        
        # Average TF-IDF per word in this class
        avg_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()
        class_tfidf_scores[cls] = dict(zip(feature_names, avg_tfidf))
    
    # Find words with low TF-IDF in ALL classes
    all_words = set()
    for scores in class_tfidf_scores.values():
        all_words.update(scores.keys())
    
    stopword_candidates = []
    for word in all_words:
        # Get TF-IDF for this word across all classes
        scores = [class_tfidf_scores[cls].get(word, 0) for cls in class_tfidf_scores]
        max_tfidf = max(scores)
        
        # If max TF-IDF across all classes is low, it's a stopword candidate
        if max_tfidf < np.percentile([max(class_tfidf_scores[cls].get(w, 0) 
                                          for cls in class_tfidf_scores) 
                                      for w in all_words], percentile):
            stopword_candidates.append(word)
    
    return set(stopword_candidates)
```

---

### Strategy 5: **Error Analysis - What Got Removed?**

#### Diagnostic Code:
```python
# Compare what each method removes
general_stopwords = set(['이', '가', '은', '는', ...])  # Load your general list

# What did your method remove that general didn't?
extra_removed_by_exp_b = set_b - general_stopwords

# Sort by impact on classification
# (Re-run classification incrementally removing each word)
```

#### Expected Findings:
You'll likely find words like:
- 사건, 사실, 주장, 인정, 판단 (case-related, but common to all types)
- 피고, 원고, 법원 (discriminative - should NOT remove!)

---

### Strategy 6: **POS-Tag Based Filtering (Most Conservative)**

Only remove grammatical particles, never content words:

```python
from konlpy.tag import Mecab

mecab = Mecab()

def get_grammar_only_stopwords(candidates):
    """
    Only keep particles (Josa) and endings (Eomi)
    """
    safe_stopwords = []
    
    for word in candidates:
        try:
            pos = mecab.pos(word)
            if pos:
                pos_tag = pos[0][1]
                # Only particles, endings, suffixes
                if pos_tag.startswith('J') or pos_tag.startswith('E') or pos_tag.startswith('XS'):
                    safe_stopwords.append(word)
        except:
            pass
    
    return set(safe_stopwords)

# Apply to your candidates
safe_exp_a = get_grammar_only_stopwords(set_a)
safe_exp_b = get_grammar_only_stopwords(set_b)  
safe_exp_c = get_grammar_only_stopwords(set_c)
```

---

## 🚀 Recommended Action Plan

### Phase 1: Quick Fix (1 hour)
1. ✅ Apply Strategy 1 (Domain-Aware Filtering)
2. ✅ Re-run classification with filtered lists
3. ✅ Compare results

### Phase 2: Systematic Improvement (1 day)
4. ✅ Implement Strategy 2 (Class-Conditional Analysis)
5. ✅ Implement Strategy 6 (POS-Tag Filtering)
6. ✅ Create hybrid: `General ∩ (Your Methods + Domain Filter)`

### Phase 3: Advanced (2-3 days)
7. ✅ Error analysis on misclassified cases
8. ✅ Implement Strategy 4 (Class-Weighted TF-IDF)
9. ✅ Manual review of top 100 removed words

---

## 📈 Expected Improvements

### Conservative Estimate:
```
Current Best: 0.8762 (General)
Target:       0.8750-0.8800 (Domain-aware filtered)
```

### Why Modest Gains?
- Your baseline (0.873) is already strong
- Stopword removal is a **preprocessing step**, not a model
- Gains typically 0.5-2% in well-tuned systems

### Real Value:
- ⚡ **Speed**: Your Exp E was fastest (3.48s vs 3.74s baseline)
- 💾 **Model Size**: Smaller vocabulary
- 🎯 **Interpretability**: Cleaner feature space

---

## 🔬 Validation Strategy

### A/B Test Framework:
```python
stopword_variants = {
    'V1_Baseline': None,
    'V2_General': general_stopwords,
    'V3_ExpE_Original': set_e,
    'V4_ExpE_Filtered': filter_with_domain_knowledge(set_e, LEGAL_PROTECTED_TERMS),
    'V5_POS_Only': get_grammar_only_stopwords(set_e),
    'V6_Hybrid': general_stopwords.intersection(
        filter_with_domain_knowledge(set_a.union(set_b).union(set_c), LEGAL_PROTECTED_TERMS)
    )
}

# Run classification for each
results = {}
for name, stopwords in stopword_variants.items():
    acc, f1, time = run_classification_with_stopwords(stopwords)
    results[name] = {'accuracy': acc, 'f1': f1, 'time': time, 'count': len(stopwords) if stopwords else 0}
```

---

## ✅ Success Criteria

Your improved stopwords should:

1. **Performance**: F1 >= 0.8741 (match or beat general)
2. **Efficiency**: Processing time < 3.5s
3. **Domain Validity**: No legal protected terms removed
4. **Linguistic Validity**: >70% particles/functional words (POS analysis)
5. **Stability**: Consistent across different train/test splits (CV)

---

## 🎓 Key Lessons

### What Went Wrong:
❌ Pure frequency-based methods don't understand **domain semantics**
❌ High frequency ≠ non-discriminative in specialized domains
❌ Statistical methods need domain constraints

### What To Do:
✅ Start conservative (grammar only)
✅ Add domain knowledge (protected terms)
✅ Validate on downstream tasks
✅ Prioritize precision over recall in stopword identification

---

## 📚 References for Legal NLP

- Legal IR often keeps more terms than general NLP
- Court case classification benefits from entity preservation
- Consider task-specific stopwords (retrieval vs classification)

Good luck! Start with Phase 1 and let me know the results.
