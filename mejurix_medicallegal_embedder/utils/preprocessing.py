"""
텍스트 전처리 유틸리티 함수
"""

import re
import unicodedata
import logging

# 로거 설정
logger = logging.getLogger(__name__)

def normalize_text(text):
    """
    텍스트 정규화 함수
    
    Args:
        text (str): 정규화할 텍스트
        
    Returns:
        str: 정규화된 텍스트
    """
    if text is None:
        return ""
    
    # 유니코드 정규화
    text = unicodedata.normalize('NFKC', text)
    
    # 공백 정규화
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def clean_medical_text(text):
    """
    의학 텍스트 정제 함수
    
    Args:
        text (str): 정제할 의학 텍스트
        
    Returns:
        str: 정제된 텍스트
    """
    if text is None:
        return ""
    
    # 기본 정규화
    text = normalize_text(text)
    
    # 의학 약어 정규화 (예시)
    replacements = {
        r'\bDx\b': 'diagnosis',
        r'\bTx\b': 'treatment',
        r'\bHx\b': 'history',
        r'\bPt\b': 'patient',
        r'\bRx\b': 'prescription',
        r'\bSx\b': 'symptom',
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    
    return text

def clean_legal_text(text):
    """
    법률 텍스트 정제 함수
    
    Args:
        text (str): 정제할 법률 텍스트
        
    Returns:
        str: 정제된 텍스트
    """
    if text is None:
        return ""
    
    # 기본 정규화
    text = normalize_text(text)
    
    # 법률 약어 정규화 (예시)
    replacements = {
        r'\bplf\b': 'plaintiff',
        r'\bdef\b': 'defendant',
        r'\bw/\b': 'with',
        r'\bw/o\b': 'without',
        r'\be\.g\.\b': 'for example',
        r'\bi\.e\.\b': 'that is',
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    
    return text

def extract_medical_entities(text):
    """
    의학 텍스트에서 주요 개체 추출 함수 (기본 구현)
    
    Args:
        text (str): 개체를 추출할 텍스트
        
    Returns:
        dict: 추출된 개체 정보
    """
    entities = {
        'diseases': [],
        'symptoms': [],
        'medications': [],
        'treatments': []
    }
    
    # 여기에 개체 추출 로직을 구현
    # 이는 기본 구현으로, 실제로는 더 정교한 NER 모델이나 규칙 기반 추출이 필요합니다.
    
    return entities

def chunk_text(text, max_length=120, overlap=20):
    """
    긴 텍스트를 처리 가능한 청크로 분할
    
    Args:
        text (str): 분할할 텍스트
        max_length (int): 최대 청크 길이 (단어 기준)
        overlap (int): 청크 간 중첩 단어 수
        
    Returns:
        list: 텍스트 청크 리스트
    """
    if text is None or text.strip() == "":
        return []
    
    words = text.split()
    chunks = []
    
    if len(words) <= max_length:
        return [text]
    
    for i in range(0, len(words), max_length - overlap):
        chunk = ' '.join(words[i:i + max_length])
        chunks.append(chunk)
        
        if i + max_length >= len(words):
            break
    
    return chunks

def preprocess_for_embedding(text, entity_type=None):
    """
    임베딩을 위한 텍스트 전처리
    
    Args:
        text (str): 전처리할 텍스트
        entity_type (str, optional): 텍스트 엔티티 유형 ('medical' 또는 'legal')
        
    Returns:
        str: 전처리된 텍스트
        
    Raises:
        ValueError: entity_type이 유효하지 않은 경우
    """
    if text is None:
        return ""
    
    # 기본 정규화
    text = normalize_text(text)
    
    # 엔티티 유형에 따른 추가 전처리
    if entity_type is None:
        return text
    elif entity_type.lower() == 'medical':
        return clean_medical_text(text)
    elif entity_type.lower() == 'legal':
        return clean_legal_text(text)
    else:
        raise ValueError(f"지원되지 않는 엔티티 유형: {entity_type}. 'medical' 또는 'legal'만 사용 가능합니다.")

def batch_preprocess(texts, entity_types=None):
    """
    텍스트 배치 전처리
    
    Args:
        texts (list): 전처리할 텍스트 리스트
        entity_types (list, optional): 각 텍스트의 엔티티 유형 리스트
        
    Returns:
        list: 전처리된 텍스트 리스트
    """
    processed_texts = []
    
    for i, text in enumerate(texts):
        entity_type = None
        if entity_types is not None and i < len(entity_types):
            entity_type = entity_types[i]
        
        processed_text = preprocess_for_embedding(text, entity_type)
        processed_texts.append(processed_text)
    
    return processed_texts 