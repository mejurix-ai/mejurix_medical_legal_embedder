"""
텍스트 처리를 위한 유틸리티 함수들

이 모듈은 의료-법률 텍스트 처리를 위한 다양한 유틸리티 함수를 제공합니다.
"""

import re
import unicodedata
from typing import List, Dict, Tuple, Optional


def normalize_text(text: str) -> str:
    """
    텍스트를 정규화합니다.
    
    1. 유니코드 정규화 (NFKC)
    2. 연속된 공백 제거
    3. 앞뒤 공백 제거
    
    Args:
        text (str): 정규화할 텍스트
        
    Returns:
        str: 정규화된 텍스트
    """
    # 유니코드 정규화
    text = unicodedata.normalize('NFKC', text)
    
    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # 앞뒤 공백 제거
    text = text.strip()
    
    return text


def split_into_sentences(text: str) -> List[str]:
    """
    텍스트를 문장 단위로 분리합니다.
    
    Args:
        text (str): 분리할 텍스트
        
    Returns:
        List[str]: 문장 목록
    """
    # 문장 끝 패턴
    pattern = r'(?<![0-9a-zA-Z])[.!?]+(?=\s+[A-Z가-힣]|\s*$|\s*["\')\]}])'
    
    # 문장 분리
    sentences = re.split(pattern, text)
    
    # 빈 문장 제거 및 정규화
    sentences = [normalize_text(sent) for sent in sentences if sent.strip()]
    
    return sentences


def extract_medical_terms(text: str) -> List[str]:
    """
    텍스트에서 의학 용어를 추출합니다.
    
    Args:
        text (str): 처리할 텍스트
        
    Returns:
        List[str]: 추출된 의학 용어 목록
    """
    # 의학 용어 패턴 (예시)
    patterns = [
        r'[a-zA-Z0-9]+ 골절',  # 골절 관련 용어
        r'[a-zA-Z0-9]+ 증후군',  # 증후군 관련 용어
        r'[a-zA-Z0-9]+ 손상',   # 손상 관련 용어
        r'[a-zA-Z0-9]+ 장애',   # 장애 관련 용어
    ]
    
    medical_terms = []
    
    # 각 패턴에 대해 매칭되는 용어 추출
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            medical_terms.append(match.group(0))
    
    return medical_terms


def extract_legal_terms(text: str) -> List[str]:
    """
    텍스트에서 법률 용어를 추출합니다.
    
    Args:
        text (str): 처리할 텍스트
        
    Returns:
        List[str]: 추출된 법률 용어 목록
    """
    # 법률 용어 패턴 (예시)
    patterns = [
        r'배상 청구',          # 배상 청구 관련 용어
        r'손해 배상',          # 손해 배상 관련 용어
        r'장애 등급',          # 장애 등급 관련 용어
        r'보험금 청구',        # 보험금 청구 관련 용어
    ]
    
    legal_terms = []
    
    # 각 패턴에 대해 매칭되는 용어 추출
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            legal_terms.append(match.group(0))
    
    return legal_terms


def find_entity_connections(text: str) -> List[Tuple[str, str, str]]:
    """
    텍스트에서 개체 간의 관계를 추출합니다.
    
    Args:
        text (str): 처리할 텍스트
        
    Returns:
        List[Tuple[str, str, str]]: (개체1, 관계, 개체2) 형태의 관계 목록
    """
    # 의학 및 법률 용어 추출
    medical_terms = extract_medical_terms(text)
    legal_terms = extract_legal_terms(text)
    
    connections = []
    
    # 간단한 구현: 모든 의학 용어와 법률 용어 간의 관계를 "관련됨"으로 연결
    for med_term in medical_terms:
        for legal_term in legal_terms:
            connections.append((med_term, "관련됨", legal_term))
    
    return connections 