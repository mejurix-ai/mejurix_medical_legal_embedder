"""
Mejurix 의료-법률 임베딩 모델

NER 기반 접근법을 사용하여 의료 및 법률 문서를 위한 텍스트 임베딩을 생성합니다.
"""

from .embedder import MejurixMedicalLegalEmbedder

__version__ = "0.1.0"
__author__ = "Mejurix"
__all__ = ["MejurixMedicalLegalEmbedder"] 