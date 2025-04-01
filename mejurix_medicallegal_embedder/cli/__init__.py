"""
Mejurix 의료-법률 임베딩 모델의 명령행 인터페이스

이 패키지는 Mejurix 의료-법률 임베딩 모델을 명령행에서 사용할 수 있게 해주는 도구를 제공합니다.
"""

from .embed import main as embed_main

__all__ = ["embed_main"] 