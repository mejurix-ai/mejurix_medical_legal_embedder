"""
Mejurix 의료-법률 임베딩 모델 구현
"""

import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
import logging

# 로거 설정
logger = logging.getLogger(__name__)

class MejurixMedicalLegalEmbedder:
    """NER 기반 의료-법률 임베딩 모델
    
    이 모델은 의료 및 법률 텍스트를 의미 있는 벡터 공간으로 임베딩합니다.
    NER(개체명 인식) 정보를 활용하여 임베딩의 품질을 향상시킵니다.
    
    Attributes:
        model_path (str): 모델 가중치와 설정이 저장된 경로
        config (dict): 모델 설정
        tokenizer: Hugging Face 토크나이저
        bert_model: 기본 BERT 모델
        projection_dim (int): 임베딩 투영 차원
        dropout_rate (float): 드롭아웃 비율
    """
    
    def __init__(self, model_path=None):
        """
        NER 기반 의료-법률 임베딩 모델 초기화
        
        Args:
            model_path (str, optional): 모델 경로. None인 경우 패키지 내장 모델 사용
        """
        if model_path is None:
            # 패키지에 포함된 기본 모델 경로
            model_path = os.path.join(os.path.dirname(__file__), 'pretrained/model')
            
            # 모델이 존재하지 않는 경우 안내 메시지
            if not os.path.exists(model_path):
                logger.warning(
                    "내장 모델을 찾을 수 없습니다. 'download_model()' 메서드를 사용하여 모델을 다운로드하세요."
                )
                model_path = None
        
        if model_path:
            # 모델 경로가 존재하는 경우 모델 로드
            self._load_model(model_path)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.bert_model.to(self.device)
            self.projection.to(self.device)
            self.entity_type_embedding.to(self.device)
            self.entity_type_projection.to(self.device)
            
            logger.info(f"모델을 성공적으로 로드했습니다. 장치: {self.device}")
        else:
            self.is_loaded = False
            logger.warning("모델이 로드되지 않았습니다.")
    
    def _load_model(self, model_path):
        """
        지정된 경로에서 모델 로드
        
        Args:
            model_path (str): 모델 디렉토리 경로
        """
        self.config_path = os.path.join(model_path, 'embedder_config.json')
        self.state_path = os.path.join(model_path, 'embedder_state.pt')
        self.bert_path = os.path.join(model_path, 'bert_model')
        
        # 설정 로드
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            # 토크나이저 및 BERT 모델 로드
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
            self.bert_model = AutoModel.from_pretrained(self.bert_path)
            
            # 임베더 상태 로드
            self.projection_dim = self.config.get('projection_dim', 256)
            self.dropout_rate = self.config.get('dropout_rate', 0.5)
            
            # 모델 구조 초기화
            self._init_model_structure()
            
            # 학습된 가중치 로드
            self._load_weights()
            
            # 평가 모드로 설정
            self.eval()
            
            self.is_loaded = True
            logger.info(f"모델 로드 완료: {model_path}")
            
        except Exception as e:
            self.is_loaded = False
            logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            raise ValueError(f"모델 로드 실패: {str(e)}")
    
    def _init_model_structure(self):
        """모델 구조 초기화"""
        # BERT 출력 차원
        self.bert_dim = self.bert_model.config.hidden_size
        
        # 투영 레이어
        self.projection = torch.nn.Linear(self.bert_dim, self.projection_dim)
        
        # 엔티티 타입 임베딩
        self.entity_type_embedding = torch.nn.Embedding(10, 64)
        self.entity_type_projection = torch.nn.Linear(64, self.projection_dim)
        
        # 드롭아웃
        self.dropout = torch.nn.Dropout(self.dropout_rate)
    
    def _load_weights(self):
        """학습된 가중치 로드"""
        state_dict = torch.load(self.state_path, map_location='cpu')
        
        # 투영 레이어 가중치 로드
        self.projection.weight.data = state_dict['projection.weight']
        self.projection.bias.data = state_dict['projection.bias']
        
        # 엔티티 타입 임베딩 가중치 로드
        self.entity_type_embedding.weight.data = state_dict['entity_type_embedding.weight']
        self.entity_type_projection.weight.data = state_dict['entity_type_projection.weight']
        self.entity_type_projection.bias.data = state_dict['entity_type_projection.bias']
    
    def eval(self):
        """모델을 평가 모드로 설정"""
        self.bert_model.eval()
        self.projection.eval()
        self.entity_type_embedding.eval()
        self.entity_type_projection.eval()
    
    @classmethod
    def download_model(cls, target_dir=None):
        """
        기본 모델 다운로드 및 설치
        
        Args:
            target_dir (str, optional): 저장할 디렉토리. None이면 패키지 내부 디렉토리 사용
            
        Returns:
            str: 모델이 설치된 경로
        """
        if target_dir is None:
            target_dir = os.path.join(os.path.dirname(__file__), 'pretrained/model')
        
        os.makedirs(target_dir, exist_ok=True)
        
        # 여기에 모델 파일 다운로드 코드 구현
        # 예: 원격 서버에서 모델 파일 다운로드
        
        # 샘플 코드 (실제 구현 필요)
        source_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                      "models/ner_model_20250329_193235/best_model")
        
        if os.path.exists(source_model_path):
            # 모델 디렉토리가 존재하면 복사
            for item in ['bert_model', 'embedder_config.json', 'embedder_state.pt']:
                source_item = os.path.join(source_model_path, item)
                target_item = os.path.join(target_dir, item)
                
                if os.path.isdir(source_item):
                    # 디렉토리인 경우
                    if not os.path.exists(target_item):
                        os.makedirs(target_item, exist_ok=True)
                    
                    # bert_model 디렉토리 내용 복사
                    for file_name in os.listdir(source_item):
                        source_file = os.path.join(source_item, file_name)
                        target_file = os.path.join(target_item, file_name)
                        
                        if os.path.isfile(source_file):
                            import shutil
                            shutil.copy2(source_file, target_file)
                else:
                    # 파일인 경우
                    import shutil
                    shutil.copy2(source_item, target_item)
            
            logger.info(f"모델이 {target_dir}에 성공적으로 설치되었습니다.")
        else:
            logger.error(f"모델 소스 경로를 찾을 수 없습니다: {source_model_path}")
            raise FileNotFoundError(f"모델 소스 경로가 존재하지 않습니다: {source_model_path}")
        
        return target_dir
    
    def encode(self, texts, entity_types=None, batch_size=8):
        """
        텍스트를 임베딩 벡터로 인코딩
        
        Args:
            texts (str or list): 인코딩할 텍스트 또는 텍스트 리스트
            entity_types (int or list, optional): 엔티티 타입 ID 또는 ID 리스트
            batch_size (int): 배치 처리 크기
        
        Returns:
            numpy.ndarray: 임베딩 벡터
        """
        if not self.is_loaded:
            raise RuntimeError("모델이 로드되지 않았습니다. download_model()을 통해 모델을 설치하세요.")
        
        # 단일 텍스트를 리스트로 변환
        if isinstance(texts, str):
            texts = [texts]
            if entity_types is not None and not isinstance(entity_types, list):
                entity_types = [entity_types]
        
        # 배치 처리를 위한 준비
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # 토큰화
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                    max_length=128, return_tensors="pt")
            
            # 장치 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 추론
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                hidden_states = outputs.last_hidden_state
                
                # CLS 토큰 임베딩
                cls_embeddings = hidden_states[:, 0, :]
                
                # 투영 및 드롭아웃
                projected = self.projection(cls_embeddings)
                
                # 엔티티 타입 정보 추가 (있는 경우)
                if entity_types is not None:
                    batch_entity_types = entity_types[i:i+batch_size]
                    entity_embeds = self.entity_type_embedding(torch.tensor(batch_entity_types, device=self.device))
                    entity_projected = self.entity_type_projection(entity_embeds)
                    
                    # 결합 (덧셈)
                    projected = projected + entity_projected
                
                # L2 정규화
                normalized = torch.nn.functional.normalize(projected, p=2, dim=1)
                
                # CPU로 이동 후 NumPy 변환
                embeddings.append(normalized.cpu().numpy())
        
        # 모든 배치 결합
        if len(embeddings) == 1:
            return embeddings[0]
        return np.vstack(embeddings)
    
    def compute_similarity(self, text1, text2, entity_type1=None, entity_type2=None):
        """
        두 텍스트 간의 코사인 유사도 계산
        
        Args:
            text1 (str): 첫 번째 텍스트
            text2 (str): 두 번째 텍스트
            entity_type1 (int, optional): 첫 번째 텍스트의 엔티티 타입
            entity_type2 (int, optional): 두 번째 텍스트의 엔티티 타입
        
        Returns:
            float: 코사인 유사도 (0~1)
        """
        if not self.is_loaded:
            raise RuntimeError("모델이 로드되지 않았습니다. download_model()을 통해 모델을 설치하세요.")
        
        # 임베딩 계산
        embedding1 = self.encode([text1], [entity_type1] if entity_type1 is not None else None)[0]
        embedding2 = self.encode([text2], [entity_type2] if entity_type2 is not None else None)[0]
        
        # 코사인 유사도 계산 (1 - 코사인 거리)
        similarity = 1 - cosine(embedding1, embedding2)
        
        return similarity
    
    def batch_compute_similarity(self, texts1, texts2, entity_types1=None, entity_types2=None):
        """
        두 텍스트 리스트 간의 코사인 유사도 일괄 계산
        
        Args:
            texts1 (list): 첫 번째 텍스트 리스트
            texts2 (list): 두 번째 텍스트 리스트
            entity_types1 (list, optional): 첫 번째 텍스트 리스트의 엔티티 타입
            entity_types2 (list, optional): 두 번째 텍스트 리스트의 엔티티 타입
            
        Returns:
            numpy.ndarray: 코사인 유사도 행렬
        """
        if not self.is_loaded:
            raise RuntimeError("모델이 로드되지 않았습니다. download_model()을 통해 모델을 설치하세요.")
        
        # 임베딩 계산
        embeddings1 = self.encode(texts1, entity_types1)
        embeddings2 = self.encode(texts2, entity_types2)
        
        # 코사인 유사도 행렬 계산
        similarity_matrix = np.zeros((len(texts1), len(texts2)))
        
        for i in range(len(texts1)):
            for j in range(len(texts2)):
                similarity_matrix[i, j] = 1 - cosine(embeddings1[i], embeddings2[j])
        
        return similarity_matrix
    
    def get_entity_types(self):
        """
        지원되는 엔티티 타입 목록 반환
        
        Returns:
            dict: 엔티티 타입 ID와 이름 매핑
        """
        # 엔티티 타입 정의
        return {
            0: "NONE",
            1: "DISEASE",
            2: "SYMPTOM",
            3: "MEDICATION",
            4: "TREATMENT",
            5: "DIAGNOSIS",
            6: "OUTCOME",
            7: "SEVERITY",
            8: "LEGAL_TERM",
            9: "COMPENSATION"
        } 