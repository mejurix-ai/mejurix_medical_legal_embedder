#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
명령줄 인터페이스 모듈
"""

import sys
import argparse
import logging
import os
import numpy as np
from .embedder import MejurixMedicalLegalEmbedder
from .utils.preprocessing import preprocess_for_embedding

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def encode_text(args):
    """
    텍스트 인코딩 명령 처리
    
    Args:
        args: 명령줄 인자
    """
    # 모델 로드
    try:
        embedder = MejurixMedicalLegalEmbedder(args.model_path)
    except Exception as e:
        logger.error(f"모델 로드 중 오류 발생: {str(e)}")
        if args.download:
            logger.info("모델 다운로드를 시도합니다...")
            try:
                model_path = MejurixMedicalLegalEmbedder.download_model()
                embedder = MejurixMedicalLegalEmbedder(model_path)
                logger.info(f"모델이 다운로드되고 로드되었습니다: {model_path}")
            except Exception as download_error:
                logger.error(f"모델 다운로드 중 오류 발생: {str(download_error)}")
                sys.exit(1)
        else:
            logger.error("모델을 로드하지 못했습니다. --download 옵션을 사용하여 모델을 다운로드하세요.")
            sys.exit(1)
    
    # 텍스트 읽기
    if args.text:
        texts = [args.text]
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                if args.line_by_line:
                    texts = [line.strip() for line in f if line.strip()]
                else:
                    texts = [f.read()]
        except Exception as e:
            logger.error(f"파일 읽기 중 오류 발생: {str(e)}")
            sys.exit(1)
    else:
        logger.error("텍스트 또는 파일을 지정해야 합니다.")
        sys.exit(1)
    
    # 전처리
    if args.preprocess:
        entity_types = [args.entity_type] * len(texts) if args.entity_type else None
        texts = [preprocess_for_embedding(text, args.entity_type) for text in texts]
    
    # 임베딩 생성
    try:
        entity_type_ids = None
        if args.entity_type_id is not None:
            entity_type_ids = [args.entity_type_id] * len(texts)
        
        embeddings = embedder.encode(texts, entity_type_ids, batch_size=args.batch_size)
        
        # 결과 저장
        if args.output:
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            if args.format == 'npy':
                np.save(args.output, embeddings)
            elif args.format == 'txt':
                np.savetxt(args.output, embeddings, delimiter=',')
            logger.info(f"임베딩이 {args.output}에 저장되었습니다.")
        else:
            for i, embedding in enumerate(embeddings):
                print(f"임베딩 {i+1}/{len(embeddings)}: {embedding.shape}")
                if args.verbose:
                    print(embedding)
    
    except Exception as e:
        logger.error(f"임베딩 생성 중 오류 발생: {str(e)}")
        sys.exit(1)

def calculate_similarity(args):
    """
    유사도 계산 명령 처리
    
    Args:
        args: 명령줄 인자
    """
    # 모델 로드
    try:
        embedder = MejurixMedicalLegalEmbedder(args.model_path)
    except Exception as e:
        logger.error(f"모델 로드 중 오류 발생: {str(e)}")
        if args.download:
            logger.info("모델 다운로드를 시도합니다...")
            try:
                model_path = MejurixMedicalLegalEmbedder.download_model()
                embedder = MejurixMedicalLegalEmbedder(model_path)
                logger.info(f"모델이 다운로드되고 로드되었습니다: {model_path}")
            except Exception as download_error:
                logger.error(f"모델 다운로드 중 오류 발생: {str(download_error)}")
                sys.exit(1)
        else:
            logger.error("모델을 로드하지 못했습니다. --download 옵션을 사용하여 모델을 다운로드하세요.")
            sys.exit(1)
    
    # 텍스트 읽기 및 처리
    text1 = args.text1
    text2 = args.text2
    
    if args.preprocess:
        text1 = preprocess_for_embedding(text1, args.entity_type1)
        text2 = preprocess_for_embedding(text2, args.entity_type2)
    
    # 유사도 계산
    try:
        similarity = embedder.compute_similarity(
            text1, 
            text2, 
            args.entity_type_id1, 
            args.entity_type_id2
        )
        
        print(f"유사도: {similarity:.4f}")
        
        # 임계값 기반 판단
        if args.threshold is not None:
            if similarity >= args.threshold:
                print(f"유사하다고 판단합니다 (임계값: {args.threshold})")
            else:
                print(f"유사하지 않다고 판단합니다 (임계값: {args.threshold})")
    
    except Exception as e:
        logger.error(f"유사도 계산 중 오류 발생: {str(e)}")
        sys.exit(1)

def download_model_cmd(args):
    """
    모델 다운로드 명령 처리
    
    Args:
        args: 명령줄 인자
    """
    try:
        model_path = MejurixMedicalLegalEmbedder.download_model(args.target_dir)
        logger.info(f"모델이 {model_path}에 성공적으로 다운로드되었습니다.")
    except Exception as e:
        logger.error(f"모델 다운로드 중 오류 발생: {str(e)}")
        sys.exit(1)

def main():
    """
    메인 함수 - 명령줄 인자 파싱 및 처리
    """
    # 메인 파서
    parser = argparse.ArgumentParser(
        description="Mejurix 의료-법률 임베딩 모델 CLI",
        epilog="예시: medicallegal-embedder encode --text '환자는 요추 3번 골절 진단을 받았습니다'"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='명령')
    
    # 인코딩 명령
    encode_parser = subparsers.add_parser('encode', help='텍스트를 벡터로 인코딩')
    encode_parser.add_argument('--text', type=str, help='인코딩할 텍스트')
    encode_parser.add_argument('--file', type=str, help='인코딩할 텍스트 파일')
    encode_parser.add_argument('--line-by-line', action='store_true', help='파일을 라인별로 처리')
    encode_parser.add_argument('--model-path', type=str, help='모델 경로')
    encode_parser.add_argument('--output', type=str, help='출력 파일 경로')
    encode_parser.add_argument('--format', type=str, choices=['npy', 'txt'], default='npy', help='출력 형식')
    encode_parser.add_argument('--batch-size', type=int, default=8, help='배치 크기')
    encode_parser.add_argument('--preprocess', action='store_true', help='텍스트 전처리 수행')
    encode_parser.add_argument('--entity-type', type=str, choices=['medical', 'legal'], help='엔티티 유형')
    encode_parser.add_argument('--entity-type-id', type=int, help='엔티티 타입 ID')
    encode_parser.add_argument('--download', action='store_true', help='모델이 없는 경우 다운로드')
    encode_parser.add_argument('--verbose', action='store_true', help='상세 출력')
    
    # 유사도 계산 명령
    similarity_parser = subparsers.add_parser('similarity', help='두 텍스트 간의 유사도 계산')
    similarity_parser.add_argument('--text1', type=str, required=True, help='첫 번째 텍스트')
    similarity_parser.add_argument('--text2', type=str, required=True, help='두 번째 텍스트')
    similarity_parser.add_argument('--model-path', type=str, help='모델 경로')
    similarity_parser.add_argument('--preprocess', action='store_true', help='텍스트 전처리 수행')
    similarity_parser.add_argument('--entity-type1', type=str, choices=['medical', 'legal'], help='첫 번째 텍스트의 엔티티 유형')
    similarity_parser.add_argument('--entity-type2', type=str, choices=['medical', 'legal'], help='두 번째 텍스트의 엔티티 유형')
    similarity_parser.add_argument('--entity-type-id1', type=int, help='첫 번째 텍스트의 엔티티 타입 ID')
    similarity_parser.add_argument('--entity-type-id2', type=int, help='두 번째 텍스트의 엔티티 타입 ID')
    similarity_parser.add_argument('--threshold', type=float, help='유사도 임계값')
    similarity_parser.add_argument('--download', action='store_true', help='모델이 없는 경우 다운로드')
    
    # 모델 다운로드 명령
    download_parser = subparsers.add_parser('download', help='모델 다운로드')
    download_parser.add_argument('--target-dir', type=str, help='다운로드 대상 디렉토리')
    
    # 인자 파싱
    args = parser.parse_args()
    
    # 명령 처리
    if args.command == 'encode':
        encode_text(args)
    elif args.command == 'similarity':
        calculate_similarity(args)
    elif args.command == 'download':
        download_model_cmd(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main() 