#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mejurix 의료-법률 임베딩 모델 명령행 인터페이스

이 스크립트는 명령행에서 Mejurix 의료-법률 임베딩 모델을 사용할 수 있게 해줍니다.
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

from ..embedder import MejurixMedicalLegalEmbedder


def parse_args():
    """명령행 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description="Mejurix 의료-법률 임베딩 모델을 사용하여 텍스트를 임베딩합니다."
    )
    
    # 입력 옵션
    input_group = parser.add_argument_group("입력 옵션")
    input_mutex = input_group.add_mutually_exclusive_group(required=True)
    input_mutex.add_argument("--text", type=str, help="임베딩할 텍스트")
    input_mutex.add_argument("--file", type=str, help="임베딩할 텍스트 파일 경로")
    input_mutex.add_argument("--compare", type=str, nargs=2, metavar=("TEXT1", "TEXT2"), 
                         help="유사도를 비교할 두 텍스트")
    
    # 출력 옵션
    output_group = parser.add_argument_group("출력 옵션")
    output_group.add_argument("--output", type=str, help="출력 파일 경로")
    output_group.add_argument("--format", type=str, choices=["json", "npy", "txt"], 
                          default="json", help="출력 형식 (기본값: json)")
    
    # 모델 옵션
    model_group = parser.add_argument_group("모델 옵션")
    model_group.add_argument("--model_path", type=str, 
                         help="사용자 정의 모델 경로 (기본값: 내장 모델)")
    
    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()
    
    # 모델 로드
    try:
        if args.model_path:
            embedder = MejurixMedicalLegalEmbedder(model_path=args.model_path)
        else:
            embedder = MejurixMedicalLegalEmbedder()
        print(f"모델을 성공적으로 로드했습니다.", file=sys.stderr)
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 텍스트 임베딩 또는 유사도 계산
    try:
        if args.text:
            # 단일 텍스트 임베딩
            embedding = embedder.encode(args.text)
            result = {"embedding": embedding.tolist()}
            
            # 출력
            if args.output:
                save_output(result, args.output, args.format)
            else:
                print_output(result, args.format)
                
        elif args.file:
            # 파일에서 텍스트 읽기
            try:
                with open(args.file, "r", encoding="utf-8") as f:
                    text = f.read().strip()
            except Exception as e:
                print(f"파일 읽기 오류: {e}", file=sys.stderr)
                sys.exit(1)
                
            # 텍스트 임베딩
            embedding = embedder.encode(text)
            result = {"embedding": embedding.tolist()}
            
            # 출력
            if args.output:
                save_output(result, args.output, args.format)
            else:
                print_output(result, args.format)
                
        elif args.compare:
            # 두 텍스트 간 유사도 계산
            text1, text2 = args.compare
            similarity = embedder.compute_similarity(text1, text2)
            result = {
                "text1": text1,
                "text2": text2,
                "similarity": float(similarity)
            }
            
            # 출력
            if args.output:
                save_output(result, args.output, args.format)
            else:
                if args.format == "txt":
                    print(f"{similarity:.4f}")
                else:
                    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    except Exception as e:
        print(f"처리 중 오류 발생: {e}", file=sys.stderr)
        sys.exit(1)


def save_output(result, output_path, format_type):
    """결과를 파일로 저장합니다."""
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        elif format_type == "npy":
            if "embedding" in result:
                np.save(output_path, np.array(result["embedding"]))
            else:
                np.save(output_path, np.array([result["similarity"]]))
        elif format_type == "txt":
            with open(output_path, "w", encoding="utf-8") as f:
                if "similarity" in result:
                    f.write(f"{result['similarity']:.4f}\n")
                else:
                    f.write(str(result["embedding"]))
        
        print(f"결과가 {output_path}에 저장되었습니다.", file=sys.stderr)
    
    except Exception as e:
        print(f"결과 저장 중 오류 발생: {e}", file=sys.stderr)
        sys.exit(1)


def print_output(result, format_type):
    """결과를 표준 출력으로 출력합니다."""
    if format_type == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif format_type == "txt":
        if "embedding" in result:
            print(str(result["embedding"]))
        else:
            print(f"{result['similarity']:.4f}")
    elif format_type == "npy":
        print("NPY 형식은 파일 출력만 지원합니다. JSON 형식으로 출력합니다.", file=sys.stderr)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main() 