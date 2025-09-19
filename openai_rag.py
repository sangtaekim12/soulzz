#!/usr/bin/env python3
"""
OpenAI API를 활용한 고품질 RAG 시스템
"""

import os
import logging
from improved_rag import ImprovedTourismRAG
import requests
import json

logger = logging.getLogger(__name__)

class OpenAITourismRAG(ImprovedTourismRAG):
    def __init__(self, data_folder="./data", similarity_threshold=0.1, api_key=None):
        super().__init__(data_folder, similarity_threshold)
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
    def call_openai_api(self, messages, model="gpt-4o-mini"):
        """OpenAI API 호출"""
        if not self.api_key:
            return None
            
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": 800,
            "temperature": 0.3,
            "top_p": 0.9
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"OpenAI API 호출 실패: {e}")
            return None

    def generate_answer_with_openai(self, question):
        """OpenAI API를 사용한 고품질 답변 생성"""
        retrieved_docs, sources = self.query(question, top_k=5)
        
        if not retrieved_docs:
            return "관련된 정보를 찾을 수 없습니다.", []
        
        # 컨텍스트 준비
        context = "\n\n".join(retrieved_docs[:3])
        
        messages = [
            {
                "role": "system",
                "content": """당신은 한국 관광 전문 AI 도우미입니다. 

주어진 자료를 바탕으로 정확하고 유용한 정보를 제공하세요.

답변 원칙:
1. 제공된 자료만을 근거로 답변
2. 자료에 없는 내용은 추측하지 말 것
3. 구조화되고 읽기 쉬운 형태로 작성
4. 핵심 내용을 명확히 전달
5. 한국어로 자연스럽게 답변
6. 가능하면 숫자나 구체적 데이터 포함"""
            },
            {
                "role": "user",
                "content": f"""다음 관광 연구 자료를 바탕으로 질문에 답해주세요:

【자료】
{context}

【질문】
{question}

【답변 요구사항】
- 핵심 내용을 3-5개 포인트로 구조화
- 구체적인 데이터나 수치가 있다면 포함
- 200-400자 내외로 간결하게
- 이해하기 쉽도록 정리"""
            }
        ]
        
        result = self.call_openai_api(messages)
        
        if result and 'choices' in result:
            answer = result['choices'][0]['message']['content']
            return answer, sources[:3]
        else:
            # API 실패 시 폴백
            return self.structure_answer(question, retrieved_docs, sources)

    def generate_answer(self, question):
        """메인 답변 생성 (OpenAI 우선, 폴백 지원)"""
        if not self.is_initialized:
            return "시스템이 아직 초기화되지 않았습니다.", []
        
        try:
            # OpenAI API 사용 시도
            if self.api_key:
                return self.generate_answer_with_openai(question)
            else:
                # 폴백: 개선된 구조화 답변
                return super().generate_answer(question)
                
        except Exception as e:
            logger.error(f"답변 생성 중 오류: {e}")
            return "답변 생성 중 오류가 발생했습니다.", []

# 무료 대안: Hugging Face Inference API
class HuggingFaceRAG(ImprovedTourismRAG):
    def __init__(self, data_folder="./data", similarity_threshold=0.1):
        super().__init__(data_folder, similarity_threshold)
        
    def call_huggingface_api(self, text, model="microsoft/DialoGPT-medium"):
        """Hugging Face 무료 API 호출"""
        # 무료 모델 사용 (제한적)
        try:
            # 간단한 텍스트 요약 시도
            return self.simple_text_processing(text)
        except Exception as e:
            logger.error(f"Hugging Face API 오류: {e}")
            return None
    
    def simple_text_processing(self, text):
        """간단한 텍스트 처리"""
        sentences = text.split('.')
        # 중요해 보이는 문장들 선택 (키워드 기반)
        important_keywords = ['특성', '특징', '한류', '관광객', '여행객', '소비', '방문']
        
        important_sentences = []
        for sentence in sentences:
            if any(keyword in sentence for keyword in important_keywords):
                important_sentences.append(sentence.strip())
        
        return '. '.join(important_sentences[:3])
    
    def generate_answer(self, question):
        """Hugging Face 기반 답변 생성"""
        retrieved_docs, sources = self.query(question, top_k=3)
        
        if not retrieved_docs:
            return "관련된 정보를 찾을 수 없습니다.", []
        
        # 텍스트 처리 및 요약
        context = " ".join(retrieved_docs)
        processed_text = self.simple_text_processing(context)
        
        if processed_text:
            # 구조화된 답변 생성
            lines = processed_text.split('.')
            formatted_answer = "📋 주요 내용:\n\n"
            for i, line in enumerate(lines[:4], 1):
                if line.strip():
                    formatted_answer += f"{i}. {line.strip()}\n"
            
            return formatted_answer, sources[:3]
        else:
            return super().generate_answer(question)