#!/usr/bin/env python3
"""
답변 품질 개선 모듈
"""

import re
import logging

logger = logging.getLogger(__name__)

class AnswerEnhancer:
    def __init__(self):
        self.korean_keywords = {
            '한류여행객특성': {
                'patterns': ['한류', '여행객', '특성', '외국인', 'K-pop', '드라마'],
                'template': '한류 여행객의 주요 특성'
            },
            '관광소비패턴': {
                'patterns': ['소비', '쇼핑', '지출', '구매'],
                'template': '관광 소비 패턴'
            },
            '방문행동': {
                'patterns': ['방문', '활동', '체험', '이용'],
                'template': '방문 및 활동 행동'
            }
        }
    
    def enhance_answer(self, question, raw_answer, sources):
        """답변 품질 개선"""
        try:
            # 1. 질문 유형 분석
            answer_type = self.analyze_question_type(question)
            
            # 2. 원본 답변에서 핵심 정보 추출
            key_info = self.extract_key_information(raw_answer, answer_type)
            
            # 3. 구조화된 답변 생성
            enhanced_answer = self.structure_answer(question, key_info, answer_type)
            
            return enhanced_answer, sources
            
        except Exception as e:
            logger.error(f"답변 개선 중 오류: {e}")
            return raw_answer, sources
    
    def analyze_question_type(self, question):
        """질문 유형 분석"""
        question_lower = question.lower()
        
        if any(keyword in question_lower for keyword in ['특성', '특징', '성향']):
            return 'characteristics'
        elif any(keyword in question_lower for keyword in ['소비', '쇼핑', '구매']):
            return 'consumption'
        elif any(keyword in question_lower for keyword in ['방문', '활동', '여행']):
            return 'behavior'
        else:
            return 'general'
    
    def extract_key_information(self, text, answer_type):
        """텍스트에서 핵심 정보 추출"""
        # 문장 분리
        sentences = re.split(r'[.!?]\s+', text)
        
        key_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            
            # 최소 길이 및 의미 있는 문장 필터링
            if (len(sentence) > 30 and len(sentence) < 200 and
                not sentence.startswith('관련 정보를') and
                '목차' not in sentence and
                '연구방법' not in sentence and
                len(re.findall(r'[가-힣]', sentence)) > 10):
                
                key_sentences.append(sentence)
        
        # 답변 유형별 필터링
        if answer_type == 'characteristics':
            # 특성 관련 키워드가 포함된 문장 우선
            filtered = [s for s in key_sentences if any(k in s for k in ['특성', '특징', '비율', '%', '높음', '낮음'])]
            return filtered[:4] if filtered else key_sentences[:4]
        
        elif answer_type == 'consumption':
            # 소비 관련 키워드가 포함된 문장 우선
            filtered = [s for s in key_sentences if any(k in s for k in ['소비', '구매', '쇼핑', '지출', '상품'])]
            return filtered[:4] if filtered else key_sentences[:4]
        
        else:
            return key_sentences[:4]
    
    def structure_answer(self, question, key_info, answer_type):
        """구조화된 답변 생성"""
        if not key_info:
            return "죄송합니다. 관련된 구체적인 정보를 찾을 수 없습니다."
        
        # 질문에 따른 인트로 생성
        if '특성' in question or '특징' in question:
            intro = "📊 **한류 여행객의 주요 특성**"
        elif '소비' in question or '쇼핑' in question:
            intro = "💰 **한류 여행객의 소비 패턴**"
        elif '행동' in question or '활동' in question:
            intro = "🎯 **한류 여행객의 행동 특성**"
        else:
            intro = "📋 **주요 정보**"
        
        # 구조화된 답변 구성
        answer_parts = [intro, ""]
        
        for i, info in enumerate(key_info, 1):
            # 문장 정리
            clean_info = self.clean_sentence(info)
            if clean_info:
                answer_parts.append(f"**{i}.** {clean_info}")
        
        # 추가 정보 안내
        if len(key_info) >= 3:
            answer_parts.append("")
            answer_parts.append("💡 *더 자세한 정보는 출처 문서를 참고하세요.*")
        
        return "\n".join(answer_parts)
    
    def clean_sentence(self, sentence):
        """문장 정리 및 다듬기"""
        if not sentence:
            return ""
        
        # 기본 정리
        sentence = sentence.strip()
        
        # 불완전한 문장 제거
        if (len(sentence) < 20 or 
            sentence.endswith('등이') or
            sentence.endswith('것은') or
            '다음과 같' in sentence):
            return ""
        
        # 문장 끝 정리
        if not sentence.endswith(('.', '다', '음', '함', '됨')):
            sentence += "."
        
        # 숫자와 단위 정리
        sentence = re.sub(r'(\d+)\s*%', r'\1%', sentence)
        sentence = re.sub(r'(\d+)\s*(명|개|건)', r'\1\2', sentence)
        
        return sentence

# 전역 인스턴스 생성
enhancer = AnswerEnhancer()