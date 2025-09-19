#!/usr/bin/env python3
"""
스마트 RAG 시스템 - 무료 로컬 자연어 생성
트랜스포머 없이 고급 규칙 기반 자연어 생성
"""

import os
import logging
import re
import random
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

class SmartTourismRAG:
    def __init__(self, data_folder="./data", similarity_threshold=0.1):
        self.data_folder = data_folder
        self.docs = []
        self.metainfo = []
        
        # 한국어 최적화된 TF-IDF 벡터라이저
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 3),
            lowercase=True,
            min_df=1,
            max_df=0.95,
            token_pattern=r'(?u)\b\w\w+\b|[가-힣]+',
            stop_words=None
        )
        
        self.doc_vectors = None
        self.similarity_threshold = similarity_threshold
        self.is_initialized = False
        
        # 자연어 생성을 위한 템플릿과 패턴
        self.setup_nlg_templates()
        
    def setup_nlg_templates(self):
        """자연어 생성을 위한 템플릿 설정"""
        
        # 답변 시작 패턴
        self.answer_starters = [
            "질문하신 내용에 대해 말씀드리면,",
            "관련 정보를 살펴보니,", 
            "자료에 따르면,",
            "조사 결과를 보면,",
            "문서 분석 결과,",
            "수집된 데이터에 의하면,"
        ]
        
        # 연결어
        self.connectors = [
            "또한", "그리고", "더불어", "아울러", "한편", "특히", "무엇보다"
        ]
        
        # 마무리 표현
        self.conclusion_phrases = [
            "종합해보면,", "정리하면,", "요약하자면,", 
            "결론적으로,", "이를 통해 알 수 있듯이,"
        ]
        
        # 질문 유형별 답변 패턴
        self.question_patterns = {
            'what': ['특성', '특징', '개념', '정의'],
            'how': ['방법', '과정', '절차', '방식'],
            'why': ['이유', '원인', '배경', '근거'],
            'where': ['장소', '위치', '지역', '곳'],
            'when': ['시기', '시간', '기간', '때'],
            'who': ['대상', '주체', '사람', '그룹'],
            'comparison': ['차이', '비교', '대비', '구분'],
            'trend': ['변화', '증감', '추이', '경향']
        }

    def classify_question_type(self, question):
        """질문 유형 분류"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['뭐', '무엇', '어떤', '특성', '특징']):
            return 'what'
        elif any(word in question_lower for word in ['어떻게', '방법', '과정']):
            return 'how'  
        elif any(word in question_lower for word in ['왜', '이유', '원인']):
            return 'why'
        elif any(word in question_lower for word in ['어디', '장소', '위치']):
            return 'where'
        elif any(word in question_lower for word in ['언제', '시기', '시간']):
            return 'when'
        elif any(word in question_lower for word in ['누구', '대상', '주체']):
            return 'who'
        elif any(word in question_lower for word in ['차이', '비교', '대비']):
            return 'comparison'
        elif any(word in question_lower for word in ['변화', '증감', '추이', '경향']):
            return 'trend'
        else:
            return 'general'

    def clean_korean_text(self, text):
        """한국어 텍스트 전처리"""
        if not text:
            return ""
        
        # 1. 기본 정리
        text = re.sub(r'\s+', ' ', text)
        
        # 2. 띄어쓰기 문제 해결 (한국어 특화)
        text = re.sub(r'([가-힣])\s+([가-힣])(?=\s|[가-힣]|$)', r'\1\2', text)
        
        # 3. 숫자와 단위 결합
        text = re.sub(r'(\d+)\s*([%명개건천만억원달러])', r'\1\2', text)
        
        # 4. 영어 단어 결합
        text = re.sub(r'([A-Za-z])\s+([A-Za-z])\s+([A-Za-z])', r'\1\2\3', text)
        
        # 5. 불필요한 문자 제거
        text = re.sub(r'[^\w\s가-힣.,!?%-]', ' ', text)
        
        return text.strip()

    def extract_key_facts(self, text_list, question_type):
        """컨텍스트에서 핵심 사실 추출"""
        facts = []
        
        for text in text_list:
            # 문장 단위로 분리
            sentences = re.split(r'[.!?]\s*', text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20 or len(sentence) > 300:
                    continue
                
                # 질문 유형에 따른 관련 문장 점수 계산
                score = 0
                
                # 숫자 정보가 있으면 가산점
                if re.search(r'\d+[%명개건천만억원달러]', sentence):
                    score += 2
                
                # 질문 유형별 키워드 매칭
                if question_type in self.question_patterns:
                    for keyword in self.question_patterns[question_type]:
                        if keyword in sentence:
                            score += 3
                
                # 일반적으로 유용한 키워드
                useful_keywords = ['특성', '특징', '비율', '증가', '감소', '높음', '낮음', 
                                '선호', '경향', '변화', '주요', '중요', '대표적']
                for keyword in useful_keywords:
                    if keyword in sentence:
                        score += 1
                
                if score >= 2:
                    facts.append((sentence, score))
        
        # 점수순으로 정렬하여 상위 팩트 반환
        facts.sort(key=lambda x: x[1], reverse=True)
        return [fact[0] for fact in facts[:5]]

    def generate_natural_answer(self, question, key_facts, question_type):
        """자연스러운 답변 생성"""
        if not key_facts:
            return "죄송합니다. 질문과 관련된 구체적인 정보를 찾을 수 없습니다. 다른 키워드로 질문해보시겠어요?"
        
        # 답변 구성 요소
        starter = random.choice(self.answer_starters)
        connector = random.choice(self.connectors)
        
        # 답변 구조 생성
        answer_parts = [starter]
        
        # 첫 번째 핵심 사실
        if len(key_facts) > 0:
            answer_parts.append(key_facts[0])
        
        # 두 번째 사실이 있으면 연결어와 함께 추가
        if len(key_facts) > 1:
            answer_parts.append(f"{connector} {key_facts[1]}")
        
        # 세 번째 사실이 있으면 추가
        if len(key_facts) > 2:
            second_connector = random.choice(self.connectors)
            answer_parts.append(f"{second_connector} {key_facts[2]}")
        
        # 질문 유형에 따른 마무리
        if question_type == 'trend':
            conclusion = "이러한 추세를 통해 현재 상황을 파악할 수 있습니다."
        elif question_type == 'comparison':
            conclusion = "각각의 차이점과 특성을 고려해보시기 바랍니다."
        elif question_type == 'what':
            conclusion = "이상이 주요 특성과 내용입니다."
        else:
            conclusion = "참고하시기 바랍니다."
        
        # 최종 답변 조합
        answer = " ".join(answer_parts) + " " + conclusion
        
        # 답변 길이 조정
        if len(answer) > 400:
            answer = answer[:400] + "..."
        
        return answer

    def initialize(self):
        """시스템 초기화"""
        try:
            logger.info("스마트 RAG 시스템 초기화 시작...")
            self.load_pdfs()
            
            if self.docs:
                self.build_vectors()
                self.is_initialized = True
                logger.info("스마트 RAG 시스템 초기화 완료!")
            else:
                logger.warning("문서를 로드할 수 없습니다.")
                
        except Exception as e:
            logger.error(f"초기화 실패: {e}")
            raise

    def load_pdfs(self):
        """PDF 로드 및 전처리"""
        logger.info("PDF 파일 로드 시작...")
        
        if not os.path.exists(self.data_folder):
            return
            
        pdf_files = [f for f in os.listdir(self.data_folder) if f.endswith('.pdf')]
        
        for pdf in tqdm(pdf_files, desc="PDF 처리"):
            try:
                path = os.path.join(self.data_folder, pdf)
                reader = PdfReader(path)
                
                for i, page in enumerate(reader.pages):
                    raw_text = page.extract_text()
                    if raw_text and len(raw_text.strip()) > 50:
                        
                        # 한국어 최적화 전처리
                        cleaned = self.clean_korean_text(raw_text)
                        
                        if len(cleaned) > 100:
                            # 적당한 크기로 분할
                            if len(cleaned) > 1000:
                                chunks = self.split_text(cleaned, 800)
                                for j, chunk in enumerate(chunks):
                                    self.docs.append(chunk)
                                    self.metainfo.append(f"{pdf} - page {i+1} - part {j+1}")
                            else:
                                self.docs.append(cleaned)
                                self.metainfo.append(f"{pdf} - page {i+1}")
                                
            except Exception as e:
                logger.error(f"PDF {pdf} 처리 실패: {e}")
                
        logger.info(f"총 {len(self.docs)}개 문서 청크 로드 완료")

    def split_text(self, text, max_length):
        """텍스트를 적절한 크기로 분할"""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        sentences = re.split(r'[.!?]\s*', text)
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk + sentence) <= max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def build_vectors(self):
        """벡터 DB 구축"""
        logger.info("벡터 데이터베이스 구축 중...")
        
        if not self.docs:
            return
        
        try:
            self.doc_vectors = self.vectorizer.fit_transform(self.docs)
            logger.info(f"벡터 DB 구축 완료: {self.doc_vectors.shape}")
            
        except Exception as e:
            logger.error(f"벡터화 실패: {e}")
            raise

    def search(self, question, top_k=4):
        """문서 검색"""
        if not self.is_initialized or self.doc_vectors is None:
            return [], []
        
        try:
            # 질문 전처리
            cleaned_question = self.clean_korean_text(question)
            
            # 벡터화
            query_vector = self.vectorizer.transform([cleaned_question])
            
            # 유사도 계산
            similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
            
            # 상위 결과 선택
            top_indices = similarities.argsort()[-top_k*2:][::-1]
            
            retrieved = []
            sources = []
            
            for idx in top_indices:
                if similarities[idx] >= self.similarity_threshold and len(retrieved) < top_k:
                    retrieved.append(self.docs[idx])
                    score = round(similarities[idx], 3)
                    sources.append(f"{self.metainfo[idx]} (유사도 {score})")
            
            return retrieved, sources
            
        except Exception as e:
            logger.error(f"검색 오류: {e}")
            return [], []

    def generate_answer(self, question):
        """최종 답변 생성"""
        if not self.is_initialized:
            return "시스템이 초기화되지 않았습니다.", []
        
        try:
            # 1. 관련 문서 검색
            retrieved_docs, sources = self.search(question, top_k=4)
            
            if not retrieved_docs:
                return "관련된 정보를 찾을 수 없습니다. 다른 키워드로 질문해보세요.", []
            
            # 2. 질문 유형 분류
            question_type = self.classify_question_type(question)
            
            # 3. 핵심 사실 추출
            key_facts = self.extract_key_facts(retrieved_docs, question_type)
            
            # 4. 자연스러운 답변 생성
            natural_answer = self.generate_natural_answer(question, key_facts, question_type)
            
            return natural_answer, sources[:3]
            
        except Exception as e:
            logger.error(f"답변 생성 오류: {e}")
            return "답변 생성 중 오류가 발생했습니다.", []

    def health_check(self):
        """시스템 상태"""
        return {
            "initialized": self.is_initialized,
            "documents_loaded": len(self.docs),
            "model_loaded": True,  # 규칙 기반 시스템
            "model_name": "smart-nlg-rag",
            "vector_db_ready": self.doc_vectors is not None,
            "version": "smart_natural_language_generation",
            "nlg_templates": len(self.answer_starters)
        }