#!/usr/bin/env python3
"""
수정된 RAG 시스템 - 한국어 처리 개선
"""

import os
import logging
import re
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

class FixedTourismRAG:
    def __init__(self, data_folder="./data", similarity_threshold=0.1):
        self.data_folder = data_folder
        self.docs = []
        self.metainfo = []
        
        # 한국어 최적화된 TF-IDF 벡터라이저
        self.vectorizer = TfidfVectorizer(
            max_features=1500,
            ngram_range=(1, 2),  # 1-gram, 2-gram
            lowercase=True,
            min_df=1,  # 최소 문서 빈도를 1로 설정
            max_df=0.95,
            token_pattern=r'(?u)\b\w\w+\b|[가-힣]+',  # 한국어 + 영어 토큰
            stop_words=None  # 한국어에는 불용어 제거 안함
        )
        
        self.doc_vectors = None
        self.similarity_threshold = similarity_threshold
        self.is_initialized = False

    def clean_korean_text(self, text):
        """한국어 텍스트 전처리"""
        if not text:
            return ""
        
        # 1. 기본 정리
        text = re.sub(r'\s+', ' ', text)
        
        # 2. 띄어쓰기 문제 해결 (한국어 특화)
        # "한 류" -> "한류", "관 광 객" -> "관광객"
        text = re.sub(r'([가-힣])\s+([가-힣])(?=\s|[가-힣]|$)', r'\1\2', text)
        
        # 3. 숫자와 단위 결합
        text = re.sub(r'(\d+)\s*([%명개건천만억원달러])', r'\1\2', text)
        
        # 4. 영어 단어 결합
        text = re.sub(r'([A-Za-z])\s+([A-Za-z])\s+([A-Za-z])', r'\1\2\3', text)
        
        # 5. 불필요한 문자 제거
        text = re.sub(r'[^\w\s가-힣.,!?%-]', ' ', text)
        
        return text.strip()

    def initialize(self):
        """시스템 초기화"""
        try:
            logger.info("수정된 RAG 시스템 초기화 시작...")
            self.load_pdfs()
            
            if self.docs:
                self.build_vectors()
                self.is_initialized = True
                logger.info("수정된 RAG 시스템 초기화 완료!")
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

    def search(self, question, top_k=5):
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

    def generate_structured_answer(self, question, retrieved_docs, sources):
        """구조화된 답변 생성"""
        if not retrieved_docs:
            return "관련된 정보를 찾을 수 없습니다. 다른 키워드로 질문해보세요.", []
        
        # 질문 유형 분석
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['특성', '특징']):
            title = "📊 **주요 특성**"
        elif any(word in question_lower for word in ['소비', '구매', '쇼핑']):
            title = "💰 **소비 패턴**"
        elif any(word in question_lower for word in ['방문', '여행', '활동']):
            title = "🎯 **방문 행동**"
        else:
            title = "📋 **관련 정보**"
        
        # 핵심 정보 추출
        key_points = []
        
        for doc in retrieved_docs[:3]:
            # 의미있는 문장들 추출
            sentences = [s.strip() for s in re.split(r'[.!?]', doc) if s.strip()]
            
            for sentence in sentences:
                if (len(sentence) > 30 and len(sentence) < 150 and
                    any(word in sentence for word in ['특성', '비율', '높음', '낮음', '증가', '감소', '선호']) and
                    not sentence.startswith('표') and
                    not sentence.startswith('그림')):
                    
                    key_points.append(sentence)
                    
                if len(key_points) >= 4:
                    break
                    
            if len(key_points) >= 4:
                break
        
        # 답변 구성
        if key_points:
            answer_parts = [title, ""]
            
            for i, point in enumerate(key_points[:4], 1):
                # 문장 정리
                clean_point = point
                if not clean_point.endswith('.'):
                    clean_point += "."
                
                answer_parts.append(f"**{i}.** {clean_point}")
            
            answer_parts.append("")
            answer_parts.append("💡 *더 자세한 내용은 출처 문서를 참고하세요.*")
            
            return "\n".join(answer_parts), sources[:3]
        else:
            # 폴백: 원문의 일부를 정리해서 반환
            context = " ".join(retrieved_docs[:2])
            return f"{title}\n\n{context[:300]}...", sources[:3]

    def generate_answer(self, question):
        """최종 답변 생성"""
        if not self.is_initialized:
            return "시스템이 초기화되지 않았습니다.", []
        
        try:
            retrieved_docs, sources = self.search(question, top_k=5)
            return self.generate_structured_answer(question, retrieved_docs, sources)
            
        except Exception as e:
            logger.error(f"답변 생성 오류: {e}")
            return "답변 생성 중 오류가 발생했습니다.", []

    def health_check(self):
        """시스템 상태"""
        return {
            "initialized": self.is_initialized,
            "documents_loaded": len(self.docs),
            "model_loaded": self.doc_vectors is not None,
            "vector_db_ready": self.doc_vectors is not None,
            "version": "fixed_korean"
        }