#!/usr/bin/env python3
"""
개선된 RAG 시스템 - 고품질 답변 생성
"""

import os
import logging
import re
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import requests
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedTourismRAG:
    def __init__(self, data_folder="./data", similarity_threshold=0.1):
        self.data_folder = data_folder
        self.docs = []
        self.metainfo = []
        self.vectorizer = TfidfVectorizer(
            max_features=2000,  # 특징 수 증가
            stop_words='english',
            ngram_range=(1, 3),  # 3-gram까지 확장
            lowercase=True,
            min_df=1,
            max_df=0.95
        )
        self.doc_vectors = None
        self.similarity_threshold = similarity_threshold
        self.is_initialized = False
        
        # 핵심 키워드 사전
        self.keywords_dict = {
            '한류': ['K-pop', '드라마', '영화', '한류콘텐츠', 'BTS', '블랙핑크'],
            '관광': ['방문', '여행', '관광객', '외국인', '관광지', '여행객'],
            '특성': ['특징', '성향', '행동', '패턴', '경향'],
            '소비': ['구매', '지출', '소비패턴', '소비성향'],
            '문화': ['전통문화', '문화체험', '문화콘텐츠'],
        }

    def initialize(self):
        """RAG 시스템 초기화"""
        try:
            logger.info("개선된 RAG 시스템 초기화 시작...")
            self.load_pdfs()
            if self.docs:
                self.build_tfidf_vectors()
                self.is_initialized = True
                logger.info("개선된 RAG 시스템 초기화 완료!")
            else:
                logger.warning("PDF 문서가 없습니다.")
        except Exception as e:
            logger.error(f"RAG 시스템 초기화 실패: {e}")
            raise

    def clean_text(self, text):
        """향상된 텍스트 전처리"""
        # 불필요한 문자 및 공백 정리
        text = re.sub(r'\s+', ' ', text)  # 여러 공백을 하나로
        text = re.sub(r'[^\w\s가-힣.,!?()%-]', ' ', text)  # 특수문자 정리
        text = re.sub(r'\b\d{4}\b', ' ', text)  # 연도 제거
        text = re.sub(r'\b[A-Z]{2,}\b', lambda m: m.group().lower(), text)  # 대문자 정리
        return text.strip()

    def load_pdfs(self):
        """PDF 파일들을 로드하고 텍스트를 추출"""
        logger.info("PDF 로드 및 텍스트 추출 시작")
        
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
            return
            
        pdf_files = [f for f in os.listdir(self.data_folder) if f.endswith('.pdf')]
        
        for pdf in tqdm(pdf_files, desc="PDF 처리 중"):
            try:
                path = os.path.join(self.data_folder, pdf)
                reader = PdfReader(path)
                
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and len(text.strip()) > 50:
                        # 문단별로 분할하여 저장
                        paragraphs = text.split('\n\n')
                        for j, paragraph in enumerate(paragraphs):
                            cleaned_text = self.clean_text(paragraph)
                            if len(cleaned_text) > 100:  # 최소 길이 확보
                                self.docs.append(cleaned_text)
                                self.metainfo.append(f"{pdf} - page {i+1}, para {j+1}")
                                
            except Exception as e:
                logger.error(f"PDF 파일 {pdf} 처리 중 오류: {e}")
                
        logger.info(f"총 {len(self.docs)}개의 문단을 추출했습니다.")

    def build_tfidf_vectors(self):
        """TF-IDF 벡터 생성"""
        logger.info("TF-IDF 벡터 생성 중...")
        if self.docs:
            self.doc_vectors = self.vectorizer.fit_transform(self.docs)
            logger.info("TF-IDF 벡터 생성 완료")

    def query(self, question, top_k=5):
        """향상된 문서 검색"""
        if not self.is_initialized or self.doc_vectors is None:
            return [], []
            
        # 질문 전처리 및 키워드 확장
        expanded_question = self.expand_question(question)
        query_vector = self.vectorizer.transform([expanded_question])
        
        # 코사인 유사도 계산
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        # 상위 k개 문서 선택
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        retrieved = []
        sources = []
        
        for idx in top_indices:
            if similarities[idx] >= self.similarity_threshold:
                retrieved.append(self.docs[idx])
                sim_score = round(similarities[idx], 3)
                sources.append(f"{self.metainfo[idx]} (유사도 {sim_score})")
        
        return retrieved, sources

    def expand_question(self, question):
        """질문 키워드 확장"""
        expanded = question
        
        # 키워드 사전을 활용한 확장
        for key, synonyms in self.keywords_dict.items():
            if key in question:
                expanded += " " + " ".join(synonyms)
        
        return expanded

    def extract_key_information(self, text, question):
        """핵심 정보 추출"""
        # 문장 단위로 분할
        sentences = re.split(r'[.!?]\s+', text)
        
        # 질문과 관련된 문장들 추출
        relevant_sentences = []
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        
        for sentence in sentences:
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            if len(question_words & sentence_words) >= 2 and len(sentence.strip()) > 20:
                relevant_sentences.append(sentence.strip())
        
        return relevant_sentences[:3]  # 상위 3개 문장만

    def structure_answer(self, question, retrieved_docs, sources):
        """구조화된 답변 생성"""
        if not retrieved_docs:
            return "죄송합니다. 관련된 정보를 찾을 수 없습니다.", []
        
        # 핵심 정보 추출
        key_info = []
        for doc in retrieved_docs[:3]:
            extracted = self.extract_key_information(doc, question)
            key_info.extend(extracted)
        
        # 중복 제거 및 정리
        unique_info = list(dict.fromkeys(key_info))[:5]
        
        # 답변 구조화
        if unique_info:
            answer_parts = []
            
            # 주요 내용 요약
            if '특성' in question or '특징' in question:
                answer_parts.append("📊 주요 특성:")
                for i, info in enumerate(unique_info[:3], 1):
                    answer_parts.append(f"  {i}. {info}")
            
            elif '한류' in question:
                answer_parts.append("🎭 한류 관련 정보:")
                for i, info in enumerate(unique_info[:3], 1):
                    answer_parts.append(f"  • {info}")
            
            else:
                answer_parts.append("💡 관련 정보:")
                for i, info in enumerate(unique_info[:3], 1):
                    answer_parts.append(f"  - {info}")
            
            if len(unique_info) > 3:
                answer_parts.append("\n📋 추가 정보가 더 있습니다.")
            
            final_answer = "\n".join(answer_parts)
        else:
            # 폴백: 원본 텍스트의 일부만 정리해서 반환
            context = " ".join(retrieved_docs[:2])
            sentences = re.split(r'[.!?]', context)[:3]
            final_answer = "관련 정보:\n" + "\n".join([f"• {s.strip()}" for s in sentences if s.strip()])
        
        return final_answer, sources[:3]

    def generate_answer(self, question):
        """최종 답변 생성"""
        if not self.is_initialized:
            return "시스템이 아직 초기화되지 않았습니다.", []
            
        try:
            retrieved_docs, sources = self.query(question, top_k=8)
            return self.structure_answer(question, retrieved_docs, sources)
            
        except Exception as e:
            logger.error(f"답변 생성 중 오류: {e}")
            return "답변 생성 중 오류가 발생했습니다.", []

    def health_check(self):
        """시스템 상태 확인"""
        return {
            "initialized": self.is_initialized,
            "documents_loaded": len(self.docs),
            "model_loaded": self.doc_vectors is not None,
            "vector_db_ready": self.doc_vectors is not None
        }