#!/usr/bin/env python3
"""
의미 기반 RAG 시스템 - Sentence Transformers 활용
고품질 의미 검색과 컨텍스트 기반 답변 생성
"""

import os
import logging
import re
import random
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

logger = logging.getLogger(__name__)

class SemanticTourismRAG:
    def __init__(self, data_folder="./data", similarity_threshold=0.3):
        self.data_folder = data_folder
        self.docs = []
        self.metainfo = []
        
        # Sentence Transformer 모델 (다국어 지원)
        self.model = None
        self.doc_embeddings = None
        
        # TF-IDF 백업 (하이브리드 검색용)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1500,
            ngram_range=(1, 2),
            lowercase=True,
            min_df=1,
            max_df=0.95,
            token_pattern=r'(?u)\b\w\w+\b|[가-힣]+'
        )
        self.tfidf_vectors = None
        
        self.similarity_threshold = similarity_threshold
        self.is_initialized = False
        
        # 자연어 생성을 위한 템플릿
        self.setup_nlg_templates()
        
    def setup_nlg_templates(self):
        """자연어 생성을 위한 템플릿 설정"""
        self.answer_starters = [
            "관련 연구자료에 따르면,",
            "제공된 문서를 분석해보면,", 
            "관련 정보를 종합하면,",
            "연구 결과에 의하면,",
            "자료 분석 결과,"
        ]
        
        self.connectors = [
            "또한", "그리고", "더불어", "아울러", "한편으로는", "추가적으로"
        ]

    def setup_model(self):
        """Sentence Transformer 모델 설정"""
        try:
            logger.info("의미 검색 모델 로드 중...")
            
            # 다국어 지원 모델들 (한국어 포함, 경량 순)
            model_options = [
                "all-MiniLM-L6-v2",  # 경량 기본
                "paraphrase-multilingual-MiniLM-L12-v2",  # 다국어 경량
            ]
            
            for model_name in model_options:
                try:
                    logger.info(f"모델 시도: {model_name}")
                    self.model = SentenceTransformer(model_name)
                    logger.info(f"모델 로드 성공: {model_name}")
                    return True
                except Exception as e:
                    logger.warning(f"모델 {model_name} 로드 실패: {e}")
                    continue
            
            logger.error("모든 Sentence Transformer 모델 로드 실패")
            return False
            
        except Exception as e:
            logger.error(f"모델 설정 실패: {e}")
            return False

    def clean_korean_text(self, text):
        """한국어 텍스트 전처리"""
        if not text:
            return ""
        
        # 기본 정리
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(\d+)\s*([%명개건천만억원달러톤kg])', r'\1\2', text)
        text = re.sub(r'[^\w\s가-힣.,!?%-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def initialize(self):
        """시스템 초기화"""
        try:
            logger.info("의미 기반 RAG 시스템 초기화 시작...")
            
            # 1. 모델 설정
            if not self.setup_model():
                raise Exception("Sentence Transformer 모델 로드 실패")
            
            # 2. 문서 로드
            self.load_pdfs()
            
            # 3. 벡터 DB 구축
            if self.docs:
                self.build_embeddings()
                self.build_tfidf_backup()
                self.is_initialized = True
                logger.info("의미 기반 RAG 시스템 초기화 완료!")
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
                            # 의미 단위로 분할 (더 큰 청크 사용)
                            if len(cleaned) > 1500:
                                chunks = self.split_text_semantic(cleaned, 1200)
                                for j, chunk in enumerate(chunks):
                                    self.docs.append(chunk)
                                    self.metainfo.append(f"{pdf} - page {i+1} - part {j+1}")
                            else:
                                self.docs.append(cleaned)
                                self.metainfo.append(f"{pdf} - page {i+1}")
                                
            except Exception as e:
                logger.error(f"PDF {pdf} 처리 실패: {e}")
                
        logger.info(f"총 {len(self.docs)}개 문서 청크 로드 완료")

    def split_text_semantic(self, text, max_length):
        """의미 단위 기반 텍스트 분할"""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        # 문단 단위로 먼저 분할
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            if len(current_chunk + paragraph) <= max_length:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # 문단이 너무 길면 문장 단위로 분할
                if len(paragraph) > max_length:
                    sentences = re.split(r'[.!?]\s*', paragraph)
                    temp_chunk = ""
                    for sentence in sentences:
                        if len(temp_chunk + sentence) <= max_length:
                            temp_chunk += sentence + ". "
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = sentence + ". "
                    if temp_chunk:
                        current_chunk = temp_chunk
                    else:
                        current_chunk = ""
                else:
                    current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def build_embeddings(self):
        """의미 임베딩 구축"""
        logger.info("의미 임베딩 구축 중...")
        
        if not self.docs or not self.model:
            return
        
        try:
            # 배치 처리로 임베딩 생성
            batch_size = 32
            embeddings = []
            
            for i in tqdm(range(0, len(self.docs), batch_size), desc="임베딩 생성"):
                batch = self.docs[i:i+batch_size]
                batch_embeddings = self.model.encode(
                    batch, 
                    convert_to_tensor=False,
                    show_progress_bar=False
                )
                embeddings.extend(batch_embeddings)
            
            self.doc_embeddings = np.array(embeddings)
            logger.info(f"의미 임베딩 구축 완료: {self.doc_embeddings.shape}")
            
        except Exception as e:
            logger.error(f"임베딩 구축 실패: {e}")
            raise

    def build_tfidf_backup(self):
        """TF-IDF 백업 벡터 구축"""
        logger.info("TF-IDF 백업 시스템 구축 중...")
        
        try:
            self.tfidf_vectors = self.tfidf_vectorizer.fit_transform(self.docs)
            logger.info(f"TF-IDF 백업 구축 완료: {self.tfidf_vectors.shape}")
        except Exception as e:
            logger.error(f"TF-IDF 백업 구축 실패: {e}")

    def semantic_search(self, question, top_k=5):
        """의미 기반 검색"""
        if not self.is_initialized or self.doc_embeddings is None:
            return [], []
        
        try:
            # 질문 임베딩
            question_embedding = self.model.encode([question], convert_to_tensor=False)
            
            # 코사인 유사도 계산
            similarities = cosine_similarity(question_embedding, self.doc_embeddings).flatten()
            
            # 상위 결과 선택
            top_indices = similarities.argsort()[-top_k*2:][::-1]
            
            retrieved = []
            sources = []
            
            for idx in top_indices:
                if similarities[idx] >= self.similarity_threshold and len(retrieved) < top_k:
                    retrieved.append(self.docs[idx])
                    score = round(similarities[idx], 3)
                    sources.append(f"{self.metainfo[idx]} (의미유사도 {score})")
            
            return retrieved, sources
            
        except Exception as e:
            logger.error(f"의미 검색 오류: {e}")
            return [], []

    def hybrid_search(self, question, top_k=5):
        """하이브리드 검색 (의미 + TF-IDF)"""
        # 의미 기반 검색
        semantic_docs, semantic_sources = self.semantic_search(question, top_k//2 + 2)
        
        # TF-IDF 검색 (백업)
        tfidf_docs = []
        tfidf_sources = []
        
        if self.tfidf_vectors is not None:
            try:
                cleaned_question = self.clean_korean_text(question)
                query_vector = self.tfidf_vectorizer.transform([cleaned_question])
                similarities = cosine_similarity(query_vector, self.tfidf_vectors).flatten()
                
                top_indices = similarities.argsort()[-top_k:][::-1]
                
                for idx in top_indices:
                    if similarities[idx] > 0.05:  # 낮은 임계값
                        tfidf_docs.append(self.docs[idx])
                        score = round(similarities[idx], 3)
                        tfidf_sources.append(f"{self.metainfo[idx]} (키워드매칭 {score})")
                        
                        if len(tfidf_docs) >= top_k//2:
                            break
            except Exception as e:
                logger.warning(f"TF-IDF 검색 실패: {e}")
        
        # 결과 통합 (중복 제거)
        combined_docs = semantic_docs[:]
        combined_sources = semantic_sources[:]
        
        for i, doc in enumerate(tfidf_docs):
            if doc not in combined_docs and len(combined_docs) < top_k:
                combined_docs.append(doc)
                combined_sources.append(tfidf_sources[i])
        
        return combined_docs[:top_k], combined_sources[:top_k]

    def extract_relevant_info(self, docs, question):
        """관련 정보 추출 및 정리"""
        if not docs:
            return []
        
        relevant_info = []
        
        # 질문 키워드 추출
        question_keywords = self.extract_keywords_from_question(question)
        
        for doc in docs:
            # 문장 단위로 분리
            sentences = re.split(r'[.!?]\s*', doc)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20 or len(sentence) > 300:
                    continue
                
                # 관련성 점수 계산
                relevance_score = self.calculate_relevance_score(sentence, question_keywords)
                
                if relevance_score > 0:
                    relevant_info.append((sentence, relevance_score))
        
        # 점수순 정렬 후 상위 반환
        relevant_info.sort(key=lambda x: x[1], reverse=True)
        return [info[0] for info in relevant_info[:6]]

    def extract_keywords_from_question(self, question):
        """질문에서 핵심 키워드 추출"""
        # 불용어 제거 및 키워드 추출
        stopwords = {'은', '는', '이', '가', '을', '를', '의', '에', '어떻게', '무엇', '뭐'}
        words = re.findall(r'[가-힣]+', question)
        keywords = [word for word in words if len(word) > 1 and word not in stopwords]
        return keywords

    def calculate_relevance_score(self, sentence, keywords):
        """문장의 관련성 점수 계산"""
        score = 0
        
        # 키워드 매칭
        for keyword in keywords:
            if keyword in sentence:
                score += 2
        
        # 숫자와 단위 정보
        if re.search(r'\d+[%명개건천만억원달러톤kg]', sentence):
            score += 1
        
        # 전문 용어
        professional_terms = ['연구', '분석', '조사', '결과', '방법', '방식', '시스템', '모델', '데이터']
        for term in professional_terms:
            if term in sentence:
                score += 1
        
        return score

    def generate_coherent_answer(self, question, relevant_info):
        """일관성 있는 답변 생성"""
        if not relevant_info:
            return "죄송합니다. 질문과 관련된 구체적인 정보를 문서에서 찾을 수 없습니다."
        
        # 답변 구조 생성
        starter = random.choice(self.answer_starters)
        
        answer_parts = [starter]
        
        # 핵심 정보 3-4개 선택
        key_info = relevant_info[:4]
        
        for i, info in enumerate(key_info):
            if i == 0:
                answer_parts.append(info)
            elif i == len(key_info) - 1:
                connector = random.choice(self.connectors)
                answer_parts.append(f"{connector} {info}")
            else:
                connector = random.choice(self.connectors)  
                answer_parts.append(f"{connector} {info}")
        
        # 최종 조합
        answer = " ".join(answer_parts)
        
        # 길이 조정
        if len(answer) > 500:
            answer = answer[:500] + "..."
        
        return answer

    def generate_answer(self, question):
        """최종 답변 생성"""
        if not self.is_initialized:
            return "시스템이 초기화되지 않았습니다.", []
        
        try:
            # 하이브리드 검색으로 관련 문서 찾기
            retrieved_docs, sources = self.hybrid_search(question, top_k=5)
            
            if not retrieved_docs:
                return "관련된 정보를 찾을 수 없습니다. 다른 키워드로 질문해보세요.", []
            
            # 관련 정보 추출
            relevant_info = self.extract_relevant_info(retrieved_docs, question)
            
            # 일관성 있는 답변 생성
            coherent_answer = self.generate_coherent_answer(question, relevant_info)
            
            return coherent_answer, sources[:3]
            
        except Exception as e:
            logger.error(f"답변 생성 오류: {e}")
            return "답변 생성 중 오류가 발생했습니다.", []

    def health_check(self):
        """시스템 상태"""
        return {
            "initialized": self.is_initialized,
            "documents_loaded": len(self.docs),
            "model_loaded": self.model is not None,
            "model_name": "semantic-rag-multilingual",
            "embedding_ready": self.doc_embeddings is not None,
            "tfidf_backup_ready": self.tfidf_vectors is not None,
            "version": "semantic_based_rag_system"
        }