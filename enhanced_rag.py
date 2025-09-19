#!/usr/bin/env python3
"""
향상된 RAG 시스템 - 답변 품질 개선
TF-IDF + 컨텍스트 인식 + 고급 답변 생성
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

class EnhancedTourismRAG:
    def __init__(self, data_folder="./data", similarity_threshold=0.05):
        self.data_folder = data_folder
        self.docs = []
        self.metainfo = []
        
        # 향상된 TF-IDF 설정
        self.vectorizer = TfidfVectorizer(
            max_features=3000,  # 더 많은 특성
            ngram_range=(1, 3),  # 3-gram까지 확장
            lowercase=True,
            min_df=1,
            max_df=0.9,
            token_pattern=r'(?u)\b\w\w+\b|[가-힣]+',
            stop_words=None
        )
        
        self.doc_vectors = None
        self.similarity_threshold = similarity_threshold
        self.is_initialized = False
        
        # 향상된 답변 생성 시스템
        self.setup_enhanced_nlg()
        
    def setup_enhanced_nlg(self):
        """향상된 자연어 생성 시스템 설정"""
        
        # 다양한 답변 시작 패턴
        self.answer_patterns = {
            'method': [
                "관련 연구 방법론에 따르면,",
                "제시된 자료의 방법론을 살펴보면,",
                "연구에서 사용된 방식은"
            ],
            'result': [
                "연구 결과에 의하면,",
                "분석 결과 나타난 바에 따르면,",
                "조사 데이터를 보면"
            ],
            'general': [
                "관련 자료를 종합하면,",
                "문서 내용을 분석해보면,",
                "제공된 정보에 따르면"
            ]
        }
        
        # 연결 표현
        self.transitions = {
            'addition': ["또한", "그리고", "더불어", "아울러"],
            'detail': ["구체적으로", "세부적으로", "상세히 살펴보면"],
            'contrast': ["한편", "반면", "다른 관점에서"],
            'conclusion': ["결국", "종합하면", "이를 통해"]
        }
        
        # 질문 유형별 키워드
        self.question_types = {
            'how': {
                'keywords': ['어떻게', '방법', '방식', '과정', '절차', '산정'],
                'pattern': 'method'
            },
            'what': {
                'keywords': ['무엇', '뭐', '어떤', '특성', '특징', '내용'],
                'pattern': 'general'  
            },
            'result': {
                'keywords': ['결과', '영향', '효과', '변화', '차이'],
                'pattern': 'result'
            }
        }

    def classify_question_intent(self, question):
        """질문 의도 분류"""
        question_lower = question.lower()
        
        for q_type, info in self.question_types.items():
            for keyword in info['keywords']:
                if keyword in question_lower:
                    return q_type, info['pattern']
        
        return 'general', 'general'

    def clean_korean_text(self, text):
        """한국어 텍스트 전처리 - 개선"""
        if not text:
            return ""
        
        # 기본 정리
        text = re.sub(r'\s+', ' ', text)
        
        # 숫자와 단위 결합
        text = re.sub(r'(\d+)\s*([%명개건천만억원달러톤kg])', r'\1\2', text)
        
        # 특수문자 정리 (한국어, 영문, 숫자, 기본 문장부호만 유지)
        text = re.sub(r'[^\w\s가-힣.,!?%-]', ' ', text)
        
        # 다중 공백 정리
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def initialize(self):
        """시스템 초기화"""
        try:
            logger.info("향상된 RAG 시스템 초기화 시작...")
            self.load_pdfs()
            
            if self.docs:
                self.build_vectors()
                self.is_initialized = True
                logger.info("향상된 RAG 시스템 초기화 완료!")
            else:
                logger.warning("문서를 로드할 수 없습니다.")
                
        except Exception as e:
            logger.error(f"초기화 실패: {e}")
            raise

    def load_pdfs(self):
        """PDF 로드 및 전처리 - 개선된 청크 분할"""
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
                            # 컨텍스트 보존을 위한 더 큰 청크 사용
                            if len(cleaned) > 2000:
                                chunks = self.split_text_contextual(cleaned, 1500)
                                for j, chunk in enumerate(chunks):
                                    self.docs.append(chunk)
                                    self.metainfo.append(f"{pdf} - page {i+1} - section {j+1}")
                            else:
                                self.docs.append(cleaned)
                                self.metainfo.append(f"{pdf} - page {i+1}")
                                
            except Exception as e:
                logger.error(f"PDF {pdf} 처리 실패: {e}")
                
        logger.info(f"총 {len(self.docs)}개 문서 청크 로드 완료")

    def split_text_contextual(self, text, max_length):
        """컨텍스트 보존 텍스트 분할"""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        
        # 단락 구분자로 1차 분할
        sections = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            # 현재 청크에 추가 가능한지 확인
            if len(current_chunk + "\n\n" + section) <= max_length:
                if current_chunk:
                    current_chunk += "\n\n" + section
                else:
                    current_chunk = section
            else:
                # 현재 청크 저장
                if current_chunk:
                    chunks.append(current_chunk)
                
                # 새 청크 시작
                if len(section) <= max_length:
                    current_chunk = section
                else:
                    # 섹션이 너무 길면 문장 단위로 분할
                    sentences = re.split(r'[.!?]\s+', section)
                    temp_chunk = ""
                    
                    for sentence in sentences:
                        if len(temp_chunk + sentence + ". ") <= max_length:
                            temp_chunk += sentence + ". "
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = sentence + ". "
                    
                    current_chunk = temp_chunk.strip() if temp_chunk else ""
        
        # 마지막 청크 추가
        if current_chunk:
            chunks.append(current_chunk)
        
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

    def enhanced_search(self, question, top_k=6):
        """향상된 검색 - 다중 전략"""
        if not self.is_initialized or self.doc_vectors is None:
            return [], []
        
        try:
            # 질문 전처리
            cleaned_question = self.clean_korean_text(question)
            
            # 기본 검색
            query_vector = self.vectorizer.transform([cleaned_question])
            similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
            
            # 키워드 확장 검색
            expanded_query = self.expand_query(question)
            if expanded_query != cleaned_question:
                expanded_vector = self.vectorizer.transform([expanded_query])
                expanded_similarities = cosine_similarity(expanded_vector, self.doc_vectors).flatten()
                
                # 두 검색 결과 결합 (가중평균)
                similarities = 0.7 * similarities + 0.3 * expanded_similarities
            
            # 상위 결과 선택
            top_indices = similarities.argsort()[-top_k*2:][::-1]
            
            retrieved = []
            sources = []
            
            for idx in top_indices:
                if similarities[idx] >= self.similarity_threshold and len(retrieved) < top_k:
                    retrieved.append(self.docs[idx])
                    score = round(similarities[idx], 3)
                    sources.append(f"{self.metainfo[idx]} (관련도 {score})")
            
            return retrieved, sources
            
        except Exception as e:
            logger.error(f"검색 오류: {e}")
            return [], []

    def expand_query(self, question):
        """쿼리 확장 - 동의어 및 관련어 추가"""
        expansions = {
            '온실가스': '온실가스 배출량 GHG 이산화탄소',
            '산정': '산정 계산 측정 평가 산출',
            '방법': '방법 방식 절차 과정 시스템',
            '관광산업': '관광산업 관광업 여행업 숙박업',
            '영향': '영향 효과 변화 결과 차이',
            '특성': '특성 특징 성격 양상 패턴 프로파일',
            '한류': '한류 K-POP K-pop 드라마 영화 콘텐츠',
            '관광객': '관광객 외국인 방문객 여행객 방한',
            '소비': '소비 쇼핑 구매 지출 상품',
            '집단': '집단 그룹 층 대상 계층'
        }
        
        expanded = question
        for key, expansion in expansions.items():
            if key in question:
                expanded += " " + expansion
        
        return expanded

    def extract_contextual_info(self, docs, question, question_type):
        """컨텍스트 기반 정보 추출"""
        if not docs:
            return []
        
        contextual_info = []
        question_keywords = self.extract_question_keywords(question)
        
        for doc in docs:
            # 문단 단위로 처리 (문장보다 큰 단위)
            paragraphs = re.split(r'\n\s*\n|(?<=[.!?])\s{2,}', doc)
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if len(paragraph) < 30 or len(paragraph) > 500:
                    continue
                
                # 컨텍스트 점수 계산
                context_score = self.calculate_context_score(
                    paragraph, question_keywords, question_type
                )
                
                if context_score > 1:  # 매우 관대한 임계값
                    contextual_info.append((paragraph, context_score))
        
        # 점수순 정렬
        contextual_info.sort(key=lambda x: x[1], reverse=True)
        return [info[0] for info in contextual_info[:5]]

    def extract_question_keywords(self, question):
        """질문에서 핵심 키워드 추출"""
        # 불용어
        stopwords = {'은', '는', '이', '가', '을', '를', '의', '에', '에서', '으로', '로', 
                    '어떻게', '무엇', '뭐', '어떤', '어디', '언제', '왜', '누구'}
        
        # 한국어 단어 추출
        words = re.findall(r'[가-힣]{2,}', question)
        keywords = [word for word in words if word not in stopwords]
        
        # 영어 단어도 추출
        english_words = re.findall(r'[A-Za-z]{3,}', question)
        keywords.extend(english_words)
        
        return keywords

    def calculate_context_score(self, paragraph, keywords, question_type):
        """컨텍스트 점수 계산 - 개선된 버전"""
        score = 1  # 기본 점수
        
        # 키워드 매칭 (가중치 적용)
        keyword_matches = 0
        for keyword in keywords:
            if keyword in paragraph:
                keyword_matches += 1
                score += 2  # 키워드 매칭 점수
        
        # 관련 키워드 확장 매칭
        related_keywords = {
            '한류': ['K-POP', 'K-pop', '드라마', '영화', '콘텐츠', '블랙핑크', 'BTS', '기생충'],
            '관광객': ['외국인', '방문객', '여행객', '방한', '입국'],
            '특성': ['특징', '성격', '양상', '패턴', '프로파일', '분석'],
            '소비': ['쇼핑', '구매', '지출', '상품', '브랜드']
        }
        
        for main_key, related_list in related_keywords.items():
            if main_key in ' '.join(keywords):
                for related in related_list:
                    if related in paragraph:
                        score += 1
        
        # 질문 유형별 가중치
        if question_type == 'what':
            what_indicators = ['특성', '특징', '비율', '프로파일', '분석', '조사', '결과', '집단']
            for indicator in what_indicators:
                if indicator in paragraph:
                    score += 2
        elif question_type == 'how':
            method_indicators = ['방법', '방식', '과정', '절차', '시스템', '모델', '기준']
            for indicator in method_indicators:
                if indicator in paragraph:
                    score += 2
        
        # 숫자 및 구체적 정보 (더 관대하게)
        if re.search(r'\d+', paragraph):  # 모든 숫자
            score += 1
        if re.search(r'\d+[%명개건천만억원달러톤kg]', paragraph):
            score += 2
        
        # 전문 용어 확장
        professional_terms = ['연구', '분석', '조사', '결과', '데이터', '통계', '집단', 
                             '여성', '남성', '연령', '국적', '방문', '활동', '행동']
        for term in professional_terms:
            if term in paragraph:
                score += 1
        
        # 문장 구조 품질
        if (paragraph.count('.') >= 1 and 
            not paragraph.startswith('표') and 
            not paragraph.startswith('그림') and
            len(paragraph) > 20):
            score += 1
        
        return score

    def generate_enhanced_answer(self, question, contextual_info, question_type, pattern_type):
        """향상된 답변 생성"""
        if not contextual_info:
            return "질문과 관련된 구체적인 정보를 문서에서 찾을 수 없습니다. 다른 방식으로 질문해보시겠어요?"
        
        # 핵심 정보 선별 (최대 3개)
        key_info = contextual_info[:3]
        
        # 정보를 문장 단위로 정제하고 구조화
        formatted_sentences = []
        
        for i, info in enumerate(key_info):
            # 정보를 문장으로 분리하고 정제
            sentences = self.format_to_sentences(info)
            formatted_sentences.extend(sentences)
            
            # 최대 4-5개 문장으로 제한
            if len(formatted_sentences) >= 5:
                formatted_sentences = formatted_sentences[:5]
                break
        
        # 답변 구성
        if formatted_sentences:
            # 첫 번째 문장은 자연스러운 시작으로
            if formatted_sentences[0].startswith(('분석에 활용한', '제공된 정보에 따르면', '연구 결과')):
                answer_parts = [formatted_sentences[0]]
            else:
                starter = random.choice(self.answer_patterns[pattern_type])
                answer_parts = [f"{starter} {formatted_sentences[0]}"]
            
            # 나머지 문장들 추가
            for sentence in formatted_sentences[1:]:
                if sentence.strip():
                    answer_parts.append(sentence)
        else:
            return "질문과 관련된 구체적인 정보를 문서에서 찾을 수 없습니다."
        
        # 문단 구분과 함께 최종 답변 구성
        formatted_answer = self.format_final_answer(answer_parts)
        
        return formatted_answer

    def format_to_sentences(self, text):
        """텍스트를 완성된 문장들로 변환"""
        # 불필요한 부분 제거
        text = re.sub(r'^[0-9\s\-\.]+', '', text)  # 앞쪽 번호 제거
        text = re.sub(r'\s+', ' ', text)  # 공백 정리
        text = text.strip()
        
        if not text:
            return []
        
        # 긴 문장을 적절히 나누기 위한 전처리
        # 대시나 콤마로 구분된 부분을 문장으로 분리
        text = re.sub(r'\s*-\s*', '. ', text)  # 대시를 마침표로
        text = re.sub(r'([^,]{50,}),\s*([가-힣])', r'\1. \2', text)  # 긴 문장의 콤마를 마침표로
        
        # 문장 분리 (마침표, 느낌표, 물음표 기준)
        sentences = re.split(r'[.!?]', text)
        formatted_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 15:  # 너무 짧은 문장 제외
                continue
            
            # 문장 시작 정리
            sentence = re.sub(r'^[-\s]+', '', sentence)
            
            # 너무 긴 문장 분리 (100자 이상)
            if len(sentence) > 100:
                # 자연스러운 분리점 찾기
                split_points = []
                for i, char in enumerate(sentence):
                    if char in ['는', '을', '를', '에', '로', '고', '며'] and i > 40 and i < len(sentence) - 20:
                        split_points.append(i + 1)
                
                if split_points:
                    # 가장 적절한 분리점 선택 (중간 부근)
                    mid_point = len(sentence) // 2
                    best_split = min(split_points, key=lambda x: abs(x - mid_point))
                    
                    part1 = sentence[:best_split].strip()
                    part2 = sentence[best_split:].strip()
                    
                    if part1 and len(part1) > 15:
                        formatted_sentences.append(part1 + '.')
                    if part2 and len(part2) > 15:
                        sentence = part2
            
            # 완성된 문장으로 만들기
            if sentence and not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            
            if sentence and len(sentence) > 15:
                formatted_sentences.append(sentence)
        
        # 최대 4개 문장으로 제한
        return formatted_sentences[:4]
    
    def format_final_answer(self, answer_parts):
        """최종 답변을 문단 구분과 함께 포맷"""
        if not answer_parts:
            return ""
        
        # 최대 5개 문장으로 제한하여 간결성 유지
        limited_parts = answer_parts[:5]
        
        formatted_paragraphs = []
        current_paragraph = []
        
        for i, sentence in enumerate(limited_parts):
            # 각 문장이 마침표로 끝나는지 확인
            if not sentence.rstrip().endswith(('.', '!', '?')):
                sentence = sentence.rstrip() + '.'
            
            current_paragraph.append(sentence)
            
            # 2개 문장마다 문단 구분 또는 마지막 문장
            if len(current_paragraph) >= 2 or i == len(limited_parts) - 1:
                if current_paragraph:
                    # 문단 내 문장들을 한 줄로 연결
                    paragraph_text = ' '.join(current_paragraph)
                    formatted_paragraphs.append(paragraph_text)
                    current_paragraph = []
        
        # 문단들을 줄바꿈으로 구분하여 결합
        final_answer = '\n\n'.join(formatted_paragraphs)
        
        # 길이 제한 (너무 긴 경우 첫 번째 문단만 사용)
        if len(final_answer) > 400 and len(formatted_paragraphs) > 1:
            final_answer = formatted_paragraphs[0]
            if not final_answer.endswith('.'):
                final_answer += '.'
        
        return final_answer
    
    def clean_answer_text(self, text):
        """답변 텍스트 정제 (레거시 메서드)"""
        # 불필요한 부분 제거
        text = re.sub(r'^[0-9\s\-\.]+', '', text)  # 앞쪽 번호 제거
        text = re.sub(r'\s+', ' ', text)  # 공백 정리
        
        # 문장 경계 정리
        if not text.endswith(('.', '!', '?')):
            # 마지막 완전한 문장까지만 유지
            sentences = re.split(r'[.!?]', text)
            if len(sentences) > 1:
                text = '.'.join(sentences[:-1]) + '.'
            else:
                text += '.'
        
        return text.strip()

    def generate_answer(self, question):
        """최종 답변 생성"""
        if not self.is_initialized:
            return "시스템이 초기화되지 않았습니다.", []
        
        try:
            # 질문 의도 파악
            question_type, pattern_type = self.classify_question_intent(question)
            
            # 향상된 검색
            retrieved_docs, sources = self.enhanced_search(question, top_k=6)
            
            if not retrieved_docs:
                return "관련된 정보를 찾을 수 없습니다. 다른 키워드로 질문해보세요.", []
            
            # 컨텍스트 기반 정보 추출
            contextual_info = self.extract_contextual_info(retrieved_docs, question, question_type)
            
            # 컨텍스트 추출 실패시 폴백: 원본 문서 직접 사용
            if not contextual_info and retrieved_docs:
                logger.info("컨텍스트 추출 실패, 원본 문서 직접 사용")
                contextual_info = self.fallback_extract_info(retrieved_docs, question)
            
            # 향상된 답변 생성
            enhanced_answer = self.generate_enhanced_answer(
                question, contextual_info, question_type, pattern_type
            )
            
            return enhanced_answer, sources[:3]
            
        except Exception as e:
            logger.error(f"답변 생성 오류: {e}")
            return "답변 생성 중 오류가 발생했습니다.", []

    def fallback_extract_info(self, docs, question):
        """폴백: 원본 문서에서 직접 정보 추출"""
        logger.info("폴백 정보 추출 실행")
        fallback_info = []
        
        # 질문 키워드
        keywords = self.extract_question_keywords(question)
        
        for doc in docs[:2]:  # 상위 2개 문서만 사용
            # 문장 단위로 분리
            sentences = re.split(r'[.!?]\s*', doc)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20 or len(sentence) > 400:
                    continue
                
                # 간단한 키워드 매칭
                matches = 0
                for keyword in keywords:
                    if keyword in sentence:
                        matches += 1
                
                # 한류 특화 키워드
                hallyu_keywords = ['한류', '적극', '집단', '특성', '여성', '20대', 'K-POP', '드라마']
                for hk in hallyu_keywords:
                    if hk in sentence:
                        matches += 1
                
                if matches >= 1:  # 1개 이상 키워드 매칭시 선택 (더 관대하게)
                    fallback_info.append(sentence)
                    
                if len(fallback_info) >= 4:
                    break
        
        return fallback_info[:3]

    def health_check(self):
        """시스템 상태"""
        return {
            "initialized": self.is_initialized,
            "documents_loaded": len(self.docs),
            "model_loaded": True,
            "model_name": "enhanced-contextual-rag",
            "vector_db_ready": self.doc_vectors is not None,
            "version": "enhanced_contextual_rag_system",
            "features": "contextual_search+enhanced_nlg+multi_strategy"
        }