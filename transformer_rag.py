#!/usr/bin/env python3
"""
트랜스포머 기반 RAG 시스템 - 무료 로컬 LLM 모델 사용
"""

import os
import logging
import re
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig
)
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TransformerTourismRAG:
    def __init__(self, data_folder="./data", similarity_threshold=0.1):
        self.data_folder = data_folder
        self.docs = []
        self.metainfo = []
        
        # 한국어 최적화된 TF-IDF 벡터라이저
        self.vectorizer = TfidfVectorizer(
            max_features=1500,
            ngram_range=(1, 2),
            lowercase=True,
            min_df=1,
            max_df=0.95,
            token_pattern=r'(?u)\b\w\w+\b|[가-힣]+',
            stop_words=None
        )
        
        self.doc_vectors = None
        self.similarity_threshold = similarity_threshold
        self.is_initialized = False
        
        # 트랜스포머 모델 관련
        self.model_name = None
        self.tokenizer = None
        self.model = None
        self.generator = None
        
    def setup_model(self):
        """무료 한국어 트랜스포머 모델 설정"""
        try:
            logger.info("트랜스포머 모델 로드 중...")
            
            # 한국어 지원 무료 모델들 (크기 순, 경량화)
            model_options = [
                "microsoft/DialoGPT-small",   # 매우 경량, 대화형
                "distilgpt2",                 # 경량 GPT-2
                "gpt2",                       # 기본 GPT-2
            ]
            
            # GPU 사용 가능 여부 확인
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"사용 디바이스: {device}")
            
            # 첫 번째로 성공하는 모델 사용
            for model_name in model_options:
                try:
                    logger.info(f"모델 시도: {model_name}")
                    
                    if "gpt" in model_name.lower() or "dialogpt" in model_name.lower():
                        # 생성 모델
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                        
                        # 패딩 토큰 설정
                        if self.tokenizer.pad_token is None:
                            self.tokenizer.pad_token = self.tokenizer.eos_token
                        
                        # 메모리 최적화 설정
                        if device == "cuda":
                            quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16
                            )
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                quantization_config=quantization_config,
                                device_map="auto",
                                torch_dtype=torch.float16
                            )
                        else:
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                torch_dtype=torch.float32
                            )
                            self.model.to(device)
                        
                        # 파이프라인 생성
                        self.generator = pipeline(
                            "text-generation",
                            model=self.model,
                            tokenizer=self.tokenizer,
                            device=0 if device == "cuda" else -1,
                            do_sample=True,
                            temperature=0.7,
                            max_length=512,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                        
                        self.model_name = model_name
                        logger.info(f"모델 로드 성공: {model_name}")
                        return True
                        
                except Exception as e:
                    logger.warning(f"모델 {model_name} 로드 실패: {e}")
                    continue
            
            # 모든 모델 실패시 간단한 규칙 기반 폴백
            logger.warning("모든 트랜스포머 모델 로드 실패, 규칙 기반 폴백 사용")
            self.generator = None  # 규칙 기반 사용
            self.model_name = "rule-based-fallback"
            return True
            
        except Exception as e:
            logger.error(f"모델 설정 실패: {e}")
            return False

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

    def initialize(self):
        """시스템 초기화"""
        try:
            logger.info("트랜스포머 RAG 시스템 초기화 시작...")
            
            # 1. 모델 설정
            if not self.setup_model():
                raise Exception("트랜스포머 모델 로드 실패")
            
            # 2. 문서 로드
            self.load_pdfs()
            
            # 3. 벡터 DB 구축
            if self.docs:
                self.build_vectors()
                self.is_initialized = True
                logger.info("트랜스포머 RAG 시스템 초기화 완료!")
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

    def search(self, question, top_k=3):
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

    def generate_llm_answer(self, question, context_docs):
        """트랜스포머 모델을 사용한 자연스러운 답변 생성"""
        if not self.generator:
            # 규칙 기반 폴백
            return self.generate_rule_based_answer(question, context_docs)
        
        try:
            # 컨텍스트 준비
            context = "\n".join(context_docs[:2])  # 상위 2개 문서만 사용
            context = context[:1000]  # 길이 제한
            
            # 프롬프트 구성 (한국어)
            prompt = f"""다음 정보를 바탕으로 질문에 대해 자연스럽고 도움이 되는 답변을 해주세요.

정보:
{context}

질문: {question}

답변:"""
            
            # 모델 추론
            if "gpt" in self.model_name.lower() or "dialogpt" in self.model_name.lower():
                # GPT 계열 모델
                responses = self.generator(
                    prompt,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    num_return_sequences=1
                )
                
                generated_text = responses[0]['generated_text']
                
                # 프롬프트 이후 부분만 추출
                if "답변:" in generated_text:
                    answer = generated_text.split("답변:")[-1].strip()
                else:
                    answer = generated_text[len(prompt):].strip()
                    
            else:
                # 다른 모델들 (BART, BERT 등)
                answer = "트랜스포머 모델을 통해 처리된 답변입니다."
            
            # 답변 후처리
            answer = self.post_process_answer(answer, question)
            
            return answer
            
        except Exception as e:
            logger.error(f"LLM 답변 생성 오류: {e}")
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}"

    def post_process_answer(self, answer, question):
        """답변 후처리"""
        if not answer or len(answer.strip()) < 10:
            return "죄송합니다. 적절한 답변을 생성할 수 없습니다."
        
        # 불필요한 반복 제거
        lines = answer.split('\n')
        unique_lines = []
        for line in lines:
            line = line.strip()
            if line and line not in unique_lines and len(line) > 5:
                unique_lines.append(line)
        
        # 너무 긴 답변 자르기
        processed = ' '.join(unique_lines)
        if len(processed) > 500:
            processed = processed[:500] + "..."
        
        return processed

    def generate_rule_based_answer(self, question, context_docs):
        """트랜스포머 모델이 실패한 경우 규칙 기반 답변 생성"""
        if not context_docs:
            return "관련된 정보를 찾을 수 없습니다."
        
        # 컨텍스트에서 핵심 문장 추출
        key_sentences = []
        for doc in context_docs[:2]:
            sentences = [s.strip() for s in re.split(r'[.!?]', doc) if s.strip()]
            for sentence in sentences:
                if (len(sentence) > 20 and len(sentence) < 200 and
                    any(word in sentence for word in ['특성', '비율', '높음', '낮음', '증가', '감소', '선호', '관광', '여행'])):
                    key_sentences.append(sentence)
                if len(key_sentences) >= 3:
                    break
            if len(key_sentences) >= 3:
                break
        
        if key_sentences:
            answer = f"질문: {question}\n\n"
            answer += "관련 정보:\n"
            for i, sentence in enumerate(key_sentences[:3], 1):
                answer += f"{i}. {sentence}.\n"
            answer += "\n💡 더 자세한 정보는 원본 문서를 참조하세요."
            return answer
        else:
            # 컨텍스트의 일부를 그대로 반환
            context_text = " ".join(context_docs[:1])[:300]
            return f"질문: {question}\n\n관련 내용:\n{context_text}..."

    def generate_answer(self, question):
        """최종 답변 생성"""
        if not self.is_initialized:
            return "시스템이 초기화되지 않았습니다.", []
        
        try:
            # 1. 관련 문서 검색
            retrieved_docs, sources = self.search(question, top_k=3)
            
            if not retrieved_docs:
                return "관련된 정보를 찾을 수 없습니다. 다른 키워드로 질문해보세요.", []
            
            # 2. 트랜스포머 모델로 답변 생성
            llm_answer = self.generate_llm_answer(question, retrieved_docs)
            
            return llm_answer, sources
            
        except Exception as e:
            logger.error(f"답변 생성 오류: {e}")
            return "답변 생성 중 오류가 발생했습니다.", []

    def health_check(self):
        """시스템 상태"""
        return {
            "initialized": self.is_initialized,
            "documents_loaded": len(self.docs),
            "model_loaded": self.generator is not None,
            "model_name": self.model_name,
            "vector_db_ready": self.doc_vectors is not None,
            "version": "transformer_based",
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }