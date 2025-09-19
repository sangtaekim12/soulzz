import os
import logging
import re
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleTourismRAG:
    def __init__(self, data_folder="./data", similarity_threshold=0.1):
        self.data_folder = data_folder
        self.docs = []
        self.metainfo = []  # 출처 정보
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.doc_vectors = None
        self.similarity_threshold = similarity_threshold
        self.is_initialized = False

    def initialize(self):
        """RAG 시스템 초기화"""
        try:
            logger.info("Simple RAG 시스템 초기화 시작...")
            
            # PDF 데이터 로드
            self.load_pdfs()
            
            # TF-IDF 벡터화
            if self.docs:
                self.build_tfidf_vectors()
                self.is_initialized = True
                logger.info("Simple RAG 시스템 초기화 완료!")
            else:
                logger.warning("PDF 문서가 없습니다. data 폴더에 PDF 파일을 추가해주세요.")
                
        except Exception as e:
            logger.error(f"RAG 시스템 초기화 실패: {e}")
            raise

    def clean_text(self, text):
        """텍스트 전처리"""
        # 특수 문자 제거 및 정규화
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def load_pdfs(self):
        """PDF 파일들을 로드하고 텍스트를 추출"""
        logger.info("PDF 로드 및 텍스트 추출 시작")
        
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
            logger.warning(f"{self.data_folder} 폴더가 생성되었습니다. PDF 파일을 추가해주세요.")
            return
            
        pdf_files = [f for f in os.listdir(self.data_folder) if f.endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"{self.data_folder} 폴더에 PDF 파일이 없습니다.")
            return
            
        for pdf in tqdm(pdf_files, desc="PDF 처리 중"):
            try:
                path = os.path.join(self.data_folder, pdf)
                reader = PdfReader(path)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and len(text.strip()) > 20:
                        cleaned_text = self.clean_text(text)
                        if cleaned_text:
                            self.docs.append(cleaned_text)
                            self.metainfo.append(f"{pdf} - page {i+1}")
            except Exception as e:
                logger.error(f"PDF 파일 {pdf} 처리 중 오류: {e}")
                
        logger.info(f"총 {len(self.docs)}개의 페이지에서 텍스트를 추출했습니다.")

    def build_tfidf_vectors(self):
        """TF-IDF 벡터 생성"""
        logger.info("TF-IDF 벡터 생성 중...")
        
        if not self.docs:
            logger.warning("문서가 없어서 벡터를 구축할 수 없습니다.")
            return
            
        try:
            self.doc_vectors = self.vectorizer.fit_transform(self.docs)
            logger.info("TF-IDF 벡터 생성 완료")
        except Exception as e:
            logger.error(f"TF-IDF 벡터 생성 실패: {e}")
            raise

    def query(self, question, top_k=5):
        """질문에 대한 유사한 문서들을 검색"""
        if not self.is_initialized or self.doc_vectors is None:
            return [], []
            
        try:
            # 질문을 벡터화
            query_vector = self.vectorizer.transform([self.clean_text(question)])
            
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
            
        except Exception as e:
            logger.error(f"검색 중 오류: {e}")
            return [], []

    def generate_simple_answer(self, question):
        """간단한 키워드 기반 답변 생성 (Ollama 없이)"""
        retrieved_docs, sources = self.query(question)
        
        if not retrieved_docs:
            return "죄송합니다. 관련된 정보를 찾을 수 없습니다. 다른 질문을 시도해보세요.", []
        
        # 간단한 키워드 매칭 기반 답변
        context = " ".join(retrieved_docs[:2])  # 상위 2개 문서만 사용
        
        # 키워드 기반 간단한 답변 생성
        keywords = {
            '서울': '서울은 대한민국의 수도로 다양한 관광지와 문화유산이 있습니다.',
            '부산': '부산은 해안도시로 아름다운 해변과 신선한 해산물로 유명합니다.',
            '제주': '제주도는 자연경관이 뛰어나고 독특한 문화를 가진 섬입니다.',
            '관광지': '한국에는 다양한 관광지가 있으며, 각 지역마다 특색있는 명소들이 있습니다.',
            '음식': '한국 음식은 다양하고 풍부한 맛을 자랑하며, 지역마다 특색있는 요리가 있습니다.',
            '문화': '한국의 전통문화는 오랜 역사를 가지고 있으며, 현재도 잘 보존되고 있습니다.',
        }
        
        # 질문에서 키워드 찾기
        answer_parts = []
        for keyword, description in keywords.items():
            if keyword in question:
                answer_parts.append(description)
        
        if answer_parts:
            base_answer = " ".join(answer_parts)
        else:
            base_answer = "관련 정보를 찾았습니다."
        
        # 검색된 문서에서 간단한 정보 추출
        if context:
            # 문서에서 문장 추출 (간단한 방법)
            sentences = context.split('.')[:2]  # 처음 2문장
            if sentences:
                extracted_info = ". ".join([s.strip() for s in sentences if len(s.strip()) > 10])
                if extracted_info:
                    answer = f"{base_answer}\n\n{extracted_info}."
                else:
                    answer = base_answer
            else:
                answer = base_answer
        else:
            answer = base_answer
        
        return answer, sources[:3]  # 상위 3개 출처만 반환

    def generate_answer(self, question):
        """Ollama를 사용한 답변 생성 (폴백으로 simple_answer 사용)"""
        try:
            # Ollama 사용 시도
            import ollama
            
            retrieved_docs, sources = self.query(question)
            
            if not retrieved_docs:
                return "죄송합니다. 관련된 정보를 찾을 수 없습니다. 다른 질문을 시도해보세요.", []
            
            context = "\n".join(retrieved_docs[:2])  # 상위 2개 문서만 사용
            prompt = f"""
다음 관광 관련 자료를 참고하여 한국어로 정확하고 도움이 되는 답변을 해주세요.

자료:
{context}

질문: {question}

답변 지침:
- 제공된 자료를 기반으로 정확하게 답변하세요
- 자료에 없는 내용은 추측하지 마세요
- 친근하고 도움이 되는 톤으로 답변하세요
- 답변은 200자 이내로 간결하게 작성하세요

답변:
"""
            
            response = ollama.chat(
                model="nous-hermes2", 
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response['message']['content']
            return answer, sources[:3]
            
        except Exception as e:
            logger.warning(f"Ollama 사용 실패, 간단한 답변으로 대체: {e}")
            # Ollama 실패 시 간단한 답변 사용
            return self.generate_simple_answer(question)

    def health_check(self):
        """시스템 상태 확인"""
        status = {
            "initialized": self.is_initialized,
            "documents_loaded": len(self.docs),
            "model_loaded": self.doc_vectors is not None,
            "vector_db_ready": self.doc_vectors is not None
        }
        return status