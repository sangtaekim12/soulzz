import os
import logging
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TourismRAG:
    def __init__(self, data_folder="./data", model_name="nous-hermes2", similarity_threshold=1.0):
        self.data_folder = data_folder
        self.docs = []
        self.metainfo = []  # 출처 정보
        self.model = None
        self.index = None
        self.llm_model_name = model_name
        self.similarity_threshold = similarity_threshold  # 거리 임계값 (낮을수록 엄격)
        self.is_initialized = False

    def initialize(self):
        """RAG 시스템 초기화"""
        try:
            logger.info("RAG 시스템 초기화 시작...")
            
            # 임베딩 모델 로드
            logger.info("임베딩 모델 로드 중...")
            self.model = SentenceTransformer('BAAI/bge-small-en-v1.5')
            
            # PDF 데이터 로드
            self.load_pdfs()
            
            # 벡터 DB 구축
            if self.docs:
                self.build_vector_db()
                self.is_initialized = True
                logger.info("RAG 시스템 초기화 완료!")
            else:
                logger.warning("PDF 문서가 없습니다. data 폴더에 PDF 파일을 추가해주세요.")
                
        except Exception as e:
            logger.error(f"RAG 시스템 초기화 실패: {e}")
            raise

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
                    if text and len(text.strip()) > 50:
                        self.docs.append(text.strip())
                        self.metainfo.append(f"{pdf} - page {i+1}")
            except Exception as e:
                logger.error(f"PDF 파일 {pdf} 처리 중 오류: {e}")
                
        logger.info(f"총 {len(self.docs)}개의 페이지에서 텍스트를 추출했습니다.")

    def build_vector_db(self):
        """임베딩 생성 및 벡터 DB 구축"""
        logger.info("임베딩 생성 및 벡터DB 구축")
        
        if not self.docs:
            logger.warning("문서가 없어서 벡터 DB를 구축할 수 없습니다.")
            return
            
        embeddings = self.model.encode(self.docs, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings))
        self.embeddings = embeddings
        logger.info("벡터 DB 구축 완료")

    def query(self, question, top_k=5):
        """질문에 대한 유사한 문서들을 검색"""
        if not self.is_initialized or not self.index:
            return [], []
            
        q_embed = self.model.encode([question])
        D, I = self.index.search(np.array(q_embed), k=min(top_k, len(self.docs)))

        retrieved = []
        sources = []

        for idx, dist in zip(I[0], D[0]):
            if dist < self.similarity_threshold:
                retrieved.append(self.docs[idx])
                sim_score = round(1 - dist / 2, 3)  # 거리 → 유사도 변환 (정규화)
                sources.append(f"{self.metainfo[idx]} (유사도 {sim_score})")
        
        return retrieved, list(set(sources))  # 중복 제거

    def generate_answer(self, question):
        """질문에 대한 답변 생성"""
        if not self.is_initialized:
            return "시스템이 아직 초기화되지 않았습니다. 잠시 후 다시 시도해주세요.", []
            
        try:
            retrieved_docs, sources = self.query(question)

            if not retrieved_docs:
                return "죄송합니다. 관련된 정보를 찾을 수 없습니다. 다른 질문을 시도해보세요.", []

            context = "\n".join(retrieved_docs[:3])  # 상위 3개 문서만 사용
            prompt = f"""
아래 관광 연구 자료를 참고하여 반드시 **한국어로** 정확하고 도움이 되는 답변을 해주세요.

자료:
{context}

질문: {question}

답변 지침:
- 제공된 자료를 기반으로 정확하게 답변하세요
- 자료에 없는 내용은 추측하지 마세요
- 친근하고 도움이 되는 톤으로 답변하세요
- 답변은 300자 이내로 간결하게 작성하세요

답변:
"""

            response = ollama.chat(model=self.llm_model_name, messages=[{"role": "user", "content": prompt}])
            answer = response['message']['content']
            return answer, sources[:3]  # 상위 3개 출처만 반환
            
        except Exception as e:
            logger.error(f"답변 생성 중 오류: {e}")
            return "죄송합니다. 답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.", []

    def health_check(self):
        """시스템 상태 확인"""
        status = {
            "initialized": self.is_initialized,
            "documents_loaded": len(self.docs),
            "model_loaded": self.model is not None,
            "vector_db_ready": self.index is not None
        }
        return status