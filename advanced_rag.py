import os
import logging
import re
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import unicodedata

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedTourismRAG:
    def __init__(self, data_folder="./data", similarity_threshold=0.15):
        self.data_folder = data_folder
        self.docs = []
        self.metainfo = []
        self.vectorizer = TfidfVectorizer(
            max_features=2000,  # 증가
            stop_words='english',
            ngram_range=(1, 3),  # 3-gram까지 확장
            lowercase=True,
            min_df=2,  # 최소 문서 빈도
            max_df=0.8  # 최대 문서 빈도
        )
        self.doc_vectors = None
        self.similarity_threshold = similarity_threshold
        self.is_initialized = False

    def advanced_text_cleaning(self, text):
        """고급 텍스트 전처리"""
        if not text:
            return ""
        
        # 1. 유니코드 정규화
        text = unicodedata.normalize('NFKC', text)
        
        # 2. 과도한 띄어쓰기 문제 해결
        # "2 0 2 4" -> "2024", "K C T I" -> "KCTI"
        text = re.sub(r'(\w)\s+(\w)\s+(\w)\s+(\w)', r'\1\2\3\4', text)
        text = re.sub(r'(\w)\s+(\w)\s+(\w)', r'\1\2\3', text)
        text = re.sub(r'(\w)\s+(\w)(?=\s|$)', r'\1\2', text)
        
        # 3. 한글 단어 사이 불필요한 공백 제거
        text = re.sub(r'([가-힣])\s+([가-힣])', r'\1\2', text)
        
        # 4. 숫자와 단위 결합
        text = re.sub(r'(\d+)\s*([%원달러명개천만억])', r'\1\2', text)
        
        # 5. 영어 단어 복원
        text = re.sub(r'([A-Z])\s+([A-Z])\s*([A-Z])', r'\1\2\3', text)
        
        # 6. 과도한 공백을 단일 공백으로
        text = re.sub(r'\s{2,}', ' ', text)
        
        # 7. 문장 부호 정리
        text = re.sub(r'[^\w\s가-힣.,!?;:()\[\]%-]', ' ', text)
        
        # 8. 최종 정리
        text = text.strip()
        
        return text

    def extract_meaningful_sentences(self, text):
        """의미있는 문장만 추출"""
        sentences = []
        
        # 문장 분리 (한글 기준)
        sent_patterns = [
            r'[.!?]\s+',
            r'[다음과같다음]\s*[.:]\s*',
            r'[입니다됩니다었습니다습니다]\s*[.]\s*'
        ]
        
        current_sentences = [text]
        for pattern in sent_patterns:
            temp = []
            for sent in current_sentences:
                temp.extend(re.split(pattern, sent))
            current_sentences = temp
        
        for sent in current_sentences:
            sent = sent.strip()
            # 의미있는 문장만 선별
            if (len(sent) > 20 and 
                len(sent) < 500 and 
                re.search(r'[가-힣]', sent) and  # 한글 포함
                not re.match(r'^[0-9\s.]+$', sent)):  # 숫자만 아님
                sentences.append(sent)
        
        return sentences

    def load_pdfs(self):
        """개선된 PDF 로딩"""
        logger.info("고급 PDF 처리 시작...")
        
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
            return
            
        pdf_files = [f for f in os.listdir(self.data_folder) if f.endswith('.pdf')]
        
        if not pdf_files:
            logger.warning("PDF 파일이 없습니다.")
            return
            
        for pdf in tqdm(pdf_files, desc="PDF 고급 처리 중"):
            try:
                path = os.path.join(self.data_folder, pdf)
                reader = PdfReader(path)
                
                for i, page in enumerate(reader.pages):
                    try:
                        raw_text = page.extract_text()
                        if raw_text and len(raw_text.strip()) > 50:
                            
                            # 고급 텍스트 정리
                            cleaned_text = self.advanced_text_cleaning(raw_text)
                            
                            if len(cleaned_text) > 100:
                                # 문장 단위로 분리하여 저장
                                sentences = self.extract_meaningful_sentences(cleaned_text)
                                
                                if sentences:
                                    # 긴 텍스트는 여러 청크로 분할
                                    if len(cleaned_text) > 1000:
                                        chunks = self.split_into_chunks(cleaned_text, max_length=800)
                                        for j, chunk in enumerate(chunks):
                                            self.docs.append(chunk)
                                            self.metainfo.append(f"{pdf} - page {i+1} - chunk {j+1}")
                                    else:
                                        self.docs.append(cleaned_text)
                                        self.metainfo.append(f"{pdf} - page {i+1}")
                                        
                    except Exception as e:
                        logger.warning(f"페이지 {i+1} 처리 실패: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"PDF {pdf} 처리 실패: {e}")
                continue
                
        logger.info(f"총 {len(self.docs)}개의 텍스트 청크를 추출했습니다.")

    def split_into_chunks(self, text, max_length=800, overlap=100):
        """텍스트를 청크로 분할 (중복 허용)"""
        chunks = []
        sentences = text.split('. ')
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # 중복 부분 유지
                    current_chunk = current_chunk[-overlap:] + sentence + ". "
                else:
                    current_chunk = sentence + ". "
            else:
                current_chunk += sentence + ". "
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks

    def build_tfidf_vectors(self):
        """개선된 TF-IDF 벡터 생성"""
        logger.info("고급 TF-IDF 벡터 생성 중...")
        
        if not self.docs:
            return
            
        try:
            # 한국어 텍스트에 최적화된 전처리
            processed_docs = []
            for doc in self.docs:
                # 추가 전처리
                processed = re.sub(r'[^\w\s가-힣]', ' ', doc)
                processed = re.sub(r'\s+', ' ', processed).strip()
                processed_docs.append(processed)
            
            self.doc_vectors = self.vectorizer.fit_transform(processed_docs)
            logger.info("고급 TF-IDF 벡터 생성 완료")
        except Exception as e:
            logger.error(f"벡터 생성 실패: {e}")
            raise

    def enhanced_query(self, question, top_k=5):
        """개선된 검색 시스템"""
        if not self.is_initialized or self.doc_vectors is None:
            return [], []
            
        try:
            # 질문 전처리
            processed_question = self.advanced_text_cleaning(question)
            
            # 벡터화
            query_vector = self.vectorizer.transform([processed_question])
            
            # 코사인 유사도 계산
            similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
            
            # 상위 k개 선택
            top_indices = similarities.argsort()[-top_k*2:][::-1]  # 더 많이 선택한 후 필터링
            
            retrieved = []
            sources = []
            
            for idx in top_indices:
                if similarities[idx] >= self.similarity_threshold and len(retrieved) < top_k:
                    doc_text = self.docs[idx]
                    
                    # 중복 제거 (유사한 내용 필터링)
                    is_duplicate = False
                    for existing in retrieved:
                        if self.calculate_text_similarity(doc_text, existing) > 0.8:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        retrieved.append(doc_text)
                        sim_score = round(similarities[idx], 3)
                        sources.append(f"{self.metainfo[idx]} (유사도 {sim_score})")
            
            return retrieved, sources
            
        except Exception as e:
            logger.error(f"검색 중 오류: {e}")
            return [], []

    def calculate_text_similarity(self, text1, text2):
        """두 텍스트 간 유사도 계산"""
        try:
            vectors = self.vectorizer.transform([text1, text2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return similarity
        except:
            return 0

    def generate_enhanced_answer(self, question):
        """개선된 답변 생성"""
        retrieved_docs, sources = self.enhanced_query(question)
        
        if not retrieved_docs:
            return "죄송합니다. 질문과 관련된 정보를 찾을 수 없습니다. 다른 방식으로 질문해보시거나 더 구체적인 키워드를 사용해주세요.", []
        
        # 컨텍스트 구성 (더 지능적으로)
        context_parts = []
        total_length = 0
        
        for doc in retrieved_docs:
            if total_length + len(doc) < 1500:  # 컨텍스트 길이 제한
                context_parts.append(doc)
                total_length += len(doc)
            else:
                break
        
        context = "\n\n".join(context_parts)
        
        # 질문 키워드 기반 맞춤형 답변
        answer = self.create_contextual_answer(question, context)
        
        return answer, sources[:3]

    def create_contextual_answer(self, question, context):
        """컨텍스트 기반 지능형 답변 생성"""
        # 질문에서 핵심 키워드 추출
        question_lower = question.lower()
        
        # 한류 관련 질문 처리
        if any(keyword in question_lower for keyword in ['한류', 'hallyu', 'k-pop', '한국 문화', '드라마']):
            intro = "한류와 관련하여 다음과 같은 정보를 찾았습니다:"
        # 관광 관련 질문 처리  
        elif any(keyword in question_lower for keyword in ['관광', '여행', '방문', '여행객']):
            intro = "관광 및 여행과 관련하여 다음과 같은 정보를 제공합니다:"
        # 통계/데이터 관련 질문 처리
        elif any(keyword in question_lower for keyword in ['특성', '분석', '데이터', '통계', '현황']):
            intro = "분석 데이터에 따르면 다음과 같습니다:"
        else:
            intro = "관련 정보를 찾았습니다:"
        
        # 컨텍스트에서 핵심 문장 추출
        sentences = context.split('. ')
        key_sentences = []
        
        for sentence in sentences:
            # 의미있고 완성된 문장만 선택
            if (len(sentence.strip()) > 30 and 
                len(sentence.strip()) < 300 and
                not sentence.strip().endswith('다음과 같') and
                '다음' not in sentence[:10]):
                key_sentences.append(sentence.strip())
            
            if len(key_sentences) >= 3:  # 최대 3개 문장
                break
        
        if key_sentences:
            main_content = ". ".join(key_sentences)
            if not main_content.endswith('.'):
                main_content += "."
            
            answer = f"{intro}\n\n{main_content}"
        else:
            answer = f"{intro}\n\n{context[:400]}..."
            
        return answer

    def initialize(self):
        """시스템 초기화"""
        try:
            logger.info("고급 RAG 시스템 초기화 시작...")
            
            self.load_pdfs()
            
            if self.docs:
                self.build_tfidf_vectors()
                self.is_initialized = True
                logger.info("고급 RAG 시스템 초기화 완료!")
            else:
                logger.warning("PDF 문서가 없습니다.")
                
        except Exception as e:
            logger.error(f"시스템 초기화 실패: {e}")
            raise

    def health_check(self):
        """시스템 상태 확인"""
        status = {
            "initialized": self.is_initialized,
            "documents_loaded": len(self.docs),
            "model_loaded": self.doc_vectors is not None,
            "vector_db_ready": self.doc_vectors is not None,
            "processing_level": "advanced"
        }
        return status

    # 기존 메서드와의 호환성을 위한 별칭
    def generate_answer(self, question):
        return self.generate_enhanced_answer(question)
    
    def query(self, question, top_k=5):
        return self.enhanced_query(question, top_k)