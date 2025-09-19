# 관광 지식 챗봇 🌍🤖

RAG(Retrieval-Augmented Generation) 기술과 Ollama를 활용한 관광 지식 챗봇 웹 애플리케이션입니다.

## 주요 기능

- **PDF 문서 기반 지식베이스**: PDF 파일을 업로드하여 관광 관련 지식을 구축
- **의미 기반 검색**: 문장 임베딩을 통한 정확한 정보 검색
- **실시간 웹 채팅**: 직관적인 웹 인터페이스를 통한 실시간 대화
- **출처 표시**: 답변의 근거가 되는 문서와 페이지 정보 제공
- **반응형 디자인**: 모바일과 데스크톱 모두 지원

## 시스템 요구사항

- Python 3.8+
- Ollama (nous-hermes2 모델)
- 최소 4GB RAM 권장
- GPU 권장 (CPU에서도 동작 가능)

## 설치 및 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. Ollama 설정

```bash
# Ollama 설치 (Linux/Mac)
curl -fsSL https://ollama.ai/install.sh | sh

# nous-hermes2 모델 다운로드
ollama pull nous-hermes2
```

### 3. PDF 문서 준비

`data/` 폴더에 관광 관련 PDF 문서들을 추가하세요:

```
webapp/
├── data/
│   ├── tourism_guide_seoul.pdf
│   ├── busan_travel_info.pdf
│   └── jeju_attractions.pdf
├── app.py
├── tourism_rag.py
└── ...
```

## 실행 방법

### 개발 환경

```bash
python app.py
```

### 프로덕션 환경 (Supervisor 사용)

```bash
# Supervisor 설치
pip install supervisor

# 서비스 시작
supervisord -c supervisord.conf

# 서비스 상태 확인
supervisorctl -c supervisord.conf status

# 서비스 관리
supervisorctl -c supervisord.conf start tourism-chatbot
supervisorctl -c supervisord.conf stop tourism-chatbot
supervisorctl -c supervisord.conf restart tourism-chatbot
```

### 또는 직접 실행

```bash
python server.py
```

## 웹 인터페이스 접속

서버 시작 후 웹 브라우저에서 접속:
- 로컬: http://localhost:5000
- 외부 접속: http://[서버IP]:5000

## API 엔드포인트

### 1. 메인 페이지
```
GET /
```

### 2. 시스템 상태 확인
```
GET /health
```

응답 예시:
```json
{
  "initialized": true,
  "documents_loaded": 25,
  "model_loaded": true,
  "vector_db_ready": true,
  "server_ready": true
}
```

### 3. 챗봇 대화
```
POST /chat
Content-Type: application/json

{
  "message": "서울의 주요 관광지는 어디인가요?"
}
```

응답 예시:
```json
{
  "answer": "서울의 주요 관광지로는 경복궁, 남산타워, 명동, 강남 등이 있습니다...",
  "sources": [
    "seoul_guide.pdf - page 5 (유사도 0.892)",
    "tourism_info.pdf - page 12 (유사도 0.847)"
  ],
  "error": null
}
```

## 프로젝트 구조

```
webapp/
├── app.py                 # Flask 웹 애플리케이션
├── tourism_rag.py         # RAG 시스템 핵심 로직
├── server.py              # 프로덕션 서버 실행 스크립트
├── requirements.txt       # Python 의존성
├── supervisord.conf       # Supervisor 설정
├── data/                  # PDF 문서 저장 폴더
├── templates/             # HTML 템플릿
│   └── index.html
├── static/               # 정적 파일
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
└── README.md
```

## 설정 가능한 옵션

### tourism_rag.py 설정

```python
rag = TourismRAG(
    data_folder="./data",           # PDF 문서 폴더
    model_name="nous-hermes2",      # Ollama 모델명
    similarity_threshold=1.0        # 유사도 임계값 (낮을수록 엄격)
)
```

### 모델 변경

다른 Ollama 모델을 사용하려면:

```bash
# 다른 모델 설치
ollama pull llama2
ollama pull codellama

# tourism_rag.py에서 모델명 변경
model_name="llama2"
```

## 문제 해결

### 1. Ollama 연결 오류
```bash
# Ollama 서비스 상태 확인
ollama list

# Ollama 서비스 재시작
sudo systemctl restart ollama
```

### 2. 메모리 부족
- `similarity_threshold` 값을 높여 검색 결과 수 제한
- PDF 문서 수를 줄이거나 크기가 작은 문서 사용
- 시스템 RAM 증설 권장

### 3. 느린 응답 속도
- GPU 사용 권장 (CUDA 설정)
- 더 빠른 임베딩 모델 사용
- 문서 전처리로 불필요한 내용 제거

### 4. PDF 읽기 오류
- PDF 파일이 텍스트 추출 가능한지 확인
- 스캔본 PDF는 OCR 처리 필요
- 암호화된 PDF는 사전에 잠금 해제

## 로그 확인

```bash
# 애플리케이션 로그
tail -f app.log

# 에러 로그
tail -f app_error.log

# Supervisor 로그
tail -f supervisord.log
```

## 보안 고려사항

- 프로덕션 환경에서는 HTTPS 사용 권장
- 파일 업로드 시 검증 로직 추가 필요
- API 요청 제한 설정 권장
- 민감한 정보가 포함된 PDF 주의

## 라이센스

MIT License

## 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 문의

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.