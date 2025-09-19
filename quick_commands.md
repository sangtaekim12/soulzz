# 📁 PDF 파일 관리 빠른 명령어들

## 🗑️ 기존 파일 삭제

```bash
# data 폴더로 이동
cd /home/user/webapp

# 모든 PDF 파일 삭제
rm -f data/*.pdf

# 삭제 확인
ls -la data/
```

## 📥 새 파일 업로드 방법들

### 1. URL에서 다운로드
```bash
cd /home/user/webapp/data

# wget 사용
wget "https://example.com/your-file.pdf" -O "new_filename.pdf"

# curl 사용
curl -o "new_filename.pdf" "https://example.com/your-file.pdf"
```

### 2. 로컬 파일 복사 (같은 시스템 내)
```bash
cd /home/user/webapp/data

# cp 명령어로 복사
cp "/path/to/your/file.pdf" "./new_filename.pdf"
```

### 3. 텍스트에서 PDF 생성 (임시 데이터용)
```bash
cd /home/user/webapp

# 업로드 도우미 실행
python upload_helper.py
```

## 🔄 서비스 재시작

새로운 파일을 추가한 후 반드시 실행:

```bash
cd /home/user/webapp

# 챗봇 서비스 재시작
supervisorctl -c supervisord.conf restart tourism-chatbot

# 상태 확인
supervisorctl -c supervisord.conf status

# 새로 로드된 문서 수 확인
sleep 5
curl -s http://localhost:5000/health | python -m json.tool
```

## 📋 파일 확인

```bash
cd /home/user/webapp

# PDF 파일들 확인
ls -la data/*.pdf

# 파일 크기와 함께 확인
ls -lh data/*.pdf

# PDF 파일 내용 간단 테스트
python3 -c "
from PyPDF2 import PdfReader
import os
for f in os.listdir('./data'):
    if f.endswith('.pdf'):
        reader = PdfReader(f'./data/{f}')
        print(f'{f}: {len(reader.pages)} pages')
"
```

## ⚠️ 주의사항

1. **파일 형식**: PDF 파일만 지원됩니다
2. **파일 크기**: 너무 큰 파일은 처리 시간이 길어질 수 있습니다
3. **텍스트 추출**: 스캔본 PDF는 OCR이 필요할 수 있습니다
4. **서비스 재시작**: 새 파일 추가 후 반드시 서비스를 재시작해야 합니다
5. **파일명**: 영어와 숫자, 언더스코어만 사용 권장

## 🎯 추천 워크플로우

1. 기존 파일 삭제: `rm -f data/*.pdf`
2. 새 파일 추가: URL 다운로드 또는 파일 복사
3. 파일 확인: `ls -la data/`
4. 서비스 재시작: `supervisorctl -c supervisord.conf restart tourism-chatbot`
5. 로드 확인: `curl -s http://localhost:5000/health`
6. 웹에서 테스트: 브라우저로 챗봇 접속하여 테스트