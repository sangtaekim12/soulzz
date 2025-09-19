from flask import Flask, request, jsonify, render_template
import threading
import time
import logging
from enhanced_rag import EnhancedTourismRAG
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# RAG 인스턴스 전역 변수
rag_system = None
initialization_complete = False

def initialize_rag():
    """RAG 시스템을 백그라운드에서 초기화"""
    global rag_system, initialization_complete
    try:
        rag_system = EnhancedTourismRAG(
            data_folder="./data", 
            similarity_threshold=0.03  # 향상된 컨텍스트 검색 임계값
        )
        rag_system.initialize()
        initialization_complete = True
        logger.info("향상된 컨텍스트 RAG 시스템 초기화 완료")
    except Exception as e:
        logger.error(f"RAG 시스템 초기화 실패: {e}")
        initialization_complete = False

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/health')
def health():
    """서버 상태 확인"""
    if rag_system:
        status = rag_system.health_check()
        status['server_ready'] = initialization_complete
    else:
        status = {
            'server_ready': False,
            'initialized': False,
            'documents_loaded': 0,
            'model_loaded': False,
            'vector_db_ready': False
        }
    return jsonify(status)

@app.route('/chat', methods=['POST'])
def chat():
    """챗봇 API 엔드포인트"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'error': '메시지가 필요합니다.',
                'answer': '',
                'sources': []
            }), 400
            
        user_message = data['message'].strip()
        
        if not user_message:
            return jsonify({
                'error': '빈 메시지는 처리할 수 없습니다.',
                'answer': '',
                'sources': []
            }), 400
            
        # RAG 시스템이 초기화되지 않았으면 대기
        if not initialization_complete or not rag_system:
            return jsonify({
                'error': '시스템이 아직 초기화 중입니다. 잠시 후 다시 시도해주세요.',
                'answer': '',
                'sources': []
            }), 503
            
        # 답변 생성
        answer, sources = rag_system.generate_answer(user_message)
        
        return jsonify({
            'answer': answer,
            'sources': sources,
            'error': None
        })
        
    except Exception as e:
        logger.error(f"챗봇 처리 중 오류: {e}")
        return jsonify({
            'error': '서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요.',
            'answer': '',
            'sources': []
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': '페이지를 찾을 수 없습니다.'}), 404

@app.route('/upload', methods=['GET', 'POST'])
def upload_files():
    """파일 업로드 페이지 및 처리"""
    if request.method == 'GET':
        return render_template('upload.html')
    
    try:
        # 파일 업로드 처리
        if 'files' not in request.files:
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
        
        files = request.files.getlist('files')
        uploaded_files = []
        
        for file in files:
            if file and file.filename.endswith('.pdf'):
                filename = file.filename
                filepath = os.path.join('./data', filename)
                file.save(filepath)
                uploaded_files.append(filename)
        
        if uploaded_files:
            # RAG 시스템 재초기화 (향상된 컨텍스트 버전)
            if rag_system:
                rag_system.__init__('./data', 0.03)
                rag_system.initialize()
            
            return jsonify({
                'success': True,
                'uploaded_files': uploaded_files,
                'message': f'{len(uploaded_files)}개 파일이 업로드되었습니다.'
            })
        else:
            return jsonify({'error': 'PDF 파일만 업로드 가능합니다.'}), 400
            
    except Exception as e:
        logger.error(f"파일 업로드 오류: {e}")
        return jsonify({'error': '파일 업로드 중 오류가 발생했습니다.'}), 500

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': '서버 내부 오류가 발생했습니다.'}), 500

if __name__ == '__main__':
    # RAG 시스템을 별도 스레드에서 초기화
    init_thread = threading.Thread(target=initialize_rag)
    init_thread.daemon = True
    init_thread.start()
    
    # Flask 앱 실행 (개발 환경에서만)
    app.run(host='0.0.0.0', port=5000, debug=True)