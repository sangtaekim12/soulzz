#!/usr/bin/env python3
"""
프로덕션용 Flask 서버 실행 스크립트
"""

import os
import sys
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """서버 시작"""
    try:
        logger.info("관광 챗봇 서버를 시작합니다...")
        
        # 환경 변수 설정
        os.environ['FLASK_ENV'] = 'production'
        
        # Flask 앱 임포트 및 실행
        from app import app, initialize_rag
        import threading
        
        # RAG 시스템을 별도 스레드에서 초기화
        logger.info("RAG 시스템 초기화를 시작합니다...")
        init_thread = threading.Thread(target=initialize_rag)
        init_thread.daemon = True
        init_thread.start()
        
        # Flask 앱 실행
        logger.info("Flask 서버를 포트 5000에서 시작합니다...")
        app.run(
            host='0.0.0.0', 
            port=5000, 
            debug=False,
            threaded=True
        )
        
    except Exception as e:
        logger.error(f"서버 시작 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
