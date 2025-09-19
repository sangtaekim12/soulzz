#!/usr/bin/env python3
"""
PDF 파일 업로드 도우미 스크립트
사용자가 새로운 PDF 파일들을 쉽게 추가할 수 있도록 도움
"""

import os
import shutil
from urllib.request import urlretrieve
from urllib.parse import urlparse

class PDFUploadHelper:
    def __init__(self, data_folder="./data"):
        self.data_folder = data_folder
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
    
    def clear_existing_pdfs(self):
        """기존 PDF 파일들을 모두 삭제"""
        pdf_files = [f for f in os.listdir(self.data_folder) if f.endswith('.pdf')]
        for pdf in pdf_files:
            file_path = os.path.join(self.data_folder, pdf)
            os.remove(file_path)
            print(f"삭제됨: {pdf}")
        
        if pdf_files:
            print(f"총 {len(pdf_files)}개의 PDF 파일이 삭제되었습니다.")
        else:
            print("삭제할 PDF 파일이 없습니다.")
    
    def download_from_url(self, url, filename=None):
        """URL에서 PDF 파일을 다운로드"""
        try:
            if not filename:
                # URL에서 파일명 추출
                parsed_url = urlparse(url)
                filename = os.path.basename(parsed_url.path)
                if not filename.endswith('.pdf'):
                    filename += '.pdf'
            
            file_path = os.path.join(self.data_folder, filename)
            urlretrieve(url, file_path)
            print(f"다운로드 완료: {filename} <- {url}")
            return True
        except Exception as e:
            print(f"다운로드 실패: {url} - {e}")
            return False
    
    def copy_from_path(self, source_path, new_filename=None):
        """로컬 경로에서 PDF 파일을 복사"""
        try:
            if not os.path.exists(source_path):
                print(f"파일을 찾을 수 없습니다: {source_path}")
                return False
            
            if not new_filename:
                new_filename = os.path.basename(source_path)
            
            dest_path = os.path.join(self.data_folder, new_filename)
            shutil.copy2(source_path, dest_path)
            print(f"복사 완료: {new_filename} <- {source_path}")
            return True
        except Exception as e:
            print(f"복사 실패: {source_path} - {e}")
            return False
    
    def list_current_files(self):
        """현재 data 폴더의 PDF 파일들을 나열"""
        pdf_files = [f for f in os.listdir(self.data_folder) if f.endswith('.pdf')]
        if pdf_files:
            print(f"\n현재 data 폴더의 PDF 파일들 ({len(pdf_files)}개):")
            for i, pdf in enumerate(pdf_files, 1):
                file_path = os.path.join(self.data_folder, pdf)
                size = os.path.getsize(file_path)
                print(f"  {i}. {pdf} ({size:,} bytes)")
        else:
            print("\ndata 폴더에 PDF 파일이 없습니다.")
    
    def restart_chatbot_service(self):
        """챗봇 서비스를 재시작하여 새로운 파일들을 로드"""
        try:
            import subprocess
            result = subprocess.run(
                ['supervisorctl', '-c', 'supervisord.conf', 'restart', 'tourism-chatbot'],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("✅ 챗봇 서비스가 재시작되었습니다.")
                print("새로운 PDF 파일들이 로드됩니다.")
            else:
                print(f"❌ 서비스 재시작 실패: {result.stderr}")
        except Exception as e:
            print(f"❌ 서비스 재시작 중 오류: {e}")

def main():
    print("=== PDF 파일 업로드 도우미 ===")
    helper = PDFUploadHelper()
    
    while True:
        print("\n선택하세요:")
        print("1. 현재 PDF 파일들 보기")
        print("2. 기존 PDF 파일들 모두 삭제")
        print("3. URL에서 PDF 다운로드")
        print("4. 로컬 파일 복사")
        print("5. 챗봇 서비스 재시작")
        print("0. 종료")
        
        choice = input("\n선택 (0-5): ").strip()
        
        if choice == '0':
            break
        elif choice == '1':
            helper.list_current_files()
        elif choice == '2':
            confirm = input("정말로 모든 PDF 파일을 삭제하시겠습니까? (y/N): ")
            if confirm.lower() == 'y':
                helper.clear_existing_pdfs()
        elif choice == '3':
            url = input("PDF URL을 입력하세요: ").strip()
            if url:
                filename = input("파일명 (선택사항, 엔터로 건너뛰기): ").strip() or None
                helper.download_from_url(url, filename)
        elif choice == '4':
            source_path = input("소스 파일 경로를 입력하세요: ").strip()
            if source_path:
                filename = input("새 파일명 (선택사항, 엔터로 건너뛰기): ").strip() or None
                helper.copy_from_path(source_path, filename)
        elif choice == '5':
            helper.restart_chatbot_service()
        else:
            print("잘못된 선택입니다.")

if __name__ == "__main__":
    main()