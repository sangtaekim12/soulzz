#!/usr/bin/env python3
"""
Base64 인코딩된 파일을 디코딩하여 PDF로 저장하는 스크립트
"""

import base64
import os

def decode_base64_to_pdf(base64_string, filename):
    """Base64 문자열을 PDF 파일로 디코딩"""
    try:
        # Base64 디코딩
        pdf_data = base64.b64decode(base64_string)
        
        # data 폴더 확인
        data_dir = "./data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # 파일 저장
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(pdf_data)
        
        print(f"✅ 파일 저장 완료: {filepath}")
        print(f"📄 파일 크기: {len(pdf_data):,} bytes")
        return True
        
    except Exception as e:
        print(f"❌ 디코딩 실패: {e}")
        return False

def process_base64_files():
    """Base64 파일들을 처리"""
    print("=== Base64 PDF 디코딩 도구 ===")
    print()
    
    while True:
        print("선택하세요:")
        print("1. Base64 문자열 직접 입력")
        print("2. Base64 파일에서 읽기")
        print("3. 현재 PDF 파일들 확인")
        print("0. 종료")
        
        choice = input("\n선택 (0-3): ").strip()
        
        if choice == '0':
            break
        elif choice == '1':
            filename = input("저장할 파일명을 입력하세요 (예: document.pdf): ").strip()
            if not filename.endswith('.pdf'):
                filename += '.pdf'
            
            print("Base64 문자열을 입력하세요 (Ctrl+D로 입력 완료):")
            base64_lines = []
            try:
                while True:
                    line = input()
                    base64_lines.append(line.strip())
            except EOFError:
                pass
            
            base64_string = ''.join(base64_lines)
            if base64_string:
                decode_base64_to_pdf(base64_string, filename)
            else:
                print("❌ Base64 문자열이 입력되지 않았습니다.")
                
        elif choice == '2':
            filepath = input("Base64 파일 경로를 입력하세요: ").strip()
            filename = input("저장할 PDF 파일명을 입력하세요: ").strip()
            
            if not filename.endswith('.pdf'):
                filename += '.pdf'
            
            try:
                with open(filepath, 'r') as f:
                    base64_string = f.read().strip()
                decode_base64_to_pdf(base64_string, filename)
            except FileNotFoundError:
                print(f"❌ 파일을 찾을 수 없습니다: {filepath}")
            except Exception as e:
                print(f"❌ 파일 읽기 실패: {e}")
                
        elif choice == '3':
            data_dir = "./data"
            if os.path.exists(data_dir):
                pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
                if pdf_files:
                    print(f"\n현재 PDF 파일들 ({len(pdf_files)}개):")
                    for i, pdf in enumerate(pdf_files, 1):
                        filepath = os.path.join(data_dir, pdf)
                        size = os.path.getsize(filepath)
                        print(f"  {i}. {pdf} ({size:,} bytes)")
                else:
                    print("\n📁 data 폴더에 PDF 파일이 없습니다.")
            else:
                print("\n📁 data 폴더가 존재하지 않습니다.")
        else:
            print("❌ 잘못된 선택입니다.")
        
        print()

if __name__ == "__main__":
    process_base64_files()