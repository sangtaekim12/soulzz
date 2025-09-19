#!/usr/bin/env python3
"""
샘플 관광 데이터를 생성하는 스크립트
실제 PDF가 없는 경우 테스트용 텍스트 데이터를 생성합니다.
"""

import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph

def create_sample_pdfs():
    """샘플 관광 정보 PDF 파일들을 생성"""
    
    # data 폴더 생성
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # 샘플 관광 정보 데이터
    sample_data = {
        "seoul_tourism.pdf": [
            "서울 관광 가이드",
            "",
            "경복궁은 조선왕조의 정궁으로 1395년에 창건되었습니다.",
            "경복궁에서는 매일 수문장 교대식이 진행되며, 한복을 입고 방문하면 입장료가 무료입니다.",
            "",
            "남산서울타워는 서울의 대표적인 랜드마크입니다.",
            "남산타워에서는 서울 시내 전경을 한눈에 볼 수 있으며, 특히 야경이 아름답습니다.",
            "",
            "명동은 서울의 대표적인 쇼핑거리입니다.",
            "명동에는 다양한 브랜드 매장과 화장품 가게, 맛집들이 밀집되어 있습니다.",
            "",
            "인사동은 전통문화의 거리로 유명합니다.",
            "인사동에서는 전통 차를 마실 수 있는 찻집과 한국 전통 공예품을 구입할 수 있습니다.",
        ],
        
        "busan_guide.pdf": [
            "부산 여행 가이드",
            "",
            "해운대 해수욕장은 부산의 대표 관광지입니다.",
            "해운대는 넓은 백사장과 깨끗한 바다로 유명하며, 여름철 피서지로 인기가 높습니다.",
            "",
            "광안리 해수욕장은 광안대교의 야경으로 유명합니다.",
            "광안리에서는 다양한 해산물 요리를 맛볼 수 있으며, 카페와 바가 많습니다.",
            "",
            "감천문화마을은 '한국의 마추픽추'로 불립니다.",
            "감천문화마을은 알록달록한 집들이 계단식으로 배치된 독특한 풍경을 자랑합니다.",
            "",
            "자갈치시장은 부산 최대의 수산시장입니다.",
            "자갈치시장에서는 신선한 회와 각종 해산물을 저렴하게 맛볼 수 있습니다.",
        ],
        
        "jeju_attractions.pdf": [
            "제주도 명소 안내",
            "",
            "한라산은 제주도의 상징이자 대한민국 최고봉입니다.",
            "한라산 등반은 계절별로 다른 아름다운 풍경을 선사하며, 백록담까지의 코스가 인기입니다.",
            "",
            "성산일출봉은 제주도 동쪽에 위치한 화산체입니다.",
            "성산일출봉에서 보는 일출은 제주도 대표 관광 포인트 중 하나입니다.",
            "",
            "우도는 제주도 동쪽 바다에 있는 작은 섬입니다.",
            "우도에서는 아름다운 해변과 신선한 해산물, 유명한 땅콩아이스크림을 맛볼 수 있습니다.",
            "",
            "중문관광단지는 제주도 남쪽의 대표적인 휴양지입니다.",
            "중문에는 다양한 리조트와 골프장, 테마파크가 위치해 있습니다.",
        ],
        
        "korean_food_guide.pdf": [
            "한국 음식 관광 가이드",
            "",
            "김치는 한국의 대표적인 발효식품입니다.",
            "김치는 배추, 무, 오이 등 다양한 채소로 만들며, 지역마다 맛과 종류가 다릅니다.",
            "",
            "불고기는 한국의 전통 구이 요리입니다.",
            "불고기는 얇게 썬 소고기를 달콤한 양념에 재워 구운 요리로 외국인들에게 인기가 높습니다.",
            "",
            "비빔밥은 다양한 나물과 밥을 섞어 먹는 한국 요리입니다.",
            "비빔밥은 영양 균형이 뛰어나며, 고추장과 함께 비벼 먹는 것이 특징입니다.",
            "",
            "한정식은 한국의 전통 코스 요리입니다.",
            "한정식에서는 다양한 반찬과 함께 계절 식재료를 활용한 요리를 맛볼 수 있습니다.",
        ],
        
        "cultural_sites.pdf": [
            "한국 문화유산 안내",
            "",
            "창덕궁은 조선시대 이궁으로 유네스코 세계문화유산입니다.",
            "창덕궁의 후원은 자연과 조화를 이룬 한국 전통 조경의 백미로 평가받습니다.",
            "",
            "불국사는 신라시대의 대표적인 사찰입니다.",
            "불국사에는 다보탑과 석가탑이라는 두 개의 국보급 석탑이 있습니다.",
            "",
            "석굴암은 불국사와 함께 경주의 대표적인 문화유산입니다.",
            "석굴암의 본존불은 동양 최고의 조각 작품으로 인정받고 있습니다.",
            "",
            "종묘는 조선왕조 역대 왕과 왕비의 신주를 모신 사당입니다.",
            "종묘제례악은 유네스코 무형문화유산으로 등록된 전통 궁중음악입니다.",
        ]
    }
    
    # PDF 파일 생성
    for filename, content in sample_data.items():
        filepath = os.path.join(data_dir, filename)
        
        # 간단한 텍스트 PDF 생성
        c = canvas.Canvas(filepath, pagesize=letter)
        width, height = letter
        
        y_position = height - 50
        line_height = 20
        
        for line in content:
            if y_position < 50:  # 새 페이지 시작
                c.showPage()
                y_position = height - 50
                
            if line.strip():  # 빈 줄이 아닌 경우
                c.drawString(50, y_position, line)
            
            y_position -= line_height
            
        c.save()
        print(f"생성됨: {filepath}")
    
    print(f"\n총 {len(sample_data)}개의 샘플 PDF 파일이 생성되었습니다.")
    print(f"파일 위치: {os.path.abspath(data_dir)}")

if __name__ == "__main__":
    try:
        create_sample_pdfs()
    except ImportError:
        print("reportlab 패키지가 필요합니다. 다음 명령으로 설치해주세요:")
        print("pip install reportlab")
    except Exception as e:
        print(f"오류 발생: {e}")