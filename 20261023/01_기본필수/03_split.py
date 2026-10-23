"""03 기본필수: 20행을 train 12행, valid 4행, test 4행으로 나눕니다.

실습 03: 머신러닝 데이터 분할

목적:
전체 데이터 20행을 다음과 같이 나누기

train = 12행 (60%) → 모델 학습용
valid =  4행 (20%) → 모델 성능 검증 및 설정 조정용
test  =  4행 (20%) → 최종 성능 평가용

왜 나누는가?
모델이 학습한 데이터만 잘 맞히는 것이 아니라 처음 보는 데이터도 잘 예측하는지 확인하기 위해서
"""
# ==========================================================
# 1. 필요한 라이브러리 가져오기
# ==========================================================
from pathlib import Path                                # 파일경로 처리하기 위한 표준 라이브러리 
import pandas as pd                                     # CSV를 DataFrame(표 형태의 데이터)으로 읽기 위한 라이브러리
from sklearn.model_selection import train_test_split   
# scikit-learn에서 데이터 분할 함수를 가져옴.
# sklearn
# 왜 사용하는가?
# 직접 데이터를 섞고 비율에 따라 나누는 코드를 작성하지 않아도 train_test_split()이 자동으로 데이터를 나누어 주기 때문임.
#
# sklearn = 머신러닝 관련 기능을 제공하는 라이브러리
# model_selection = 모델 평가와 데이터 분할 등을 위한 모듈


# ==========================================================
# 2. CSV 파일 읽기
# ==========================================================
# 현재 Python 파일이 있는 폴더의 절대경로를 가져옴.
HERE = Path(__file__).resolve().parent
# CSV 파일을 Pandas DataFrame으로 변환함.
# 왜 Pandas를 사용하는가? CSV의 각 행을 하나의 데이터로 관리하고, failed 같은 특정 열을 쉽게 추출하기 위해서
data = pd.read_csv(HERE / "requests.csv")

# ==========================================================
# 3. 전체 데이터에서 test 데이터 분리
# ==========================================================

# test_size=0.2는 전체 20행 중 4행을 따로 보관한다는 뜻입니다.
# 완료 B03-1: test 비율을 전체의 20%로 바꾸세요. 소수로 적습니다.
# 전체 데이터의 20%를 테스트 데이터로 사용함.
# 20 * 0.2 = 4행
# 나머지 80%인 16행은 학습/검증 데이터로 사용함.
test_ratio = 0.2



# train_test_split()의 역할:
# 데이터를 무작위로 섞어서 두 그룹으로 분리함.
#
# 매개변수 설명:
#
# data
#   → 분할할 전체 DataFrame
#
# test_size=test_ratio
#   → 두 번째 반환값에 할당할 데이터의 비율
#   → 여기서는 20%인 4행
#
# random_state=42
#   → 난수 시드를 고정함.
#   → 프로그램을 여러 번 실행해도 같은 방식으로 분할됨.
#   → 42는 특별한 숫자가 아니라 임의로 선택한 값임.
#
# stratify=data["failed"]
#   → failed 열의 값 비율을 최대한 유지하면서 분할함.
#
#   예를 들어:
#   성공(0) 15건, 실패(1) 5건이라면
#   train/test에도 이 비율이 비슷하게 유지되도록 함.
#
# 반환값:
#   train_valid = 16행
#   test        = 4행
#
# Python은 반환값 여러 개를 각각 변수에 대입할 수 있음.
# 이를 언패킹(Unpacking)이라고 함.
train_valid, test = train_test_split(
    data, test_size=test_ratio, random_state=42, stratify=data["failed"])

# ==========================================================
# 4. train_valid를 다시 train과 valid로 분리
# ==========================================================
# 앞에서 분리한 train_valid는 총 16행임.
#
# 이 중 25%를 valid로 사용함.
#
# 16 * 0.25 = 4행 (valid)
# 16 * 0.75 = 12행 (train)
#
# 주의:
# test_size=0.25는 전체 20행의 25%가 아님!
# 현재 전달한 train_valid 16행의 25%라는 뜻임.
#
# stratify=train_valid["failed"]
#   → 이번에도 성공/실패 비율을 최대한 유지함.
train, valid = train_test_split(
    train_valid, test_size=0.25, random_state=42, stratify=train_valid["failed"])

# ==========================================================
# 5. 데이터 분할 결과 확인
# ==========================================================
# len()은 DataFrame의 행 개수를 반환함.
#
# 왜 확인하는가? 의도한 비율대로 데이터가 분리되었는지 검증하기 위해서임.
print("train:", len(train))
print("valid:", len(valid))
print("test:", len(test))
# valid["request_id"]
#   → valid DataFrame에서 request_id 열만 추출함.
#   → 반환 자료형은 Pandas Series임.
#
# .tolist()
#   → Series를 일반 Python List로 변환함.
#
# 왜 출력하는가? 검증용으로 어떤 요청들이 선택되었는지 확인하기 위해서
print("valid ID:", valid["request_id"].tolist())
# test 데이터는 모델 학습이나 모델 선택에 사용하지 않음.
# 최종 평가용 데이터를 미리 사용하면 모델의 실제 일반화 성능을 공정하게 평가하기 어려워지기 때문
print("test는 오늘 학습과 예측에 사용하지 않습니다.")
