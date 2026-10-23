"""
실습 06: ColumnTransformer를 이용한 데이터 전처리

목적:숫자형 데이터와 문자형 데이터를 한 번에 전처리하기.

왜 필요한가? 실제 머신러닝 데이터에는 숫자와 문자가 섞여 있기 때문임.

예를 들어:
input_chars = 100        → 숫자형 데이터
task_type   = "qa"       → 범주형 데이터

숫자와 문자는 서로 다른 전처리 방법이 필요함.

숫자형 데이터:
StandardScaler → 평균과 표준편차를 이용해 표준화

범주형 데이터:
OneHotEncoder → 문자열을 0과 1로 변환

ColumnTransformer:
각 열에 어떤 전처리 방법을 적용할지 지정하고,
변환 결과를 하나의 데이터로 합쳐주는 도구임.
"""

# ==========================================================
# 1. 필요한 라이브러리 가져오기
# ==========================================================
# Pandas:
# 표 형태의 데이터를 생성하고 처리하기 위한 라이브러리
# DataFrame을 이용해 숫자와 문자가 섞인 데이터를 관리함.
import pandas as pd

# StandardScaler:
# 숫자형 데이터의 평균과 표준편차를 계산하고 평균 0, 표준편차 1을 기준으로 표준화하는 클래스
# OneHotEncoder:
# 문자열 등의 범주형 데이터를 머신러닝 모델이 처리할 수 있는 0과 1로 변환하는 클래스
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# ColumnTransformer:
# 여러 열에 서로 다른 전처리 방법을 적용하는 클래스
#
# 왜 사용하는가? 숫자형 열과 범주형 열을 따로 처리하는 코드를 하나의 전처리 객체로 통합할 수 있기 때문임.
#
# 예:
# input_chars → StandardScaler
# task_type   → OneHotEncoder
#
# 각각 변환한 결과를 하나의 배열로 합쳐서 반환함.
from sklearn.compose import ColumnTransformer


# ==========================================================
# 2. 학습용(train), 검증용(valid) 데이터 생성
# ==========================================================
# pd.DataFrame():
# Dictionary를 이용해 행과 열이 있는 표를 생성함.
#
# Dictionary의 Key: → DataFrame의 열 이름
#
# Dictionary의 Value: → 각 열에 들어갈 데이터 목록
#
# train 데이터:
#
# input_chars | task_type
# ------------|----------
# 100         | qa
# 200         | summary
# 300         | qa
#
# 총 3행 2열
train = pd.DataFrame({"input_chars": [100, 200, 300],
                      "task_type": ["qa", "summary", "qa"]})

# valid 데이터:
#
# input_chars | task_type
# ------------|----------
# 400         | summary
# 500         | qa
#
# 총 2행 2열
#
# 왜 train과 valid를 분리하는가?
# train에서 전처리 규칙을 학습한 다음 valid에는 학습한 규칙만 적용하기 위해서임.
valid = pd.DataFrame({"input_chars": [400, 500],
                      "task_type": ["summary", "qa"]})


# ==========================================================
# 3. ColumnTransformer 객체 생성
# ==========================================================
# ColumnTransformer():
# 각 열에 어떤 전처리 방법을 적용할지 설정함.
#
# 전달하는 값은 Tuple들의 List임.
#
# 각 Tuple의 구조:
# ("이름", 변환기 객체, 적용할 열 목록)
#
# 첫 번째 요소:
#   전처리 작업을 구분하기 위한 이름
#
# 두 번째 요소:
#   실제 전처리를 수행할 객체
#
# 세 번째 요소:
#   전처리를 적용할 DataFrame의 열 이름
#
# 여기서는 전처리 작업을 두 개 등록함.

# 제공: 숫자에는 표준화, 문자에는 원-핫 변환을 지정합니다.
preprocessor = ColumnTransformer([
    # ------------------------------------------------------
    # 첫 번째 전처리: 숫자형 열 표준화
    # ------------------------------------------------------

    # "num":
    #   전처리 작업의 이름(개발자가 자유롭게 지정할 수 있음.)
    #
    # StandardScaler():
    #   숫자의 평균과 표준편차를 이용해 표준화함.
    #
    # ["input_chars"]:
    #   input_chars 열에만 적용한다는 뜻
    #
    # 결과:
    #   input_chars의 값이 표준화된 숫자로 변환됨.
    ("num", StandardScaler(), ["input_chars"]),

    # ------------------------------------------------------
    # 두 번째 전처리: 범주형 열 원-핫 인코딩
    # ------------------------------------------------------

    # "cat":
    #   범주형 전처리 작업의 이름
    #
    # OneHotEncoder(sparse_output=False):
    #   문자열을 0과 1로 변환함.
    #
    # sparse_output=False:
    #   희소 행렬이 아닌 일반 NumPy 배열로 반환하도록 설정
    #
    # ["task_type"]:
    #   task_type 열에만 적용한다는 뜻
    #
    # train에서 발견되는 범주:
    #   qa
    #   summary
    #
    # 변환 결과:
    #   qa      → [1, 0]
    #   summary → [0, 1]
    ("cat", OneHotEncoder(sparse_output=False), ["task_type"]),
])

# ==========================================================
# 4. train 데이터로 전처리 규칙 학습
# ==========================================================
# fit(train):
# ColumnTransformer에 등록된 각 변환기의 fit()을 실행함.
#
# 내부적으로 다음 작업이 수행됨.
#
# 1. StandardScaler.fit()
#    → input_chars 열에서 평균과 표준편차 계산
#
#    train = [100, 200, 300]
#    평균 = 200
#    표준편차 ≈ 81.65
#
# 2. OneHotEncoder.fit()
#    → task_type 열에서 범주 목록 학습
#
#    범주:
#    ['qa', 'summary']
#
# 중요:
# fit()은 전처리 규칙만 학습함. 아직 valid 데이터는 변환하지 않음.
preprocessor.fit(train)



# ==========================================================
# 5. valid 데이터 변환
# ==========================================================
# transform(valid):
# train에서 학습한 전처리 규칙을 valid에 적용함.
#
# 내부적으로 다음 두 작업이 실행됨.
#
# 1. input_chars → StandardScaler.transform()
#
#    400 → (400 - 200) / 81.65 ≈ 2.449
#    500 → (500 - 200) / 81.65 ≈ 3.674
#
# 2. task_type → OneHotEncoder.transform()
#
#    summary → [0, 1]
#    qa      → [1, 0]
#
# ColumnTransformer는 두 변환 결과를
# 가로 방향으로 합쳐서 반환함.
#
# 최종 결과:
#
# input_chars | qa | summary
# ------------|----|--------
# 2.449       | 0  | 1
# 3.674       | 1  | 0
#
# 결과는 NumPy ndarray 형태임.
#
# 주의:
# valid에는 다시 fit()하지 않음.
# train에서 학습한 기준을 그대로 적용해야 하기 때문임.
# 완료 B06-1: preprocessor의 transform으로 valid를 변환하세요.
valid_ready = preprocessor.transform(valid)

# ==========================================================
# 6. 변환 전후 데이터 구조 확인
# ==========================================================

# valid.shape:
# DataFrame의 행과 열 개수를 반환함.
#
# 원본 valid는:
# 2행 2열
#
# 출력: (2, 2)
print("변환 전:", valid.shape)

# valid_ready.shape:
# 변환된 데이터의 행과 열 개수를 반환함.
#
# 변환 후:
# input_chars → 1열
# task_type   → 2열 (qa, summary)
#
# 총 3열이 됨.
#
# 출력: (2, 3)
print("변환 후:", valid_ready.shape)

# round(3):
# NumPy 배열의 각 숫자를 소수점 셋째 자리까지 반올림함.
#
# 왜 사용하는가?
# 표준화 결과에 소수점이 길게 나타날 수 있으므로 출력 결과를 보기 쉽게 만들기 위해서임.
#
# 원본 valid_ready 배열은 변경하지 않음.
print(valid_ready.round(3))
