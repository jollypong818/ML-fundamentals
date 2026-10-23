"""07~08 기본필수: 이 파일 하나를 완성하고 같은 파일로 재실행합니다."""
"""
실습 07~08: 머신러닝 파이프라인 구성 및 성능평가

목적:
1. CSV 파일을 읽어 머신러닝 데이터 준비
2. train / valid / test 데이터 분할
3. 입력 데이터(X)와 정답(y) 분리
4. 숫자형 / 범주형 데이터 전처리
5. Pipeline으로 전처리와 모델 연결
6. DummyClassifier 학습
7. valid 데이터 예측 및 정확도 평가
8. 예측 결과를 CSV 파일로 저장

주의:
DummyClassifier는 데이터의 특징을 학습하는 모델이 아니라
다른 모델과 성능을 비교하기 위한 기준 모델(Baseline)임.
"""


# ==========================================================
# 1. 필요한 라이브러리 가져오기
# ==========================================================

# 현재 Python 파일과 같은 폴더에 있는 CSV를 찾고 결과 CSV를 저장할 경로를 만들기 위해 사용함.
from pathlib import Path

# Pandas:
# CSV 파일을 DataFrame으로 읽고, 필요한 열을 선택하거나 결과를 CSV로 저장하기 위해 사용
import pandas as pd

# train_test_split:
# 데이터를 학습용, 검증용, 테스트용으로 나누기 위한 함수
from sklearn.model_selection import train_test_split

# StandardScaler:
# 숫자 데이터를 평균 0, 표준편차 1 기준으로 표준화함.
#
# OneHotEncoder:
# 문자열 등의 범주형 데이터를 0과 1로 변환함.
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ColumnTransformer:
# 각 열에 서로 다른 전처리 방법을 적용하고 결과를 하나의 데이터로 합쳐주는 클래스
from sklearn.compose import ColumnTransformer

# Pipeline:
# 여러 처리 단계를 하나의 객체로 연결하는 클래스
#
# 왜 사용하는가?
# 전처리와 모델 학습을 따로 실행하지 않고 하나의 흐름으로 관리할 수 있기 때문임.
from sklearn.pipeline import Pipeline

# DummyClassifier:
# 단순한 규칙으로 예측하는 기준 분류 모델
#
# strategy="most_frequent"를 사용하면 학습 데이터에서 가장 많이 등장한 정답만 예측함.
# 실제 특징과 정답 사이의 관계를 학습하지 않음.
from sklearn.dummy import DummyClassifier

# 제공: 이 Python 파일과 같은 폴더에 있는 CSV를 읽습니다. 작업 폴더를 바꾸어도 됩니다.
HERE = Path(__file__).resolve().parent
data = pd.read_csv(HERE / "requests.csv")


# ==========================================================
# 3. train / valid / test 데이터 분할
# ==========================================================
# 제공: 먼저 20%를 test로 보관하고, 남은 80% 중 25%를 valid로 나눕니다.
# random_state는 재현용 고정값입니다. stratify는 성공/실패 비율을 고려합니다.
# 첫 번째 분할:
# 전체 데이터의 20%를 test로 분리함.
#
# data:
#   분할할 전체 데이터
#
# test_size=0.2:
#   전체의 20%를 test로 사용
#
# random_state=42:
#   난수 시드를 고정해 동일한 분할을 재현
#
# stratify=data["failed"]:
#   성공(0)과 실패(1)의 비율을 최대한 유지
#
# 전체 20행 기준:
# train_valid = 16행
# test        = 4행
train_valid, test = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["failed"])
# 두 번째 분할:
# train_valid 16행 중 25%를 valid로 분리
#
# train = 12행
# valid = 4행
#
# 주의:
# test_size=0.25는 전체 데이터의 25%가 아니라 train_valid 데이터의 25%를 의미함.
train, valid = train_test_split(
    train_valid, test_size=0.25, random_state=42, stratify=train_valid["failed"])

# 최종 분할:
# train = 12행 (60%)
# valid =  4행 (20%)
# test  =  4행 (20%)
#
# test 데이터는 최종 평가용이므로 이번 실습에서는 개수 확인 외에 사용하지 않음.

# ==========================================================
# 4. 입력 데이터(X)와 정답 데이터(y) 분리
# ==========================================================
# features:
# 머신러닝 모델의 입력으로 사용할 열 이름 목록
#
# 왜 request_id와 failed는 제외하는가?
#
# request_id:
#   단순 식별자이므로 예측에 사용할 특징으로 적절하지 않음.
#
# failed:
#   모델이 예측해야 하는 정답이므로 입력에 넣으면 안 됨.
features = ["input_chars", "task_type"]
# X_train:
# 학습에 사용할 입력 데이터
#
# train[features]:
# DataFrame에서 features에 지정한 열만 선택함.
#
# 결과:
# input_chars, task_type
X_train = train[features]
# y_train:
# 모델이 학습할 정답 데이터
#
# 0 = 성공
# 1 = 실패
#
# failed 열 하나만 선택하므로 Pandas Series가 반환됨.
y_train = train["failed"]
# X_valid:
# 학습이 끝난 모델의 예측 성능을 확인할 입력 데이터
#
# train과 동일한 입력 열을 사용함.
X_valid = valid[features]


# ==========================================================
# 5. 데이터 전처리 규칙 생성
# ==========================================================

# ColumnTransformer:
# 입력 데이터의 열마다 서로 다른 전처리를 적용함.
#
# 각 Tuple의 구조:
# ("작업 이름", 변환기 객체, 적용할 열 목록)
preprocessor = ColumnTransformer([
    # 숫자형 열:
    # input_chars를 StandardScaler로 표준화
    #
    # fit 시 train의 평균과 표준편차를 계산함.
    ("num", StandardScaler(), ["input_chars"]),
    # 범주형 열:
    # task_type을 OneHotEncoder로 변환
    #
    # 예:
    # qa      → [1, 0]
    # summary → [0, 1]
    #
    # sparse_output=False:
    # 일반 NumPy 배열 형태로 결과를 반환함.
    ("cat", OneHotEncoder(sparse_output=False), ["task_type"]),
])



# ==========================================================
# 6. 머신러닝 Pipeline 구성
# ==========================================================

# Dummy는 가장 많은 정답만 반복하는 비교 출발점입니다. 실제 AI 성능이 아닙니다.
# Pipeline:
# 여러 처리 단계를 순서대로 연결하는 클래스
#
# 첫 번째 단계:
# "pre" → ColumnTransformer를 이용한 데이터 전처리
#
# 두 번째 단계:
# "model" → DummyClassifier를 이용한 모델 학습
#
# Pipeline의 장점:
# 1. 전처리와 학습을 하나의 객체로 관리
# 2. train에만 전처리 기준을 학습하도록 구성 가능
# 3. 예측할 때도 동일한 전처리 자동 적용
#
# DummyClassifier(strategy="most_frequent"):
# train에서 가장 많이 등장한 정답을 반복 예측함.
#
# 예:
# train 정답이
# 0, 0, 0, 1, 0, 1 이라면
#
# 가장 많은 정답은 0이므로 이후 모든 입력에 대해 0을 예측함.
pipeline = Pipeline([("pre", preprocessor),
                     ("model", DummyClassifier(strategy="most_frequent"))])

# ==========================================================
# 7. Pipeline 학습
# ==========================================================

# pipeline.fit(X_train, y_train):
# 학습 입력 X_train과 정답 y_train을 전달함.
#
# Pipeline 내부에서 다음 순서로 실행됨.
#
# 1. preprocessor.fit_transform(X_train)
#    → 숫자형/범주형 데이터 전처리 규칙 학습 및 변환
#
# 2. DummyClassifier.fit(변환된 X_train, y_train)
#    → y_train에서 가장 많이 등장하는 정답 확인
#
# 중요한 점:
# valid나 test 데이터로 학습하지 않음.
#
# 완료 B07-1: pipeline.fit(X_train, y_train)으로 train에서만 학습하세요. pass를 지우고 작성합니다.
pipeline.fit(X_train, y_train)

# ==========================================================
# 8. valid 데이터 예측 
# ==========================================================

# pipeline.predict(X_valid):
# 학습한 Pipeline으로 검증 데이터를 예측함.
#
# 내부 처리:
#
# 1. preprocessor.transform(X_valid)
#    → train에서 학습한 전처리 기준을 적용
#
# 2. DummyClassifier.predict()
#    → 가장 많이 등장한 정답으로 예측
#
# 반환값:
# NumPy ndarray 형태의 예측 결과
#
# 실제 값은 train 데이터의 정답 분포에 따라 달라짐.
# 완료 B07-2: pipeline.predict(X_valid)로 검증 입력을 예측하세요. 현재 값은 임시값입니다.
prediction = pipeline.predict(X_valid)

# ==========================================================
# 9. 예측 결과를 DataFrame으로 정리
# ==========================================================

# valid[["request_id", "failed"]]:
# 검증 데이터에서 ID와 실제 정답만 선택함.
#
# .copy():
# 원본 valid와 별도로 수정할 수 있는 복사본을 생성함.
#
# 왜 복사하는가?
# 예측 결과를 추가할 때 원본 valid를 직접 변경하지 않기 위해서임.
# 제공: 예측을 원래 ID·정답과 나란히 놓고, 이 파일 옆에 저장합니다.
result = valid[["request_id", "failed"]].copy()

# predicted라는 새로운 열을 생성하고
# 모델이 예측한 결과를 저장함.
#
# 예:
#
# request_id | failed | predicted
# -----------|--------|----------
# 3          | 0      | 0
# 8          | 1      | 0
#
# failed    = 실제 정답
# predicted = 모델의 예측값
result["predicted"] = prediction

# ==========================================================
# 10. 예측 결과를 CSV 파일로 저장
# ==========================================================

# to_csv():
# Pandas DataFrame을 CSV 파일로 저장하는 메서드
#
# HERE / "valid_predictions.csv":
# 현재 Python 파일과 같은 폴더에 결과 저장
#
# index=False:
# Pandas의 행 인덱스를 CSV에 추가하지 않음.
#
# encoding="utf-8-sig":
# UTF-8 BOM을 포함해 저장함.
# Excel 등에서 한글을 읽을 때 호환성을 높이기 위해 사용함.
result.to_csv(HERE / "valid_predictions.csv", index=False, encoding="utf-8-sig")
# ==========================================================
# 11. 모델 정확도(Accuracy) 계산
# ==========================================================

# result["failed"] == result["predicted"]:
# 실제 정답과 예측 결과를 행별로 비교함.
#
# 일치하면 True
# 일치하지 않으면 False
# .mean():
# Boolean에서 True는 1, False는 0으로 계산되므로
# 전체 중 True의 비율을 구할 수 있음.
#
# 주의: 정확도가 높아도 실패 데이터가 적은 경우에는 실패 예측 성능이 낮을 수 있음.
accuracy = (result["failed"] == result["predicted"]).mean()

# ==========================================================
# 12. 실행 결과 출력
# ==========================================================

# train, valid, test의 행 개수 출력
print("분할:", len(train), len(valid), len(test))


# to_string(index=False):
# DataFrame 전체를 문자열 형태로 출력
#
# index=False:
# DataFrame의 행 인덱스를 표시하지 않음.
print(result.to_string(index=False))
# round(accuracy, 2):
# 정확도를 소수점 둘째 자리까지 반올림
print("valid 정확도:", round(accuracy, 2))
# 결과 파일이 저장되었다는 메시지 출력 위에서 예외가 발생하면 아래 코드는 실행되지 않음.
print("저장: valid_predictions.csv / test는 사용하지 않았습니다.")
