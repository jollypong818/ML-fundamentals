"""06 심화(기존 응용): 기준선 지표와 새 요청 예측을 연결합니다."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

# 제공: data/는 입력 자료, outputs/는 실행 결과를 보관합니다. 이 파일의 위치를 기준으로 읽습니다.
HERE = Path(__file__).resolve().parent
data = pd.read_csv(HERE / "data" / "requests.csv")
# 제공: 먼저 20%를 test로 보관하고, 남은 80% 중 25%를 valid로 나눕니다.
# random_state는 재현용 고정값입니다. stratify는 성공/실패 비율을 고려합니다.
train_valid, test = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["failed"])
train, valid = train_test_split(
    train_valid, test_size=0.25, random_state=42, stratify=train_valid["failed"])
# test는 개수 확인 외에 사용하지 않습니다.
# 제공: 숫자 2열은 중앙값 대체 후 표준화합니다.
numeric = Pipeline([
    ("fill", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
# 제공: 문자 2열은 최빈값 대체 후 0/1 열로 바꿉니다.
categorical = Pipeline([
    ("fill", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("num", numeric, ["input_chars", "allowed_tools"]),
    ("cat", categorical, ["task_type", "priority"]),
])
features = ["input_chars", "allowed_tools", "task_type", "priority"]

pipeline = Pipeline([("pre", preprocessor),
                     ("model", DummyClassifier(strategy="most_frequent"))])
pipeline.fit(train[features], train["failed"])
# TODO H06-1: valid의 입력 4열을 예측하세요.
prediction = [1] * len(valid)
accuracy = (prediction == valid["failed"].to_numpy()).mean()
found = ((prediction == 1) & (valid["failed"].to_numpy() == 1)).sum()
new_request = pd.DataFrame({"input_chars": [900, 1500], "allowed_tools": [2, 3],
                            "task_type": ["translation", "qa"],
                            "priority": ["urgent", "normal"]})
# TODO H06-2: 새 요청 2행 new_request를 예측하세요.
new_prediction = pipeline.predict(valid[features].iloc[:2])
print("valid 정확도:", round(float(accuracy), 2))
print("실패 발견:", int(found), "/", int((valid["failed"] == 1).sum()))
print("새 요청 예측:", new_prediction.tolist())
