"""07~08 응용: 별도 모듈 없이 이 파일 하나로 완성합니다."""
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
OUTPUTS = HERE / "outputs"
OUTPUTS.mkdir(exist_ok=True)
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
# TODO A07-1: Pipeline으로 valid의 입력 4열을 예측하세요.
prediction = [1] * len(valid)
result = valid[["request_id", "failed"]].copy()
result["predicted"] = prediction
result.to_csv(OUTPUTS / "valid_predictions.csv", index=False, encoding="utf-8-sig")
# TODO A07-2: 실제값과 예측값이 같은 비율을 구하세요.
accuracy = 0
found = ((result["failed"] == 1) & (result["predicted"] == 1)).sum()
print("valid 정확도:", round(accuracy, 2))
print("실패 발견:", int(found), "/", int((result["failed"] == 1).sum()))
print("저장 행 수:", len(result))
print("저장: outputs/valid_predictions.csv / test는 사용하지 않았습니다.")
