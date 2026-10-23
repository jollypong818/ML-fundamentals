"""06 기본필수: 숫자 1열과 문자 1열을 함께 변환합니다."""
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

train = pd.DataFrame({"input_chars": [100, 200, 300],
                      "task_type": ["qa", "summary", "qa"]})
valid = pd.DataFrame({"input_chars": [400, 500],
                      "task_type": ["summary", "qa"]})
# 제공: 숫자에는 표준화, 문자에는 원-핫 변환을 지정합니다.
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), ["input_chars"]),
    ("cat", OneHotEncoder(sparse_output=False), ["task_type"]),
])
preprocessor.fit(train)
# TODO B06-1: preprocessor의 transform으로 valid를 변환하세요.
valid_ready = preprocessor.transform(valid)

print("변환 전:", valid.shape)
print("변환 후:", valid_ready.shape)
print(valid_ready.round(3))
