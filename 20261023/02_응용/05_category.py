"""05 응용: 결측은 최빈값으로, 처음 보는 종류는 0/1 열의 0으로 처리합니다."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

train = pd.DataFrame({"task_type": ["qa", "qa", "research", np.nan]})
valid = pd.DataFrame({"task_type": ["summary", np.nan]})
categorical = Pipeline([
    ("fill", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
categorical.fit(train)
# TODO A05-1: 학습된 categorical로 valid를 변환하세요.
valid_ready = np.zeros((2, 2))
print("대체값:", categorical["fill"].statistics_.tolist())
print("열 순서:", categorical["encode"].categories_[0].tolist())
print("valid 변환:", valid_ready.tolist())
