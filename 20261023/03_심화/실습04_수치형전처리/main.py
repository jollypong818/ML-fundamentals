"""04 심화(기존 응용): 수치 경로의 학습 통계가 변하지 않는지 확인합니다."""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

columns = ["input_chars", "allowed_tools"]
numeric = Pipeline([("fill", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
train_ready = numeric.fit_transform(train[columns])
median_before = numeric["fill"].statistics_.copy()
mean_before = numeric["scale"].mean_.copy()
# TODO H04-1: numeric으로 valid의 두 숫자 열을 변환하세요.
valid_ready = valid[columns].to_numpy()
median_same = np.array_equal(median_before, numeric["fill"].statistics_)
mean_same = np.array_equal(mean_before, numeric["scale"].mean_)
print("중앙값:", median_before.tolist())
print("train/valid 모양:", train_ready.shape, valid_ready.shape)
print("중앙값/평균 유지:", median_same, mean_same)
print("모든 결과가 유한수:", bool(np.isfinite(valid_ready).all()))
