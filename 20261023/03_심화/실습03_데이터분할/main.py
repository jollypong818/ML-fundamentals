"""03 심화(기존 응용): seed와 ID를 기록하고 행 순서의 영향을 비교합니다."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

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

# 제공: 정답을 합성하거나 test를 모델에 넣지 않고, 분할 소속만 기록합니다.
manifest = data[["request_id"]].copy()
manifest["split"] = ""
manifest.loc[train.index, "split"] = "train"
manifest.loc[valid.index, "split"] = "valid"
manifest.loc[test.index, "split"] = "test"
manifest.to_csv(OUTPUTS / "split_manifest.csv", index=False, encoding="utf-8-sig")
# 같은 원본을 역순으로 놓은 별도 시연입니다. 원본 파일은 바꾸지 않습니다.
reordered = data.iloc[::-1]
other_dev, other_test = train_test_split(
    reordered, test_size=0.2, random_state=42, stratify=reordered["failed"])
other_train, other_valid = train_test_split(
    other_dev, test_size=0.25, random_state=42, stratify=other_dev["failed"])
# TODO H03-1: 두 train ID 집합의 대칭차집합(^) 개수를 구하세요.
changed = 0
print("분할:", len(train), len(valid), len(test))
print("valid 클래스:", valid["failed"].value_counts().sort_index().to_dict())
print("행 역순 뒤 train 소속 변경 ID 수:", changed)
print("저장: outputs/split_manifest.csv / seed 42·42만 아니라 원본과 행 순서도 보존")
