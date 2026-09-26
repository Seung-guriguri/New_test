"""Example and benchmark datasets. BibMon's ship inside the package (offline); chelo's are downloaded on demand."""
import numpy as np
import pandas as pd

# Downs & Vogel (1993) Tennessee Eastman variables.
_XMEAS = ["A 공급 유량", "D 공급 유량", "E 공급 유량", "A·C 공급 유량", "재순환 유량", "반응기 공급 유량", "반응기 압력", "반응기 액위",
          "반응기 온도", "퍼지 유량", "분리기 온도", "분리기 액위", "분리기 압력", "분리기 하부 유량", "스트리퍼 액위", "스트리퍼 압력",
          "스트리퍼 하부 유량(제품)", "스트리퍼 온도", "스트리퍼 스팀 유량", "압축기 동력", "반응기 냉각수 출구 온도", "응축기 냉각수 출구 온도"] + \
         [f"반응기 공급 성분 {c}" for c in "ABCDEF"] + [f"퍼지 가스 성분 {c}" for c in "ABCDEFGH"] + [f"제품 성분 {c}" for c in "DEFGH"]
_XMV = ["D 공급 유량 (조작)", "E 공급 유량 (조작)", "A 공급 유량 (조작)", "A·C 공급 유량 (조작)", "압축기 재순환 밸브", "퍼지 밸브",
        "분리기 액 유량 (조작)", "스트리퍼 제품 유량 (조작)", "스트리퍼 스팀 밸브", "반응기 냉각수 유량", "응축기 냉각수 유량"]
TE_NAMES = {**{f"XMEAS({i})": f"XMEAS({i}) {n}" for i, n in enumerate(_XMEAS, 1)},
            **{f"XMV({i})": f"XMV({i}) {n}" for i, n in enumerate(_XMV, 1)}}
TE_FAULTS = {
    0: "정상 운전 (고장 없음)",
    1: "A/C 공급비 계단 변화 (B 조성 일정, 흐름 4)", 2: "B 조성 계단 변화 (A/C 비 일정, 흐름 4)", 3: "D 공급 온도 계단 변화 (흐름 2)",
    4: "반응기 냉각수 입구 온도 계단 변화", 5: "응축기 냉각수 입구 온도 계단 변화", 6: "A 공급 손실 (흐름 1)",
    7: "C 헤더 압력 손실 (흐름 4)", 8: "A·B·C 공급 조성 랜덤 변동 (흐름 4)", 9: "D 공급 온도 랜덤 변동 (흐름 2)",
    10: "C 공급 온도 랜덤 변동 (흐름 4)", 11: "반응기 냉각수 입구 온도 랜덤 변동", 12: "응축기 냉각수 입구 온도 랜덤 변동",
    13: "반응 속도(kinetics) 서서히 변화", 14: "반응기 냉각수 밸브 고착", 15: "응축기 냉각수 밸브 고착",
    16: "원인 미공개", 17: "원인 미공개", 18: "원인 미공개", 19: "원인 미공개", 20: "원인 미공개",
    21: "흐름 4 밸브가 정상 위치에 고정",
}
TE_FAULT_SAMPLE = 160  # in the standard test sets the fault is introduced after 8 h (160 samples of 3 min)


def tennessee_eastman(fault):
    """Normal training run followed by the fault's test run on one 3-minute time axis. Returns (df, fault_start or None)."""
    import bibmon
    train, test = bibmon.load_tennessee_eastman(train_id=0, test_id=fault)
    test.index = train.index[-1] + pd.to_timedelta(np.arange(1, len(test) + 1) * 3, unit="min")
    df = pd.concat([train, test]).rename(columns=TE_NAMES)
    df.index.name = "시간"
    return df, (test.index[TE_FAULT_SAMPLE] if fault else None)


def petrobras_real():
    import bibmon
    df = bibmon.load_real_data()
    # Same rule as uploaded files: status strings such as "Inp OutRange" become missing values.
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    df.index.name = "시간"
    return df


CHELO = {"cstr": "CSTR 반응기 농도 (chelo · 인터넷)", "coal": "석탄화력 발전소 열성능 (chelo · 인터넷 · Kaggle 인증)"}


def chelo_dataset(name):
    """Downloads into ~/.chelo on first use. CSTR has no timestamps and chelo drops the coal plant's dates (and rows with
    missing values), so both get a sequential index: row order is real, the time spacing is not."""
    if name == "cstr":
        from chelo.datasets.cstr_dataset import CSTRDataset as cls
        freq = "min"
    else:
        from chelo.datasets.coal_fired_plant import CoalFiredPlantDataset as cls
        freq = "D"
    ds = cls()
    ds.load_data()
    df = pd.DataFrame(ds.raw_targets).apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    df.index = pd.date_range("2020-01-01", periods=len(df), freq=freq, name="시간")
    return df


if __name__ == "__main__":
    df, start = tennessee_eastman(1)
    assert df.shape == (1460, 52) and df.index.is_monotonic_increasing and df.index.is_unique
    assert start == df.index[500 + TE_FAULT_SAMPLE] and "XMEAS(9) 반응기 온도" in df and "XMV(11) 응축기 냉각수 유량" in df
    assert (df.index.to_series().diff().dropna() == pd.Timedelta("3min")).all()
    assert tennessee_eastman(0)[1] is None and len(TE_FAULTS) == 22 and len(TE_NAMES) == 52
    real = petrobras_real()
    assert real.shape[0] > 3000 and isinstance(real.index, pd.DatetimeIndex)
    assert all(pd.api.types.is_numeric_dtype(t) for t in real.dtypes) and real.resample("1h").mean().shape[0] > 0
    print("ok")
