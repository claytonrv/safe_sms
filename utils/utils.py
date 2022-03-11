from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
dateformat = '%Y-%m-%d %H:%M:%S'


def fit(o):
    scaler.fit(o)
    return scaler.fit(o)


def transform(o):
    return scaler.transform(o)


def fit_transform(o):
    fit(o)
    return transform(o)