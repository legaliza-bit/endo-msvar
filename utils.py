def qtm(q):
    if q == 1:
        return "01"
    elif q == 2:
        return "04"
    elif q == 3:
        return "07"
    else:
        return "10"


def mtm(m):
    if len(str(m)) == 1:
        return "0" + str(m)
    return str(m)


def assemble_date(df, q=True):
    y = df.copy()
    if q:
        y["date"] = y["year"].astype(str) + "-" + y["q"].apply(qtm) + "-01"
    else:
        y["date"] = y["year"].astype(str) + "-" + y["m"].apply(mtm) + "-01"
    return y


def transform_data(y, X_m, max_date="2023-01-01", n_points=12, n_lags=2):
    y = assemble_date(y.copy()).sort_values(by="date").copy()
    X_m = assemble_date(X_m.copy(), q=False).sort_values(by="date")

    y = y[y["date"] <= max_date].tail(n_points + n_lags)
    start_date = y["date"].iloc[0]
    y = y.drop(columns=["year", "q", "date"])
    y_cols = y.columns.values
    endog = y.to_numpy()
    X_m = X_m[(X_m["date"] <= max_date)].tail(n_points * 3 + 5)
    latent_start_date = X_m["date"].iloc[0]
    X_m = X_m.drop(columns=["year", "m", "date"])
    x_cols = X_m.columns.values
    high_freq = X_m.to_numpy()

    return {
        "endog": endog,
        "high_freq": high_freq,
        "info": {
            "start_date": start_date,
            "cols": y_cols,
            "latent_start_date": latent_start_date,
            "x_cols": x_cols,
        },
    }
