import streamlit as st
import pandas as pd
import numpy as np
import requests
import zipfile
import io
from pathlib import Path
from datetime import datetime, date, timedelta

st.set_page_config(
    page_title="NSE Bullish Breakout Screener",
    page_icon="🚀",
    layout="wide",
)

DATA_DIR = Path("data")
STOCKLIST_FILE = Path("stocklist.txt")

DEFAULT_PRD = 5
DEFAULT_BO_LEN = 200
DEFAULT_CWIDTH_PCT = 3.0
DEFAULT_MINTEST = 2
HISTORY_TRADING_DAYS = 260

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


@st.cache_data
def load_stocklist():
    if not STOCKLIST_FILE.exists():
        return set()

    symbols = set()
    with open(STOCKLIST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            symbol = line.strip().upper()
            if symbol and not set(symbol) <= {"-"}:
                symbols.add(symbol)
    return symbols


def cached_trading_dates():
    dates = []
    DATA_DIR.mkdir(exist_ok=True)

    for f in DATA_DIR.glob("*.csv"):
        try:
            dates.append(datetime.strptime(f.stem, "%Y%m%d").date())
        except ValueError:
            pass

    return sorted(set(dates))


def previous_trading_day(d):
    candidates = [x for x in cached_trading_dates() if x < d]
    return max(candidates) if candidates else None


def next_trading_day(d):
    candidates = [x for x in cached_trading_dates() if x > d]
    return min(candidates) if candidates else None


def nse_url(dt):
    return (
        "https://nsearchives.nseindia.com/content/cm/"
        f"BhavCopy_NSE_CM_0_0_0_{dt.strftime('%Y%m%d')}_F_0000.csv.zip"
    )


def download_nse_day(dt, session):
    DATA_DIR.mkdir(exist_ok=True)
    target = DATA_DIR / f"{dt.strftime('%Y%m%d')}.csv"

    if target.exists():
        return True, "cached"

    try:
        r = session.get(nse_url(dt), timeout=30)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"

        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csvs:
                return False, "CSV not found in ZIP"
            target.write_bytes(z.read(csvs[0]))

        return True, "downloaded"
    except Exception as e:
        return False, str(e)


def ensure_nse_history(target_date):
    session = requests.Session()
    session.headers.update(HEADERS)

    required = HISTORY_TRADING_DAYS
    found = 0
    checked = 0
    downloaded = 0
    dt = target_date

    box = st.empty()
    bar = st.progress(0)

    while found < required and checked < required * 5:
        checked += 1
        path = DATA_DIR / f"{dt.strftime('%Y%m%d')}.csv"

        if path.exists():
            found += 1
        else:
            ok, status = download_nse_day(dt, session)
            if ok:
                found += 1
                if status == "downloaded":
                    downloaded += 1

        bar.progress(min(found / required, 1.0))
        box.write(
            f"NSE history: {found}/{required} trading days ready • "
            f"checking {dt.strftime('%d-%b-%Y')}"
        )
        dt -= timedelta(days=1)

    box.empty()
    bar.empty()

    target_exists = (
        DATA_DIR / f"{target_date.strftime('%Y%m%d')}.csv"
    ).exists()

    return found, downloaded, target_exists


@st.cache_data
def load_nse_data():
    frames = []

    for file in sorted(DATA_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(file, low_memory=False)

            required = {
                "TradDt", "TckrSymb", "FinInstrmTp", "SctySrs",
                "OpnPric", "HghPric", "LwPric", "ClsPric",
                "PrvsClsgPric", "TtlTradgVol",
            }

            if not required.issubset(df.columns):
                continue

            df = df[
                (df["FinInstrmTp"] == "STK") &
                (df["SctySrs"].isin(["EQ", "BE", "BZ", "SM", "ST", "SZ"]))
            ].copy()

            df["TckrSymb"] = (
                df["TckrSymb"].astype(str).str.strip().str.upper()
            )

            frames.append(df[list(required)])

        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)

    data["TradDt"] = pd.to_datetime(data["TradDt"], errors="coerce")

    for col in [
        "OpnPric", "HghPric", "LwPric", "ClsPric",
        "PrvsClsgPric", "TtlTradgVol"
    ]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(
        subset=["TradDt", "TckrSymb", "OpnPric", "HghPric", "LwPric", "ClsPric"]
    )

    data = data.sort_values(["TckrSymb", "TradDt"])
    return data.drop_duplicates(
        subset=["TckrSymb", "TradDt"], keep="last"
    )


def find_bullish_breakout(group, target_date, prd, bo_len, cwidth_pct, mintest):
    group = group.sort_values("TradDt").reset_index(drop=True).copy()

    n = len(group)
    if n < bo_len + (prd * 2) + 5:
        return None

    high = group["HghPric"].to_numpy(float)
    low = group["LwPric"].to_numpy(float)
    close = group["ClsPric"].to_numpy(float)
    opn = group["OpnPric"].to_numpy(float)

    # Rolling values corresponding to Pine's highest/lowest over lll bars.
    rolling_high = pd.Series(high).rolling(300, min_periods=1).max().to_numpy()
    rolling_low = pd.Series(low).rolling(300, min_periods=1).min().to_numpy()

    phval = []
    phloc = []

    for i in range(n):
        pivot_i = i - prd

        # Pine ta.pivothigh(high, prd, prd)
        if pivot_i >= prd and pivot_i + prd < n:
            window = high[pivot_i-prd:pivot_i+prd+1]
            if np.isfinite(high[pivot_i]) and high[pivot_i] == np.nanmax(window):
                phval.insert(0, float(high[pivot_i]))
                phloc.insert(0, int(pivot_i))

                while phloc and i - phloc[-1] > bo_len:
                    phloc.pop()
                    phval.pop()

        if i < prd:
            continue

        chwidth = (rolling_high[i] - rolling_low[i]) * (cwidth_pct / 100.0)
        hgst = np.nanmax(high[i-prd:i])

        # Pine bullish candle + close above previous Period highs.
        if not (close[i] > opn[i] and close[i] > hgst):
            continue

        if len(phval) < mintest:
            continue

        bomax = phval[0]
        xx = 0

        for x in range(len(phval)):
            if phval[x] >= close[i]:
                break
            xx = x
            bomax = max(bomax, phval[x])

        if xx < mintest or opn[i] > bomax:
            continue

        num = 0
        bostart = i

        for x in range(xx + 1):
            if phval[x] <= bomax and phval[x] >= bomax - chwidth:
                num += 1
                bostart = phloc[x]

        if num < mintest or hgst >= bomax:
            continue

        if group.loc[i, "TradDt"].date() == target_date:
            close_price = float(group.loc[i, "ClsPric"])
            prev_close = float(group.loc[i, "PrvsClsgPric"])
            volume = float(group.loc[i, "TtlTradgVol"])

            pct = np.nan
            if np.isfinite(prev_close) and prev_close != 0:
                pct = (close_price / prev_close - 1) * 100

            return {
                "Stock Symbol": group.loc[i, "TckrSymb"],
                "Closing Price": close_price,
                "% Change": pct,
                "Volume": volume,
                "Breakout Price": bomax,
                "Tests": num,
                "Breakout Start": group.loc[bostart, "TradDt"].strftime("%d-%b-%Y"),
            }

    return None


def scan_watchlist(target_date, prd, bo_len, cwidth_pct, mintest):
    stocklist = load_stocklist()
    data = load_nse_data()

    if not stocklist:
        raise RuntimeError("stocklist.txt not found or empty.")
    if data.empty:
        raise RuntimeError("No NSE CSV data available.")

    data = data[
        data["TckrSymb"].isin(stocklist) &
        (data["TradDt"] <= pd.Timestamp(target_date))
    ].copy()

    results = []
    grouped = data.groupby("TckrSymb", sort=False)

    progress = st.progress(0)
    status = st.empty()
    total = len(grouped)

    for i, (symbol, group) in enumerate(grouped, 1):
        result = find_bullish_breakout(
            group, target_date, prd, bo_len, cwidth_pct, mintest
        )
        if result:
            results.append(result)

        if i == total or i % 100 == 0:
            progress.progress(i / max(total, 1))
            status.write(f"Scanning {i:,} / {total:,} stocks...")

    status.empty()
    progress.empty()
    return pd.DataFrame(results)


# ---------------- UI ----------------
st.title("🚀 NSE Bullish Breakout Screener")
st.caption("TradingView Breakout Finder • Bullish Breakout only")

with st.sidebar:
    st.header("⚙️ Breakout Settings")

    prd = st.number_input("Period", 2, 50, DEFAULT_PRD)
    bo_len = st.number_input("Max B", 30, 300, DEFAULT_BO_LEN)
    cwidth_pct = st.number_input("Thre. %", 1.0, 10.0, DEFAULT_CWIDTH_PCT, step=0.1)
    mintest = st.number_input("Min Tests", 1, 20, DEFAULT_MINTEST)

    st.divider()
    st.metric("Stocklist", f"{len(load_stocklist()):,}")

    dates = cached_trading_dates()
    if dates:
        st.write(
            f"Cached: **{min(dates).strftime('%d-%b-%Y')}** → "
            f"**{max(dates).strftime('%d-%b-%Y')}**"
        )

today = date.today()

if "date_text_input" not in st.session_state:
    st.session_state.date_text_input = today.strftime("%d-%b-%Y")


def go_previous():
    try:
        current = datetime.strptime(
            st.session_state.date_text_input, "%d-%b-%Y"
        ).date()
    except ValueError:
        current = today

    p = previous_trading_day(current)
    if p:
        st.session_state.date_text_input = p.strftime("%d-%b-%Y")


def go_next():
    try:
        current = datetime.strptime(
            st.session_state.date_text_input, "%d-%b-%Y"
        ).date()
    except ValueError:
        current = today

    n = next_trading_day(current)
    if n:
        st.session_state.date_text_input = n.strftime("%d-%b-%Y")


date_text = st.text_input(
    "📅 DD-MMM-YYYY",
    key="date_text_input",
    help="Example: 14-Aug-2026"
).strip()

try:
    typed_date = datetime.strptime(date_text, "%d-%b-%Y").date()
    valid_date = True
except ValueError:
    typed_date = None
    valid_date = False

cached_dates = cached_trading_dates()
selected_date = typed_date

if valid_date and cached_dates and typed_date not in cached_dates:
    p = previous_trading_day(typed_date)
    if p:
        selected_date = p
        st.info(
            f"{typed_date.strftime('%d-%b-%Y')} was not a trading day. "
            f"Using previous trading day: **{selected_date.strftime('%d-%b-%Y')}**"
        )

if valid_date:
    st.write(
        f"Selected Trading Date: **{selected_date.strftime('%d-%b-%Y')}**"
    )

    c1, c2, c3 = st.columns([1, 2, 1])

    with c1:
        st.button(
            "◀ Previous Trading Day",
            use_container_width=True,
            on_click=go_previous,
            disabled=previous_trading_day(selected_date) is None
        )

    with c2:
        get_watchlist = st.button(
            "🔎 GET WATCHLIST",
            type="primary",
            use_container_width=True
        )

    with c3:
        st.button(
            "Next Trading Day ▶",
            use_container_width=True,
            on_click=go_next,
            disabled=next_trading_day(selected_date) is None
        )
else:
    get_watchlist = False
    st.error("Invalid date. Use DD-MMM-YYYY, e.g. 14-Aug-2026.")


if get_watchlist and valid_date:
    if selected_date > today:
        st.error("Future date is not allowed.")
        st.stop()

    with st.spinner("Checking NSE data and downloading missing history..."):
        ready, downloaded, target_exists = ensure_nse_history(selected_date)

    if not target_exists:
        st.error(
            f"NSE Bhavcopy is not available for {selected_date.strftime('%d-%b-%Y')}. "
            "Please select an NSE trading day."
        )
        st.stop()

    load_nse_data.clear()

    with st.spinner("Scanning Bullish Breakouts..."):
        try:
            result = scan_watchlist(
                selected_date,
                int(prd),
                int(bo_len),
                float(cwidth_pct),
                int(mintest),
            )
        except Exception as e:
            st.error(f"Scanner error: {e}")
            st.stop()

    st.divider()

    if result.empty:
        st.warning(
            f"No Bullish Breakout found on "
            f"{selected_date.strftime('%d-%b-%Y')}."
        )
    else:
        result = result.sort_values(
            "% Change", ascending=False
        ).reset_index(drop=True)

        st.success(
            f"🟢 {len(result)} Bullish Breakout stock(s) found"
        )

        display = result[
            ["Stock Symbol", "Closing Price", "% Change", "Volume"]
        ].copy()

        display["Closing Price"] = display["Closing Price"].map(
            lambda x: f"{x:,.2f}"
        )
        display["% Change"] = display["% Change"].map(
            lambda x: f"{x:+.2f}%" if pd.notna(x) else "-"
        )

        def fmt_volume(x):
            if x >= 1_000_000:
                return f"{x/1_000_000:.2f}M"
            if x >= 100_000:
                return f"{x/100_000:.2f}L"
            if x >= 1_000:
                return f"{x/1_000:.2f}K"
            return f"{x:,.0f}"

        display["Volume"] = display["Volume"].map(fmt_volume)

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            height=600
        )

        csv = result[
            ["Stock Symbol", "Closing Price", "% Change", "Volume"]
        ].to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Watchlist CSV",
            csv,
            file_name=(
                f"bullish_breakout_watchlist_"
                f"{selected_date.strftime('%Y-%m-%d')}.csv"
            ),
            mime="text/csv",
            use_container_width=True
        )

        with st.expander("🔍 Show Breakout Details"):
            st.dataframe(
                result,
                use_container_width=True,
                hide_index=True
            )

st.divider()
st.caption(
    "Bullish side only. Parameters default to Period 5, Max B 200, "
    "Threshold 3%, Min Tests 2."
)
