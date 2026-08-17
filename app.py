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

# sec_bhavdata_full report (with delivery data) has been continuously
# published by NSE since 01-Jan-2016 — unlike the newer CM-UDiFF bhavcopy,
# it wasn't affected by the Jul-2024 format change.
EARLIEST_BHAVCOPY_DATE = date(2016, 1, 1)

# How many raw daily bars go into 1 resampled bar, per timeframe.
TIMEFRAME_MULTIPLIER = {"Daily": 1, "Weekly": 5, "Monthly": 21}
TIMEFRAME_RESAMPLE_RULE = {"Weekly": "W-FRI", "Monthly": "ME"}

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
def build_cache_zip(_dates_signature):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(DATA_DIR.glob("*.csv")):
            z.write(f, arcname=f.name)
    buf.seek(0)
    return buf.getvalue()


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
        "https://nsearchives.nseindia.com/products/content/"
        f"sec_bhavdata_full_{dt.strftime('%d%m%Y')}.csv"
    )


def download_nse_day(dt, session):
    DATA_DIR.mkdir(exist_ok=True)
    target = DATA_DIR / f"{dt.strftime('%Y%m%d')}.csv"

    if target.exists():
        # Guard against files cached before a data-source change (e.g. the
        # old CM-UDiFF schema) — don't trust a same-named file blindly.
        try:
            head = target.read_text(encoding="utf-8", errors="ignore")[:200]
        except Exception:
            head = ""
        if "SYMBOL" in head:
            return True, "cached"
        target.unlink(missing_ok=True)

    try:
        r = session.get(nse_url(dt), timeout=30)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"
        if not r.content or b"SYMBOL" not in r.content[:2000]:
            return False, "unexpected content"

        target.write_bytes(r.content)
        return True, "downloaded"
    except Exception as e:
        return False, str(e)


def ensure_nse_history(target_date, required=HISTORY_TRADING_DAYS):
    session = requests.Session()
    session.headers.update(HEADERS)

    found = 0
    checked = 0
    downloaded = 0
    consecutive_fail = 0
    dt = target_date

    box = st.empty()
    bar = st.progress(0)
    hit_floor = False

    while found < required and checked < required * 5:
        if dt < EARLIEST_BHAVCOPY_DATE:
            hit_floor = True
            break

        checked += 1
        path = DATA_DIR / f"{dt.strftime('%Y%m%d')}.csv"

        if path.exists():
            found += 1
            consecutive_fail = 0
        else:
            ok, status = download_nse_day(dt, session)
            if ok:
                found += 1
                consecutive_fail = 0
                if status == "downloaded":
                    downloaded += 1
            else:
                consecutive_fail += 1

        bar.progress(min(found / required, 1.0))
        box.write(
            f"NSE history: {found}/{required} trading days ready • "
            f"checking {dt.strftime('%d-%b-%Y')}"
        )

        # Safety net: many failures in a row (e.g. weekends/holidays are
        # normal and skipped quickly, but a long unbroken failure streak
        # usually means we've wandered into a date range with no usable
        # data) — stop instead of grinding through thousands of dead
        # network requests.
        if consecutive_fail >= 15:
            break

        dt -= timedelta(days=1)

    box.empty()
    bar.empty()

    if hit_floor:
        st.info(
            f"NSE bhavcopy (this data source) only goes back to "
            f"{EARLIEST_BHAVCOPY_DATE.strftime('%d-%b-%Y')}. Got "
            f"{found} trading day(s) of history — reduce 'Max B' if you "
            f"need results with this much history."
        )

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
            df.columns = df.columns.str.strip()

            required = {
                "SYMBOL", "SERIES", "DATE1", "PREV_CLOSE",
                "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE",
                "TTL_TRD_QNTY",
            }

            if not required.issubset(df.columns):
                continue

            # Source CSV has a leading space after every comma, so
            # string columns come in as " EQ", " 20MICRONS", etc.
            for col in ["SYMBOL", "SERIES", "DATE1"]:
                df[col] = df[col].astype(str).str.strip()

            df = df[
                df["SERIES"].isin(["EQ", "BE", "BZ", "SM", "ST", "SZ"])
            ].copy()

            df["SYMBOL"] = df["SYMBOL"].str.upper()

            df = df.rename(columns={
                "SYMBOL": "TckrSymb",
                "DATE1": "TradDt",
                "PREV_CLOSE": "PrvsClsgPric",
                "OPEN_PRICE": "OpnPric",
                "HIGH_PRICE": "HghPric",
                "LOW_PRICE": "LwPric",
                "CLOSE_PRICE": "ClsPric",
                "TTL_TRD_QNTY": "TtlTradgVol",
            })

            keep_cols = [
                "TradDt", "TckrSymb", "OpnPric", "HghPric", "LwPric",
                "ClsPric", "PrvsClsgPric", "TtlTradgVol",
            ]
            frames.append(df[keep_cols])

        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True)

    data["TradDt"] = pd.to_datetime(
        data["TradDt"], format="%d-%b-%Y", errors="coerce"
    )

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


def resample_ohlc(group, timeframe):
    """Resample a per-stock daily dataframe to Weekly/Monthly OHLCV bars."""
    if timeframe == "Daily":
        return group.reset_index(drop=True)

    rule = TIMEFRAME_RESAMPLE_RULE[timeframe]
    g = group.set_index("TradDt").sort_index()

    resampled = g.resample(rule).agg({
        "TckrSymb": "last",
        "OpnPric": "first",
        "HghPric": "max",
        "LwPric": "min",
        "ClsPric": "last",
        "TtlTradgVol": "sum",
    })
    resampled = resampled.dropna(subset=["OpnPric"])
    resampled["PrvsClsgPric"] = resampled["ClsPric"].shift(1)
    resampled = resampled.reset_index()
    return resampled


def find_bullish_breakout(
    group, target_date, prd, bo_len, cwidth_pct, mintest,
    require_exact_date=True,
):
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

        is_match = (
            group.loc[i, "TradDt"].date() == target_date
            if require_exact_date
            else i == n - 1
        )

        if is_match:
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


def scan_watchlist(target_date, prd, bo_len, cwidth_pct, mintest, timeframe="Daily"):
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

    require_exact_date = timeframe == "Daily"

    results = []
    grouped = data.groupby("TckrSymb", sort=False)

    progress = st.progress(0)
    status = st.empty()
    total = len(grouped)

    for i, (symbol, group) in enumerate(grouped, 1):
        tf_group = resample_ohlc(group, timeframe)
        result = find_bullish_breakout(
            tf_group, target_date, prd, bo_len, cwidth_pct, mintest,
            require_exact_date=require_exact_date,
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

    timeframe = st.selectbox("🕒 Timeframe", ["Daily", "Weekly", "Monthly"], index=0)

    prd = st.number_input("Period", 2, 50, DEFAULT_PRD)
    bo_len = st.number_input("Max B", 5, 300, DEFAULT_BO_LEN)
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

        st.download_button(
            "⬇️ Download Cached Data (ZIP)",
            build_cache_zip(tuple(dates)),
            file_name=f"nse_bhavcopy_cache_{date.today().strftime('%Y%m%d')}.zip",
            mime="application/zip",
            use_container_width=True,
            help=(
                "Streamlit Cloud storage is temporary — download this "
                "before the app sleeps/restarts if you want to keep it."
            ),
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

    # Weekly/Monthly bars each need several raw daily bars, so scale the
    # required daily-history window accordingly (with some buffer).
    buffer_bars = 5 if timeframe != "Daily" else 20
    bars_needed = int(bo_len) + (int(prd) * 2) + buffer_bars
    required_daily_days = bars_needed * TIMEFRAME_MULTIPLIER[timeframe]
    required_daily_days = max(required_daily_days, HISTORY_TRADING_DAYS)

    # Accurate periods-available estimate (calendar-based, not a days/
    # multiplier approximation which under-counts).
    if timeframe == "Weekly":
        periods_available = (
            (selected_date - EARLIEST_BHAVCOPY_DATE).days // 7
        )
    elif timeframe == "Monthly":
        periods_available = (
            (selected_date.year - EARLIEST_BHAVCOPY_DATE.year) * 12
            + (selected_date.month - EARLIEST_BHAVCOPY_DATE.month)
            + 1
        )
    else:
        periods_available = None

    if periods_available is not None:
        max_bars_possible = max(
            periods_available - (int(prd) * 2) - 5, 0
        )
        if int(bo_len) > max_bars_possible:
            st.warning(
                f"⚠️ NSE bhavcopy (this data source) is only available from "
                f"{EARLIEST_BHAVCOPY_DATE.strftime('%d-%b-%Y')} onward — "
                f"~{periods_available} {timeframe.lower()} bars exist total. "
                f"With Period={prd}, 'Max B' can realistically go up to "
                f"~**{max_bars_possible}**. Your Max B of {bo_len} won't be "
                f"fully met; lower it for reliable results."
            )

    with st.spinner("Checking NSE data and downloading missing history..."):
        ready, downloaded, target_exists = ensure_nse_history(
            selected_date, required=required_daily_days
        )

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
                timeframe=timeframe,
            )
        except Exception as e:
            st.error(f"Scanner error: {e}")
            st.stop()

    st.divider()

    if result.empty:
        st.warning(
            f"No Bullish Breakout ({timeframe}) found on "
            f"{selected_date.strftime('%d-%b-%Y')}."
        )
    else:
        result = result.sort_values(
            "% Change", ascending=False
        ).reset_index(drop=True)

        st.success(
            f"🟢 {len(result)} Bullish Breakout ({timeframe}) stock(s) found"
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
                f"bullish_breakout_watchlist_{timeframe.lower()}_"
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
