import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

dates = pd.date_range("2005-01-01", "2015-12-31", freq="D")
dates = dates[~((dates.month == 2) & (dates.day == 29))]  # drop leap days

day_of_year = dates.dayofyear.where(~((dates.is_leap_year) & (dates.dayofyear > 59)),
                                     dates.dayofyear - 1)

# Seasonal signal: warmest ~day 200 (mid-July), coldest ~day 15 (mid-Jan)
seasonal = 15 + 15 * np.sin(2 * np.pi * (day_of_year - 110) / 365)
year_trend = (dates.year - 2005) * 0.03  # slight warming trend

tmax = seasonal + year_trend + rng.normal(3, 3, len(dates))
tmin = seasonal + year_trend + rng.normal(-4, 3, len(dates))
tmin = np.minimum(tmin, tmax - 1)  # keep min below max

weather = pd.DataFrame({
    "Date": dates,
    "Day_of_Year": day_of_year,
    "TMAX": tmax.round(1),
    "TMIN": tmin.round(1),
})

# Force a handful of 2015 days to break records, so the scatter overlay has something to show
is_2015 = weather["Date"].dt.year == 2015
break_high_idx = weather[is_2015].sample(12, random_state=1).index
break_low_idx = weather[is_2015].sample(12, random_state=2).index
weather.loc[break_high_idx, "TMAX"] += rng.uniform(6, 10, len(break_high_idx))
weather.loc[break_low_idx, "TMIN"] -= rng.uniform(6, 10, len(break_low_idx))

weather.to_csv("../data/synthetic_daily_temps_2005_2015.csv", index=False)
weather.head()

baseline = weather[weather["Date"].dt.year.between(2005, 2014)]
y2015 = weather[weather["Date"].dt.year == 2015].copy()

record_high = baseline.groupby("Day_of_Year")["TMAX"].max()
record_low = baseline.groupby("Day_of_Year")["TMIN"].min()

y2015["record_high_0514"] = y2015["Day_of_Year"].map(record_high)
y2015["record_low_0514"] = y2015["Day_of_Year"].map(record_low)

broke_high = y2015[y2015["TMAX"] > y2015["record_high_0514"]]
broke_low = y2015[y2015["TMIN"] < y2015["record_low_0514"]]

len(broke_high), len(broke_low)

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BLUE = "#2a78d6"   # record low
ORANGE = "#eb6834"  # record high
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

# Map day-of-year -> a plain 2015 date so the x-axis gets real Jan-Dec tick labels
x_axis_dates = pd.to_datetime(record_high.index, unit="D", origin="2014-12-31")

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(x_axis_dates, record_high.values, color=ORANGE, linewidth=2,
        label="Record high (2005-2014)")
ax.plot(x_axis_dates, record_low.values, color=BLUE, linewidth=2,
        label="Record low (2005-2014)")
ax.fill_between(x_axis_dates, record_low.values, record_high.values,
                 color=BLUE, alpha=0.10, linewidth=0)

ax.scatter(broke_high["Date"].map(lambda d: d.replace(year=2015)), broke_high["TMAX"],
           s=45, facecolor=ORANGE, edgecolor="white", linewidth=1.2, zorder=5,
           marker="^", label="2015 day broke record high")
ax.scatter(broke_low["Date"].map(lambda d: d.replace(year=2015)), broke_low["TMIN"],
           s=45, facecolor=BLUE, edgecolor="white", linewidth=1.2, zorder=5,
           marker="v", label="2015 day broke record low")

ax.set_title("Record Daily High and Low Temperatures (2005-2014)\nwith 2015 Record-Breaking Days",
             fontsize=14, color=INK, pad=14)
ax.set_xlabel("Month", color=MUTED)
ax.set_ylabel("Temperature (°C)", color=MUTED)

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.set_xlim(x_axis_dates.min(), x_axis_dates.max())

ax.grid(axis="y", color=GRID, linewidth=1)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)

ax.tick_params(colors=MUTED)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)

plt.tight_layout()
plt.savefig("record_highs_lows_2015.png", dpi=150, bbox_inches="tight")
plt.show()