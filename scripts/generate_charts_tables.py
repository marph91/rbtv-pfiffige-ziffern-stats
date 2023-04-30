import math
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# TODO:
# - Rekorde: https://pufopedia.info/?title=Verflixxte_Klixx#Rekorde
# - Abweichung vom tatsächlichen Preis
# - Unterschätzt/Überschätzt
# - Unterboten/Überboten
# - Bessere erste Schätzungen
# - Undercover-Karte


# Prepare folder structure.
Path("docs/img").mkdir(parents=True, exist_ok=True)

# Prepare jinja templates.
env = Environment(
    loader=FileSystemLoader("templates"),
    keep_trailing_newline=True,  # https://github.com/pallets/jinja/issues/848
)


def render_template(name, **kwargs):
    template = env.get_template(name)
    rendered_template = template.render(**kwargs)
    with open("docs" / Path(name).with_suffix(""), "w") as outfile:
        outfile.write(rendered_template)


# General preprocessing.

table = pd.read_csv("data/guesses.csv", converters={"undercover_produkt": bool})

table["lars_difference"] = table["real_price"] - table["lars_guess"]
table["florentin_difference"] = table["real_price"] - table["florentin_guess"]
table["winner"] = np.where(
    abs(table["lars_difference"]) < abs(table["florentin_difference"]),
    "Lars",
    "Florentin",
)
table["winner"] = np.where(
    np.isclose(abs(table["lars_difference"]), abs(table["florentin_difference"])),
    "Unentschieden",
    table["winner"],
)
table["winner"] = table["winner"].astype("string")
table["exact_guess"] = np.where(
    np.isclose(table["lars_difference"], 0.0)
    | np.isclose(table["florentin_difference"], 0.0),
    True,
    False,
)
table["geier_ratio"] = table["lars_guess"] / table["florentin_guess"]


def calculate_score(dataframe):
    if dataframe["winner"] == "Unentschieden":
        return 0
    if dataframe["exact_guess"]:
        return math.ceil(
            dataframe["lars_guess"]
            if dataframe["winner"] == "Lars"
            else dataframe["florentin_guess"]
        )
    # TODO: +1 or *2 for senfkarte?
    return 1 + 1 * int(dataframe["winner"] in str(dataframe["senfkarte"]))


table["score"] = table.apply(lambda row: calculate_score(row), axis=1)

#######################################
# guesses.md
#######################################


def get_exact_guesses(dataframe, name):
    df_filtered_rows = dataframe[
        dataframe["exact_guess"] & (dataframe["winner"] == name)
    ]
    df_filtered_columns = df_filtered_rows[["episode", f"{name.lower()}_guess"]]
    df_filtered_columns.columns = ["Episode", "Schätzung"]  # for better readability
    return df_filtered_columns.set_index("Episode")


#######################################


def get_price_distribution(dataframe):
    price_distribution = (
        dataframe.apply(lambda x: round(x, ndigits=2))  # round to two decimal places
        .value_counts()
        .sort_index()
    )
    return pd.DataFrame(
        {"Preis [€]": price_distribution.index, "Anzahl": price_distribution.values}
    )


def plot_price_distribution(dataframe, filename):
    plt.figure()
    ax = dataframe.plot(
        kind="scatter",
        x="Preis [€]",
        y="Anzahl",
        logx=True,
        grid=True,
    )

    # Always show non-scientific ticks.
    # https://docs.python.org/3/library/string.html#format-specification-mini-language
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: f"{y:.16g}"))
    # https://stackoverflow.com/a/34880501/7410886
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    # Limit the range for better visibility.
    ax.set_xlim(dataframe["Preis [€]"].min(), dataframe["Preis [€]"].max() + 100)
    # ax.set(adjustable="box")

    plt.savefig(filename)


def get_top_guesses(dataframe, count=10):
    # dataframe.loc[real_price_distribution["Anzahl"] > 1]
    return (
        dataframe.sort_values(by="Anzahl", ascending=False)
        .head(count)
        .set_index("Preis [€]")
    )


real_price_distribution = get_price_distribution(table["real_price"])
lars_guess_distribution = get_price_distribution(table["lars_guess"])
florentin_guess_distribution = get_price_distribution(table["florentin_guess"])

plot_price_distribution(real_price_distribution, "docs/img/distribution_price.png")
plot_price_distribution(lars_guess_distribution, "docs/img/distribution_lars.png")
plot_price_distribution(
    florentin_guess_distribution,
    "docs/img/distribution_florentin.png",
)

#######################################

plt.figure()
# plt.close("all")
groups = table.groupby("first_guess")
for name, group in groups:
    plt.plot(
        group["lars_guess"],
        group["florentin_guess"],
        label=name,
        marker=".",
        linestyle="",
    )
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Schätzung Lars")
plt.ylabel("Schätzung Florentin")
plt.legend()

#######################################

# Geierkorridor
geier_ratio = 13 / 16
reference_x = [1.0, 1000]
reference_y_lower = [x * geier_ratio for x in reference_x]
reference_y_upper = [x * 1 / geier_ratio for x in reference_x]
plt.plot(reference_x, reference_y_lower, reference_x, reference_y_upper, color="black")

ax = plt.gca()
ax.grid(True)
# Limit the range for better visibility. This truncates some points.
ax.set_xlim(10**0, 10**3)
ax.set_ylim(10**0, 10**3)
# Always show non-scientific ticks.
# https://stackoverflow.com/a/49751075/7410886
ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())

plt.savefig("docs/img/geierkorridor.png")

#######################################

geier_attempts_lars = table[
    # fmt: off
    (table["first_guess"] == "Florentin")
    & (table["geier_ratio"] < 1 / geier_ratio)
    & (table["geier_ratio"] > geier_ratio)
    # fmt: on
][["episode", "lars_guess", "florentin_guess"]]
geier_attempts_lars.columns = ["Episode", "Schätzung Lars", "Schätzung Florentin"]
geier_attempts_florentin = table[
    # fmt: off
    (table["first_guess"] == "Lars")
    & (table["geier_ratio"] < 1 / geier_ratio)
    & (table["geier_ratio"] > geier_ratio)
    # fmt: on
][["episode", "lars_guess", "florentin_guess"]]
geier_attempts_florentin.columns = ["Episode", "Schätzung Lars", "Schätzung Florentin"]

#######################################

senfkarten_stats = {
    "Lars": {
        "win": len(
            table[
                (table["winner"] == "Lars")
                & table["senfkarte"].str.contains("Lars")
                & table["senfkarte"].notnull()
            ]
        ),
        "lose": len(
            table[
                (table["winner"] == "Florentin")
                & table["senfkarte"].str.contains("Lars")
                & table["senfkarte"].notnull()
            ]
        ),
        "win_ratio_total": len(table[table["winner"] == "Lars"]) / len(table),
    },
    "Florentin": {
        "win": len(
            table[
                (table["winner"] == "Florentin")
                & table["senfkarte"].str.contains("Florentin")
                & table["senfkarte"].notnull()
            ]
        ),
        "lose": len(
            table[
                (table["winner"] == "Lars")
                & table["senfkarte"].str.contains("Florentin")
                & table["senfkarte"].notnull()
            ]
        ),
        "win_ratio_total": len(table[table["winner"] == "Florentin"]) / len(table),
    },
}


# Render template
render_template(
    "guesses.md.j2",
    distribution_table_price=get_top_guesses(real_price_distribution).to_markdown(),
    distribution_table_lars=get_top_guesses(lars_guess_distribution).to_markdown(),
    distribution_table_florentin=get_top_guesses(
        florentin_guess_distribution
    ).to_markdown(),
    exact_guesses_lars=get_exact_guesses(table, "Lars").to_markdown(),
    exact_guesses_florentin=get_exact_guesses(table, "Florentin").to_markdown(),
    geier_ratio=geier_ratio,
    geier_attempts_lars=geier_attempts_lars.to_markdown(index=False),
    geier_attempts_florentin=geier_attempts_florentin.to_markdown(index=False),
    senfkarten_stats=senfkarten_stats,
)


#######################################
# episodes.md
#######################################

stats_per_episode = []
for episode in set(table["episode"].tolist()):
    guesses = table[table["episode"] == episode]
    stats_per_episode.append(
        {
            "index": episode,
            "guesses": len(guesses),
            "score_lars": sum(guesses[guesses["winner"] == "Lars"]["score"]),
            "score_florentin": sum(guesses[guesses["winner"] == "Florentin"]["score"]),
        }
    )
table_episodes = pd.DataFrame(stats_per_episode)

table_episodes["winner"] = np.where(
    table_episodes["score_lars"] > table_episodes["score_florentin"],
    "Lars",
    "Florentin",
)
table_episodes["winner"] = np.where(
    table_episodes["score_lars"] == table_episodes["score_florentin"],
    "Unentschieden",
    table_episodes["winner"],
)
table_episodes["winner"] = table_episodes["winner"].astype("string")

print(table_episodes)

#######################################

fig = plt.figure()
ax = table_episodes.plot(
    kind="bar",
    xlabel="Episode",
    ylabel="Gültige Schätzungen",
    y="guesses",
    legend=False,
)

ax.yaxis.grid()
ax.set_axisbelow(True)
# https://stackoverflow.com/a/34880501/7410886
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

plt.savefig("docs/img/guesses_per_episode.png")

#######################################

fig = plt.figure()
table_episodes_filtered = table_episodes[["score_lars", "score_florentin"]]
table_episodes_filtered.columns = ["Lars", "Florentin"]  # for better readability
ax = table_episodes_filtered.plot(kind="bar", xlabel="Episode", ylabel="Punkte")

# https://stackoverflow.com/a/23358722/7410886
ax.yaxis.grid()
ax.set_axisbelow(True)

plt.savefig("docs/img/points_per_episode.png")

# Render template
render_template(
    "episodes.md.j2",
    total_episodes=table.iloc[-1]["episode"],
    total_guesses=len(table),
    total_score=table["score"].sum(),
)


#######################################
# comparison.md
#######################################


table_episodes_total = pd.DataFrame(
    [
        {
            "Metrik": "Kronen",
            "Lars": table_episodes["winner"].value_counts()["Lars"],
            "Florentin": table_episodes["winner"].value_counts()["Florentin"],
        },
        {
            "Metrik": "Punkte pro Episode",
            "Lars": table_episodes["score_lars"].mean().round(2),
            "Florentin": table_episodes["score_florentin"].mean().round(2),
        },
        {
            "Metrik": "Gesamtpunkte",
            "Lars": table_episodes["score_lars"].sum(),
            "Florentin": table_episodes["score_florentin"].sum(),
        },
        {
            "Metrik": "Gewonnene Schätzungen",
            "Lars": table["winner"].value_counts()["Lars"],
            "Florentin": table["winner"].value_counts()["Florentin"],
        },
        {
            "Metrik": "Exakte Treffer",
            "Lars": len(get_exact_guesses(table, "Lars")),
            "Florentin": len(get_exact_guesses(table, "Florentin")),
        },
        {
            "Metrik": "Erste Schätzungen",
            "Lars": table["first_guess"].value_counts()["Lars"],
            "Florentin": table["first_guess"].value_counts()["Florentin"],
        },
        {
            "Metrik": "Geier-Versuche",
            "Lars": len(geier_attempts_lars),
            "Florentin": len(geier_attempts_florentin),
        },
    ]
)

#######################################

plt.figure()
table["winner"].value_counts().plot(
    kind="pie",
    ylabel="",
    autopct=lambda p: "{:.0f}".format(p * len(table.index) / 100),
)
plt.savefig("docs/img/winner.png")

# Render template
render_template(
    "comparison.md.j2",
    total_comparison=table_episodes_total.to_markdown(),
)


#######################################
# index.md
#######################################


table_episodes_meta = pd.read_csv("data/episodes.csv")

render_template(
    "index.md.j2", last_episode=table_episodes_meta.tail(1).to_dict("records")[0]
)
