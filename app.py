import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.express as px

st.set_page_config(page_title="AI Match Forecast", layout="wide", page_icon="⚽")

st.title("⚽ AI Match Forecast Engine")
st.markdown("**Pre-match AI prediction using Poisson xG model**")

# ====================== SIDEBAR ======================
st.sidebar.header("Match Parameters")

home_team = st.sidebar.text_input("Home Team", "Manchester City")
away_team = st.sidebar.text_input("Away Team", "Arsenal")

home_exp = st.sidebar.slider(f"{home_team} Expected Goals (xG)", 0.0, 6.0, 2.1, 0.1)
away_exp = st.sidebar.slider(f"{away_team} Expected Goals (xG)", 0.0, 6.0, 1.4, 0.1)

# Manual context inputs (you fill these before match)
st.sidebar.markdown("---")
st.sidebar.subheader("Real Context (fill before match)")
home_form = st.sidebar.text_input("Home Recent Form (e.g. W-D-W-L-W)", "W-W-D-W-L")
away_form = st.sidebar.text_input("Away Recent Form", "D-L-W-D-L")
h2h = st.sidebar.text_area("H2H Last 5 matches", "2-1, 1-1, 3-0, 0-2, 1-1")
season_stats = st.sidebar.text_area("Season Stats", 
    f"{home_team}: 18 wins, 4 draws, 3 losses | Avg xG 2.3\n"
    f"{away_team}: 14 wins, 7 draws, 4 losses | Avg xG 1.7")
absences = st.sidebar.text_area("Key Absences", "Home: No major\nAway: Striker suspended")

# ====================== CALCULATIONS ======================
MAX_GOALS = 8
h_probs = [poisson.pmf(i, home_exp) for i in range(MAX_GOALS + 1)]
a_probs = [poisson.pmf(j, away_exp) for j in range(MAX_GOALS + 1)]
grid = np.outer(h_probs, a_probs)

home_win = np.sum(np.tril(grid, -1))
draw = np.trace(grid)
away_win = np.sum(np.triu(grid, 1))
total = home_win + draw + away_win

btts = (1 - h_probs[0]) * (1 - a_probs[0])
over_25 = 1 - np.sum(grid[:3, :3])

# Most likely score
max_idx = np.argmax(grid)
most_home = max_idx // (MAX_GOALS + 1)
most_away = max_idx % (MAX_GOALS + 1)
most_prob = grid[most_home, most_away]

# ====================== MAIN DISPLAY ======================
col1, col2, col3 = st.columns(3)
col1.metric("🏠 Home Win", f"{home_win:.1%}", f"{home_win*100:.1f}%")
col2.metric("🤝 Draw", f"{draw:.1%}")
col3.metric("🚀 Away Win", f"{away_win:.1%}")

st.subheader("Both Teams To Score (BTTS)")
if btts > 0.55:
    st.success(f"**YES** ({btts:.1%})")
elif btts < 0.45:
    st.error(f"**NO** ({btts:.1%})")
else:
    st.warning(f"**50/50** ({btts:.1%})")

st.subheader(f"Most Likely Score: **{home_team} {most_home}–{most_away} {away_team}** ({most_prob:.1%})")

# Probability Heatmap
st.write("### Result Probability Chart")
fig = px.imshow(grid,
                x=list(range(MAX_GOALS+1)), y=list(range(MAX_GOALS+1)),
                labels=dict(x=f"{away_team} Goals", y=f"{home_team} Goals", color="Probability"),
                color_continuous_scale="Viridis",
                text_auto=".1%")
fig.update_traces(texttemplate="%{z:.1%}")
st.plotly_chart(fig, use_container_width=True)

st.caption(f"Probabilities cover scores up to {MAX_GOALS}–{MAX_GOALS} (sums to {total:.1%})")

# ====================== TABS ======================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Recent Form", "H2H Record", "Season Statistics", "Key Players & Absences", "Tactical Insights"])

with tab1:
    st.write(f"**{home_team}**: {home_form}")
    st.write(f"**{away_team}**: {away_form}")

with tab2:
    st.write(h2h)

with tab3:
    st.write(season_stats)

with tab4:
    st.write(absences)

with tab5:
    if home_win > 0.55:
        st.success(f"**{home_team}** are clear favourites. Expect them to dominate possession and create more chances.")
    elif away_win > 0.55:
        st.success(f"**{away_team}** have the edge. Counter-attack likely to be key.")
    else:
        st.info("Evenly matched game. Tactical discipline and set-pieces will decide the winner.")
    
    st.write(f"Over 2.5 goals probability: **{over_25:.1%}**")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Poisson xG model • Free to use & share")
