"""
app.py — Dashboard Streamlit
============================
"The Invisible Rain" — Ratio ET/P en Tunisie 2000-2024
Déficit hydrique agricole et accélération climatique

Usage : streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# ── CONFIG PAGE ───────────────────────────────────────────────
st.set_page_config(
    page_title="The Invisible Rain — Tunisie",
    page_icon="🌧️",
    layout="wide"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        border-radius: 8px 8px 0 0;
        font-weight: 500;
    }
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 5px solid;
        margin-bottom: 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    }
</style>
""", unsafe_allow_html=True)

# ── CHARGEMENT DONNÉES ────────────────────────────────────────
@st.cache_data
def load_data():
    annual  = pd.read_csv("data/tunisia_annual_clean.csv")
    monthly = pd.read_csv("data/tunisia_monthly_clean.csv")
    return annual, monthly

df_annual, df_monthly = load_data()

# ── HEADER ────────────────────────────────────────────────────
st.title("🌧️ The Invisible Rain — Tunisie")
st.markdown(
    "*Quelle part des précipitations tunisiennes est perdue par évapotranspiration "
    "avant de bénéficier aux sols agricoles ?*  \n"
    "📡 Source : **ERA5-Land CDS Copernicus** · Résolution 0.1° (~9 km) · 2000–2024"
)
st.divider()

# ── SIDEBAR ───────────────────────────────────────────────────
st.sidebar.header("🎛️ Filtres globaux")
year_range = st.sidebar.slider("Période d'analyse", min_value=2000, max_value=2024, value=(2000, 2024))
variable = st.sidebar.selectbox(
    "Variable principale",
    options=["ratio_ET_P_annual","precip_annual_mm","pot_ET_annual_mm","temp_mean_C","soil_moisture_mean"],
    format_func=lambda x: {
        "ratio_ET_P_annual":"🔴 Ratio ET/P (déficit)",
        "precip_annual_mm":"🔵 Précipitations (mm/an)",
        "pot_ET_annual_mm":"🟠 ET Potentielle (mm/an)",
        "temp_mean_C":"🌡️ Température (°C)",
        "soil_moisture_mean":"🟤 Humidité du sol"
    }[x]
)
selected_year = st.sidebar.selectbox("Année — snapshot carte", options=sorted(df_annual["year"].unique(), reverse=True))
st.sidebar.divider()
st.sidebar.caption("ERA5-Land · CDS Copernicus\n5 variables · 300 mois · 3116 pixels\nRésolution : 0.1° × 0.1° (~9 km)")

# ── FILTRE ────────────────────────────────────────────────────
mask = (df_annual["year"] >= year_range[0]) & (df_annual["year"] <= year_range[1])
df_f = df_annual[mask].copy()
df_f["region"] = np.where(df_f["latitude"] >= 34, "Nord (≥34°N)", "Sud (<34°N)")
df_f["decade"] = pd.cut(df_f["year"], bins=[1999,2009,2019,2024],
                         labels=["2000–2009","2010–2019","2020–2024"])

# ── KPIs ──────────────────────────────────────────────────────
ratio_med   = df_f["ratio_ET_P_annual"].median()
temp_mean   = df_f["temp_mean_C"].mean()
pct_deficit = (df_f["ratio_ET_P_annual"] > 5).mean() * 100
ts_temp     = df_f.groupby("year")["temp_mean_C"].mean()
temp_trend  = np.polyfit(ts_temp.index, ts_temp.values, 1)[0] * 25
precip_mean = df_f.groupby("year")["precip_annual_mm"].median().mean()
df_yr       = df_f.groupby("year")["ratio_ET_P_annual"].median()
worst_year  = int(df_yr.idxmax())
best_year   = int(df_yr.idxmin())

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Ratio ET/P médian", f"{ratio_med:.1f}x")
k2.metric("Précipitations moy.", f"{precip_mean:.0f} mm/an")
k3.metric("Température moy.", f"{temp_mean:.1f} °C")
k4.metric("Hausse T° sur 25 ans", f"+{temp_trend:.1f} °C", delta=f"+{temp_trend:.1f}°C", delta_color="inverse")
k5.metric("Pixels en déficit fort", f"{pct_deficit:.0f}%")
st.divider()

# ── ONGLETS ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Carte & Spatial",
    "📈 Séries temporelles",
    "🌿 Saisonnalité",
    "🔬 Analyse statistique",
    "⚠️ Anomalies & Décennies"
])

month_names = {1:"Jan",2:"Fév",3:"Mar",4:"Avr",5:"Mai",6:"Jun",
               7:"Jul",8:"Aoû",9:"Sep",10:"Oct",11:"Nov",12:"Déc"}

# ══ ONGLET 1 — CARTE & SPATIAL ═══════════════════════════════
with tab1:
    st.subheader(f"🗺️ Carte interactive — {selected_year}")
    col_map, col_info = st.columns([1.4, 1])

    with col_map:
        df_map = df_annual[df_annual["year"] == selected_year].copy()
        cscales = {"ratio_ET_P_annual":"RdYlBu_r","precip_annual_mm":"Blues",
                   "pot_ET_annual_mm":"OrRd","temp_mean_C":"RdYlBu_r","soil_moisture_mean":"YlGnBu"}
        fig_map = px.scatter_mapbox(df_map, lat="latitude", lon="longitude",
                                     color=variable, color_continuous_scale=cscales[variable],
                                     zoom=5.2, center={"lat":33.8,"lon":9.5},
                                     mapbox_style="carto-positron", height=440, opacity=0.85)
        fig_map.update_traces(marker=dict(size=5))
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)

    with col_info:
        nord = df_map[df_map["latitude"] >= 34]["ratio_ET_P_annual"].median()
        sud  = df_map[df_map["latitude"] <  34]["ratio_ET_P_annual"].median()
        fig_ns_bar = go.Figure(go.Bar(
            x=["Nord (≥34°N)","Sud (<34°N)"], y=[nord, sud],
            marker_color=["#2196F3","#FF5722"],
            text=[f"{nord:.1f}x", f"{sud:.1f}x"], textposition="outside"
        ))
        fig_ns_bar.update_layout(height=220, margin=dict(t=10,b=10,l=0,r=0), yaxis_title="Ratio ET/P")
        st.markdown(f"**Nord vs Sud — {selected_year}**")
        st.plotly_chart(fig_ns_bar, use_container_width=True)
        st.markdown("**Statistiques spatiales**")
        st.dataframe(df_map[variable].describe().round(2).rename({
            "count":"N pixels","mean":"Moyenne","std":"Écart-type",
            "min":"Min","25%":"Q1","50%":"Médiane","75%":"Q3","max":"Max"
        }), use_container_width=True)

    st.divider()
    st.subheader("🧭 Nord vs Sud — Évolution 2000–2024")
    ns_ts = df_f.groupby(["year","region"])["ratio_ET_P_annual"].median().reset_index()
    fig_ns = px.line(ns_ts, x="year", y="ratio_ET_P_annual", color="region",
                     color_discrete_map={"Nord (≥34°N)":"#2196F3","Sud (<34°N)":"#FF5722"},
                     markers=True, height=300)
    fig_ns.update_layout(margin=dict(t=10,b=20,l=0,r=0), legend=dict(orientation="h",y=-0.2),
                          yaxis_title="Ratio ET/P médian", hovermode="x unified")
    st.plotly_chart(fig_ns, use_container_width=True)
    st.caption("Le Sud souffre d'un déficit 2 à 3× plus élevé que le Nord — gradient climatique méditerranéen → saharien.")

# ══ ONGLET 2 — SÉRIES TEMPORELLES ════════════════════════════
with tab2:
    st.subheader("📈 Évolution temporelle")
    col_ts1, col_ts2 = st.columns(2)

    with col_ts1:
        ts = df_f.groupby("year").agg(
            median_val=(variable,"median"),
            q25=(variable, lambda x: x.quantile(0.25)),
            q75=(variable, lambda x: x.quantile(0.75)),
        ).reset_index()
        z = np.polyfit(ts["year"], ts["median_val"], 1)
        ts["trend"] = np.poly1d(z)(ts["year"])
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=ts["year"], y=ts["q75"], fill=None, mode="lines",
                                     line=dict(width=0), showlegend=False))
        fig_ts.add_trace(go.Scatter(x=ts["year"], y=ts["q25"], fill="tonexty", mode="lines",
                                     line=dict(width=0), fillcolor="rgba(31,119,180,0.12)", name="Q25–Q75"))
        fig_ts.add_trace(go.Scatter(x=ts["year"], y=ts["median_val"], name="Médiane",
                                     mode="lines+markers", line=dict(color="#1f77b4",width=2.5), marker=dict(size=5)))
        fig_ts.add_trace(go.Scatter(x=ts["year"], y=ts["trend"], name="Tendance",
                                     mode="lines", line=dict(color="#d62728",width=2,dash="dash")))
        fig_ts.update_layout(height=340, margin=dict(t=10,b=20,l=0,r=0),
                              yaxis_title=variable, hovermode="x unified", legend=dict(orientation="h",y=-0.2))
        st.plotly_chart(fig_ts, use_container_width=True)
        st.caption("Série temporelle avec intervalle interquartile et tendance linéaire.")

    with col_ts2:
        ts_r = df_f.groupby("year")["ratio_ET_P_annual"].median().reset_index()
        ts_r["rolling_3"] = ts_r["ratio_ET_P_annual"].rolling(window=3, center=True).mean()
        ts_r["rolling_5"] = ts_r["ratio_ET_P_annual"].rolling(window=5, center=True).mean()
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Bar(x=ts_r["year"], y=ts_r["ratio_ET_P_annual"],
                                 name="Valeur annuelle", marker_color="rgba(180,180,180,0.4)"))
        fig_ma.add_trace(go.Scatter(x=ts_r["year"], y=ts_r["rolling_3"], name="Moy. mobile 3 ans",
                                     mode="lines", line=dict(color="#FF9800",width=2)))
        fig_ma.add_trace(go.Scatter(x=ts_r["year"], y=ts_r["rolling_5"], name="Moy. mobile 5 ans",
                                     mode="lines", line=dict(color="#d62728",width=2.5,dash="dash")))
        fig_ma.update_layout(height=340, margin=dict(t=10,b=20,l=0,r=0),
                              yaxis_title="Ratio ET/P", hovermode="x unified", legend=dict(orientation="h",y=-0.2))
        st.plotly_chart(fig_ma, use_container_width=True)
        st.caption("La moyenne mobile révèle la tendance structurelle au-delà des variations annuelles.")

    st.divider()
    col_ts3, col_ts4 = st.columns(2)

    with col_ts3:
        st.subheader("🌡️ Anomalie de température")
        temp_ts = df_f.groupby("year")["temp_mean_C"].mean().reset_index()
        zt = np.polyfit(temp_ts["year"], temp_ts["temp_mean_C"], 1)
        temp_ts["trend"] = np.poly1d(zt)(temp_ts["year"])
        tmean = temp_ts["temp_mean_C"].mean()
        fig_t = go.Figure()
        fig_t.add_trace(go.Bar(x=temp_ts["year"], y=temp_ts["temp_mean_C"], name="Température",
                                marker_color=["#d62728" if t > tmean else "#4e9af1"
                                              for t in temp_ts["temp_mean_C"]]))
        fig_t.add_trace(go.Scatter(x=temp_ts["year"], y=temp_ts["trend"], name="Tendance",
                                    mode="lines", line=dict(color="black",width=2,dash="dash")))
        fig_t.add_hline(y=tmean, line_dash="dot", line_color="gray", annotation_text="Moyenne")
        fig_t.update_layout(height=310, margin=dict(t=10,b=20,l=0,r=0),
                             yaxis_title="°C", legend=dict(orientation="h",y=-0.2))
        st.plotly_chart(fig_t, use_container_width=True)
        st.caption("Rouge = année plus chaude que la moyenne de la période.")

    with col_ts4:
        st.subheader("💧 Bilan hydrique annuel")
        bilan = df_f.groupby("year").agg(precip=("precip_annual_mm","median"),
                                          ET=("pot_ET_annual_mm","median")).reset_index()
        fig_wf = go.Figure()
        fig_wf.add_trace(go.Bar(x=bilan["year"], y=bilan["precip"],
                                 name="Précipitations reçues", marker_color="#4e9af1"))
        fig_wf.add_trace(go.Bar(x=bilan["year"], y=-bilan["ET"],
                                 name="ET Potentielle perdue", marker_color="#e55"))
        fig_wf.add_hline(y=0, line_color="black", line_width=1)
        fig_wf.update_layout(barmode="overlay", height=310, margin=dict(t=10,b=20,l=0,r=0),
                              yaxis_title="mm/an", legend=dict(orientation="h",y=-0.2), hovermode="x unified")
        st.plotly_chart(fig_wf, use_container_width=True)
        st.caption("Les barres rouges (ET) dépassent systématiquement les bleues — c'est 'The Invisible Rain'.")

# ══ ONGLET 3 — SAISONNALITÉ ══════════════════════════════════
with tab3:
    st.subheader("🌿 Analyse saisonnière")
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        seasonal = df_monthly.groupby("month").agg(
            precip=("precip_mm","median"), pot_ET=("pot_ET_mm","median")).reset_index()
        seasonal["mois"] = seasonal["month"].map(month_names)
        fig_sea = go.Figure()
        fig_sea.add_trace(go.Bar(x=seasonal["mois"], y=seasonal["precip"],
                                  name="Précipitations (mm)", marker_color="#4e9af1"))
        fig_sea.add_trace(go.Scatter(x=seasonal["mois"], y=seasonal["pot_ET"],
                                      name="ET Potentielle (mm)", mode="lines+markers",
                                      line=dict(color="#e55",width=2.5)))
        fig_sea.add_annotation(text="⚠️ L'ET dépasse les précipitations chaque mois",
                                xref="paper", yref="paper", x=0.5, y=1.08,
                                showarrow=False, font=dict(size=11,color="red"))
        fig_sea.update_layout(height=330, margin=dict(t=40,b=20,l=0,r=0),
                               yaxis_title="mm/mois", legend=dict(orientation="h",y=-0.2))
        st.plotly_chart(fig_sea, use_container_width=True)
        st.caption("Cycle saisonnier médian 2000–2024.")

    with col_s2:
        df_monthly["season"] = df_monthly["month"].map({
            12:"Hiver",1:"Hiver",2:"Hiver",3:"Printemps",4:"Printemps",5:"Printemps",
            6:"Été",7:"Été",8:"Été",9:"Automne",10:"Automne",11:"Automne"
        })
        fig_sbox = px.box(df_monthly, x="season", y="ratio_ET_P",
                           category_orders={"season":["Hiver","Printemps","Été","Automne"]},
                           color="season",
                           color_discrete_map={"Hiver":"#2196F3","Printemps":"#4CAF50",
                                               "Été":"#FF5722","Automne":"#FF9800"},
                           points="outliers", height=330)
        fig_sbox.add_hline(y=1, line_dash="dot", line_color="black", annotation_text="Équilibre ET=P")
        fig_sbox.update_layout(margin=dict(t=10,b=20,l=0,r=0),
                                yaxis_title="Ratio ET/P mensuel", showlegend=False)
        st.plotly_chart(fig_sbox, use_container_width=True)
        st.caption("L'été montre des valeurs extrêmes — eau quasi totalement perdue par évaporation.")

    st.divider()
    col_s3, col_s4 = st.columns(2)

    with col_s3:
        st.subheader("🗓️ Heatmap Mois × Année")
        hm_data = df_monthly.groupby(["year","month"])["ratio_ET_P_capped"].median().reset_index()
        hm_pivot = hm_data.pivot(index="month", columns="year", values="ratio_ET_P_capped")
        hm_pivot.index = [month_names[m] for m in hm_pivot.index]
        fig_hm2 = px.imshow(hm_pivot, color_continuous_scale="RdYlBu_r",
                             aspect="auto", height=360, labels=dict(color="Ratio ET/P"))
        fig_hm2.update_layout(margin=dict(t=10,b=20,l=0,r=0),
                               xaxis_title="Année", yaxis_title="Mois")
        st.plotly_chart(fig_hm2, use_container_width=True)
        st.caption("Rouge = déficit extrême. Juin–août systématiquement critiques.")

    with col_s4:
        st.subheader("📦 Distribution mensuelle")
        df_monthly["mois_nom"] = df_monthly["month"].map(month_names)
        fig_mbox = px.box(df_monthly, x="mois_nom", y="ratio_ET_P_capped",
                           category_orders={"mois_nom":list(month_names.values())},
                           color_discrete_sequence=["#e55"], points=False, height=360)
        fig_mbox.update_layout(margin=dict(t=10,b=20,l=0,r=0),
                                xaxis_title="Mois", yaxis_title="Ratio ET/P (plafonné à 15)")
        st.plotly_chart(fig_mbox, use_container_width=True)
        st.caption("Échelle plafonnée à 15 pour la lisibilité — les valeurs réelles d'été dépassent 100.")

# ══ ONGLET 4 — ANALYSE STATISTIQUE ═══════════════════════════
with tab4:
    st.subheader("🔬 Relations entre variables")
    col_a1, col_a2 = st.columns(2)

    with col_a1:
        df_sc = df_f.groupby("year").agg(temp=("temp_mean_C","mean"),
                                          pot_ET=("pot_ET_annual_mm","median"),
                                          precip=("precip_annual_mm","median"),
                                          ratio=("ratio_ET_P_annual","median")).reset_index()
        fig_sc = px.scatter(df_sc, x="temp", y="pot_ET", size="ratio", color="year",
                             color_continuous_scale="RdYlBu_r",
                             hover_data={"year":True,"precip":":.1f","ratio":":.1f"},
                             labels={"temp":"Température (°C)","pot_ET":"ET Potentielle (mm/an)",
                                     "year":"Année","ratio":"Ratio ET/P"},
                             trendline="ols", height=370)
        fig_sc.update_layout(margin=dict(t=10,b=20,l=0,r=0))
        st.plotly_chart(fig_sc, use_container_width=True)
        st.caption("Chaque point = une année. Taille = intensité du déficit. T° ↑ → ET ↑.")

    with col_a2:
        df_corr = df_f.groupby("year").agg(
            Précipitations=("precip_annual_mm","median"),
            ET_Potentielle=("pot_ET_annual_mm","median"),
            ET_Réelle=("actual_ET_annual_mm","median"),
            Température=("temp_mean_C","mean"),
            Humidité_Sol=("soil_moisture_mean","mean"),
            Ratio_ET_P=("ratio_ET_P_annual","median")
        )
        fig_hm = px.imshow(df_corr.corr().round(2), text_auto=True,
                            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                            aspect="auto", height=370)
        fig_hm.update_layout(margin=dict(t=10,b=20,l=0,r=0))
        st.plotly_chart(fig_hm, use_container_width=True)
        st.caption("Rouge = corrélation positive · Bleu = négative · Coefficients de Pearson.")

    st.divider()
    col_a3, col_a4 = st.columns(2)

    with col_a3:
        st.subheader("💧 Précipitations vs Humidité du sol")
        df_sc2 = df_f.groupby("year").agg(precip=("precip_annual_mm","median"),
                                            soil=("soil_moisture_mean","mean"),
                                            ratio=("ratio_ET_P_annual","median")).reset_index()
        fig_sc2 = px.scatter(df_sc2, x="precip", y="soil", color="ratio", size="ratio",
                              color_continuous_scale="RdYlBu",
                              hover_data={"year":True},
                              labels={"precip":"Précipitations (mm/an)",
                                      "soil":"Humidité du sol (m³/m³)","ratio":"Ratio ET/P"},
                              trendline="ols", height=320)
        fig_sc2.update_layout(margin=dict(t=10,b=20,l=0,r=0))
        st.plotly_chart(fig_sc2, use_container_width=True)
        st.caption("Plus les précipitations baissent, plus le sol s'assèche — lien direct avec la sécurité alimentaire.")

    with col_a4:
        st.subheader("📊 Indicateurs — Regard critique")
        temp_recent = df_f[df_f["year"] >= 2020]["temp_mean_C"].mean()
        temp_early  = df_f[df_f["year"] <= 2005]["temp_mean_C"].mean()
        pct_critical = (df_f["ratio_ET_P_annual"] > 50).mean() * 100
        ts_p = df_f.groupby("year")["precip_annual_mm"].median()
        slope_p = np.polyfit(ts_p.index, ts_p.values, 1)[0]
        st.markdown(f"""
        <div class="kpi-card" style="border-color:#d62728;">
            <b>📅 Année la plus critique</b><br>
            <span style="font-size:24px;font-weight:bold;color:#d62728;">{worst_year}</span>
            &nbsp;— Ratio ET/P : <b>{df_yr[worst_year]:.1f}x</b>
        </div>
        <div class="kpi-card" style="border-color:#4CAF50;">
            <b>✅ Meilleure année hydrique</b><br>
            <span style="font-size:24px;font-weight:bold;color:#2e7d32;">{best_year}</span>
            &nbsp;— Ratio ET/P : <b>{df_yr[best_year]:.1f}x</b>
        </div>
        <div class="kpi-card" style="border-color:#e91e63;">
            <b>🌡️ Hausse T° (2020–2024 vs 2000–2005)</b><br>
            <span style="font-size:24px;font-weight:bold;color:#c62828;">+{temp_recent-temp_early:.2f} °C</span>
        </div>
        <div class="kpi-card" style="border-color:#2196F3;">
            <b>⚠️ Pixels en déficit extrême (ratio &gt; 50)</b><br>
            <span style="font-size:24px;font-weight:bold;color:#1565c0;">{pct_critical:.1f}%</span> des pixels
        </div>
        <div class="kpi-card" style="border-color:#FF9800;">
            <b>📉 Tendance précipitations</b><br>
            <span style="font-size:24px;font-weight:bold;color:#e65100;">{slope_p:+.1f} mm/an</span>
        </div>
        """, unsafe_allow_html=True)

# ══ ONGLET 5 — ANOMALIES & DÉCENNIES ═════════════════════════
with tab5:
    st.subheader("⚠️ Détection d'anomalies & Comparaison décennale")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("**🔴 Années anomales — Z-score > 2**")
        ts_z = df_f.groupby("year")["ratio_ET_P_annual"].median().reset_index()
        z_scores = np.abs(stats.zscore(ts_z["ratio_ET_P_annual"]))
        ts_z["z_score"] = z_scores
        ts_z["anomalie"] = z_scores > 2
        fig_z = go.Figure()
        fig_z.add_trace(go.Bar(
            x=ts_z["year"], y=ts_z["ratio_ET_P_annual"],
            marker_color=["#d62728" if a else "#4e9af1" for a in ts_z["anomalie"]],
            text=[f"⚠️ Z={z:.1f}" if a else "" for z, a in zip(ts_z["z_score"], ts_z["anomalie"])],
            textposition="outside"
        ))
        fig_z.update_layout(height=330, margin=dict(t=10,b=20,l=0,r=0),
                             yaxis_title="Ratio ET/P médian")
        st.plotly_chart(fig_z, use_container_width=True)
        anomalies = ts_z[ts_z["anomalie"]]["year"].tolist()
        st.caption(f"Années anomales (Z > 2) : **{anomalies}** — déficit exceptionnellement élevé.")

    with col_d2:
        st.markdown("**📊 Comparaison par décennie**")
        dec = df_f.dropna(subset=["decade"]).groupby("decade").agg(
            ratio_med=("ratio_ET_P_annual","median"),
            precip_med=("precip_annual_mm","median"),
            temp_mean=("temp_mean_C","mean"),
            ET_med=("pot_ET_annual_mm","median")
        ).reset_index()
        fig_dec = go.Figure()
        for var, name, color in zip(
            ["ratio_med","precip_med","temp_mean","ET_med"],
            ["Ratio ET/P","Précip (mm)","Température (°C)","ET Pot. (mm)"],
            ["#d62728","#4e9af1","#FF9800","#e55"]
        ):
            fig_dec.add_trace(go.Bar(name=name, x=dec["decade"], y=dec[var],
                                      marker_color=color, opacity=0.85))
        fig_dec.update_layout(barmode="group", height=330, margin=dict(t=10,b=20,l=0,r=0),
                               legend=dict(orientation="h",y=-0.2), yaxis_title="Valeur médiane")
        st.plotly_chart(fig_dec, use_container_width=True)
        st.caption("Le déficit s'accélère décennie après décennie.")

    st.divider()
    st.subheader("📦 Distribution du ratio ET/P par décennie")
    df_dec_box = df_f.dropna(subset=["decade"])
    df_dec_box = df_dec_box[df_dec_box["ratio_ET_P_annual"] <= 100]
    fig_dec_box = px.box(df_dec_box, x="decade", y="ratio_ET_P_annual", color="decade",
                          color_discrete_map={"2000–2009":"#4e9af1","2010–2019":"#FF9800","2020–2024":"#d62728"},
                          points="outliers", height=320)
    fig_dec_box.add_hline(y=1, line_dash="dot", line_color="black", annotation_text="Équilibre ET=P")
    fig_dec_box.update_layout(margin=dict(t=10,b=20,l=0,r=0),
                               yaxis_title="Ratio ET/P annuel", showlegend=False)
    st.plotly_chart(fig_dec_box, use_container_width=True)
    st.caption("La distribution s'élève et s'élargit chaque décennie — aggravation et variabilité croissante.")

# ── FOOTER ────────────────────────────────────────────────────
st.divider()
st.caption(
    "📡 ERA5-Land Monthly Means · CDS Copernicus · 0.1°×0.1° · 2000–2024 · "
    "Variables : tp, pev, e, t2m, swvl1 · Projet : Données climatiques R→Python · 2026"
)
