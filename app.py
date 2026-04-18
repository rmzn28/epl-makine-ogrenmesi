import streamlit as st
import pandas as pd
import shap
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import poisson
from itertools import permutations

# Plotly toolbar gizleme (native app hissi)
PLOTLY_CFG = {'displayModeBar': False}

# Sayfa Ayarları
st.set_page_config(page_title="EPL Match Predictor & Analytics", layout="wide", page_icon="⚽")

# ==================== SABIT SÖZLÜKLER ====================
TEAM_COLORS = {
    'Arsenal': '#EF0107', 'Aston Villa': '#670E36', 'Bournemouth': '#DA291C',
    'Brentford': '#E30613', 'Brighton': '#0057B8', 'Burnley': '#6C1D45',
    'Cardiff': '#0070B5', 'Chelsea': '#034694', 'Crystal Palace': '#1B458F',
    'Everton': '#003399', 'Fulham': '#000000', 'Huddersfield': '#0E63AD',
    'Hull': '#F5A623', 'Ipswich': '#0044AA', 'Leeds': '#FFCD00',
    'Leicester': '#003090', 'Liverpool': '#C8102E', 'Manchester City': '#6CABDD',
    'Manchester United': '#DA291C', 'Middlesbrough': '#E11B22', 'Newcastle': '#241F20',
    'Norwich': '#00A650', 'Nottingham Forest': '#DD0000', 'Queens Park Rangers': '#1D5BA4',
    'Sheffield United': '#EE2737', 'Southampton': '#D71920', 'Stoke': '#E03A3E',
    'Sunderland': '#EB172B', 'Swansea': '#121212', 'Tottenham': '#132257',
    'Watford': '#FBEE23', 'West Brom': '#122F67', 'West Ham': '#7A263A', 'Wolves': '#FDB913',
}
DEFAULT_COLOR = '#555555'

def get_team_color(name):
    return TEAM_COLORS.get(name, DEFAULT_COLOR)

FEATURE_LABELS = {
    'Home_AvgScored_3': 'Ev Sahibi: Son 3 Maç Atılan Gol Ort.',
    'Home_AvgScored_5': 'Ev Sahibi: Son 5 Maç Atılan Gol Ort.',
    'Home_AvgConceded_3': 'Ev Sahibi: Son 3 Maç Yenilen Gol Ort.',
    'Home_AvgConceded_5': 'Ev Sahibi: Son 5 Maç Yenilen Gol Ort.',
    'Home_AvgShots_3': 'Ev Sahibi: Son 3 Maç İsabetli Şut Ort.',
    'Home_AvgShots_5': 'Ev Sahibi: Son 5 Maç İsabetli Şut Ort.',
    'Home_AvgPoints_3': 'Ev Sahibi: Son 3 Maç Puan Ort.',
    'Home_AvgPoints_5': 'Ev Sahibi: Son 5 Maç Puan Ort.',
    'Away_AvgScored_3': 'Deplasman: Son 3 Maç Atılan Gol Ort.',
    'Away_AvgScored_5': 'Deplasman: Son 5 Maç Atılan Gol Ort.',
    'Away_AvgConceded_3': 'Deplasman: Son 3 Maç Yenilen Gol Ort.',
    'Away_AvgConceded_5': 'Deplasman: Son 5 Maç Yenilen Gol Ort.',
    'Away_AvgShots_3': 'Deplasman: Son 3 Maç İsabetli Şut Ort.',
    'Away_AvgShots_5': 'Deplasman: Son 5 Maç İsabetli Şut Ort.',
    'Away_AvgPoints_3': 'Deplasman: Son 3 Maç Puan Ort.',
    'Away_AvgPoints_5': 'Deplasman: Son 5 Maç Puan Ort.',
    'H2H_Home_WinRate_5': 'İkili: Ev Sahibi Son 5 H2H Galibiyet %',
}

def get_readable_label(raw):
    return FEATURE_LABELS.get(raw, raw)

# ==================== MODEL YÜKLEME ====================
@st.cache_resource
def load_assets():
    try:
        clf_models = joblib.load('models/clf_models.pkl')
        reg_models = joblib.load('models/reg_models.pkl')
        metrics_data = joblib.load('models/evaluation_metrics.pkl')
        feature_columns = joblib.load('models/feature_columns.pkl')
        le = joblib.load('models/label_encoder.pkl')
        latest_stats = joblib.load('models/latest_stats.pkl')
        return clf_models, reg_models, metrics_data, feature_columns, le, latest_stats
    except FileNotFoundError:
        st.error("Model dosyaları bulunamadı. Lütfen önce `python train_model.py` çalıştırın.")
        return None, None, None, None, None, None

clf_models, reg_models, evaluation_metrics, feature_columns, le, latest_stats = load_assets()
if clf_models is None:
    st.stop()

team_stats = latest_stats['team_stats']
h2h_stats = latest_stats['h2h_stats']
team_last_date = latest_stats.get('team_last_date', {})
teams = sorted(latest_stats.get('current_teams', list(team_stats.keys())))
current_points = latest_stats.get('current_points', {})
played_fixtures = latest_stats.get('played_fixtures', [])
played_results = latest_stats.get('played_results', [])
historical_records = latest_stats.get('historical_records', [])

# ==================== YARDIMCI FONKSİYONLAR ====================
def get_rolling_avg(team, stat_name, window):
    history = team_stats[team].get(stat_name, [])
    if len(history) == 0: return 0.0
    elif len(history) < window: return float(np.mean(history))
    else: return float(np.mean(history[-window:]))

def get_form_guide(team, n=5):
    form = team_stats[team].get('form', [])
    recent = form[-n:] if len(form) >= n else form
    return ' '.join({'W': '🟩', 'D': '🟨', 'L': '🟥'}.get(r, '⬜') for r in recent)

def build_feature_vector(home, away):
    row_features = {}
    for window in [3, 5]:
        row_features[f'Home_AvgScored_{window}'] = get_rolling_avg(home, 'scored', window)
        row_features[f'Home_AvgConceded_{window}'] = get_rolling_avg(home, 'conceded', window)
        row_features[f'Home_AvgShots_{window}'] = get_rolling_avg(home, 'shots', window)
        row_features[f'Home_AvgPoints_{window}'] = get_rolling_avg(home, 'points', window)
        row_features[f'Away_AvgScored_{window}'] = get_rolling_avg(away, 'scored', window)
        row_features[f'Away_AvgConceded_{window}'] = get_rolling_avg(away, 'conceded', window)
        row_features[f'Away_AvgShots_{window}'] = get_rolling_avg(away, 'shots', window)
        row_features[f'Away_AvgPoints_{window}'] = get_rolling_avg(away, 'points', window)
    h2h_key = tuple(sorted([home, away]))
    past_h2h = h2h_stats.get(h2h_key, [])[-5:]
    row_features['H2H_Home_WinRate_5'] = sum(1 for w in past_h2h if w == home) / len(past_h2h) if past_h2h else 0.5
    
    df_raw = pd.DataFrame([row_features])
    df_encoded = pd.DataFrame(columns=feature_columns)
    df_encoded.loc[0] = 0
    for col in df_raw.columns:
        if col in df_encoded.columns:
            df_encoded.at[0, col] = df_raw.at[0, col]
    home_col = f'HomeTeam_{home}'
    away_col = f'AwayTeam_{away}'
    if home_col in df_encoded.columns: df_encoded.at[0, home_col] = 1
    if away_col in df_encoded.columns: df_encoded.at[0, away_col] = 1
    return df_encoded

# ==================== STREAMLIT ARAYÜZÜ ====================
st.title("⚽ Premier League Match Predictor & Analytics")
st.markdown("Canlı verilerle beslenen, tamamen özelleştirilebilir model analitik paneli.")

if 'simulated' not in st.session_state:
    st.session_state.simulated = False

# --- Sidebar ---
st.sidebar.header("Match Selection")
home_team = st.sidebar.selectbox("🏠 Ev Sahibi", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
away_team = st.sidebar.selectbox("✈️ Deplasman", teams, index=teams.index("Chelsea") if "Chelsea" in teams else 1)

if home_team == away_team:
    st.sidebar.error("Lütfen farklı iki takım seçin.")
    st.stop()

st.sidebar.markdown("---")
clf_choice = st.sidebar.selectbox("🧠 Sınıflandırma Modeli (Sonuç)", list(clf_models.keys()), index=list(clf_models.keys()).index("Logistic Regression") if "Logistic Regression" in clf_models else 0)
reg_choice = st.sidebar.selectbox("📈 Regresyon Modeli (Skor/İstat)", list(reg_models.keys()), index=list(reg_models.keys()).index("Random Forest") if "Random Forest" in reg_models else 0)

selected_clf = clf_models[clf_choice]
selected_reg_models = reg_models[reg_choice]

st.sidebar.markdown("---")
form_html_h = f"<div style='white-space: nowrap;'><b>{home_team} Form:</b> {get_form_guide(home_team)}</div>"
form_html_a = f"<div style='white-space: nowrap;'><b>{away_team} Form:</b> {get_form_guide(away_team)}</div>"
st.sidebar.markdown(form_html_h, unsafe_allow_html=True)
st.sidebar.markdown(form_html_a, unsafe_allow_html=True)
st.sidebar.caption("🟩 Galibiyet | 🟨 Beraberlik | 🟥 Mağlubiyet")

st.sidebar.markdown("---")
if st.sidebar.button("Simüle Et", use_container_width=True):
    st.session_state.simulated = True

if not st.session_state.simulated:
    st.info("👆 Lütfen yan menüden bir takım ve modelleri seçip **'Simüle Et'** butonuna tıklayın.")
    st.stop()

# --- Tahminler ---
feature_vector = build_feature_vector(home_team, away_team)
probabilities = selected_clf.predict_proba(feature_vector)[0]
class_labels = le.classes_
prob_dict = {l: p for l, p in zip(class_labels, probabilities)}
home_prob, draw_prob, away_prob = prob_dict.get('H', 0), prob_dict.get('D', 0), prob_dict.get('A', 0)

pred_hg = float(selected_reg_models['FTHG'].predict(feature_vector)[0])
pred_ag = float(selected_reg_models['FTAG'].predict(feature_vector)[0])
exact_h, exact_a = max(0, int(round(pred_hg))), max(0, int(round(pred_ag)))

# ==================== TABS ====================
tab1, tab2, tab3, tab4 = st.tabs(["⚽ Maç Tahmini", "📈 Tarihsel Üstünlük", "🏆 Sezon Simülasyonu", "📊 Model Değerlendirme"])

# ==================== TAB 1: MAÇ TAHMİNİ ====================
with tab1:
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.subheader(f"📊 Olasılıklar ({clf_choice})")
        pie_colors = [get_team_color(home_team), '#808080', get_team_color(away_team)]
        fig = go.Figure(data=[go.Pie(
            labels=[home_team, 'Beraberlik', away_team], values=[home_prob, draw_prob, away_prob],
            hole=.4, marker_colors=pie_colors, textinfo='label+percent', textfont_size=13
        )])
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)

    with col2:
        st.subheader(f"🎯 Kesin Skor Tahmini ({reg_choice})")
        hc, ac = get_team_color(home_team), get_team_color(away_team)
        st.markdown(
            f"<div style='text-align:center; padding:15px;'>"
            f"<span style='font-size:1.3em; font-weight:bold; color:{hc};'>{home_team}</span><br>"
            f"<span style='font-size:3em; font-weight:bold; color:{hc};'>{exact_h}</span>"
            f"<span style='font-size:2em; color:#888;'> - </span>"
            f"<span style='font-size:3em; font-weight:bold; color:{ac};'>{exact_a}</span><br>"
            f"<span style='font-size:1.3em; font-weight:bold; color:{ac};'>{away_team}</span></div>",
            unsafe_allow_html=True
        )
        st.caption(f"Raw: {pred_hg:.2f} - {pred_ag:.2f}")

    # Beklenen Maç İstatistikleri
    if set(['HC', 'AC', 'HST', 'AST', 'HS', 'AS', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']).issubset(selected_reg_models.keys()):
        st.write("---")
        st.subheader(f"📊 Beklenen Maç İstatistikleri ({reg_choice})")
        pred_stats = {}
        for sk in ['HC', 'AC', 'HST', 'AST', 'HS', 'AS', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']:
            pred_stats[sk] = round(max(0, float(selected_reg_models[sk].predict(feature_vector)[0])), 1)

        categories = ['Kırmızı Kart (Red Cards)', 'Sarı Kart (Yel. Cards)', 'Faul (Fouls)', 'Korner (Corners)', 'İsabetli Şut (SoT)', 'Toplam Şut (Shots)']
        home_values = [pred_stats['HR'], pred_stats['HY'], pred_stats['HF'], pred_stats['HC'], pred_stats['HST'], pred_stats['HS']]
        away_values = [pred_stats['AR'], pred_stats['AY'], pred_stats['AF'], pred_stats['AC'], pred_stats['AST'], pred_stats['AS']]

        fig_stats = go.Figure()
        fig_stats.add_trace(go.Bar(
            y=categories, x=home_values, name=home_team, orientation='h',
            marker_color=get_team_color(home_team), text=home_values, textposition='outside',
            textfont=dict(size=13, color=get_team_color(home_team))
        ))
        fig_stats.add_trace(go.Bar(
            y=categories, x=away_values, name=away_team, orientation='h',
            marker_color=get_team_color(away_team), text=away_values, textposition='outside',
            textfont=dict(size=13, color=get_team_color(away_team))
        ))
        fig_stats.update_layout(
            barmode='group', margin=dict(t=10, b=0, l=0, r=40),
            legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='right', x=1),
            yaxis=dict(tickfont=dict(size=13)), height=450
        )
        fig_stats.update_xaxes(title_text='')
        st.plotly_chart(fig_stats, use_container_width=True, config=PLOTLY_CFG)

    st.write("---")

    # Poisson + Radar
    ch, cr = st.columns(2)
    with ch:
        st.subheader(f"🔥 Skor Olasılık Matrisi (Poisson)")
        mg = 6
        lh, la = max(0.01, pred_hg), max(0.01, pred_ag)
        pm = np.array([[poisson.pmf(i, lh) * poisson.pmf(j, la) for j in range(mg)] for i in range(mg)])
        fig_h = px.imshow(np.round(pm * 100, 2), labels=dict(x=f"{away_team} Gol", y=f"{home_team} Gol", color="%"),
                          x=[str(i) for i in range(mg)], y=[str(i) for i in range(mg)],
                          color_continuous_scale='YlOrRd', text_auto=True, aspect='equal')
        fig_h.update_layout(margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig_h, use_container_width=True, config=PLOTLY_CFG)

    with cr:
        st.subheader("📡 Radar Karşılaştırması")
        cats = ['Atılan Gol', 'Savunma Gücü', 'İsabetli Şut', 'Puan']
        hc_raw = get_rolling_avg(home_team, 'conceded', 5)
        ac_raw = get_rolling_avg(away_team, 'conceded', 5)
        mc = max(hc_raw, ac_raw, 2.5)
        hv = [get_rolling_avg(home_team, 'scored', 5), mc - hc_raw + 0.2,
              get_rolling_avg(home_team, 'shots', 5), get_rolling_avg(home_team, 'points', 5)]
        av = [get_rolling_avg(away_team, 'scored', 5), mc - ac_raw + 0.2,
              get_rolling_avg(away_team, 'shots', 5), get_rolling_avg(away_team, 'points', 5)]
        fr = go.Figure()
        for vals, name in [(hv, home_team), (av, away_team)]:
            fr.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]], fill='toself',
                                          name=name, line_color=get_team_color(name),
                                          fillcolor=get_team_color(name), opacity=0.35))
        fr.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(max(hv), max(av)) * 1.2])),
            margin=dict(t=30, b=0, l=60, r=60), showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5))
        st.plotly_chart(fr, use_container_width=True, config=PLOTLY_CFG)

    st.write("---")

    # SHAP
    st.subheader(f"🔬 Yerel Etmenler (SHAP) - {clf_choice}")
    with st.spinner("SHAP hesaplanıyor..."):
        bg = pd.DataFrame(np.zeros((1, feature_vector.shape[1])), columns=feature_columns)
        pidx = int(np.argmax(probabilities))
        plabel = class_labels[pidx]
        try:
            exp = shap.TreeExplainer(selected_clf)
            sv_raw = exp.shap_values(feature_vector)
            if isinstance(sv_raw, list): sv = sv_raw[pidx][0]
            elif len(sv_raw.shape) == 3: sv = sv_raw[0, :, pidx]
            else: sv = sv_raw[0]
        except Exception:
            try:
                exp = shap.KernelExplainer(selected_clf.predict_proba, bg)
                sv_raw = exp.shap_values(feature_vector)
                if isinstance(sv_raw, list): sv = sv_raw[pidx][0]
                else: sv = sv_raw[0, :, pidx] if len(sv_raw.shape) == 3 else sv_raw[0]
            except Exception:
                sv = np.zeros(feature_vector.shape[1])

        fi = [{'Feature': get_readable_label(f), 'SHAP Value': float(s), 'Abs': abs(float(s))}
              for f, s in zip(feature_columns, sv)
              if not (f.startswith('HomeTeam_') or f.startswith('AwayTeam_') or f.startswith('Team_'))]
        fdf = pd.DataFrame(fi).sort_values('Abs', ascending=False).head(10)
        fdf['Yön'] = fdf['SHAP Value'].apply(lambda x: 'Arttırdı ↑' if x > 0 else 'Azalttı ↓')

        if plabel == 'H': stitle = f'Ev Sahibi ({home_team}) Galibiyetini Tetikleyen 10 Faktör'
        elif plabel == 'A': stitle = f'Deplasman ({away_team}) Galibiyetini Tetikleyen 10 Faktör'
        else: stitle = 'Beraberlik Sonucunu Tetikleyen 10 Faktör'

        ff = px.bar(fdf, x='SHAP Value', y='Feature', orientation='h', color='Yön',
                    color_discrete_map={'Arttırdı ↑': '#2ca02c', 'Azalttı ↓': '#d62728'}, title=stitle)
        ff.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(ff, use_container_width=True, config=PLOTLY_CFG)
    st.info("**💡** Bu lokal SHAP analizi, modelin TAM BU MAÇ için hangi istatistiklerden etkilendiğini gösterir. "
            "Sağa (yeşil) çekenler olasılığı arttırmış, sola (kırmızı) çekenler azaltmıştır.")

# ==================== TAB 2: TARİHSEL ÜSTÜNLÜK ====================
with tab2:
    st.subheader(f"📈 Tarihsel Üstünlük: {home_team} vs {away_team}")
    st.markdown("Son 10+ yıllık 38 maçlık hareketli ortalama puanlar ile takımların güç dönemlerini karşılaştırın.")

    if historical_records:
        hist_df = pd.DataFrame(historical_records)
        hist_df['Date'] = pd.to_datetime(hist_df['Date'])

        team_timeline = {}
        for team_name in [home_team, away_team]:
            home_mask = hist_df['HomeTeam'] == team_name
            away_mask = hist_df['AwayTeam'] == team_name
            home_rows = hist_df[home_mask][['Date', 'FTR']].copy()
            home_rows['Points'] = home_rows['FTR'].map({'H': 3, 'D': 1, 'A': 0})
            away_rows = hist_df[away_mask][['Date', 'FTR']].copy()
            away_rows['Points'] = away_rows['FTR'].map({'A': 3, 'D': 1, 'H': 0})
            combined = pd.concat([home_rows, away_rows]).sort_values('Date').reset_index(drop=True)
            combined['RollingAvg'] = combined['Points'].rolling(window=38, min_periods=10).mean()
            combined = combined.dropna(subset=['RollingAvg'])
            team_timeline[team_name] = combined

        def hex_to_rgba(hex_col, alpha=0.5):
            hex_col = hex_col.lstrip('#')
            if len(hex_col) == 6:
                return f'rgba({int(hex_col[:2], 16)}, {int(hex_col[2:4], 16)}, {int(hex_col[4:], 16)}, {alpha})'
            return hex_col

        fig_hist = go.Figure()
        for tn in [home_team, away_team]:
            if tn in team_timeline and len(team_timeline[tn]) > 0:
                tl = team_timeline[tn]
                fig_hist.add_trace(go.Scatter(
                    x=tl['Date'], y=tl['RollingAvg'], name=tn,
                    fill='tozeroy', line=dict(color=get_team_color(tn), width=2),
                    fillcolor=hex_to_rgba(get_team_color(tn), 0.5), opacity=0.5
                ))
        fig_hist.update_layout(
            title="38 Maçlık Hareketli Ortalama Puan (Maç Başına)",
            xaxis_title="Tarih", yaxis_title="Ortalama Puan / Maç",
            hovermode='x unified', margin=dict(t=40, b=0, l=0, r=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
        )
        st.plotly_chart(fig_hist, use_container_width=True, config=PLOTLY_CFG)
        st.info("**💡** 38 maçlık pencere tam bir sezona denk gelir. Zirve dönemleri, kadro yatırımları ve düşüşler grafikten takip edilebilir.")
    else:
        st.warning("Tarihsel veri bulunamadı. Lütfen `train_model.py` dosyasını yeniden çalıştırın.")

# ==================== TAB 3: MONTE CARLO SİMÜLASYONU ====================
with tab3:
    st.subheader("🏆 Sezon Sonu Simülasyonu (Monte Carlo)")
    st.markdown("Kalan maçlar belirtilen regresyon ve sınıflandırma kurallarıyla Poisson dağılımı kullanılarak simüle edilir.")

    @st.cache_data(show_spinner=False)
    def run_monte_carlo(_sel_reg_models, _feature_columns, _teams, _played_fixtures, _current_points, _team_stats, _h2h_stats, _team_last_date, n_sims=10000):
        full_fixtures = [(h, a) for h, a in permutations(_teams, 2)]
        played_set = set(_played_fixtures)
        remaining = [(h, a) for h, a in full_fixtures if (h, a) not in played_set]
        if len(remaining) == 0:
            return None
        fixture_lambdas = []
        for h, a in remaining:
            fv = _build_sim_vector(h, a, _feature_columns, _team_stats, _h2h_stats, _team_last_date)
            lh = max(0.1, float(_sel_reg_models['FTHG'].predict(fv)[0]))
            la = max(0.1, float(_sel_reg_models['FTAG'].predict(fv)[0]))
            fixture_lambdas.append((h, a, lh, la))
            
        title_counts = {t: 0 for t in _teams}
        top4_counts = {t: 0 for t in _teams}
        relegation_counts = {t: 0 for t in _teams}
        total_points_sum = {t: 0 for t in _teams}
        rng = np.random.default_rng(42)
        for _ in range(n_sims):
            sim_points = dict(_current_points)
            for h, a, lh, la in fixture_lambdas:
                hg = rng.poisson(lh)
                ag = rng.poisson(la)
                if hg > ag: sim_points[h] = sim_points.get(h, 0) + 3
                elif ag > hg: sim_points[a] = sim_points.get(a, 0) + 3
                else:
                    sim_points[h] = sim_points.get(h, 0) + 1
                    sim_points[a] = sim_points.get(a, 0) + 1
            ranking = sorted(sim_points.items(), key=lambda x: x[1], reverse=True)
            for i, (team, pts) in enumerate(ranking):
                total_points_sum[team] += pts
                if i == 0: title_counts[team] += 1
                if i < 4: top4_counts[team] += 1
                if i >= len(ranking) - 3: relegation_counts[team] += 1
        results = []
        for t in _teams:
            results.append({
                'Takım': t, 'Ort. Puan': round(total_points_sum[t] / n_sims, 1),
                'Şampiyon %': round(title_counts[t] / n_sims * 100, 1),
                'İlk 4 (UCL) %': round(top4_counts[t] / n_sims * 100, 1),
                'Küme Düşme %': round(relegation_counts[t] / n_sims * 100, 1),
                'Mevcut Puan': _current_points.get(t, 0)
            })
        return pd.DataFrame(results).sort_values('Ort. Puan', ascending=False).reset_index(drop=True)

    def _build_sim_vector(home, away, feat_cols, t_stats, h2h_s, t_last_date):
        def _avg(team, stat, w):
            h = t_stats[team].get(stat, [])
            if len(h) == 0: return 0.0
            return float(np.mean(h[-w:])) if len(h) >= w else float(np.mean(h))
        rf = {}
        for w in [3, 5]:
            rf[f'Home_AvgScored_{w}'] = _avg(home, 'scored', w)
            rf[f'Home_AvgConceded_{w}'] = _avg(home, 'conceded', w)
            rf[f'Home_AvgShots_{w}'] = _avg(home, 'shots', w)
            rf[f'Home_AvgPoints_{w}'] = _avg(home, 'points', w)
            rf[f'Away_AvgScored_{w}'] = _avg(away, 'scored', w)
            rf[f'Away_AvgConceded_{w}'] = _avg(away, 'conceded', w)
            rf[f'Away_AvgShots_{w}'] = _avg(away, 'shots', w)
            rf[f'Away_AvgPoints_{w}'] = _avg(away, 'points', w)
        hk = tuple(sorted([home, away]))
        ph = h2h_s.get(hk, [])[-5:]
        rf['H2H_Home_WinRate_5'] = sum(1 for w in ph if w == home) / len(ph) if ph else 0.5
        de = pd.DataFrame(columns=feat_cols)
        de.loc[0] = 0
        for c in rf:
            if c in de.columns: de.at[0, c] = rf[c]
        hc = f'HomeTeam_{home}'
        ac = f'AwayTeam_{away}'
        if hc in de.columns: de.at[0, hc] = 1
        if ac in de.columns: de.at[0, ac] = 1
        return de

    if len(teams) == 20 and current_points:
        with st.spinner("🎲 10.000 sezon simüle ediliyor..."):
            sim_df = run_monte_carlo(
                selected_reg_models, feature_columns, teams,
                played_fixtures, current_points, team_stats, h2h_stats, team_last_date
            )

        if sim_df is not None:
            st.markdown(f"**Oynanan Maç:** {len(played_fixtures)} / 380 | **Kalan:** {380 - len(played_fixtures)}")

            st.dataframe(sim_df, use_container_width=True, height=740)

            top6 = sim_df.head(6)
            fig_mc = px.bar(top6, x='Takım', y='Şampiyon %',
                            color='Takım', color_discrete_map={t: get_team_color(t) for t in top6['Takım']},
                            title=f'Şampiyonluk Olasılığı (Poisson + {reg_choice})')
            fig_mc.update_layout(showlegend=False, margin=dict(t=40, b=0))
            st.plotly_chart(fig_mc, use_container_width=True, config=PLOTLY_CFG)

            st.write("---")

            # ==================== xPTS LUCK FACTOR ====================
            st.subheader(f"🎯 Performans ve Şans Faktörü (xPTS Analizi - {clf_choice})")
            st.markdown("Model, oynanan her maç için beklenen puanları (xPTS) hesaplar. **Gerçek puan - xPTS** farkı, şansın ne kadar etkili olduğunu gösterir.")

            if played_results:
                xpts = {t: 0.0 for t in teams}
                for match in played_results:
                    h, a = match['HomeTeam'], match['AwayTeam']
                    if h not in teams or a not in teams:
                        continue
                    try:
                        fv = _build_sim_vector(h, a, feature_columns, team_stats, h2h_stats, team_last_date)
                        probs = getattr(selected_clf, "predict_proba")(fv)[0]
                        prob_d = {cl: p for cl, p in zip(class_labels, probs)}
                        ph, pd_val, pa = prob_d.get('H', 0), prob_d.get('D', 0), prob_d.get('A', 0)
                        xpts[h] += ph * 3 + pd_val * 1
                        xpts[a] += pa * 3 + pd_val * 1
                    except Exception:
                        pass

                luck_data = []
                for t in teams:
                    actual = current_points.get(t, 0)
                    expected = round(xpts.get(t, 0), 1)
                    diff = round(actual - expected, 1)
                    luck_data.append({'Takım': t, 'Gerçek Puan': float(actual), 'xPTS': expected, 'Fark': diff})

                luck_df = pd.DataFrame(luck_data).sort_values('Fark', ascending=False)
                overachievers = luck_df.head(3)
                underachievers = luck_df.tail(3).sort_values('Fark')

                col_over, col_under = st.columns(2)
                with col_over:
                    st.markdown("##### 🍀 Beklentiyi Aşanlar (Overachievers)")
                    for _, row in overachievers.iterrows():
                        st.markdown(
                            f"<div style='padding:6px 12px; margin:4px 0; border-left:4px solid #2ca02c; background:#2ca02c11;'>"
                            f"<b>{row['Takım']}</b> — Gerçek: {row['Gerçek Puan']:.1f} | xPTS: {row['xPTS']:.1f} | "
                            f"<span style='color:#2ca02c; font-weight:bold;'>+{row['Fark']:.1f}</span></div>",
                            unsafe_allow_html=True
                        )
                with col_under:
                    st.markdown("##### 😤 Şanssızlar (Underachievers)")
                    for _, row in underachievers.iterrows():
                        st.markdown(
                            f"<div style='padding:6px 12px; margin:4px 0; border-left:4px solid #d62728; background:#d6272811;'>"
                            f"<b>{row['Takım']}</b> — Gerçek: {row['Gerçek Puan']:.1f} | xPTS: {row['xPTS']:.1f} | "
                            f"<span style='color:#d62728; font-weight:bold;'>{row['Fark']:.1f}</span></div>",
                            unsafe_allow_html=True
                        )

                st.info("**💡** xPTS, seçilen modelin her maçtaki tahmin olasılıklarından hesaplanır. "
                        "Pozitif fark 'şanslı/klinik' takımları, negatif fark 'şanssız/puan kaybeden' takımları gösterir.")
        else:
            st.success("🏁 Sezon tamamlanmış görünüyor. Tüm maçlar oynanmış!")
    else:
        st.warning("Monte Carlo simülasyonu için 20 aktif takım ve mevcut puan verisi gereklidir. `train_model.py` dosyasını çalıştırın.")

# ==================== TAB 4: MODEL DEĞERLENDİRME ====================
with tab4:
    st.subheader("Makine Öğrenmesi Model Performansları")
    st.markdown("Hem Sınıflandırma hem de Regresyon modellerinin Test seti üzerindeki performans değerlendirmeleri aşağıda verilmiştir.")
    
    st.markdown("#### 🧠 Sınıflandırma Modelleri (Match Outcome)")
    st.dataframe(evaluation_metrics['classification'], use_container_width=True)
    
    st.markdown("#### 📈 Regresyon Modelleri (Score & Statistics)")
    st.dataframe(evaluation_metrics['regression'], use_container_width=True)
