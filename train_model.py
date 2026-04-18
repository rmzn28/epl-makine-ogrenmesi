import pandas as pd
import numpy as np
import glob
import os
import joblib
import warnings
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, r2_score, brier_score_loss
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# ==================== TAKIM İSİM STANDARTLAŞTIRMA ====================
TEAM_MAPPING = {
    'Man United': 'Manchester United',
    'Man City': 'Manchester City',
    'Newcastle United': 'Newcastle',
    'Spurs': 'Tottenham',
    "Nott'm Forest": 'Nottingham Forest',
    'Sheff Utd': 'Sheffield United',
    'WBA': 'West Brom',
    'QPR': 'Queens Park Rangers'
}

def load_and_clean_data(data_path="Data/*.csv"):
    """
    Yerel CSV'leri yükler ve temizler (VPN/URL engeli nedeniyle sadece lokal).
    """
    print("Yerel veriler yükleniyor...")
    all_files = glob.glob(os.path.join(os.path.dirname(__file__), data_path))
    if not all_files:
        raise FileNotFoundError(f"Hiç CSV dosyası bulunamadı: {data_path}")
        
    df_list = []
    for f in all_files:
        temp_df = pd.read_csv(f, encoding='utf-8', on_bad_lines='skip')
        df_list.append(temp_df)
        
    df = pd.concat(df_list, ignore_index=True)
    
    # Gerekli sütunların olduğundan emin olalım
    required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kritik sütun eksik: {col}")
            
    # Eksik istatistik sütunlarını 0 ile doldur
    stat_cols = ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
    for sc in stat_cols:
        if sc not in df.columns: df[sc] = 0
    
    df = df[required_cols + stat_cols].copy()
    
    # Tarih formatını standartlaştır
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format="mixed")
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    # Takım isimlerini standartlaştır
    df['HomeTeam'] = df['HomeTeam'].replace(TEAM_MAPPING)
    df['AwayTeam'] = df['AwayTeam'].replace(TEAM_MAPPING)
    
    df = df.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], keep='last').reset_index(drop=True)
    
    print(f"Toplam benzersiz maç sayısı: {len(df)}")
    return df

def feature_engineering(df):
    """
    Geçmişe dönük hareketli ortalamaları (rolling averages) ve form puanlarını hesaplar.
    """
    print("Özellik mühendisliği (Feature Engineering) uygulanıyor...")
    features_df = []
    
    team_stats = {}
    h2h_stats = {}
    team_last_date = {}
    
    for index, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        date = row['Date']
        fthg = row['FTHG']
        ftag = row['FTAG']
        ftr = row['FTR']
        
        # Dictionary Initialization
        if home_team not in team_stats:
            team_stats[home_team] = {'scored': [], 'conceded': [], 'points': [], 'shots': [], 'form': []}
        if away_team not in team_stats:
            team_stats[away_team] = {'scored': [], 'conceded': [], 'points': [], 'shots': [], 'form': []}
            
        h2h_key = tuple(sorted([home_team, away_team]))
        if h2h_key not in h2h_stats:
            h2h_stats[h2h_key] = []
            
        row_features = {
            'Date': date, 'HomeTeam': home_team, 'AwayTeam': away_team, 
            'FTR': ftr, 'FTHG': fthg, 'FTAG': ftag,
            'HS': row['HS'], 'AS': row['AS'], 'HST': row['HST'],
            'AST': row['AST'], 'HC': row['HC'], 'AC': row['AC'],
            'HF': row['HF'], 'AF': row['AF'], 'HY': row['HY'],
            'AY': row['AY'], 'HR': row['HR'], 'AR': row['AR']
        }
        
        def calculate_rolling_avg(team, stat_name, window_size):
            history = team_stats[team][stat_name]
            if len(history) < window_size:
                return np.nan
            return float(np.mean(history[-window_size:]))
            
        for window in [3, 5]:
            row_features[f'Home_AvgScored_{window}'] = calculate_rolling_avg(home_team, 'scored', window)
            row_features[f'Home_AvgConceded_{window}'] = calculate_rolling_avg(home_team, 'conceded', window)
            row_features[f'Home_AvgShots_{window}'] = calculate_rolling_avg(home_team, 'shots', window)
            row_features[f'Home_AvgPoints_{window}'] = calculate_rolling_avg(home_team, 'points', window)
            
            row_features[f'Away_AvgScored_{window}'] = calculate_rolling_avg(away_team, 'scored', window)
            row_features[f'Away_AvgConceded_{window}'] = calculate_rolling_avg(away_team, 'conceded', window)
            row_features[f'Away_AvgShots_{window}'] = calculate_rolling_avg(away_team, 'shots', window)
            row_features[f'Away_AvgPoints_{window}'] = calculate_rolling_avg(away_team, 'points', window)

        past_h2h = h2h_stats[h2h_key][-5:]
        if len(past_h2h) > 0:
            row_features['H2H_Home_WinRate_5'] = sum(1 for w in past_h2h if w == home_team) / len(past_h2h)
        else:
            row_features['H2H_Home_WinRate_5'] = 0.5

        features_df.append(row_features)
        
        # Güncellemeler (Sadece geçmiş için, current row dahil edilmez -> Veri sızıntısını önlemek için)
        team_stats[home_team]['scored'].append(fthg)
        team_stats[home_team]['conceded'].append(ftag)
        team_stats[home_team]['shots'].append(row['HST'])
        team_stats[away_team]['scored'].append(ftag)
        team_stats[away_team]['conceded'].append(fthg)
        team_stats[away_team]['shots'].append(row['AST'])
        
        if ftr == 'H':
            team_stats[home_team]['points'].append(3)
            team_stats[home_team]['form'].append('W')
            team_stats[away_team]['points'].append(0)
            team_stats[away_team]['form'].append('L')
            h2h_stats[h2h_key].append(home_team)
        elif ftr == 'A':
            team_stats[home_team]['points'].append(0)
            team_stats[home_team]['form'].append('L')
            team_stats[away_team]['points'].append(3)
            team_stats[away_team]['form'].append('W')
            h2h_stats[h2h_key].append(away_team)
        else:
            team_stats[home_team]['points'].append(1)
            team_stats[home_team]['form'].append('D')
            team_stats[away_team]['points'].append(1)
            team_stats[away_team]['form'].append('D')
            h2h_stats[h2h_key].append('Draw')
            
        team_last_date[home_team] = date
        team_last_date[away_team] = date

    features_df = pd.DataFrame(features_df)
    features_df = features_df.dropna().reset_index(drop=True)
    return features_df, team_stats, h2h_stats, team_last_date

def multi_class_brier_score(y_true, y_prob):
    y_true_dummies = pd.get_dummies(y_true).values
    return brier_score_loss(y_true_dummies.ravel(), y_prob.ravel())

def train_and_optimize(features_df, team_stats, h2h_stats, team_last_date, raw_df):
    print("Model eğitimi hazırlığı başlıyor...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(features_df['FTR'])
    
    y_home_goals = features_df['FTHG']
    y_away_goals = features_df['FTAG']
    
    stat_targets = {
        'HS': features_df['HS'], 'AS': features_df['AS'],
        'HST': features_df['HST'], 'AST': features_df['AST'],
        'HC': features_df['HC'], 'AC': features_df['AC'],
        'HF': features_df['HF'], 'AF': features_df['AF'],
        'HY': features_df['HY'], 'AY': features_df['AY'],
        'HR': features_df['HR'], 'AR': features_df['AR']
    }
    
    cols_to_drop = ['Date', 'FTR', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
    X_raw = features_df.drop(columns=cols_to_drop)
    
    X = pd.get_dummies(X_raw, columns=['HomeTeam', 'AwayTeam'])
    feature_columns = X.columns.tolist()

    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    
    y_train, y_test = y_encoded[:split_index], y_encoded[split_index:]
    y_train_hg, y_test_hg = y_home_goals[:split_index], y_home_goals[split_index:]
    y_train_ag, y_test_ag = y_away_goals[:split_index], y_away_goals[split_index:]
    
    print(f"Eğitim Seti Boyutu: {len(X_train)}, Test Seti Boyutu: {len(X_test)}")
    
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, multi_class='multinomial'),
            'params': {'C': [0.1, 1.0, 10.0]}
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}
        },
        'XGBoost': {
            'model': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
            'params': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
        }
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    print("Sınıflandırma Modelleri Eğitim ve Optimizasyonu başlıyor...")
    trained_clf_models = {}
    clf_metrics = []
    
    for name, mp in models.items():
        print(f"Denetleniyor: {name}")
        clf = GridSearchCV(mp['model'], mp['params'], cv=tscv, scoring='f1_macro', n_jobs=-1)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted')
        brier = multi_class_brier_score(y_test, y_prob)
        
        print(f"[{name}] Test Seti Performansı - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1(W): {f1:.4f}")
        
        trained_clf_models[name] = clf.best_estimator_
        clf_metrics.append({'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1, 'Brier': brier})

    clf_metrics_df = pd.DataFrame(clf_metrics)

    print("\nRegresyon Modelleri Eğitiliyor...")
    
    reg_models_dict = {'Linear Regression': {}, 'Random Forest': {}}
    reg_metrics = []
    
    regression_targets = [
        ('FTHG', y_train_hg, y_test_hg),
        ('FTAG', y_train_ag, y_test_ag)
    ]
    
    stat_labels = {'HS': 'Home Shots', 'AS': 'Away Shots', 'HST': 'Home SoT',
                   'AST': 'Away SoT', 'HC': 'Home Corners', 'AC': 'Away Corners',
                   'HF': 'Home Fouls', 'AF': 'Away Fouls', 'HY': 'Home Yellow',
                   'AY': 'Away Yellow', 'HR': 'Home Red', 'AR': 'Away Red'}
                   
    for st_col, _ in stat_labels.items():
        if st_col in stat_targets:
            y_st = stat_targets[st_col]
            regression_targets.append((st_col, y_st[:split_index], y_st[split_index:]))
            
    for target_name, y_tr, y_te in regression_targets:
        print(f"  Hedef Öğreniliyor: {target_name}")
        
        # 1. Random Forest
        rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf_reg.fit(X_train, y_tr)
        rf_pred = rf_reg.predict(X_test)
        rf_mae, rf_mse, rf_r2 = mean_absolute_error(y_te, rf_pred), mean_squared_error(y_te, rf_pred), r2_score(y_te, rf_pred)
        reg_models_dict['Random Forest'][target_name] = rf_reg
        reg_metrics.append({'Model': 'Random Forest', 'Hedef': target_name, 'MAE': rf_mae, 'MSE': rf_mse, 'R2': rf_r2})
        
        # 2. Linear Regression
        lr_reg = LinearRegression()
        lr_reg.fit(X_train, y_tr)
        lr_pred = lr_reg.predict(X_test)
        lr_mae, lr_mse, lr_r2 = mean_absolute_error(y_te, lr_pred), mean_squared_error(y_te, lr_pred), r2_score(y_te, lr_pred)
        reg_models_dict['Linear Regression'][target_name] = lr_reg
        reg_metrics.append({'Model': 'Linear Regression', 'Hedef': target_name, 'MAE': lr_mae, 'MSE': lr_mse, 'R2': lr_r2})
        
    reg_metrics_df = pd.DataFrame(reg_metrics)
    
    # Kaydetme
    os.makedirs('models', exist_ok=True)
    joblib.dump(trained_clf_models, 'models/clf_models.pkl')
    joblib.dump(reg_models_dict, 'models/reg_models.pkl')
    joblib.dump({'classification': clf_metrics_df, 'regression': reg_metrics_df}, 'models/evaluation_metrics.pkl')
    joblib.dump(feature_columns, 'models/feature_columns.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    
    # Güncel sezon takımları (Tarih bazlı)
    max_date = raw_df['Date'].max()
    season_start_year = max_date.year if max_date.month >= 8 else max_date.year - 1
    season_start = pd.Timestamp(f'{season_start_year}-08-01')
    current_season_df = raw_df[raw_df['Date'] >= season_start]
    current_teams = sorted(set(current_season_df['HomeTeam'].unique()) | set(current_season_df['AwayTeam'].unique()))
    print(f"Güncel Sezon Takımları ({len(current_teams)}): {current_teams}")
    
    # Oynanan maçlar listesi (Monte Carlo için)
    played_fixtures = list(zip(
        current_season_df['HomeTeam'].tolist(),
        current_season_df['AwayTeam'].tolist()
    ))
    
    # Oynanan maçlar + sonuçlar (xPTS Luck Factor için)
    played_results = []
    for _, r in current_season_df.iterrows():
        played_results.append({
            'HomeTeam': r['HomeTeam'], 'AwayTeam': r['AwayTeam'], 'FTR': r['FTR']
        })
    
    # Güncel sezon puan tablosu (Monte Carlo başlangıç noktası)
    current_points = {}
    for team in current_teams:
        current_points[team] = 0
    for _, r in current_season_df.iterrows():
        h, a = r['HomeTeam'], r['AwayTeam']
        if h in current_points and a in current_points:
            if r['FTR'] == 'H':
                current_points[h] += 3
            elif r['FTR'] == 'A':
                current_points[a] += 3
            else:
                current_points[h] += 1
                current_points[a] += 1
    
    # Tarihsel veri (Historical Dominance tab için)
    historical_records = []
    for _, r in raw_df.iterrows():
        historical_records.append({
            'Date': r['Date'], 'HomeTeam': r['HomeTeam'], 'AwayTeam': r['AwayTeam'],
            'FTHG': r['FTHG'], 'FTAG': r['FTAG'], 'FTR': r['FTR']
        })
    
    latest_stats = {
        'team_stats': team_stats,
        'h2h_stats': h2h_stats,
        'team_last_date': team_last_date,
        'current_teams': current_teams,
        'played_fixtures': played_fixtures,
        'played_results': played_results,
        'current_points': current_points,
        'historical_records': historical_records
    }
    joblib.dump(latest_stats, 'models/latest_stats.pkl')
    print("Eğitim tamamlandı! Gereksinimler 'models/' klasörüne kaydedildi.")

if __name__ == "__main__":
    try:
        raw_df = load_and_clean_data()
        features_df, t_stats, h2h_stats, t_last_date = feature_engineering(raw_df)
        train_and_optimize(features_df, t_stats, h2h_stats, t_last_date, raw_df)
    except Exception as e:
        print(f"Bir hata oluştu: {str(e)}")
