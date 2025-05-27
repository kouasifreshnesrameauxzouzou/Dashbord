import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="💎 Diamond Analytics Dashboard",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration des couleurs personnalisées
colors = {
    'primary_100': '#FF6600',
    'primary_200': '#ff983f', 
    'primary_300': '#ffffa1',
    'accent_100': '#F5F5F5',
    'accent_200': '#929292',
    'text_100': '#FFFFFF',
    'text_200': '#e0e0e0',
    'bg_100': '#1D1F21',
    'bg_200': '#2c2e30',
    'bg_300': '#444648'
}

# CSS personnalisé pour le design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6600, #ff983f);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
    
    .kpi-container {
        background: linear-gradient(135deg, #2c2e30, #444648);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        border-left: 5px solid #FF6600;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .kpi-title {
        font-size: 0.9rem;
        color: #e0e0e0;
        margin-bottom: 0.5rem;
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: bold;
        color: #FF6600;
        margin: 0;
    }
    
    .section-header {
        background: linear-gradient(90deg, #1D1F21, #2c2e30);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF6600;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: #2c2e30;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #444648;
    }
</style>
""", unsafe_allow_html=True)

# Template Plotly personnalisé
import plotly.io as pio

pio.templates["custom_dark"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=colors['bg_100'],
        plot_bgcolor=colors['bg_200'],
        font=dict(color=colors['text_100'], family='Arial, sans-serif'),
        colorway=[colors['primary_100'], colors['primary_200'], colors['primary_300'], 
                 colors['accent_200'], colors['text_200']],
        title=dict(font=dict(size=16, color=colors['text_100'])),
        xaxis=dict(
            gridcolor=colors['bg_300'],
            linecolor=colors['accent_200'],
            tickcolor=colors['accent_200'],
            title=dict(font=dict(color=colors['text_200']))
        ),
        yaxis=dict(
            gridcolor=colors['bg_300'],
            linecolor=colors['accent_200'],
            tickcolor=colors['accent_200'],
            title=dict(font=dict(color=colors['text_200']))
        )
    )
)

# Fonction pour charger les données
@st.cache_data
def load_data():
    # Remplacez ce chemin par votre fichier CSV
    # df = pd.read_csv('diamonds.csv')
    
    # Pour la démo, créons des données fictives
    np.random.seed(42)
    n_samples = 10000
    
    cuts = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    colors = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
    clarities = ['FL', 'IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2']
    
    df = pd.DataFrame({
        'carat': np.random.exponential(0.5, n_samples) + 0.2,
        'cut': np.random.choice(cuts, n_samples),
        'color': np.random.choice(colors, n_samples),
        'clarity': np.random.choice(clarities, n_samples),
        'depth': np.random.normal(61.8, 1.5, n_samples),
        'table': np.random.normal(57.5, 2, n_samples),
        'x': np.random.normal(5.7, 1.2, n_samples),
        'y': np.random.normal(5.7, 1.2, n_samples),
        'z': np.random.normal(3.5, 0.8, n_samples)
    })
    
    # Génération du prix basé sur les caractéristiques
    cut_multiplier = {'Fair': 0.8, 'Good': 0.9, 'Very Good': 1.0, 'Premium': 1.1, 'Ideal': 1.2}
    color_multiplier = {'J': 0.8, 'I': 0.85, 'H': 0.9, 'G': 0.95, 'F': 1.0, 'E': 1.05, 'D': 1.1}
    
    df['price'] = (
        (df['carat'] ** 2) * 3000 * 
        df['cut'].map(cut_multiplier) * 
        df['color'].map(color_multiplier) * 
        np.random.normal(1, 0.1, n_samples)
    ).round(0).astype(int)
    
    return df

# Chargement des données
df = load_data()

# En-tête principal
st.markdown("""
<div class="main-header">
    <h1>💎 Diamond Analytics Dashboard</h1>
    <p>Analyse Complète des Diamants - Descriptive | Diagnostique | Prescriptive | Prédictive</p>
</div>
""", unsafe_allow_html=True)

# Sidebar pour les filtres
st.sidebar.header("🔍 Filtres")

# Filtres
price_range = st.sidebar.slider(
    "Gamme de Prix ($)",
    min_value=int(df['price'].min()),
    max_value=int(df['price'].max()),
    value=(int(df['price'].min()), int(df['price'].max()))
)

selected_cuts = st.sidebar.multiselect(
    "Qualité de Taille",
    options=df['cut'].unique(),
    default=df['cut'].unique()
)

carat_range = st.sidebar.slider(
    "Gamme de Carat",
    min_value=float(df['carat'].min()),
    max_value=float(df['carat'].max()),
    value=(float(df['carat'].min()), float(df['carat'].max())),
    step=0.1
)

# Application des filtres
filtered_df = df[
    (df['price'] >= price_range[0]) & 
    (df['price'] <= price_range[1]) &
    (df['cut'].isin(selected_cuts)) &
    (df['carat'] >= carat_range[0]) &
    (df['carat'] <= carat_range[1])
]

# KPIs principaux
st.markdown("## 📊 Indicateurs Clés de Performance")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-title">💎 Diamants Analysés</div>
        <div class="kpi-value">{len(filtered_df):,}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    avg_price = filtered_df['price'].mean()
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-title">💰 Prix Moyen</div>
        <div class="kpi-value">${avg_price:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    avg_carat = filtered_df['carat'].mean()
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-title">⚖️ Carat Moyen</div>
        <div class="kpi-value">{avg_carat:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    median_price = filtered_df['price'].median()
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-title">📈 Prix Médian</div>
        <div class="kpi-value">${median_price:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    max_price = filtered_df['price'].max()
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-title">👑 Prix Maximum</div>
        <div class="kpi-value">${max_price:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

# Analyse Descriptive
st.markdown("""
<div class="section-header">
    <h2>🔍 Analyse Descriptive</h2>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Distribution des prix
    fig1 = px.histogram(filtered_df, x='price', nbins=50, 
                       title='Distribution des Prix des Diamants')
    fig1.update_layout(template="custom_dark", height=400)
    fig1.update_traces(marker_color=colors['primary_100'])
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Distribution par cut
    cut_counts = filtered_df['cut'].value_counts()
    fig2 = px.pie(values=cut_counts.values, names=cut_counts.index,
                  title='Répartition par Qualité de Taille',
                  color_discrete_sequence=[colors['primary_100'], colors['primary_200'], 
                                         colors['primary_300'], colors['accent_200'], colors['text_200']])
    fig2.update_layout(template="custom_dark", height=400)
    st.plotly_chart(fig2, use_container_width=True)

# Matrice de corrélation
numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
correlation_matrix = filtered_df[numeric_cols].corr()

fig3 = px.imshow(correlation_matrix, 
                 title='Matrice de Corrélation',
                 color_continuous_scale=['#1D1F21', colors['primary_200'], colors['primary_100']],
                 aspect='auto')
fig3.update_layout(template="custom_dark", height=500)
st.plotly_chart(fig3, use_container_width=True)

# Analyse Diagnostique
st.markdown("""
<div class="section-header">
    <h2>🔬 Analyse Diagnostique</h2>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Prix par carat et cut
    sample_df = filtered_df.sample(min(5000, len(filtered_df)))
    fig4 = px.scatter(sample_df, x='carat', y='price', color='cut',
                      title='Relation Prix-Carat par Qualité de Taille',
                      color_discrete_sequence=[colors['primary_100'], colors['primary_200'], 
                                             colors['primary_300'], colors['accent_200'], colors['text_200']])
    fig4.update_layout(template="custom_dark", height=400)
    st.plotly_chart(fig4, use_container_width=True)

with col2:
    # Box plot prix par couleur
    fig5 = px.box(filtered_df, x='color', y='price',
                  title='Distribution des Prix par Couleur')
    fig5.update_layout(template="custom_dark", height=400)
    fig5.update_traces(marker_color=colors['primary_100'], line_color=colors['primary_100'])
    st.plotly_chart(fig5, use_container_width=True)

# Détection d'anomalies
q1 = filtered_df['price'].quantile(0.25)
q3 = filtered_df['price'].quantile(0.75)
iqr = q3 - q1
outliers = filtered_df[(filtered_df['price'] < q1 - 1.5*iqr) | (filtered_df['price'] > q3 + 1.5*iqr)]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🚨 Outliers Détectés", f"{len(outliers):,}", f"{len(outliers)/len(filtered_df)*100:.1f}%")
with col2:
    st.metric("📉 Prix Min Outliers", f"${outliers['price'].min():,}" if len(outliers) > 0 else "N/A")
with col3:
    st.metric("📈 Prix Max Outliers", f"${outliers['price'].max():,}" if len(outliers) > 0 else "N/A")

# Analyse Prescriptive
st.markdown("""
<div class="section-header">
    <h2>💡 Analyse Prescriptive</h2>
</div>
""", unsafe_allow_html=True)

# Segmentation par gamme de prix
filtered_df['price_segment'] = pd.cut(filtered_df['price'], 5, labels=['Économique', 'Abordable', 'Moyen', 'Premium', 'Luxe'])

# Treemap des segments
fig6 = px.treemap(filtered_df.groupby(['price_segment', 'cut']).size().reset_index(name='count'),
                  path=['price_segment', 'cut'], values='count',
                  title='Répartition par Segment de Prix et Qualité',
                  color='count',
                  color_continuous_scale=[colors['bg_200'], colors['primary_200'], colors['primary_100']])
fig6.update_layout(template="custom_dark", height=500)
st.plotly_chart(fig6, use_container_width=True)

# Recommandations
col1, col2, col3 = st.columns(3)

best_value_cut = filtered_df.groupby('cut')['price'].mean().sort_values().index[0]
most_popular_color = filtered_df['color'].mode().iloc[0]
optimal_carat_range = f"{filtered_df['carat'].quantile(0.25):.2f} - {filtered_df['carat'].quantile(0.75):.2f}"

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h4>🏆 Meilleur Rapport Qualité-Prix</h4>
        <p style="font-size: 1.2rem; color: #FF6600;"><strong>{best_value_cut}</strong></p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h4>🌈 Couleur la Plus Demandée</h4>
        <p style="font-size: 1.2rem; color: #FF6600;"><strong>{most_popular_color}</strong></p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h4>⚖️ Gamme Carat Optimale</h4>
        <p style="font-size: 1.2rem; color: #FF6600;"><strong>{optimal_carat_range}</strong></p>
    </div>
    """, unsafe_allow_html=True)

# Analyse Prédictive
st.markdown("""
<div class="section-header">
    <h2>🔮 Analyse Prédictive</h2>
</div>
""", unsafe_allow_html=True)

# Modèle de prédiction
@st.cache_data
def train_model(df):
    df_ml = df.copy()
    
    # Encodage des variables catégorielles
    le_cut = LabelEncoder()
    le_color = LabelEncoder()
    le_clarity = LabelEncoder()
    
    df_ml['cut_encoded'] = le_cut.fit_transform(df_ml['cut'])
    df_ml['color_encoded'] = le_color.fit_transform(df_ml['color'])
    df_ml['clarity_encoded'] = le_clarity.fit_transform(df_ml['clarity'])
    
    # Sélection des features
    features = ['carat', 'depth', 'table', 'x', 'y', 'z', 'cut_encoded', 'color_encoded', 'clarity_encoded']
    X = df_ml[features]
    y = df_ml['price']
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entraînement du modèle
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Prédictions
    y_pred = rf_model.predict(X_test)
    
    # Métriques
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Importance des features
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    return rf_model, mae, r2, feature_importance, y_test, y_pred

model, mae, r2, feature_importance, y_test, y_pred = train_model(df)

# Métriques du modèle
col1, col2 = st.columns(2)
with col1:
    st.metric("🤖 R² Score", f"{r2:.3f}")
with col2:
    st.metric("📊 Erreur Absolue Moyenne", f"${mae:,.0f}")

col1, col2 = st.columns(2)

with col1:
    # Importance des features
    fig7 = px.bar(feature_importance, x='importance', y='feature',
                  orientation='h',
                  title='Importance des Caractéristiques',
                  color='importance',
                  color_continuous_scale=[colors['bg_200'], colors['primary_200'], colors['primary_100']])
    fig7.update_layout(template="custom_dark", height=400)
    st.plotly_chart(fig7, use_container_width=True)

with col2:
    # Prédictions vs Réalité
    comparison_df = pd.DataFrame({
        'Réel': y_test.iloc[:1000],
        'Prédit': y_pred[:1000]
    })
    
    fig8 = px.scatter(comparison_df, x='Réel', y='Prédit',
                      title='Prédictions vs Prix Réels')
    fig8.update_traces(marker_color=colors['primary_100'])
    fig8.add_shape(type="line", 
                   x0=comparison_df['Réel'].min(), y0=comparison_df['Réel'].min(),
                   x1=comparison_df['Réel'].max(), y1=comparison_df['Réel'].max(),
                   line=dict(color=colors['primary_200'], width=2, dash="dash"))
    fig8.update_layout(template="custom_dark", height=400)
    st.plotly_chart(fig8, use_container_width=True)

# Simulateur de prédiction
st.markdown("### 🎯 Simulateur de Prix")

col1, col2, col3 = st.columns(3)

with col1:
    sim_carat = st.slider("Carat", 0.2, 5.0, 1.0, 0.1)
    sim_depth = st.slider("Profondeur", 50.0, 80.0, 61.8, 0.1)
    sim_table = st.slider("Table", 50.0, 70.0, 57.5, 0.1)

with col2:
    sim_x = st.slider("Longueur (x)", 3.0, 10.0, 5.7, 0.1)
    sim_y = st.slider("Largeur (y)", 3.0, 10.0, 5.7, 0.1)
    sim_z = st.slider("Hauteur (z)", 2.0, 7.0, 3.5, 0.1)

with col3:
    sim_cut = st.selectbox("Qualité de Taille", df['cut'].unique())
    sim_color = st.selectbox("Couleur", df['color'].unique())
    sim_clarity = st.selectbox("Pureté", df['clarity'].unique())

# Prédiction en temps réel
if st.button("💰 Prédire le Prix", type="primary"):
    # Encodage pour la prédiction
    le_cut = LabelEncoder()
    le_color = LabelEncoder()
    le_clarity = LabelEncoder()
    
    le_cut.fit(df['cut'])
    le_color.fit(df['color'])
    le_clarity.fit(df['clarity'])
    
    sim_features = [[
        sim_carat, sim_depth, sim_table, sim_x, sim_y, sim_z,
        le_cut.transform([sim_cut])[0],
        le_color.transform([sim_color])[0],
        le_clarity.transform([sim_clarity])[0]
    ]]
    
    predicted_price = model.predict(sim_features)[0]
    
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #FF6600, #ff983f); padding: 2rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
        <h2 style="color: white; margin: 0;">Prix Prédit: ${predicted_price:,.0f}</h2>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #929292; padding: 1rem;">
    <p>💎 Diamond Analytics Dashboard | Analyse Complète des Données de Diamants</p>
    <p>Développé avec Streamlit & Plotly | Modèle de Machine Learning intégré</p>
</div>
""", unsafe_allow_html=True)
