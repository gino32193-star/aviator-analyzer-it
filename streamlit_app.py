# streamlit_app.py
import io, time, numpy as np, pandas as pd, streamlit as st
from model import train_model, predict_next
from sklearn.model_selection import train_test_split

st.set_page_config(page_title='Aviator Analyzer (IT)', layout='wide')
st.title('Aviator Analyzer — Analisi empirica (ITALIANO)')
st.markdown('Carica un CSV con colonna `multiplier` o incolla i moltiplicatori. Questo strumento è a scopo educativo e non garantisce predizioni certe.')

uploaded = st.file_uploader('Carica CSV', type=['csv'])
text = st.text_area('Oppure incolla i moltiplicatori (separati da virgola, nuova riga o spazio)')
use_sample = st.checkbox('Usa dataset di esempio (demo)')

def parse_text(s):
    parts = [p.strip() for p in s.replace(',', '\\n').split() if p.strip()]
    arr = []
    for p in parts:
        try:
            arr.append(float(p))
        except:
            pass
    return np.array(arr, dtype=float)

multipliers = np.array([])
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        if 'multiplier' in df.columns:
            multipliers = pd.to_numeric(df['multiplier'], errors='coerce').dropna().values
        else:
            st.error(\"Il CSV deve contenere la colonna 'multiplier'\")
    except Exception as e:
        st.error(f'Errore lettura CSV: {e}')
elif text.strip():
    multipliers = parse_text(text)
elif use_sample:
    rng = np.random.default_rng(123)
    base = rng.random(3000) * 0.5 + 1.0
    tail = np.random.pareto(2.5, size=200) + 1.0
    multipliers = np.concatenate([base, tail])

if multipliers.size == 0:
    st.warning('Nessun dato: carica CSV, incolla multipliers o usa il dataset di esempio.')
    st.stop()

# Statistiche base
st.subheader('Statistiche riassuntive')
stats = {
    'count': int(len(multipliers)),
    'mean': float(np.mean(multipliers)),
    'median': float(np.median(multipliers)),
    'std': float(np.std(multipliers, ddof=1)),
    'min': float(np.min(multipliers)),
    'max': float(np.max(multipliers)),
}
cols = st.columns(6)
keys = ['count','mean','median','std','min','max']
for c,k in zip(cols, keys):
    c.metric(k, f\"{stats.get(k, '—')}\")

st.write('Percentili: p50, p75, p90, p99')
st.write({
    'p50': float(np.percentile(multipliers,50)),
    'p75': float(np.percentile(multipliers,75)),
    'p90': float(np.percentile(multipliers,90)),
    'p99': float(np.percentile(multipliers,99)),
})

# Grafici
st.subheader('Visualizzazioni')
import matplotlib.pyplot as plt
fig1, ax1 = plt.subplots(figsize=(7,3))
ax1.hist(multipliers, bins=80)
ax1.set_xlabel('Multiplier')
ax1.set_ylabel('Conteggio')
ax1.set_title('Istogramma multipliers')
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(7,3))
sorted_m = np.sort(multipliers)
cdf = np.arange(1, len(sorted_m)+1) / len(sorted_m)
ax2.plot(sorted_m, cdf)
ax2.set_xlabel('Multiplier')
ax2.set_ylabel('CDF')
ax2.set_title('CDF empirica')
st.pyplot(fig2)

# Probabilità empirica
st.subheader('Probabilità empiriche')
threshold = st.number_input('Soglia per P(m >= x)', value=2.0, step=0.1)
prob = float(np.mean(multipliers >= threshold))
st.write(f'P(multiplier >= {threshold}) = {prob:.6f} ({prob*100:.4f}%)')

# Simulazione strategie
st.subheader('Simulazione strategie (Monte Carlo)')
sim_strategy = st.selectbox('Strategia', ['flat','martingale','kelly'])
stake = st.number_input('Stake iniziale (flat/martingale)', value=1.0, min_value=0.01)
trials = st.number_input('Numero di trial (Monte Carlo)', value=2000, min_value=100, step=100)
rounds = st.number_input('Rounds per trial', value=200, min_value=10)
mart_mult = st.number_input('Moltiplicatore martingale', value=2.0, min_value=1.1)
kelly_frac = st.number_input('Frazione Kelly (0-1)', value=0.5, min_value=0.0, max_value=1.0)
initial_bankroll = st.number_input('Bankroll iniziale per trial', value=100.0, min_value=1.0)
mcap = st.number_input('Cap scommessa martingale', value=1000.0, min_value=stake)

def empirical_prob(arr, th):
    return float(np.mean(arr >= th))

def simulate_strategy(multipliers, strategy='flat', stake=1.0, trials=2000, rounds_per_trial=200,
                      martingale_multiplier=2.0, kelly_fraction=1.0, target_cashout=2.0,
                      initial_bankroll=100.0, martingale_cap=1000.0):
    rng = np.random.default_rng(int(time.time()))
    results = []
    for t in range(trials):
        bankroll = initial_bankroll
        for r in range(rounds_per_trial):
            if strategy == 'flat':
                bet = stake
                m = rng.choice(multipliers)
                if m >= target_cashout:
                    bankroll += bet * (target_cashout - 1.0)
                else:
                    bankroll -= bet
            elif strategy == 'martingale':
                current_bet = stake
                while True:
                    m = rng.choice(multipliers)
                    if m >= target_cashout:
                        bankroll += current_bet * (target_cashout - 1.0)
                        break
                    else:
                        bankroll -= current_bet
                        current_bet *= martingale_multiplier
                        if current_bet > martingale_cap or bankroll < current_bet or bankroll <= 0:
                            break
            elif strategy == 'kelly':
                p_target = empirical_prob(multipliers, target_cashout)
                b = target_cashout - 1.0
                if b <= 0 or p_target == 0:
                    f = 0.0
                else:
                    f_unscaled = (p_target * (b + 1) - 1) / b
                    f = max(0.0, f_unscaled) * kelly_fraction
                bet = bankroll * f
                if bet <= 0:
                    continue
                m = rng.choice(multipliers)
                if m >= target_cashout:
                    bankroll += bet * (target_cashout - 1.0)
                else:
                    bankroll -= bet
            if bankroll <= 0:
                bankroll = 0.0
                break
        results.append(bankroll)
    return np.array(results)

if st.button('Esegui simulazione'):
    with st.spinner('Esecuzione simulazioni...'):
        sim_res = simulate_strategy(multipliers, strategy=sim_strategy, stake=stake, trials=int(trials),
                                    rounds_per_trial=int(rounds), martingale_multiplier=float(mart_mult),
                                    kelly_fraction=float(kelly_frac), target_cashout=float(threshold),
                                    initial_bankroll=float(initial_bankroll), martingale_cap=float(mcap))
    st.success('Simulazione completata')
    st.write('Media final bankroll:', float(sim_res.mean()))
    st.write('Mediana final bankroll:', float(np.median(sim_res)))
    st.write('Percentuale rovinati:', float(np.mean(sim_res <= 0.0)))
    fig3, ax3 = plt.subplots(figsize=(7,3))
    ax3.hist(sim_res, bins=50)
    ax3.set_xlabel('Final bankroll')
    ax3.set_ylabel('Conteggio')
    st.pyplot(fig3)
    df_sim = pd.DataFrame({'final_bankroll': sim_res})
    csv_bytes = df_sim.to_csv(index=False).encode('utf-8')
    st.download_button('Scarica risultati simulazione (CSV)', data=csv_bytes, file_name='simulation_results.csv', mime='text/csv')

# ML opzionale (educativo)
st.markdown('---')
st.subheader('ML opzionale (educativo)')
run_ml = st.checkbox('Allena modello ML (sperimentale)')
if run_ml:
    WINDOW = st.number_input('Dimensione finestra (passati rounds)', min_value=1, max_value=100, value=5)
    X = []
    y = []
    for i in range(WINDOW, len(multipliers)):
        window = multipliers[i-WINDOW:i]
        feats = [window.mean(), window.std(), window[-1], np.min(window), np.max(window)]
        X.append(feats)
        y.append(multipliers[i])
    X = np.array(X); y = np.array(y)
    if len(y) < 50:
        st.warning('Pochi dati per ML (servono almeno 50 esempi).')
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model, metrics = train_model(X_train, y_train, X_test, y_test)
        st.write('Metriche:', metrics)
        next_feats = np.array([[np.mean(multipliers[-WINDOW:]), np.std(multipliers[-WINDOW:]), multipliers[-1], np.min(multipliers[-WINDOW:]), np.max(multipliers[-WINDOW:])]])
        pred = predict_next(model, next_feats)
        st.write('Stima ML per il prossimo multiplier (valore puntuale):', float(pred))
        st.write('Nota: su sistemi casuali la predizione è probabilmente inaffidabile.')


st.markdown('---')
st.write('Note legali: questo strumento è solo a scopo educativo. Non tentare di manipolare o aggirare sistemi di gioco.')