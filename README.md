# Aviator Web Analyzer (Italiano) — Deploy pronto

Questo repository contiene un'app Streamlit per analizzare storici dei moltiplicatori di Aviator (SNAI) e per sperimentare semplici modelli ML a scopo educativo.

**Importante:** non esiste un modo legale o tecnico per prevedere i moltiplicatori con certezza. Usa questo strumento solo per analisi ed esercitazione.

## File
 streamlit==1.25.0
pandas
numpy
scikit-learn
matplotlib>=3.6.0,<3.11
Pillow>=9.4.0

## Esecuzione locale
1. Crea e attiva un ambiente virtuale (opzionale):
   - mac/linux: `python -m venv venv && source venv/bin/activate`
   - windows: `python -m venv venv && venv\\Scripts\\activate`
2. Installa dipendenze: `pip install -r requirements.txt`
3. Avvia: `streamlit run streamlit_app.py`
4. Apri il browser su `http://localhost:8501`

## Deploy rapido
- **Streamlit Cloud**: crea una nuova app e collega il repository GitHub.
- **Render**: crea un nuovo Web Service, collega GitHub e deploya.
- **Heroku**: usa Docker o il buildpack Python + Procfile.

## Note legali e di sicurezza
- Non fare scraping se la fonte proibisce l'accesso automatico.
- Non usare questo strumento per attività illegali o per aggirare sistemi di gioco.
- I modelli ML sono sperimentali e possono fornire stime errate.
