# ───────────────────────── imports ─────────────────────────
import os, json, logging, re, requests, time
import pandas as pd, pandas_ta as ta
import numpy as np
from numpy.linalg import norm
from typing import Optional
from google.cloud import storage, bigquery, secretmanager

# embeddings / NLP
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from gensim.models import LdaModel

# ──────────────────── FinBERT singleton ───────────────────
_FINBERT_NAME = "ProsusAI/finbert"
logging.info("Loading FinBERT weights once …")
FINBERT_TOKENIZER = AutoTokenizer.from_pretrained(_FINBERT_NAME, local_files_only=True)
FINBERT_MODEL = AutoModelForSequenceClassification.from_pretrained(
    _FINBERT_NAME, torch_dtype=torch.float32, low_cpu_mem_usage=False, local_files_only=True
).eval()

# ───────────────────────── init ────────────────────────────
PROJECT_ID = os.environ.get("PROJECT_ID")
bq_client = bigquery.Client()
storage_client = storage.Client()
secret_client = secretmanager.SecretManagerServiceClient()
logging.basicConfig(level=logging.INFO)

# ─────────────────────── helpers ───────────────────────────
def get_fmp_api_key() -> str | None:
    try:
        name = f"projects/{PROJECT_ID}/secrets/FMP_API_KEY/versions/latest"
        return secret_client.access_secret_version(request={"name": name}).payload.data.decode()
    except Exception as e:
        logging.error("get_fmp_api_key: %s", e)
        return None

# ---------- NEW: earnings-call date pulled from transcript ----------
def get_earnings_call_date(transcript_gcs_path: str) -> dict:
    try:
        bucket, blob = transcript_gcs_path.removeprefix("gs://").split("/", 1)
        raw = storage_client.bucket(bucket).blob(blob).download_as_string()
        date_str = json.loads(raw).get("date")
        if date_str:
            date_str = date_str.split(" ")[0]  # YYYY-MM-DD
        return {"earnings_call_date": date_str}
    except Exception as e:
        logging.error("get_earnings_call_date: %s", e, exc_info=True)
        return {"earnings_call_date": None}

# ---------- NEW: simple EPS-surprise helper ----------
def get_eps_surprise(api_key: Optional[str], ticker: str) -> dict:
    if not api_key:
        return {"eps_surprise": None, "eps_missing": True}
    url = f"https://financialmodelingprep.com/api/v3/earnings-surprises/{ticker}?apikey={api_key}"
    try:
        data = requests.get(url, timeout=10).json()
        if not data:
            raise ValueError("empty response")
        actual   = data[0]["actualEarningResult"]
        estimate = data[0]["estimatedEarning"]
        surprise_pct = (actual - estimate) / abs(estimate) if estimate else None
        return {"eps_surprise": float(surprise_pct) if surprise_pct is not None else None,
                "eps_missing": False}
    except Exception as e:
        logging.error("get_eps_surprise: %s", e)
        return {"eps_surprise": None, "eps_missing": True}

# ──────────────────── vector embeddings ───────────────────
def get_embeddings(summary_gcs_path: str) -> dict:
    key = "summary_embedding"
    try:
        bucket, blob = summary_gcs_path.removeprefix("gs://").split("/", 1)
        txt = storage_client.bucket(bucket).blob(blob).download_as_bytes() \
              .decode("utf-8", errors="replace").strip()
        if not txt:
            logging.warning("Empty summary → embedding=None")
            return {key: None}

        txt = re.sub(r"\s+", " ", txt)
        txt = "".join(c for c in txt if ord(c) >= 32)

        tokenizer, model = FINBERT_TOKENIZER, FINBERT_MODEL

        def embed(text: str):
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            if enc["input_ids"].shape[1] <= 512:
                with torch.no_grad():
                    h = model(**enc, output_hidden_states=True).hidden_states[-1]
                return h.mean(dim=1).squeeze().cpu().numpy()

            tokens = tokenizer.encode(text, add_special_tokens=False)
            chunk, embs = 510, []
            for i in range(0, len(tokens), chunk):
                tok = tokens[i:i+chunk]
                tok = tokenizer.build_inputs_with_special_tokens(tok)
                enc = {"input_ids": torch.tensor([tok]),
                       "attention_mask": torch.ones(1, len(tok), dtype=torch.long)}
                with torch.no_grad():
                    h = model(**enc, output_hidden_states=True).hidden_states[-1]
                embs.append(h.mean(dim=1).squeeze().cpu())
            return np.stack(embs).mean(axis=0)

        for attempt in range(3):
            try:
                vec = embed(txt)
                if vec.shape != (768,): raise ValueError("bad dim")
                vec = vec / np.linalg.norm(vec)
                return {key: vec.tolist()}
            except Exception as e:
                logging.error("Embed attempt %d failed: %s", attempt+1, e)
                if attempt == 2: return {key: None}
                time.sleep(2 ** attempt)
    except Exception as e:
        logging.error("get_embeddings: %s", e, exc_info=True)
        return {key: None}

# ─────────────── FinBERT sentiment (singleton) ────────────
def get_finbert_sentiments(summary_gcs_path: str) -> dict:
    bucket, blob = summary_gcs_path.removeprefix("gs://").split("/", 1)
    text = storage_client.bucket(bucket).blob(blob).download_as_text()

    tokenizer, model = FINBERT_TOKENIZER, FINBERT_MODEL
    def score(t: str):
        if not t.strip(): return [0.0, 0.0, 1.0]
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=512)
        if enc["input_ids"].shape[1] <= 512:
            with torch.no_grad(): logits = model(**enc).logits
            return torch.softmax(logits, dim=-1)[0].tolist()

        tokens, chunk, probs = tokenizer.encode(t, add_special_tokens=True), 510, []
        for i in range(0, len(tokens), chunk):
            tok = tokens[i:i+chunk]
            tok = tokenizer.build_inputs_with_special_tokens(tok)
            enc = {"input_ids": torch.tensor([tok]),
                   "attention_mask": torch.ones(1, len(tok), dtype=torch.long)}
            with torch.no_grad(): logits = model(**enc).logits
            probs.append(torch.softmax(logits, dim=-1)[0])
        return torch.stack(probs).mean(dim=0).tolist()

    try:
        p_pos, p_neg, p_neu = score(text)
        return {"pos_prob": p_pos, "neg_prob": p_neg, "neu_prob": p_neu}
    except Exception as e:
        logging.error("get_finbert_sentiments: %s", e, exc_info=True)
        return {"pos_prob": 0.0, "neg_prob": 0.0, "neu_prob": 1.0}

def get_stock_metadata(ticker: str) -> dict:
    try:
        table = os.getenv("METADATA_TABLE", "profitscout-lx6bb.profit_scout.stock_metadata")
        q = f"SELECT sector, industry FROM `{table}` WHERE ticker = @t LIMIT 1"
        df = bq_client.query(q, job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("t", "STRING", ticker)]
        )).to_dataframe()
        return df.iloc[0].to_dict() if not df.empty else {"sector": None, "industry": None}
    except Exception as e:
        logging.error("get_stock_metadata: %s", e)
        return {"sector": None, "industry": None}

def get_price_technicals(ticker: str, call_date: str) -> dict:
    try:
        table = os.environ["PRICE_TABLE"]
        q = (f"SELECT date, open, high, low, adj_close "
             f"FROM `{table}` WHERE ticker=@t AND date<=@d ORDER BY date DESC LIMIT 100")
        df = bq_client.query(q, job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("t", "STRING", ticker),
                    bigquery.ScalarQueryParameter("d", "DATE", call_date.split()[0])
                ])).to_dataframe().set_index("date").sort_index()
        if df.empty: return {}
        df.rename(columns={"open":"Open","high":"High","low":"Low","adj_close":"Close"}, inplace=True)
        df["Adj Close"] = df["Close"]
        df.ta.sma(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.adx(length=14, append=True, high=df["High"], low=df["Low"], close=df["Close"])
        cur, prev = df.iloc[-1], df.iloc[-2] if len(df) > 1 else df.iloc[-1]
        d = lambda c,p: float(c-p) if pd.notna(c) and pd.notna(p) else None
        return {
            "adj_close_on_call_date": float(cur["Adj Close"]),
            "sma_20": float(cur["SMA_20"]) if pd.notna(cur["SMA_20"]) else None,
            "ema_50": float(cur["EMA_50"]) if pd.notna(cur["EMA_50"]) else None,
            "rsi_14": float(cur["RSI_14"]) if pd.notna(cur["RSI_14"]) else None,
            "adx_14": float(cur["ADX_14"]) if pd.notna(cur["ADX_14"]) else None,
            "sma_20_delta": d(cur["SMA_20"], prev["SMA_20"]),
            "ema_50_delta": d(cur["EMA_50"], prev["EMA_50"]),
            "rsi_14_delta": d(cur["RSI_14"], prev["RSI_14"]),
            "adx_14_delta": d(cur["ADX_14"], prev["ADX_14"]),
        }
    except Exception as e:
        logging.error("get_price_technicals: %s", e)
        return {}

def get_max_close_30d(ticker: str, call_date: str) -> dict:
    try:
        start = pd.Timestamp(call_date.split()[0])
        end   = start + pd.Timedelta(days=30)
        if pd.Timestamp.now().floor("D") < end:
            return {"max_close_30d": None}
        table = os.environ["PRICE_TABLE"]
        q = (f"SELECT MAX(adj_close) AS m "
             f"FROM `{table}` WHERE ticker=@t AND date BETWEEN @s AND @e")
        df = bq_client.query(q, job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("t","STRING",ticker),
                    bigquery.ScalarQueryParameter("s","DATE",start.strftime("%Y-%m-%d")),
                    bigquery.ScalarQueryParameter("e","DATE",end.strftime("%Y-%m-%d"))
                ])).to_dataframe()
        return {"max_close_30d": float(df.at[0,"m"]) if not df.empty else None}
    except Exception as e:
        logging.error("get_max_close_30d: %s", e)
        return {"max_close_30d": None}

# ─────────────────────── LDA topics ───────────────────────
def get_lda_topics(summary_gcs_path: str) -> dict:
    stop_words = {  # abridged for brevity
        'i','me','my','we','our','you',"you're","your",'he','she','it','they',
        'what','which','who','this','that','these','those','am','is','are','was',
        'be','been','being','have','has','had','do','does','did','a','an','the',
        'and','but','if','or','because','as','until','while','of','at','by','for',
        'with','about','against','between','into','through','during','before',
        'after','above','below','to','from','up','down','in','out','on','off',
        'over','under','again','further','then','once','here','there','when',
        'where','why','how','all','any','both','each','few','more','most','other',
        'some','such','no','nor','not','only','own','same','so','than','too','very'
    }
    try:
        bucket, blob = summary_gcs_path.removeprefix("gs://").split("/", 1)
        text = storage_client.bucket(bucket).blob(blob).download_as_text()
        if not text.strip():
            return {f"lda_topic_{i}": 0.0 for i in range(10)}
        tokens = [w for w in simple_preprocess(text) if w not in stop_words and len(w) > 2]
        if not tokens:
            return {f"lda_topic_{i}": 0.0 for i in range(10)}
        dictionary = corpora.Dictionary([tokens])
        corpus = [dictionary.doc2bow(tokens)]
        lda = LdaModel(corpus=corpus, id2word=dictionary,
                       num_topics=10, passes=10, iterations=100, random_state=42)
        probs = [0.0]*10
        for tid, p in lda.get_document_topics(corpus[0], minimum_probability=0.0):
            probs[tid] = float(p)
        return {f"lda_topic_{i}": probs[i] for i in range(10)}
    except Exception as e:
        logging.error("get_lda_topics: %s", e)
        return {f"lda_topic_{i}": None for i in range(10)}

# ─────────────────── feature orchestrator ──────────────────
def create_features(message: dict) -> dict | None:
    need = {"ticker", "quarter_end_date", "summary_gcs_path", "transcript_gcs_path"}
    if not need.issubset(message) or any(message[k] is None for k in need):
        logging.error("Missing required keys: %s", message)
        return None

    tkr = message["ticker"]
    row = {"ticker": tkr, "quarter_end_date": message["quarter_end_date"]}

    # --- metadata & embeddings ---
    row.update(get_stock_metadata(tkr))
    row.update(get_embeddings(message["summary_gcs_path"]))

    # --- call date & EPS surprise ---
    api_key = get_fmp_api_key()
    row.update(get_earnings_call_date(message["transcript_gcs_path"]))
    row.update(get_eps_surprise(api_key, tkr))

    # --- sentiment ---
    fin = get_finbert_sentiments(message["summary_gcs_path"])
    row["sentiment_score"] = fin["pos_prob"]
    row.update(fin)

    # --- price features (only if call date found) ---
    if row.get("earnings_call_date"):
        row.update(get_price_technicals(tkr, row["earnings_call_date"]))
        row.update(get_max_close_30d(tkr, row["earnings_call_date"]))
    else:
        logging.warning("No earnings_call_date; skipping price features for %s", tkr)

    # --- engineered features (unchanged) ---
    row["industry_sector"] = f"{row.get('sector','UNK_SEC')}_{row.get('industry','UNK_IND')}"
    vec = np.array(row["summary_embedding"]) if row.get("summary_embedding") else np.array([])
    if vec.size == 768 and np.all(np.isfinite(vec)):
        row["summary_norm"] = float(norm(vec))
        row["summary_mean"] = float(vec.mean())
        row["summary_std"]  = float(vec.std())
    else:
        row["summary_norm"] = row["summary_mean"] = row["summary_std"] = 0.0

    row["eps_surprise_isnull"] = int(row.get("eps_surprise") is None)
    if row.get("adj_close_on_call_date") and row.get("sma_20"):
        row["price_sma20_ratio"] = row["adj_close_on_call_date"]/row["sma_20"]
    if row.get("ema_50") and row.get("sma_20"):
        row["ema50_sma20_ratio"] = row["ema_50"]/row["sma_20"]
    if row.get("rsi_14") is not None:
        row["rsi_centered"] = row["rsi_14"] - 50
        if row.get("sentiment_score") is not None:
            row["sent_rsi"] = row["sentiment_score"] * row["rsi_centered"]
    if row.get("adx_14") is not None:
        row["adx_log1p"] = np.log1p(row["adx_14"])

    row.update(get_lda_topics(message["summary_gcs_path"]))

    if not {"earnings_call_date", "sentiment_score"}.issubset(row):
        logging.error("Critical features missing for %s", tkr)
        return None
    return row