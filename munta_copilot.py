import os
import io
import wave
import time
import json
import queue
import asyncio
import threading
import uuid
from collections import deque
import numpy as np
import pyaudiowpatch as pyaudio
import httpx
from google import genai
from google.genai import types
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

# =====================================================================
# 1. KONFIGURATION OCH SYSTEMPROMPT (MED ALL DATA FRÅN DIN INLÄMNING)
# =====================================================================
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "9003f4476a1bdbd292c9e9ac030dea15b584cbfc")
deepgram_client = httpx.Client(timeout=10.0)

# Google Cloud Vertex AI (Gemini 2.5 Flash - Snabbaste & smartaste modellen)
GOOGLE_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "project-2b716df5-ad88-44bb-b58")
GOOGLE_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
gemini_client = genai.Client(vertexai=True, project=GOOGLE_PROJECT, location=GOOGLE_LOCATION)

# Ljudinställningar för automatisk tystnadsdetektering (VAD)
SILENCE_THRESHOLD = 150      # Känslighet (sänkt från 350 för att inte missa normal samtalsvolym)
SILENCE_DURATION = 1.0       # Sekunder av tystnad innan frågan skickas
MIN_SPEECH_SECONDS = 0.8     # Minsta längd på ett yttrande för att transkriberas

SYSTEM_PROMPT = """Du är en realtids-copilot åt en student som just nu har muntlig examination i kursen "Maskininlärning och Deep Learning".
Läraren ställer frågor. Din uppgift är att DIREKT ge studenten korta, naturliga, extremt tydliga talepunkter på svenska som studenten kan läsa rakt av eller med egna ord.

REGLER FÖR DINA SVAR:
1. Svara ALLTID med 1 kort mening, raka punkter. Fetmarkera nyckelord och exakta siffror.
2. Inget intro ("Här är svaret:"), ingen utfyllnad. Skriv i JAG-form ("Jag valde...", "Jag upptäckte...") så studenten kan läsa det rakt av!
3. Använd EXAKT den här sanningen från studentens inlämnade Google Colab-notebook:

FAKTA OM STUDENTENS PROJEKT (customer_churn_exam.csv):
- Problem: Binär klassificering av kundbortfall inom 30 dagar (`churned_30d`, 1 = slutar, 0 = stannar) utifrån snapshot `2026-08-01`.
- Storlek & Dubbletter: Ursprungligen 1818 rader, 20 kolumner. Jag hittade 18 exakta dubbletter på `customer_id` och tog bort dem FÖRE train_test_split (kvar: 1800 unika kunder), annars hade samma kund kunnat hamna i både train och test.
- Klassobalans: 323 kunder churnade (17,94 %) och 1477 stannade (82,06 %).
- Data Leakage (SUPER-VIKTIGT):
  1) `days_until_churn`: Har bara värden (1-30 dagar) för de 330 rader där `churned_30d == 1` och är NaN för alla som stannade. Vid prediktionstillfället vet man inte om/när kunden slutar i framtiden. Tar man med den får modellen 100 % genom fusk. Droppades!
  2) `retention_offer_sent`: Har 0.55 korrelation med churn (64,6 % av de som fick erbjudandet slutade, mot 8,4 % för övriga). Räddningserbjudanden skickas oftast EFTER att kunden redan flaggat att den vill sluta. Droppades pga hög läckagerisk!
  3) Övriga droppade kolumner: `customer_id` (bara ett ID), `snapshot_date` (konstant 2026-08-01), `signup_date` (finns redan som `tenure_months`).
- Saknade värden (NaN): `nps_score` (141), `age` (91), `avg_session_minutes` (60), `region` (47), `feature_usage_score` (45), `payment_method` (26).
- Train/Test & Preprocessing Pipeline:
  * 80/20 `train_test_split` med `stratify=y` och `random_state=42` (1440 train, 360 test varav 65 churn).
  * `ColumnTransformer` + `Pipeline` som enbart kör `fit` på `X_train` (förhindrar dataläckage från testdatan).
  * Numeriska (9 st): `SimpleImputer(strategy='median')` (tåligt mot outliers) + `StandardScaler()`.
  * Kategoriska (4 st): `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(drop='first', handle_unknown='ignore')`.
- Mätetal (Metrics):
  * Accuracy är missvisande pga obalanserad data (82 % stannar).
  * **Recall** är viktigast affärsmässigt: En **False Negative** betyder att vi missar en kund som faktiskt säger upp sig (mycket dyrt att tappa en kund). En **False Positive** betyder bara att vi skickar ett erbjudande/kontaktar en nöjd kund i onödan (låg kostnad). Jag kollade även på F1-score och ROC-AUC.
- Modellresultat på testdatan (360 rader):
  1) **Baseline (`DummyClassifier`, most_frequent)**: Accuracy 81,9 %, Precision 0,0 %, Recall 0,0 %, F1 0,000, ROC-AUC 0,500.
  2) **Logistic Regression (`class_weight='balanced'`) [VALD MODELL]**: Accuracy 69,7 %, Precision 32,5 %, **Recall 63,1 %**, **F1 0,429**, **ROC-AUC 0,766**. (5-fold CV på train gav Recall 68,3 % och F1 0,447).
  3) **Random Forest (`max_depth=5`, `min_samples_leaf=5`, `class_weight='balanced'`)**: Accuracy 72,8 %, Precision 33,7 %, Recall 52,3 %, F1 0,410, ROC-AUC 0,729.
  4) **PyTorch Neuralt Nätverk (`ChurnMLP`)**: Accuracy 71,1 %, Precision 31,4 %, Recall 50,8 %, F1 0,388, ROC-AUC 0,696.
- Confusion Matrix för Logistic Regression (Testdata, 360 kunder):
  * True Negatives (Stannar korrekt): 210 st
  * False Positives (Stannar, men modellen gissar churn): 85 st
  * False Negatives (Slutar, men modellen missar dem): 24 st
  * True Positives (Slutar, och modellen fångar dem): 41 st (41 av 65 = 63,1 % Recall). Vill man fånga fler av de 24 FN kan man sänka threshold från 0.5 till t.ex. 0.4.
- Viktigaste variablerna (Koefficienter i LogReg):
  * Ökar churn mest: Månadsavtal (`contract_type_monthly`), hög månadsavgift (`monthly_fee`), faktura (`payment_method_invoice`), sena betalningar (`late_payments_12m`) och supportärenden (`support_tickets_90d`).
  * Minskar churn mest: Tvåårsavtal (`contract_type_two_year`), många inloggningsdagar (`login_days_30d`), hög `feature_usage_score` och hög `nps_score`.
- Deep Learning & PyTorch-detaljer:
  * Arkitektur: Input (23 features efter OHE) -> Linear(23, 32) + ReLU + Dropout(0.2) -> Linear(32, 16) + ReLU -> Linear(16, 1) som ger råa **logits**.
  * Loss & Optimizer: `BCEWithLogitsLoss` med `pos_weight` (ca 4.57, dvs antal 0:or delat på 1:or) för att hantera obalansen. `Adam` optimizer (lr=0.005), batch_size=32, 30 epochs.
  * Varför klassisk ML vann över Deep Learning: Tabulär data med bara 1800 rader och relativt linjära samband. Neurala nätverk kräver mer data, overfittar lättare på tabeller och är svårare att tolka (black box).
  * När Deep Learning är rätt: Ostrukturerad data som bilder (CNN), ljud/sensorer/sekvenser (Conv1d/RNN) eller fritext (NLP/Transformers).
- Transfer Learning:
  * Att ta en modell förtränad på enorma datamängder (t.ex. ResNet för bild eller BERT för text) och återanvända kunskapen.
  * *Feature Extraction*: Frysa basmodellens vikter och bara träna sista lagret.
  * *Fine-tuning*: Träna vidare delar av eller hela nätverket med låg learning rate.
  * Ej relevant för ren tabulär data med företagsunika kolumner, men hade kunnat användas om vi hade fritext från supportärenden (via t.ex. KB-BERT).
- MLOps:
  * Sparade hela pipelinen (`preprocessor` + `LogisticRegression`) i en fil med `joblib.dump` (`churn_pipeline_v1.joblib`).
  * Ny data skickas in som en rå DataFrame med 14 kolumner – pipelinen sköter imputering, skalning och OHE automatiskt.
  * Övervaka **Data Drift** (om indatans fördelning ändras, t.ex. nya priser/avtal) och **Concept Drift** (om kundbeteenden ändras). Utvärdera prediktionerna efter 30 dagar när facit finns; sjunker Recall/ROC-AUC tränas modellen om. Versionshantera kod, data-snapshot och `requirements.txt`.
"""

# =====================================================================
# 2. GLOBAL STATE FÖR LJUD & WEBSOCKETS
# =====================================================================
audio_frames = []
is_speaking = False
last_voice_time = 0.0
force_flush = False
connected_clients = set()
event_loop = None
active_device_name = "hörlurar"

def broadcast_sync(payload: dict):
    """Skickar JSON-meddelanden från ljudtråden till alla öppna webbläsarfönster."""
    if event_loop and connected_clients:
        msg = json.dumps(payload, ensure_ascii=False)
        for ws in list(connected_clients):
            asyncio.run_coroutine_threadsafe(ws.send_text(msg), event_loop)

# =====================================================================
# 3. TRANSKRIBERING (WHISPER) + AI-SVAR (LLAMA 3.3 70B)
# =====================================================================
def process_audio_chunk(raw_bytes: bytes, sample_rate: int, channels: int, sample_width: int):
    try:
        broadcast_sync({"type": "status", "text": "Transkriberar med Deepgram..."})
        
        # Skapa WAV-fil i minnet
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(raw_bytes)
        wav_buffer.seek(0)
        audio_content = wav_buffer.read()

        # 1. Transkribera med Deepgram Nova-3 (svenska, noll hallucinationer)
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/wav"
        }
        params = {
            "model": "nova-3",
            "language": "sv",
            "smart_format": "true",
            "punctuate": "true"
        }
        resp = deepgram_client.post(
            "https://api.deepgram.com/v1/listen",
            headers=headers,
            params=params,
            content=audio_content
        )
        if resp.status_code != 200:
            print(f"[DEEPGRAM FEL {resp.status_code}] {resp.text}", flush=True)
            broadcast_sync({"type": "status", "text": f"Deepgram fel: {resp.status_code}"})
            return

        res_json = resp.json()
        question_text = res_json["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
        print(f"[DEEPGRAM TRANSKRIBERAT] {question_text}", flush=True)

        if len(question_text) < 3:
            broadcast_sync({"type": "status", "text": f"Lyssnar på: {active_device_name}..."})
            return

        broadcast_sync({"type": "transcript", "text": question_text})
        # 2. Generera svar med Gemini 2.5 Flash med realtids token-streaming!
        stream_gemini_response(question_text)

    except Exception as e:
        print(f"[PROCESS AUDIO FEL] {e}", flush=True)
        broadcast_sync({"type": "status", "text": f"Fel: {str(e)}"})

def stream_gemini_response(question_text: str):
    """Genererar svar med Gemini 2.5 Flash och strömmar orden i realtid via WebSocket."""
    card_id = str(uuid.uuid4())[:8]
    broadcast_sync({"type": "start_answer", "id": card_id, "question": question_text})
    broadcast_sync({"type": "status", "text": "Skriver svar med Gemini..."})

    try:
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            max_output_tokens=300,
            temperature=0.2
        )
        stream = gemini_client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=f"Läraren sa/frågade precis: '{question_text}'. Ge mig talepunkter att svara med direkt!",
            config=config
        )
        for chunk in stream:
            if chunk.text:
                broadcast_sync({"type": "stream_chunk", "id": card_id, "chunk": chunk.text})
    except Exception as e:
        print(f"[GEMINI FEL] {e}", flush=True)
        broadcast_sync({"type": "stream_chunk", "id": card_id, "chunk": f"\n\n[Gemini Fel: {str(e)}]"})
    finally:
        broadcast_sync({"type": "stream_end", "id": card_id})
        broadcast_sync({"type": "status", "text": f"Lyssnar på: {active_device_name}..."})

# =====================================================================
# 4. WINDOWS WASAPI LOOPBACK LJUDFÅNGARE
# =====================================================================
def loopback_listener_thread():
    global audio_frames, is_speaking, last_voice_time, force_flush, active_device_name

    with pyaudio.PyAudio() as p:
        try:
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            print("Kunde inte hitta WASAPI. Kör du på Windows?")
            return

        loopback_devices = list(p.get_loopback_device_info_generator())
        if not loopback_devices:
            print("Kunde inte hitta några WASAPI Loopback-enheter.")
            broadcast_sync({"type": "status", "text": "Fel: Inga loopback-enheter hittades"})
            return

        # Sök efter hörlurar / headset (t.ex. Trust GXT Headset)
        target_device = None
        keywords = ["headset", "hörlur", "headphone", "earphone", "trust"]
        for dev in loopback_devices:
            name_lower = dev["name"].lower()
            if any(k in name_lower for k in keywords):
                target_device = dev
                break

        # Fallback till standard-utgång om hörlurar inte matchades på namn
        if not target_device:
            default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
            for dev in loopback_devices:
                if default_speakers["name"] in dev["name"]:
                    target_device = dev
                    break

        # Sista fallback till första tillgängliga loopback-enhet
        if not target_device:
            target_device = loopback_devices[0]

        active_device_name = target_device["name"].split(" [")[0]
        sample_rate = int(target_device["defaultSampleRate"])
        channels = target_device["maxInputChannels"]
        sample_width = p.get_sample_size(pyaudio.paInt16)
        chunk_size = int(sample_rate * 0.1)  # 100 ms per chunk

        print(f"[LOOPBACK AKTIV] Lyssnar på hörlurar: {target_device['name']} ({sample_rate} Hz, {channels} kanaler)", flush=True)
        broadcast_sync({"type": "status", "text": f"Lyssnar på: {active_device_name}..."})

        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            frames_per_buffer=chunk_size,
            input=True,
            input_device_index=target_device["index"]
        )

        pre_buffer = deque(maxlen=6)  # Sparar senaste 600 ms så inte början av meningen klipps bort

        while True:
            data = stream.read(chunk_size, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
            now = time.time()

            if rms > SILENCE_THRESHOLD:
                if not is_speaking:
                    is_speaking = True
                    broadcast_sync({"type": "status", "text": "Läraren pratar... (spelar in)"})
                    audio_frames.extend(pre_buffer)
                    pre_buffer.clear()
                last_voice_time = now
                audio_frames.append(data)
            else:
                if not is_speaking:
                    pre_buffer.append(data)
                else:
                    audio_frames.append(data)
                    speech_duration = (len(audio_frames) * chunk_size) / sample_rate

                    # Om det varit tyst i SILENCE_DURATION sekunder eller om användaren tryckt "Svara nu"
                    if (now - last_voice_time > SILENCE_DURATION) or force_flush:
                        is_speaking = False
                        force_flush = False
                        captured_bytes = b"".join(audio_frames)
                        audio_frames = []

                        if speech_duration >= MIN_SPEECH_SECONDS:
                            threading.Thread(
                                target=process_audio_chunk,
                                args=(captured_bytes, sample_rate, channels, sample_width),
                                daemon=True
                            ).start()
                        else:
                            broadcast_sync({"type": "status", "text": f"Lyssnar på: {active_device_name}..."})

            if force_flush and audio_frames:
                is_speaking = False
                force_flush = False
                captured_bytes = b"".join(audio_frames)
                audio_frames = []
                threading.Thread(
                    target=process_audio_chunk,
                    args=(captured_bytes, sample_rate, channels, sample_width),
                    daemon=True
                ).start()

# =====================================================================
# 5. FASTAPI SERVER & LIVE-DASHBOARD (HTML/JS)
# =====================================================================
app = FastAPI()

DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="sv">
<head>
    <meta charset="UTF-8">
    <title>Munta Copilot – Live Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background: #0d1117;
            color: #e6edf3;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            padding: 18px;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #161b22;
            padding: 12px 18px;
            border-radius: 10px;
            border: 1px solid #30363d;
            margin-bottom: 14px;
        }
        .status {
            font-size: 14px;
            color: #3fb950;
            font-weight: 600;
        }
        .controls {
            display: flex;
            gap: 10px;
        }
        button {
            background: #238636;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            font-size: 14px;
        }
        button:hover { background: #2ea043; }
        .manual-box {
            display: flex;
            gap: 8px;
            margin-bottom: 14px;
        }
        input[type="text"] {
            flex: 1;
            background: #161b22;
            border: 1px solid #30363d;
            color: #e6edf3;
            padding: 10px 14px;
            border-radius: 8px;
            font-size: 14px;
        }
        #feed {
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }
        .card {
            background: #161b22;
            border: 1px solid #30363d;
            border-left: 4px solid #58a6ff;
            border-radius: 10px;
            padding: 16px 20px;
            animation: fadeIn 0.2s ease-in;
        }
        .question {
            font-size: 14px;
            color: #8b949e;
            margin-bottom: 10px;
            font-style: italic;
        }
        .answer {
            font-size: 18px;
            line-height: 1.55;
            color: #f0f6fc;
            white-space: pre-wrap;
        }
        .answer strong, .answer b {
            color: #79c0ff;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-6px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <strong>Munta Copilot</strong> &nbsp;|&nbsp; <span id="status" class="status">Ansluter...</span>
        </div>
        <div class="controls">
            <button onclick="flushAudio()" title="Kortkommando: Mellanslag">⚡ Svara nu (Space)</button>
        </div>
    </div>

    <div class="manual-box">
        <input type="text" id="manualInput" placeholder="Eller skriv en fråga/nyckelord manuellt (t.ex. 'transfer learning' eller 'confusion matrix') och tryck Enter..." onkeydown="if(event.key==='Enter') sendManual()">
        <button onclick="sendManual()">Skicka</button>
    </div>

    <div id="feed"></div>

    <script>
        const ws = new WebSocket(`ws://${location.host}/ws`);
        const statusEl = document.getElementById('status');
        const feedEl = document.getElementById('feed');
        let activeCards = {};

        ws.onopen = () => { statusEl.innerText = "Lyssnar på systemljud (Loopback)..."; };
        ws.onclose = () => { statusEl.innerText = "Frånkopplad"; statusEl.style.color = "#f85149"; };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === "status") {
                statusEl.innerText = data.text;
            } else if (data.type === "start_answer") {
                const card = document.createElement('div');
                card.className = 'card';
                card.id = `card-${data.id}`;
                card.innerHTML = `
                    <div class="question">🎧 Lärare: "${data.question}"</div>
                    <div class="answer" id="answer-${data.id}"></div>
                `;
                feedEl.prepend(card);
                activeCards[data.id] = { el: document.getElementById(`answer-${data.id}`), raw: "" };
            } else if (data.type === "stream_chunk") {
                const item = activeCards[data.id];
                if (item) {
                    item.raw += data.chunk;
                    item.el.innerHTML = item.raw.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                }
            } else if (data.type === "stream_end") {
                delete activeCards[data.id];
            } else if (data.type === "answer") {
                const card = document.createElement('div');
                card.className = 'card';
                const formattedAnswer = data.answer.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                card.innerHTML = `
                    <div class="question">🎧 Lärare: "${data.question}"</div>
                    <div class="answer">${formattedAnswer}</div>
                `;
                feedEl.prepend(card);
            }
        };

        function flushAudio() {
            ws.send(JSON.stringify({ action: "flush" }));
        }

        function sendManual() {
            const input = document.getElementById('manualInput');
            if (input.value.trim() !== "") {
                ws.send(JSON.stringify({ action: "manual", text: input.value.trim() }));
                input.value = "";
            }
        }

        // Tryck Spacebar (när man inte skriver i textrutan) för att tvinga fram svar direkt
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Space' && document.activeElement.id !== 'manualInput') {
                e.preventDefault();
                flushAudio();
            }
        });
    </script>
</body>
</html>
"""

@app.get("/")
async def index():
    return HTMLResponse(DASHBOARD_HTML)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global force_flush
    await websocket.accept()
    connected_clients.add(websocket)
    await websocket.send_text(json.dumps({"type": "status", "text": f"Lyssnar på: {active_device_name}..."}, ensure_ascii=False))
    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            if data.get("action") == "flush":
                force_flush = True
            elif data.get("action") == "manual":
                question_text = data.get("text", "")
                threading.Thread(target=stream_gemini_response, args=(question_text,), daemon=True).start()
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

@app.on_event("startup")
async def startup_event():
    global event_loop
    event_loop = asyncio.get_running_loop()
    threading.Thread(target=loopback_listener_thread, daemon=True).start()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)