# Munta Copilot 🎓⚡

Realtids-copilot för muntliga examinationer i Machine Learning och Deep Learning. 

Systemet fångar systemljud/lärarens röst i realtid från hörlurar (WASAPI Loopback), transkriberar talet med **Deepgram Nova-3**, och genererar omedelbart skarpa, precisa talepunkter på svenska via **Google Gemini 2.5 Flash** med realtids-streaming över WebSocket till en interaktiv dashboard.

## Funktioner

- 🎧 **Windows WASAPI Loopback**: Lyssnar direkt på ljudet i dina hörlurar/headset (eller mikrofon).
- 🎙️ **Deepgram Nova-3**: Blixtsnabb svensk tal-till-text (STT) utan hallucinationer vid tystnad.
- 🧠 **Google Gemini 2.5 Flash**: Snabb och intelligent resonemangsförmåga med realtids token-streaming.
- ⚡ **WebSocket Live Dashboard**: Svaren strömmas ord för ord på bråkdelen av en sekund.
- 🎯 **Skräddarsydd kunskapsbas**: Tränad på projektdata, dataläckagerisker, mätetal (Recall, F1), preprocessing pipelines och modelljämförelser (LogReg vs Random Forest vs PyTorch MLP).

## Installation

1. Klona repot:
```bash
git clone https://github.com/yoeltew/exam.git
cd exam
```

2. Skapa och aktivera virtuell miljö:
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Installera beroenden:
```bash
pip install -r requirements.txt
```

4. Konfigurera inloggningsuppgifter:
- Logga in med Google Cloud ADC för Gemini 2.5 Flash:
  ```bash
  gcloud auth application-default login
  ```
- Sätt din Deepgram API-nyckel:
  ```powershell
  $env:DEEPGRAM_API_KEY = "din_deepgram_nyckel"
  ```

## Körning

Starta servern:
```bash
python munta_copilot.py
```

Öppna webbläsaren på:
**[http://127.0.0.1:8000](http://127.0.0.1:8000)**
