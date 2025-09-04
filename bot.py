# bot.py
import os
import asyncio
import yfinance as yf
import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# === CONFIGURACIÓN (no toques aquí) ===
TOKEN = os.getenv("TELEGRAM_TOKEN")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
CHAT_ID = os.getenv("CHAT_ID")

# Verificar que tengamos todo
if not TOKEN or not QWEN_API_KEY or not CHAT_ID:
    raise Exception("Faltan variables de entorno. Revisa .env o Railway.")

# --- Función: obtener EUR/USD ---
def get_eurusd():
    try:
        url = "https://api.exchangerate-api.com/v4/latest/EUR"
        response = requests.get(url)
        data = response.json()
        return round(data['rates']['USD'], 4)
    except:
        return None

# --- Función: generar gráfico EUR/USD ---
def get_eurusd_chart():
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        data = yf.download("EURUSD=X", start=start_date, end=end_date)

        plt.figure(figsize=(10, 5))
        plt.plot(data.index, data['Close'], color='blue', linewidth=2)
        plt.title('EUR/USD - Últimos 7 días', fontsize=16)
        plt.xlabel('Fecha')
        plt.ylabel('Precio')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        chart_path = "eurusd_chart.png"
        plt.savefig(chart_path)
        plt.close()

        return chart_path
    except Exception as e:
        print(f"Error generando gráfico: {e}")
        return None

# --- Función: preguntar a Qwen ---
def ask_qwen(prompt):
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "qwen-turbo",
        "input": {
            "messages": [
                {"role": "user", "content": prompt}
            ]
        },
        "parameters": {
            "temperature": 0.7,
            "max_tokens": 200
        }
    }
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            result = response.json()
            return result['output']['text']
        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return "No pude conectar con Qwen."

# --- Comando /start ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hola, soy tu *Asistente Financiero con IA (Qwen)*.\n"
        "Puedo:\n"
        "• Responder preguntas sobre EUR/USD\n"
        "• Mostrar el tipo de cambio (/eurusd)\n"
        "• Enviar un gráfico (/grafico)\n\n"
        "Y estoy vigilando el mercado para alertarte ⚡"
    )

# --- Comando /eurusd ---
async def eurusd_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rate = get_eurusd()
    if rate:
        await update.message.reply_text(f"💶 EUR/USD: *{rate}*", parse_mode="Markdown")
    else:
        await update.message.reply_text("❌ No pude obtener el valor actual.")

# --- Comando /grafico ---
async def grafico_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📊 Generando gráfico del EUR/USD...")
    chart_path = get_eurusd_chart()
    if chart_path:
        with open(chart_path, 'rb') as photo:
            await update.message.reply_photo(photo=InputFile(photo), caption="EUR/USD - Últimos 7 días")
        os.remove(chart_path)
    else:
        await update.message.reply_text("❌ No se pudo generar el gráfico.")

# --- Manejar mensajes del usuario ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    user_name = update.message.from_user.first_name
    eurusd = get_eurusd()
    eurusd_info = f"Valor actual EUR/USD: {eurusd}" if eurusd else "EUR/USD: no disponible"

    prompt = f"{eurusd_info}\nUsuario ({user_name}): {user_text}\nRespuesta clara y breve en español:"

    response = ask_qwen(prompt)
    await update.message.reply_text(response)

# --- Alerta automática cada 5 minutos ---
async def send_alert(context: ContextTypes.DEFAULT_TYPE):
    rate = get_eurusd()
    if rate and rate >= 1.08:  # ← Cambia este valor si quieres otra alerta
        await context.bot.send_message(
            chat_id=CHAT_ID,
            text=f"🚨 ¡Alerta EUR/USD! \nHa llegado a {rate} \n¡Atento a movimientos!"
        )

# --- Inicio principal ---
def main():
    print("🚀 Iniciando bot financiero con Qwen...")
    app = Application.builder().token(TOKEN).build()

    # Comandos
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("eurusd", eurusd_command))
    app.add_handler(CommandHandler("grafico", grafico_command))

    # Mensajes
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Programar alertas
    app.job_queue.run_repeating(send_alert, interval=300, first=10)  # Cada 5 min

    # Iniciar bot
    app.run_polling()

if __name__ == "__main__":
    main()
