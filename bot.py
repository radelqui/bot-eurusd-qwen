# bot.py - Robot de Trading con IA (adaptado para Railway + Telegram Bot)
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import uuid
import logging
import json
import os
import joblib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler
from dotenv import load_dotenv

# === Cargar variables de entorno ===
load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN", "7718630865:AAEMclwlqzuxb5uFPqX9dyJLo7ib19QnJt8")
CHAT_ID = os.getenv("CHAT_ID", "5358902915")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")  # Para análisis con IA

# === CONFIGURACIÓN DEL ROBOT (simplificada) ===
PARES_FOREX = {
    "EURUSD=X": {"nombre": "EUR/USD", "tipo": "major"},
    "USDJPY=X": {"nombre": "USD/JPY", "tipo": "major"}
}
ARCHIVO_MODELO = "modelo_ia_final.pkl"
ARCHIVO_SCALER = "scaler_final.pkl"
OPERACIONES_DIA = 2
CONFIANZA_MINIMA = 0.75
RISK_REWARD_RATIO = 2.0

# === CONFIGURACIÓN DE LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("robot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === FUNCIONES AUXILIARES ===
def get_eurusd_price():
    try:
        data = yf.download("EURUSD=X", period="1d", interval="1m")
        return round(data['Close'].iloc[-1], 5)
    except:
        return None

def ask_qwen(prompt):
    """Consulta a Qwen para análisis inteligente"""
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "qwen-plus",
        "input": {"messages": [{"role": "user", "content": prompt}]},
        "parameters": {"temperature": 0.5, "max_tokens": 300}
    }
    try:
        response = requests.post(url, json=data, headers=headers, timeout=15)
        if response.status_code == 200:
            return response.json()['output']['text']
        else:
            return f"❌ Error Qwen: {response.status_code}"
    except Exception as e:
        return f"❌ Error conexión: {str(e)}"

# === MENÚ PRINCIPAL ===
def main_menu():
    keyboard = [
        [InlineKeyboardButton("💰 EUR/USD Hoy", callback_data='price')],
        [InlineKeyboardButton("📊 Última Señal", callback_data='signal')],
        [InlineKeyboardButton("🤖 Análisis IA", callback_data='analyze')],
        [InlineKeyboardButton("📈 Dashboard", callback_data='dashboard')]
    ]
    return InlineKeyboardMarkup(keyboard)

# === HANDLERS DEL BOT ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ✅ Enviar "hola estoy despierto" al CHAT_ID
    if CHAT_ID and TOKEN:
        try:
            await context.bot.send_message(chat_id=CHAT_ID, text="hola estoy despierto ✅")
            logger.info("Mensaje de inicio enviado")
        except Exception as e:
            logger.error(f"No se pudo enviar mensaje: {e}")

    await update.message.reply_text(
        "🚀 ¡Bienvenido al Robot de Trading Inteligente!\n"
        "Usa el menú para ver señales, análisis y más.",
        reply_markup=main_menu()
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == 'price':
        price = get_eurusd_price()
        if price:
            analysis = ask_qwen(f"EUR/USD está en {price}. Analiza en 2 líneas.")
            await query.edit_message_text(
                f"💶 **EUR/USD**: {price}\n\n"
                f"🔍 **Análisis IA**:\n{analysis}\n\n"
                "¿Qué más necesitas?",
                reply_markup=main_menu(),
                parse_mode='Markdown'
            )
        else:
            await query.edit_message_text("❌ No pude obtener el precio", reply_markup=main_menu())

    elif query.data == 'signal':
        # Simulación de señal (en tu versión real, esto viene del modelo)
        await query.edit_message_text(
            "🟢 **SEÑAL ACTIVA**\n\n"
            "Par: EUR/USD\n"
            "Dirección: COMPRA\n"
            "Precio: 1.0785\n"
            "Stop Loss: 1.0750\n"
            "Take Profit: 1.0850\n"
            "Confianza: 78%\n\n"
            "¡Operación en curso!",
            reply_markup=main_menu()
        )

    elif query.data == 'analyze':
        price = get_eurusd_price()
        if not price:
            await query.edit_message_text("❌ Sin datos", reply_markup=main_menu())
            return
        prompt = (
            f"EUR/USD: {price}\n"
            "RSI: 62 (neutral)\n"
            "Tendencia: lateral con sesgo alcista\n"
            "Analiza en 3 líneas: situación actual y recomendación clara."
        )
        analysis = ask_qwen(prompt)
        await query.edit_message_text(
            f"🤖 **Análisis Profundo**:\n{analysis}\n\n"
            "¿Qué más necesitas?",
            reply_markup=main_menu()
        )

    elif query.data == 'dashboard':
        await query.edit_message_text(
            "📈 *Dashboard*\n\n"
            "Próximamente:\n"
            "• Gráficos en vivo\n"
            "• Historial de operaciones\n"
            "• Estadísticas de rendimiento",
            reply_markup=main_menu(),
            parse_mode='Markdown'
        )

# === FUNCIONES DEL ROBOT (simplificadas) ===
class RobotTradingBot:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.load_model()

    def load_model(self):
        try:
            if os.path.exists(ARCHIVO_MODELO) and os.path.exists(ARCHIVO_SCALER):
                self.model = joblib.load(ARCHIVO_MODELO)
                self.scaler = joblib.load(ARCHIVO_SCALER)
                logger.info("Modelo cargado correctamente")
            else:
                logger.warning("Modelo no encontrado. Se usará modo simulado.")
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")

    def predict_signal(self, symbol: str):
        # Aquí iría la predicción real
        # Por ahora, simulamos una señal
        return {
            "par": "EUR/USD",
            "prediccion": 1,  # 1=COMPRA, -1=VENTA
            "confianza": 0.78,
            "precio": get_eurusd_price()
        }

# === INICIO DEL BOT ===
async def send_daily_signals(context: ContextTypes.DEFAULT_TYPE):
    """Envía señales automáticas cada 6 horas"""
    robot = RobotTradingBot()
    signal = robot.predict_signal("EURUSD=X")
    if signal and signal['confianza'] >= CONFIANZA_MINIMA:
        msg = (
            f"🔔 *SEÑAL AUTOMÁTICA*\n\n"
            f"Par: {signal['par']}\n"
            f"Dirección: {'🟢 COMPRA' if signal['prediccion'] == 1 else '🔴 VENTA'}\n"
            f"Precio: {signal['precio']}\n"
            f"Confianza: {signal['confianza']*100:.0f}%"
        )
        try:
            await context.bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Error enviando señal: {e}")

def main():
    # Verificar variables
    print(f"🔧 TOKEN: {'OK' if TOKEN else 'FALTA'}")
    print(f"🔧 CHAT_ID: {'OK' if CHAT_ID else 'FALTA'}")
    print(f"🔧 QWEN_API_KEY: {'OK' if QWEN_API_KEY else 'FALTA'}")

    # Iniciar bot
    app = Application.builder().token(TOKEN).build()

    # Handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))

    # Programar alertas automáticas
    app.job_queue.run_repeating(send_daily_signals, interval=21600, first=10)  # Cada 6h

    # ✅ Mensaje de inicio
    if CHAT_ID and TOKEN:
        try:
            asyncio.create_task(
                app.bot.send_message(chat_id=CHAT_ID, text="hola estoy despierto ✅")
            )
        except:
            pass

    logger.info("🚀 Bot iniciado y listo")
    app.run_polling()

if __name__ == "__main__":
    import asyncio
    main()
