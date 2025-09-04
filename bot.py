# bot.py - EL MEJOR BOT DE EUR/USD DEL MUNDO
import os
import yfinance as yf
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters
)
from dotenv import load_dotenv
import json

# Cargar variables
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
CHAT_ID = os.getenv("CHAT_ID")

# --- Configuración Global ---
MAIN_SYMBOL = "EURUSD=X"
NEWS_API_KEY = "tu_clave_newsapi"  # Gratis en https://newsapi.org

# --- Funciones Clave ---

def get_eurusd_full():
    """Devuelve datos avanzados del EUR/USD"""
    # Precio actual
    data = yf.download(MAIN_SYMBOL, period="1d", interval="1m")
    current_price = round(data['Close'].iloc[-1], 4)
    
    # Análisis técnico básico
    rsi = calculate_rsi(data['Close'])
    trend = "alcista" if data['Close'].iloc[-1] > data['Close'].iloc[0] else "bajista"
    
    # Noticias relevantes
    news = get_forex_news()
    
    return {
        "price": current_price,
        "rsi": rsi,
        "trend": trend,
        "news": news[:2],  # Solo 2 noticias
        "support": round(current_price - 0.005, 4),
        "resistance": round(current_price + 0.005, 4)
    }

def generate_dashboard(data):
    """Crea un gráfico interactivo con Plotly"""
    fig = go.Figure()
    
    # Datos históricos
    hist = yf.download(MAIN_SYMBOL, period="7d", interval="1h")
    
    # Gráfico principal
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist['Open'],
        high=hist['High'],
        low=hist['Low'],
        close=hist['Close'],
        name="EUR/USD"
    ))
    
    # Niveles clave
    fig.add_hline(y=data['support'], line_dash="dash", line_color="green", annotation_text="Soporte")
    fig.add_hline(y=data['resistance'], line_dash="dash", line_color="red", annotation_text="Resistencia")
    
    fig.update_layout(
        title=f"EUR/USD - Análisis Técnico ({datetime.now().strftime('%d/%m/%Y')})",
        xaxis_title="Fecha",
        yaxis_title="Precio",
        template="plotly_dark"
    )
    
    return fig.to_image(format="png")

def ask_qwen(prompt):
    """Consulta avanzada a Qwen con contexto financiero"""
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {"Authorization": f"Bearer {QWEN_API_KEY}", "Content-Type": "application/json"}
    
    data = {
        "model": "qwen-turbo",
        "input": {"messages": [{"role": "user", "content": prompt}]},
        "parameters": {"temperature": 0.3, "max_tokens": 300}
    }
    
    response = requests.post(url, json=data, headers=headers)
    return response.json()['output']['text'] if response.status_code == 200 else "Error en análisis"

# --- Menús del Bot ---

def main_menu():
    keyboard = [
        [InlineKeyboardButton("💰 EUR/USD Hoy", callback_data='price')],
        [InlineKeyboardButton("📈 Dashboard", callback_data='dashboard')],
        [InlineKeyboardButton("🔔 Alertas", callback_data='alerts')],
        [InlineKeyboardButton("📰 Noticias", callback_data='news')],
        [InlineKeyboardButton("💡 Modo Embajador", callback_data='ambassador')]
    ]
    return InlineKeyboardMarkup(keyboard)

# --- Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🚀 ¡BIENVENIDO AL MEJOR BOT DE EUR/USD DEL MUNDO!\n\n"
        "Soy tu asistente financiero inteligente, con:\n"
        "• Análisis en tiempo real\n"
        "• Alertas inteligentes\n"
        "• Dashboard profesional\n"
        "• Modo Embajador para ganar beneficios\n\n"
        "¿Qué quieres hacer hoy?",
        reply_markup=main_menu()
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    data = get_eurusd_full()
    
    if query.data == 'price':
        analysis = ask_qwen(
            f"EUR/USD: {data['price']}\n"
            f"Tendencia: {data['trend']}\n"
            f"RSI: {data['rsi']}\n"
            "Analiza en 3 líneas: situación actual y recomendación clara."
        )
        await query.edit_message_text(
            f"💶 **EUR/USD HOY**: {data['price']}\n"
            f"🟢 **Tendencia**: {data['trend'].upper()}\n"
            f"📊 **RSI**: {data['rsi']}\n\n"
            f"🔍 **Análisis Qwen**:\n{analysis}\n\n"
            "¿Qué más necesitas?",
            reply_markup=main_menu(),
            parse_mode='Markdown'
        )
    
    elif query.data == 'dashboard':
        img = generate_dashboard(data)
        await context.bot.send_photo(
            chat_id=query.message.chat_id,
            photo=img,
            caption="📈 Dashboard EUR/USD - Analiza los niveles clave y tendencias"
        )
        await query.message.reply_text("¿Qué más necesitas?", reply_markup=main_menu())
    
    # ... (otros handlers para alerts, news, ambassador)

# --- Inicio ---
def main():
    app = Application.builder().token(TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    
    print("🚀 ¡BOT DEL MUNDO INICIADO! (El mejor de todos)")
    app.run_polling()

if __name__ == "__main__":
    main()
