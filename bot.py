# bot.py - Frontend de Telegram para robot original
import os
import asyncio
import logging
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler
from dotenv import load_dotenv

# === Cargar variables ===
load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === Importar el robot original ===
try:
    from robot_trading_completo import RobotTradingFinal, PARES_FOREX, CONFIANZA_MINIMA
except Exception as e:
    logger.critical(f"Error importando robot: {e}")
    raise

# === Menú ===
def main_menu():
    keyboard = [
        [InlineKeyboardButton("📊 Ver Mejor Señal", callback_data='signal')],
        [InlineKeyboardButton("📈 Dashboard", callback_data='dashboard')]
    ]
    return InlineKeyboardMarkup(keyboard)

# === Handlers ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ✅ Enviar "hola estoy despierto"
    if CHAT_ID and TOKEN:
        try:
            await context.bot.send_message(chat_id=CHAT_ID, text="hola estoy despierto ✅")
            logger.info("Mensaje de inicio enviado")
        except Exception as e:
            logger.error(f"No se pudo enviar mensaje: {e}")

    await update.message.reply_text(
        "🚀 ¡Robot de Trading con IA activo!\n"
        "Usa el menú para ver señales de EUR/USD, USD/JPY, Oro y Petróleo.",
        reply_markup=main_menu()
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    robot = RobotTradingFinal()
    mejor_senal = None
    mejor_confianza = 0
    datos_mercado = {}

    # Obtener datos de los 4 activos
    for ticker, info in PARES_FOREX.items():
        try:
            data = robot.data_provider.get_latest_data(ticker, "1h", 100)
            if data is not None and len(data) >= 60:
                precio_actual = data['Close'].iloc[-1]
                datos_mercado[ticker] = {'data': data, 'precio_actual': precio_actual, 'info': info}
        except Exception as e:
            logger.error(f"Error con {info['nombre']}: {e}")

    if not datos_mercado:
        await query.edit_message_text("❌ Sin datos del mercado", reply_markup=main_menu())
        return

    # Buscar mejor señal
    for ticker, datos in datos_mercado.items():
        info = datos['info']
        data = datos['data']
        senal, _ = robot.predecir_senal(data, info)
        if senal and senal['confianza'] >= CONFIANZA_MINIMA and senal['prediccion'] != 0:
            filtro_valido, _ = robot.filtrar_señal_mercado(data, senal['prediccion'], ticker, senal.get('analisis_cualitativo', {}))
            if filtro_valido and senal['confianza'] > mejor_confianza:
                mejor_confianza = senal['confianza']
                mejor_senal = senal

    if mejor_senal:
        sl_tp = robot.calcular_sl_tp(
            datos_mercado[mejor_senal['ticker']]['data'],
            mejor_senal['prediccion'],
            mejor_senal['precio_actual']
        )
        if sl_tp:
            mensaje = robot.crear_mensaje_alerta(mejor_senal, sl_tp, 1000)
            mensaje_md = mensaje.replace("<b>", "*").replace("</b>", "*")
            await query.edit_message_text(mensaje_md, reply_markup=main_menu(), parse_mode='Markdown')
        else:
            await query.edit_message_text("❌ Error calculando SL/TP", reply_markup=main_menu())
    else:
        await query.edit_message_text("❌ No hay señales válidas ahora", reply_markup=main_menu())

# === Inicio ===
def main():
    print(f"🔧 TOKEN: {'OK' if TOKEN else 'FALTA'}")
    print(f"🔧 CHAT_ID: {'OK' if CHAT_ID else 'FALTA'}")
    print(f"🔧 QWEN_API_KEY: {'OK' if QWEN_API_KEY else 'FALTA'}")

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))

    # ✅ Mensaje de inicio
    if CHAT_ID and TOKEN:
        try:
            asyncio.create_task(
                app.bot.send_message(chat_id=CHAT_ID, text="hola estoy despierto ✅")
            )
        except:
            pass

    logger.info("🚀 Bot iniciado")
    app.run_polling()

if __name__ == "__main__":
    main()
