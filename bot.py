# bot.py - ROBOT TRADING CON IA Y TELEGRAM
import os
import asyncio
import logging
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler
from dotenv import load_dotenv

# === Cargar variables de entorno ===
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

# === Menú principal ===
def main_menu():
    keyboard = [
        [InlineKeyboardButton("📊 Ver Mejor Señal", callback_data='signal')],
        [InlineKeyboardButton("📈 Dashboard", callback_data='dashboard')]
    ]
    return InlineKeyboardMarkup(keyboard)

# === Handlers del bot ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ✅ Enviar "hola estoy despierto" después de que el bot esté listo
    if CHAT_ID and TOKEN:
        try:
            await context.bot.send_message(chat_id=CHAT_ID, text="hola estoy despierto ✅")
            logger.info("✅ Mensaje de inicio enviado")
        except Exception as e:
            logger.error(f"❌ No se pudo enviar mensaje: {e}")

    await update.message.reply_text(
        "🚀 ¡Bienvenido al Robot de Trading Inteligente!\n"
        "Usa el menú para ver señales de EUR/USD, USD/JPY, Oro y Petróleo.\n\n"
        "Usa /entrenar para entrenar el modelo si es necesario.",
        reply_markup=main_menu()
    )

async def entrenar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Entrena el modelo IA desde Telegram"""
    await update.message.reply_text("🛠️ Iniciando entrenamiento del modelo IA...\nEsto puede tardar 2-3 minutos.")

    try:
        robot = RobotTradingFinal()
        if robot.entrenar_modelo():
            await update.message.reply_text("✅ ¡Modelo entrenado y guardado con éxito!\nAhora puedes usar 'Ver Mejor Señal'.")
        else:
            await update.message.reply_text("❌ Error: No se pudo entrenar el modelo.\nRevisa los logs en Railway.")
    except Exception as e:
        await update.message.reply_text(f"❌ Error crítico: {str(e)}")
        logger.error(f"Error en /entrenar: {e}")

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == 'signal':
        robot = RobotTradingFinal()
        
        # Verificar si el modelo está cargado
        if not robot.model or not robot.scaler:
            await query.edit_message_text(
                "❌ Modelo no disponible. Usa /entrenar para entrenarlo.",
                reply_markup=main_menu()
            )
            return

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
                logger.error(f"Error obteniendo datos de {info['nombre']}: {e}")

        if not datos_mercado:
            await query.edit_message_text("❌ No se pudieron obtener datos del mercado", reply_markup=main_menu())
            return

        # Buscar la mejor señal válida
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

# === Función que se ejecuta al iniciar el bot ===
async def post_init(application: Application) -> None:
    if CHAT_ID and TOKEN:
        try:
            await application.bot.send_message(chat_id=CHAT_ID, text="hola estoy despierto ✅")
            logger.info("✅ Mensaje de 'hola estoy despierto' enviado")
        except Exception as e:
            logger.error(f"❌ Error enviando mensaje de inicio: {e}")

# === Inicio del bot ===
def main():
    print(f"🔧 TOKEN: {'OK' if TOKEN else 'FALTA'}")
    print(f"🔧 CHAT_ID: {'OK' if CHAT_ID else 'FALTA'}")
    print(f"🔧 QWEN_API_KEY: {'OK' if QWEN_API_KEY else 'FALTA'}")

    app = Application.builder().token(TOKEN).post_init(post_init).build()

    # Añadir handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("entrenar", entrenar))
    app.add_handler(CallbackQueryHandler(button_handler))

    # Iniciar polling
    logger.info("🚀 Bot de Telegram iniciado y listo")
    app.run_polling()

if __name__ == "__main__":
    main()
