# robot_trading_completo.py
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
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# === CONFIGURACIÓN ===
ARCHIVO_MODELO = "modelo_ia_final.pkl"
ARCHIVO_SCALER = "scaler_final.pkl"
OPERACIONES_DIA = 2
CONFIANZA_MINIMA = 0.75
RISK_REWARD_RATIO = 2.0

# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === PARES FOREX ===
PARES_FOREX = {
    'EURUSD=X': {'nombre': 'EUR/USD', 'tipo': 'major'},
    'USDJPY=X': {'nombre': 'USD/JPY', 'tipo': 'major'},
    'GC=F': {'nombre': 'Oro (XAU/USD)', 'tipo': 'commodity'},
    'CL=F': {'nombre': 'Petroleo (WTI)', 'tipo': 'commodity'}
}

# === CLASE ANALIZADOR DE PRICE ACTION ===
class PriceActionAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def detectar_patrones_velas(self, data: pd.DataFrame) -> Dict[str, bool]:
        try:
            close = data['Close']
            open_ = data['Open']
            high = data['High']
            low = data['Low']
            body = abs(close - open_)
            lower_wick = low - open_.where(close > open_, close)
            upper_wick = high - close.where(close > open_, open_)

            cuerpo_pequeno = body < body.rolling(20).mean() * 0.5
            mecha_inferior_larga = lower_wick > body * 2
            mecha_superior_larga = upper_wick > body * 2

            return {
                'martillo': bool((cuerpo_pequeno & mecha_inferior_larga).iloc[-1]),
                'estrella_inversion': bool((cuerpo_pequeno & mecha_superior_larga).iloc[-1])
            }
        except Exception as e:
            self.logger.error(f"Error detectando patrones de velas: {e}")
            return {'martillo': False, 'estrella_inversion': False}

    def detectar_tendencia(self, data: pd.DataFrame) -> str:
        try:
            ema_20 = data['Close'].ewm(span=20).mean()
            ema_50 = data['Close'].ewm(span=50).mean()
            if ema_20.iloc[-1] > ema_50.iloc[-1]:
                return "alcista"
            elif ema_20.iloc[-1] < ema_50.iloc[-1]:
                return "bajista"
            else:
                return "lateral"
        except Exception as e:
            self.logger.error(f"Error detectando tendencia: {e}")
            return "desconocida"

# === CLASE ANALIZADOR CUÁNTICO/GEOMÉTRICO ===
class QuantumGeometricAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def _calcular_alma(self, prices: pd.Series, window: int = 9, sigma: float = 6.0, offset: float = 0.85):
        try:
            alma = np.zeros(len(prices))
            for i in range(window, len(prices)):
                weights = np.exp(-((np.arange(window) - offset * (window - 1)) ** 2) / (2 * sigma ** 2))
                weights /= weights.sum()
                alma[i] = np.dot(prices[i - window:i], weights)
            return pd.Series(alma, index=prices.index)
        except Exception as e:
            self.logger.error(f"Error calculando ALMA: {e}")
            return prices.copy()

    def _calcular_quantum_kernel(self, data: pd.DataFrame) -> pd.Series:
        try:
            kernel = data['Close'].rolling(7).mean() + data['Close'].pct_change().rolling(14).std() * 100
            return kernel
        except Exception as e:
            self.logger.error(f"Error calculando Kernel: {e}")
            return data['Close'].copy()

    def detectar_cruce_alma_kernel(self, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            alma = self._calcular_alma(data['Close'])
            kernel = self._calcular_quantum_kernel(data)
            cruce = (alma > kernel) & (alma.shift(1) < kernel.shift(1))
            distancia = (alma - kernel).iloc[-1]
            return {
                'cruce': bool(cruce.iloc[-1]),
                'distancia': float(distancia)
            }
        except Exception as e:
            self.logger.error(f"Error detectando cruce ALMA/Kernel: {e}")
            return {'cruce': False, 'distancia': 0.0}

    def detectar_bloques_geometricos(self, data: pd.DataFrame) -> Dict[str, bool]:
        try:
            close = data['Close']
            open_ = data['Open']
            high = data['High']
            low = data['Low']
            body = abs(close - open_)
            lower_wick = low - open_.where(close > open_, close)
            upper_wick = high - close.where(close > open_, open_)

            cuerpo_pequeno = body < body.rolling(20).mean() * 0.5
            mecha_inferior_larga = lower_wick > body * 2
            mecha_superior_larga = upper_wick > body * 2

            return {
                'consolidacion': bool((cuerpo_pequeno & ~mecha_inferior_larga & ~mecha_superior_larga).iloc[-1]),
                'momentum': bool((~cuerpo_pequeno & (mecha_inferior_larga | mecha_superior_larga)).iloc[-1]),
                'rango_relativo': float((high - low).iloc[-1] / body.iloc[-1]) if body.iloc[-1] > 0 else 1.0
            }
        except Exception as e:
            self.logger.error(f"Error detectando bloques geométricos: {e}")
            return {'consolidacion': False, 'momentum': False, 'rango_relativo': 1.0}

# === MÓDULO DATA PROVIDER ===
class DataProvider:
    def __init__(self):
        self.symbol_mapping = {
            'EURUSD=X': {'finnhub': 'EURUSD', 'twelvedata': 'EUR/USD', 'alpha_vantage': 'EURUSD', 'category': 'forex'},
            'USDJPY=X': {'finnhub': 'USDJPY', 'twelvedata': 'USD/JPY', 'alpha_vantage': 'USDJPY', 'category': 'forex'},
            'GC=F': {'finnhub': 'GC1', 'twelvedata': 'XAU/USD', 'alpha_vantage': 'GOLD', 'category': 'commodity'},
            'CL=F': {'finnhub': 'CL1', 'twelvedata': 'WTI/USD', 'alpha_vantage': 'WTI', 'category': 'commodity'}
        }
        self.cache = {}
        self.cache_expiry = 300

    def get_historical_data(self, symbol: str, start_date: str, end_date: str, interval: str = '1h') -> Optional[pd.DataFrame]:
        cache_key = f"{symbol}_{start_date}_{end_date}_{interval}"
        if cache_key in self.cache and time.time() < self.cache[cache_key]['expiry']:
            logger.info(f"Usando datos en caché para {symbol}")
            return self.cache[cache_key]['data']

        try:
            data = yf.download(symbol, start=start_date, end=end_date, interval=interval, auto_adjust=True)
            if data.empty:
                return None

            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            self.cache[cache_key] = {
                'data': data,
                'expiry': time.time() + self.cache_expiry
            }
            logger.info(f"Descargados {len(data)} registros de {symbol}")
            return data
        except Exception as e:
            logger.error(f"Error descargando {symbol}: {e}")
            return None

    def get_latest_data(self, symbol: str, interval: str, lookback: int) -> Optional[pd.DataFrame]:
        try:
            data = yf.download(symbol, period="60d", interval=interval, auto_adjust=True)
            return data.tail(lookback) if len(data) >= lookback else data
        except Exception as e:
            logger.error(f"Error obteniendo datos de {symbol}: {e}")
            return None

# === CLASE GESTIÓN DE OPERACIONES ===
class GestionOperaciones:
    def __init__(self, archivo_operaciones="operaciones.json"):
        self.archivo = archivo_operaciones
        self.operaciones_abiertas = []
        self.operaciones_cerradas = []
        self.cargar_operaciones()

    def cargar_operaciones(self):
        try:
            if os.path.exists(self.archivo):
                with open(self.archivo, 'r') as f:
                    datos = json.load(f)
                    self.operaciones_abiertas = datos.get('abiertas', [])
                    self.operaciones_cerradas = datos.get('cerradas', [])
        except Exception as e:
            logger.error(f"Error cargando operaciones: {e}")

    def guardar_operaciones(self):
        try:
            with open(self.archivo, 'w') as f:
                json.dump({
                    'abiertas': self.operaciones_abiertas,
                    'cerradas': self.operaciones_cerradas
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando operaciones: {e}")

# === CLASE ROBOT TRADING FINAL ===
class RobotTradingFinal:
    def __init__(self):
        self.modelos_a_probar = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42)
        }
        self.model = None
        self.scaler = StandardScaler()
        self.data_provider = DataProvider()
        self.price_action = PriceActionAnalyzer()
        self.quantum_geometric = QuantumGeometricAnalyzer()
        self.gestion_operaciones = GestionOperaciones()
        self.ultimo_reentrenamiento = None
        self.cargar_modelo()

    def cargar_modelo(self):
        try:
            if os.path.exists(ARCHIVO_MODELO) and os.path.exists(ARCHIVO_SCALER):
                self.model = joblib.load(ARCHIVO_MODELO)
                self.scaler = joblib.load(ARCHIVO_SCALER)
                logger.info("Modelo cargado correctamente")
                return True
            else:
                logger.warning("Modelo no encontrado. Se usará modo simulado.")
                return False
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            return False

    def entrenar_modelo(self):
        logger.info("Iniciando entrenamiento del modelo IA...")
        try:
            todos_los_datos = self.obtener_datos_multiples_pares()
            if not todos_los_datos:
                logger.error("No se pudieron obtener datos para entrenar")
                return False

            df = pd.DataFrame(todos_los_datos)
            if df.empty or 'resultado' not in df.columns:
                logger.error("Datos insuficientes para entrenar")
                return False

            # Limpiar datos
            X = df.drop(['resultado', 'par', 'cambio_real', 'umbral_usado'], axis=1, errors='ignore')
            y = df['resultado']
            valid_rows = X.apply(lambda row: not row.isna().any() and not np.isinf(row).any(), axis=1)
            X = X[valid_rows]
            y = y[valid_rows]

            if len(X) == 0:
                logger.error("No hay muestras válidas después de limpieza")
                return False

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            mejor_modelo = None
            mejor_precision = 0
            mejor_nombre = ""

            for nombre, modelo in self.modelos_a_probar.items():
                try:
                    scores = cross_val_score(modelo, X_train_scaled, y_train, cv=5, scoring='accuracy')
                    precision = scores.mean()
                    logger.info(f"{nombre} - Precisión CV: {precision:.3f} (±{scores.std():.3f})")
                    if precision > mejor_precision:
                        mejor_precision = precision
                        mejor_modelo = modelo
                        mejor_nombre = nombre
                except Exception as e:
                    logger.error(f"Error con {nombre}: {e}")
                    continue

            if mejor_modelo is None:
                logger.error("No se pudo entrenar ningún modelo.")
                return False

            mejor_modelo.fit(X_train_scaled, y_train)
            self.model = mejor_modelo

            # Guardar modelo y scaler
            joblib.dump(self.model, ARCHIVO_MODELO)
            joblib.dump(self.scaler, ARCHIVO_SCALER)
            self.ultimo_reentrenamiento = datetime.now()
            logger.info(f"✅ Modelo entrenado y guardado: {mejor_nombre} (Precisión: {mejor_precision:.3f})")
            return True
        except Exception as e:
            logger.error(f"Error entrenando modelo: {e}")
            return False

    def predecir_senal(self, data: pd.DataFrame, info: Dict):
        try:
            if self.model is None or self.scaler is None:
                logger.error("Modelo no cargado.")
                return None, "Modelo no disponible"

            data_formateada = self.asegurar_formato_datos(data)
            if data_formateada is None or data_formateada.empty:
                return None, "Datos inválidos"

            features = self.extraer_features(data_formateada, info['ticker'])
            X = pd.DataFrame([features])
            X_scaled = self.scaler.transform(X)
            proba = self.model.predict_proba(X_scaled)[0]
            prediccion = self.model.predict(X_scaled)[0]
            confianza = max(proba)

            if confianza < CONFIANZA_MINIMA:
                return None, "Confianza baja"

            senal = {
                'ticker': info['ticker'],
                'nombre': info['nombre'],
                'tipo': info['tipo'],
                'precio_actual': data_formateada['Close'].iloc[-1],
                'prediccion': prediccion,
                'confianza': confianza,
                'analisis_cualitativo': {
                    'price_action': {
                        'patron': self.price_action.detectar_patrones_velas(data_formateada),
                        'tendencia': self.price_action.detectar_tendencia(data_formateada)
                    },
                    'quantum_geometric': self.quantum_geometric.detectar_cruce_alma_kernel(data_formateada)
                }
            }
            return senal, "Señal generada"
        except Exception as e:
            logger.error(f"Error prediciendo señal: {e}")
            return None, str(e)

    def asegurar_formato_datos(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            if data is None or data.empty:
                return None
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in data.columns:
                    return None
            return data[required_cols].copy()
        except Exception as e:
            logger.error(f"Error asegurando formato de datos: {e}")
            return None

    def extraer_features(self, data: pd.DataFrame, ticker: str) -> Dict[str, float]:
        try:
            close = data['Close']
            high = data['High']
            low = data['Low']
            volume = data['Volume']
            precio_actual = float(close.iloc[-1])

            # Indicadores técnicos
            ema_20 = EMAIndicator(close, window=20).ema_indicator()
            ema_50 = EMAIndicator(close, window=50).ema_indicator()
            rsi = RSIIndicator(close).rsi().iloc[-1]
            macd = MACD(close).macd().iloc[-1]
            macd_signal = MACD(close).macd_signal().iloc[-1]
            bb = BollingerBands(close)
            bb_upper = bb.bollinger_hband().iloc[-1]
            bb_lower = bb.bollinger_lband().iloc[-1]
            try:
                atr = AverageTrueRange(high, low, close).average_true_range().iloc[-1]
            except:
                atr = 0.001  # Valor por defecto si falla

            # Análisis cuántico
            alma = self.quantum_geometric._calcular_alma(close).iloc[-1]
            kernel = self.quantum_geometric._calcular_quantum_kernel(data).iloc[-1]

            # Price Action
            patrones = self.price_action.detectar_patrones_velas(data)
            tendencia = self.price_action.detectar_tendencia(data)

            return {
                'precio_vs_ema20': (precio_actual - ema_20.iloc[-1]) / precio_actual if not pd.isna(ema_20.iloc[-1]) else 0,
                'precio_vs_ema50': (precio_actual - ema_50.iloc[-1]) / precio_actual if not pd.isna(ema_50.iloc[-1]) else 0,
                'ema20_vs_ema50': 1 if ema_20.iloc[-1] > ema_50.iloc[-1] else 0,
                'precio_vs_bb_upper': (precio_actual - bb_upper) / precio_actual if not pd.isna(bb_upper) else 0,
                'precio_vs_bb_lower': (precio_actual - bb_lower) / precio_actual if not pd.isna(bb_lower) else 0,
                'bb_width': (bb_upper - bb_lower) / precio_actual if not pd.isna(bb_upper) and not pd.isna(bb_lower) else 0,
                'rsi': rsi / 100.0 if not pd.isna(rsi) else 0.5,
                'macd_histogram': (macd - macd_signal) if not pd.isna(macd) and not pd.isna(macd_signal) else 0,
                'kernel_vs_precio': (kernel - precio_actual) / precio_actual if not pd.isna(kernel) else 0,
                'alma_kernel_distancia': alma - kernel if not pd.isna(alma) and not pd.isna(kernel) else 0,
                'bloque_consolidacion': 1 if patrones['martillo'] else 0,
                'bloque_momentum': 1 if patrones['estrella_inversion'] else 0,
                'tendencia_alcista': 1 if tendencia == 'alcista' else 0,
                'hora': datetime
