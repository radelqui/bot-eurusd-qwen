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
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
import warnings
import joblib  # Importación corregida
warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN ---
TOKEN = "7718630865:AAEMclwlqzuxb5uFPqX9dyJLo7ib19QnJt8"
CHAT_ID = "5358902915"

PARES_FOREX = {
    "EURUSD=X": {"nombre": "EUR/USD", "tipo": "major"},
    "USDJPY=X": {"nombre": "USD/JPY", "tipo": "major"},
    "GC=F": {"nombre": "Gold USD", "tipo": "commodity"},
    "CL=F": {"nombre": "WTI Crude Oil", "tipo": "commodity"}
}

ARCHIVO_MODELO = "modelo_ia_final.pkl"
ARCHIVO_SCALER = "scaler_final.pkl"
ARCHIVO_DATOS_ENTRENAMIENTO = "datos_entrenamiento_final.json"
ARCHIVO_OPERACIONES = "operaciones_hoy.json"
OPERACIONES_DIA = 2
CONFIANZA_MINIMA = 0.75
RISK_REWARD_RATIO = 2.0
CAPITAL_INICIAL = 10000
RIESGO_POR_OPERACION = 0.02
RIESGO_MAXIMO_DIARIO = 0.05

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("robot_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RobotTrading")

# === CLASE DE ANÁLISIS DE PRICE ACTION ===
class PriceActionAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger("PriceAction")
    
    def detectar_soportes_resistencias(self, data: pd.DataFrame, ventana: int = 20) -> Dict[str, float]:
        """Detecta soportes y resistencias basados en máximos y mínimos"""
        try:
            highs = data['High'].rolling(ventana).max()
            lows = data['Low'].rolling(ventana).min()
            
            resistencia = highs.iloc[-1]
            soporte = lows.iloc[-1]
            
            # Buscar confluencias con niveles anteriores
            resistencia_confluencia = self._buscar_confluencias(data, resistencia, 'resistencia')
            soporte_confluencia = self._buscar_confluencias(data, soporte, 'soporte')
            
            return {
                'resistencia': resistencia_confluencia,
                'soporte': soporte_confluencia,
                'zona_confluencia': abs(resistencia_confluencia - soporte_confluencia) / data['Close'].iloc[-1]
            }
        except Exception as e:
            self.logger.error(f"Error detectando soportes/resistencias: {e}")
            return {'resistencia': 0, 'soporte': 0, 'zona_confluencia': 0}
    
    def _buscar_confluencias(self, data: pd.DataFrame, nivel: float, tipo: str) -> float:
        """Busca confluencias de niveles con máximos/mínimos anteriores"""
        try:
            if tipo == 'resistencia':
                # Buscar máximos cercanos al nivel
                maximos_cercanos = data['High'][(data['High'] > nivel * 0.999) & (data['High'] < nivel * 1.001)]
                if len(maximos_cercanos) > 2:
                    return nivel
            else:
                # Buscar mínimos cercanos al nivel
                minimos_cercanos = data['Low'][(data['Low'] > nivel * 0.999) & (data['Low'] < nivel * 1.001)]
                if len(minimos_cercanos) > 2:
                    return nivel
            
            # Si no hay confluencia, ajustar al nivel más cercano con múltiples toques
            return self._ajustar_nivel_confluencia(data, nivel, tipo)
        except:
            return nivel
    
    def _ajustar_nivel_confluencia(self, data: pd.DataFrame, nivel: float, tipo: str) -> float:
        """Ajusta el nivel para encontrar mejor confluencia"""
        try:
            if tipo == 'resistencia':
                # Buscar nivel con más rechazos hacia abajo
                niveles = data['High'].value_counts().head(5)
                return niveles.index[0]
            else:
                # Buscar nivel con más rechazos hacia arriba
                niveles = data['Low'].value_counts().head(5)
                return niveles.index[0]
        except:
            return nivel
    
    def detectar_patrones_velas(self, data: pd.DataFrame) -> Dict[str, str]:
        """Detecta patrones de velas clave"""
        try:
            ultima_vela = data.iloc[-1]
            penultima_vela = data.iloc[-2] if len(data) > 1 else None
            
            patron = "neutral"
            fuerza = 0
            
            # Patrón Engulfing Alcista
            if (penultima_vela is not None and 
                ultima_vela['Close'] > penultima_vela['Open'] and 
                ultima_vela['Open'] < penultima_vela['Close'] and
                ultima_vela['Close'] - ultima_vela['Open'] > abs(penultima_vela['Close'] - penultima_vela['Open']) * 1.5):
                patron = "engulfing_alcista"
                fuerza = 0.8
            
            # Patrón Engulfing Bajista
            elif (penultima_vela is not None and 
                  ultima_vela['Close'] < penultima_vela['Open'] and 
                  ultima_vela['Open'] > penultima_vela['Close'] and
                  penultima_vela['Close'] - penultima_vela['Open'] > abs(ultima_vela['Close'] - ultima_vela['Open']) * 1.5):
                patron = "engulfing_bajista"
                fuerza = 0.8
            
            # Pin Bar Alcista (Martillo)
            elif (ultima_vela['Close'] > ultima_vela['Open'] and
                  (ultima_vela['High'] - ultima_vela['Low']) > 3 * abs(ultima_vela['Close'] - ultima_vela['Open']) and
                  min(ultima_vela['Close'] - ultima_vela['Low'], ultima_vela['Open'] - ultima_vela['Low']) < abs(ultima_vela['Close'] - ultima_vela['Open']) * 0.3):
                patron = "pin_bar_alcista"
                fuerza = 0.7
            
            # Pin Bar Bajista (Martillo invertido)
            elif (ultima_vela['Close'] < ultima_vela['Open'] and
                  (ultima_vela['High'] - ultima_vela['Low']) > 3 * abs(ultima_vela['Close'] - ultima_vela['Open']) and
                  min(ultima_vela['High'] - ultima_vela['Close'], ultima_vela['High'] - ultima_vela['Open']) < abs(ultima_vela['Close'] - ultima_vela['Open']) * 0.3):
                patron = "pin_bar_bajista"
                fuerza = 0.7
            
            # Inside Bar (consolidación)
            elif (penultima_vela is not None and
                  ultima_vela['High'] < penultima_vela['High'] and
                  ultima_vela['Low'] > penultima_vela['Low']):
                patron = "inside_bar"
                fuerza = 0.5
            
            return {
                'patron': patron,
                'fuerza': fuerza,
                'direccion': 'alcista' if 'alcista' in patron else 'bajista' if 'bajista' in patron else 'neutral'
            }
        except Exception as e:
            self.logger.error(f"Error detectando patrones de velas: {e}")
            return {'patron': 'neutral', 'fuerza': 0, 'direccion': 'neutral'}
    
    def detectar_lineas_tendencia(self, data: pd.DataFrame, periodo: int = 50) -> Dict[str, float]:
        """Detecta líneas de tendencia usando mínimos y máximos"""
        try:
            # Encontrar máximos y mínimos locales
            maximos = data['High'].rolling(periodo, center=True).max()
            minimos = data['Low'].rolling(periodo, center=True).min()
            
            # Calcular pendiente de tendencia alcista (usando mínimos)
            pendiente_alcista = np.polyfit(range(len(minimos.dropna())), minimos.dropna(), 1)[0]
            
            # Calcular pendiente de tendencia bajista (usando máximos)
            pendiente_bajista = np.polyfit(range(len(maximos.dropna())), maximos.dropna(), 1)[0]
            
            # Determinar tendencia dominante
            if abs(pendiente_alcista) > abs(pendiente_bajista):
                tendencia = 'alcista' if pendiente_alcista > 0 else 'bajista'
                fuerza_tendencia = abs(pendiente_alcista)
            else:
                tendencia = 'bajista' if pendiente_bajista < 0 else 'alcista'
                fuerza_tendencia = abs(pendiente_bajista)
            
            return {
                'tendencia': tendencia,
                'fuerza': fuerza_tendencia,
                'pendiente_alcista': pendiente_alcista,
                'pendiente_bajista': pendiente_bajista
            }
        except Exception as e:
            self.logger.error(f"Error detectando líneas de tendencia: {e}")
            return {'tendencia': 'neutral', 'fuerza': 0, 'pendiente_alcista': 0, 'pendiente_bajista': 0}

# === CLASE DE INDICADORES CUÁNTICOS/GEOMÉTRICOS ===
class QuantumGeometricAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger("QuantumGeometric")
    
    def calcular_alma(self, data: pd.Series, window: int = 20, sigma: float = 6.0, offset: float = 0.85) -> pd.Series:
        """Calcula el indicador Arnaud Legoux Moving Average (ALMA)"""
        try:
            # Calcular los pesos de ALMA
            m = int(offset * (window - 1))
            s = window / sigma
            
            pesos = []
            for i in range(window):
                peso = np.exp(-1 * (i - m) ** 2 / (2 * s ** 2))
                pesos.append(peso)
            
            # Normalizar pesos
            pesos = np.array(pesos)
            pesos = pesos / pesos.sum()
            
            # Aplicar pesos ponderados
            alma = data.rolling(window).apply(lambda x: np.sum(x * pesos), raw=True)
            
            return alma
        except Exception as e:
            self.logger.error(f"Error calculando ALMA: {e}")
            return pd.Series(index=data.index)
    
    def calcular_kernel_smooth(self, data: pd.Series, window: int = 20, kernel_type: str = 'gaussian') -> pd.Series:
        """Calcula suavizado con diferentes kernels"""
        try:
            if kernel_type == 'gaussian':
                # Kernel gaussiano
                kernel = np.exp(-0.5 * (np.linspace(-3, 3, window) ** 2))
            elif kernel_type == 'epanechnikov':
                # Kernel Epanechnikov
                x = np.linspace(-1, 1, window)
                kernel = 0.75 * (1 - x ** 2)
            elif kernel_type == 'triangular':
                # Kernel triangular
                kernel = np.bartlett(window)
            else:
                # Kernel uniforme
                kernel = np.ones(window) / window
            
            # Normalizar kernel
            kernel = kernel / kernel.sum()
            
            # Aplicar convolución
            kernel_smooth = data.rolling(window).apply(lambda x: np.sum(x * kernel), raw=True)
            
            return kernel_smooth
        except Exception as e:
            self.logger.error(f"Error calculando kernel smooth: {e}")
            return pd.Series(index=data.index)
    
    def detectar_bloques_geometricos(self, data: pd.DataFrame, ventana: int = 10) -> Dict[str, Any]:
        """Detecta bloques geométricos de precio"""
        try:
            close = data['Close']
            high = data['High']
            low = data['Low']
            
            # Calcular rangos de precios
            rangos = high - low
            rango_promedio = rangos.rolling(ventana).mean()
            
            # Detectar bloques de consolidación
            consolidacion = rango_promedio < rango_promedio.quantile(0.3)
            
            # Detectar bloques de tendencia
            tendencia_alcista = (close > close.shift(ventana)).rolling(ventana).mean() > 0.7
            tendencia_bajista = (close < close.shift(ventana)).rolling(ventana).mean() > 0.7
            
            # Calcular momentum geométrico
            momentum = close.pct_change(ventana) * 100
            
            return {
                'consolidacion': consolidacion.iloc[-1],
                'tendencia_alcista': tendencia_alcista.iloc[-1],
                'tendencia_bajista': tendencia_bajista.iloc[-1],
                'momentum': momentum.iloc[-1],
                'rango_relativo': rangos.iloc[-1] / rango_promedio.iloc[-1] if rango_promedio.iloc[-1] > 0 else 1
            }
        except Exception as e:
            self.logger.error(f"Error detectando bloques geométricos: {e}")
            return {'consolidacion': False, 'tendencia_alcista': False, 'tendencia_bajista': False, 'momentum': 0, 'rango_relativo': 1}
    
    def detectar_cruce_alma_kernel(self, data: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
        """Detecta cruces entre ALMA y Kernel Smooth"""
        try:
            close = data['Close']
            
            # Calcular ALMA y Kernel
            alma = self.calcular_alma(close, window)
            kernel = self.calcular_kernel_smooth(close, window)
            
            # Detectar cruces
            cruce_alcista = (alma > kernel) & (alma.shift(1) <= kernel.shift(1))
            cruce_bajista = (alma < kernel) & (alma.shift(1) >= kernel.shift(1))
            
            # Calcular distancia entre líneas
            distancia = abs(alma - kernel) / close * 100
            
            # Determinar fuerza de la señal
            if cruce_alcista.iloc[-1]:
                señal = 'compra'
                fuerza = distancia.iloc[-1]
            elif cruce_bajista.iloc[-1]:
                señal = 'venta'
                fuerza = distancia.iloc[-1]
            else:
                señal = 'neutral'
                fuerza = 0
            
            return {
                'señal': señal,
                'fuerza': fuerza,
                'alma': alma.iloc[-1],
                'kernel': kernel.iloc[-1],
                'distancia': distancia.iloc[-1]
            }
        except Exception as e:
            self.logger.error(f"Error detectando cruce ALMA/Kernel: {e}")
            return {'señal': 'neutral', 'fuerza': 0, 'alma': 0, 'kernel': 0, 'distancia': 0}

# === MÓDULO DATA PROVIDER (Simplificado para ahorrar espacio) ===
class DataProvider:
    def __init__(self):
        self.finnhub_key = "d2o3pohr01qsrqkqrlggd2o3pohr01qsrqkqrlh0"
        self.twelvedata_key = "e0a66d799e6643878f937b5ac519fc42"
        self.alpha_vantage_key = "7BX8AK8CNXXIGV7V"
        
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
            data = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
            if not data.empty:
                self.cache[cache_key] = {'data': data, 'expiry': time.time() + self.cache_expiry}
                return data
        except Exception as e:
            logger.error(f"Error obteniendo datos de yfinance: {e}")
        
        return None
    
    def get_latest_data(self, symbol: str, interval: str = '1h', lookback: int = 100) -> Optional[pd.DataFrame]:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        data = self.get_historical_data(symbol, start_date, end_date, interval)
        
        if data is None or data.empty:
            return None
            
        return data.tail(lookback)

# === CLASE GESTIÓN DE OPERACIONES (Simplificada) ===
class GestionOperaciones:
    def __init__(self, archivo_operaciones):
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
                }, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error guardando operaciones: {e}")
    
    def agregar_operacion(self, operacion):
        operacion['id'] = str(uuid.uuid4())
        operacion['timestamp_apertura'] = datetime.now().isoformat()
        operacion['estado'] = 'abierta'
        self.operaciones_abiertas.append(operacion)
        self.guardar_operaciones()
        return operacion['id']
    
    def cerrar_operacion(self, id_operacion, precio_cierre, motivo):
        for i, op in enumerate(self.operaciones_abiertas):
            if op['id'] == id_operacion:
                operacion = self.operaciones_abiertas.pop(i)
                operacion['timestamp_cierre'] = datetime.now().isoformat()
                operacion['precio_cierre'] = precio_cierre
                operacion['motivo_cierre'] = motivo
                operacion['estado'] = 'cerrada'
                
                if operacion['direccion'] == 'COMPRA':
                    operacion['resultado'] = precio_cierre - operacion['precio_entrada']
                else:
                    operacion['resultado'] = operacion['precio_entrada'] - precio_cierre
                
                operacion['resultado_pct'] = (operacion['resultado'] / operacion['precio_entrada']) * 100
                self.operaciones_cerradas.append(operacion)
                self.guardar_operaciones()
                return True
        return False
    
    def verificar_operaciones_abiertas(self, datos_mercado):
        for operacion in self.operaciones_abiertas:
            ticker = operacion['ticker']
            if ticker in datos_mercado:
                precio_actual = datos_mercado[ticker]['precio_actual']
                
                if operacion['direccion'] == 'COMPRA':
                    if precio_actual <= operacion['stop_loss']:
                        self.cerrar_operacion(operacion['id'], precio_actual, 'Stop Loss')
                        logger.info(f"Operación cerrada por SL: {operacion['par']}")
                    elif precio_actual >= operacion['take_profit']:
                        self.cerrar_operacion(operacion['id'], precio_actual, 'Take Profit')
                        logger.info(f"Operación cerrada por TP: {operacion['par']}")
                else:
                    if precio_actual >= operacion['stop_loss']:
                        self.cerrar_operacion(operacion['id'], precio_actual, 'Stop Loss')
                        logger.info(f"Operación cerrada por SL: {operacion['par']}")
                    elif precio_actual <= operacion['take_profit']:
                        self.cerrar_operacion(operacion['id'], precio_actual, 'Take Profit')
                        logger.info(f"Operación cerrada por TP: {operacion['par']}")
    
    def obtener_estadisticas(self):
        if not self.operaciones_cerradas:
            return None
        
        operaciones = self.operaciones_cerradas
        total_operaciones = len(operaciones)
        ganadoras = [op for op in operaciones if op['resultado'] > 0]
        perdedoras = [op for op in operaciones if op['resultado'] < 0]
        
        win_rate = len(ganadoras) / total_operaciones * 100 if total_operaciones > 0 else 0
        profit_factor = sum(op['resultado'] for op in ganadoras) / abs(sum(op['resultado'] for op in perdedoras)) if perdedoras else float('inf')
        
        max_drawdown = 0
        balance = 0
        max_balance = 0
        
        for op in sorted(operaciones, key=lambda x: x['timestamp_apertura']):
            balance += op['resultado']
            if balance > max_balance:
                max_balance = balance
            drawdown = max_balance - balance
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return {
            'total_operaciones': total_operaciones,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'net_profit': sum(op['resultado'] for op in operaciones)
        }

# === CLASE PRINCIPAL DEL ROBOT ===
class RobotTradingFinal:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.features_names = []
        self.min_datos_entrenamiento = 100
        self.modelos_a_probar = {
            'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
        }
        self.gestion_operaciones = GestionOperaciones(ARCHIVO_OPERACIONES)
        self.capital_inicial = CAPITAL_INICIAL
        self.data_provider = DataProvider()
        self.price_action = PriceActionAnalyzer()
        self.quantum_geometric = QuantumGeometricAnalyzer()
        self.logger = logger

    def asegurar_formato_datos(self, data):
        if data is None or data.empty:
            return None
        if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
            close = data['Close'].iloc[:, 0] if 'Close' in data.columns else data.iloc[:, 0]
            high = data['High'].iloc[:, 0] if 'High' in data.columns else data.iloc[:, 1]
            low = data['Low'].iloc[:, 0] if 'Low' in data.columns else data.iloc[:, 2]
            volume = data['Volume'].iloc[:, 0] if 'Volume' in data.columns else data.iloc[:, 3]
        else:
            close = data['Close']
            high = data['High']
            low = data['Low']
            volume = data['Volume'] if 'Volume' in data.columns else pd.Series([1]*len(data), index=data.index)
        return pd.DataFrame({'Close': close, 'High': high, 'Low': low, 'Volume': volume})

    def calcular_sr_multitimeframe(self, ticker):
        niveles = {}
        timeframes = {"1d": "1mo", "4h": "10d", "1h": "5d"}
        for tf, periodo in timeframes.items():
            try:
                interval_tf = tf if tf != "1d" else "1d"
                data_tf = self.data_provider.get_historical_data(ticker, 
                                                              (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'), 
                                                              datetime.now().strftime('%Y-%m-%d'), 
                                                              interval_tf)
                if data_tf is None or data_tf.empty:
                    niveles[tf] = {'pp': np.nan, 'r1': np.nan, 's1': np.nan, 'r2': np.nan, 's2': np.nan}
                    continue
                data_formateada = self.asegurar_formato_datos(data_tf)
                if data_formateada is None:
                    niveles[tf] = {'pp': np.nan, 'r1': np.nan, 's1': np.nan, 'r2': np.nan, 's2': np.nan}
                    continue
                close_tf = data_formateada['Close']
                high_tf = data_formateada['High']
                low_tf = data_formateada['Low']
                if len(close_tf) < 5:
                    niveles[tf] = {'pp': np.nan, 'r1': np.nan, 's1': np.nan, 'r2': np.nan, 's2': np.nan}
                    continue
                pp = (high_tf.iloc[-1] + low_tf.iloc[-1] + close_tf.iloc[-1]) / 3
                r1 = (2 * pp) - low_tf.iloc[-1]
                s1 = (2 * pp) - high_tf.iloc[-1]
                r2 = pp + (high_tf.iloc[-1] - low_tf.iloc[-1])
                s2 = pp - (high_tf.iloc[-1] - low_tf.iloc[-1])
                niveles[tf] = {'pp': pp, 'r1': r1, 's1': s1, 'r2': r2, 's2': s2}
            except Exception as e:
                niveles[tf] = {'pp': np.nan, 'r1': np.nan, 's1': np.nan, 'r2': np.nan, 's2': np.nan}
        return niveles

    def extraer_features_sr(self, precio_actual, sr_niveles):
        features = {}
        for tf, niveles in sr_niveles.items():
            if pd.isna(niveles['pp']) or precio_actual <= 0:
                 features[f'{tf}_distancia_pp'] = 0
                 features[f'{tf}_distancia_r1'] = 0
                 features[f'{tf}_distancia_s1'] = 0
                 continue
            features[f'{tf}_distancia_pp'] = (precio_actual - niveles['pp']) / precio_actual
            features[f'{tf}_distancia_r1'] = (precio_actual - niveles['r1']) / precio_actual
            features[f'{tf}_distancia_s1'] = (precio_actual - niveles['s1']) / precio_actual
        return features

    def obtener_datos_multiples_pares(self):
        self.logger.info("Obteniendo datos de mercado...")
        todos_los_datos = []
        for ticker, info in PARES_FOREX.items():
            nombre = info['nombre']
            try:
                self.logger.info(f"Procesando {nombre} ({ticker})...")
                data = self.data_provider.get_historical_data(ticker, 
                                                              (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'), 
                                                              datetime.now().strftime('%Y-%m-%d'), 
                                                              "1h")
                self.logger.info(f"Descargados {len(data)} registros de {ticker}")
                if data is None or data.empty:
                    self.logger.warning(f"Datos vacíos para {nombre}")
                    continue
                if len(data) < 100:
                    self.logger.warning(f"{nombre}: Datos insuficientes ({len(data)} velas)")
                    continue
                self.logger.info(f"Calculando S/R multi-timeframe para {nombre}...")
                sr_niveles = self.calcular_sr_multitimeframe(ticker)
                datos_procesados = self.procesar_datos_par(data, ticker, sr_niveles)
                self.logger.info(f"{nombre}: {len(datos_procesados)} muestras procesadas")
                todos_los_datos.extend(datos_procesados)
            except Exception as e:
                self.logger.error(f"Error procesando {nombre}: {e}")
                continue
        self.logger.info(f"Total de muestras recolectadas: {len(todos_los_datos)}")
        return todos_los_datos

    def procesar_datos_par(self, data, ticker, sr_niveles):
        muestras = []
        data_formateada = self.asegurar_formato_datos(data)
        if data_formateada is None:
            return muestras
        close = data_formateada['Close']
        high = data_formateada['High']
        low = data_formateada['Low']
        volume = data_formateada['Volume']
        try:
            rsi = RSIIndicator(close, window=14).rsi()
            macd_obj = MACD(close)
            macd_line = macd_obj.macd()
            macd_signal = macd_obj.macd_signal()
            bb = BollingerBands(close)
            bb_upper = bb.bollinger_hband()
            bb_lower = bb.bollinger_lband()
            ema_20 = EMAIndicator(close, window=20).ema_indicator()
            ema_50 = EMAIndicator(close, window=50).ema_indicator()
            atr = AverageTrueRange(high, low, close).average_true_range()
            adx = ADXIndicator(high, low, close).adx()
            stochastic = StochasticOscillator(high, low, close).stoch()
            vwap = VolumeWeightedAveragePrice(high, low, close, volume).volume_weighted_average_price()
            
            # Nuevos indicadores
            alma = self.quantum_geometric.calcular_alma(close)
            kernel_smooth = self.quantum_geometric.calcular_kernel_smooth(close)
        except Exception as e:
            self.logger.error(f"Error calculando indicadores: {e}")
            return muestras
        ventana = 60
        horizonte_prediccion = 6
        if len(data) <= ventana + horizonte_prediccion:
            return muestras
        for i in range(ventana, len(data) - horizonte_prediccion):
            try:
                precio_actual = close.iloc[i]
                if pd.isna(precio_actual) or precio_actual <= 0:
                    continue
                
                # Análisis de Price Action
                sr_levels = self.price_action.detectar_soportes_resistencias(data.iloc[i-20:i+1])
                patron_velas = self.price_action.detectar_patrones_velas(data.iloc[i-3:i+1])
                lineas_tendencia = self.price_action.detectar_lineas_tendencia(data.iloc[i-50:i+1])
                
                # Análisis Cuántico/Geométrico
                bloques_geometricos = self.quantum_geometric.detectar_bloques_geometricos(data.iloc[i-10:i+1])
                cruce_alma_kernel = self.quantum_geometric.detectar_cruce_alma_kernel(data.iloc[i-20:i+1])
                
                features = {
                    'precio_vs_ema20': (precio_actual - ema_20.iloc[i]) / precio_actual if not pd.isna(ema_20.iloc[i]) and precio_actual > 0 else 0,
                    'precio_vs_ema50': (precio_actual - ema_50.iloc[i]) / precio_actual if not pd.isna(ema_50.iloc[i]) and precio_actual > 0 else 0,
                    'ema20_vs_ema50': 1 if not pd.isna(ema_20.iloc[i]) and not pd.isna(ema_50.iloc[i]) and ema_20.iloc[i] > ema_50.iloc[i] else 0,
                    'precio_vs_bb_upper': (precio_actual - bb_upper.iloc[i]) / precio_actual if not pd.isna(bb_upper.iloc[i]) and precio_actual > 0 else 0,
                    'precio_vs_bb_lower': (precio_actual - bb_lower.iloc[i]) / precio_actual if not pd.isna(bb_lower.iloc[i]) and precio_actual > 0 else 0,
                    'bb_width': (bb_upper.iloc[i] - bb_lower.iloc[i]) / precio_actual if not pd.isna(bb_upper.iloc[i]) and not pd.isna(bb_lower.iloc[i]) and precio_actual > 0 else 0,
                    'rsi': rsi.iloc[i] / 100.0 if not pd.isna(rsi.iloc[i]) else 0.5,
                    'macd_histogram': (macd_line.iloc[i] - macd_signal.iloc[i]) if not pd.isna(macd_line.iloc[i]) and not pd.isna(macd_signal.iloc[i]) else 0,
                    'momentum_3': (precio_actual / close.iloc[i-3] - 1) if i >= 3 and not pd.isna(close.iloc[i-3]) and close.iloc[i-3] > 0 else 0,
                    'momentum_7': (precio_actual / close.iloc[i-7] - 1) if i >= 7 and not pd.isna(close.iloc[i-7]) and close.iloc[i-7] > 0 else 0,
                    'atr_normalizado': atr.iloc[i] / precio_actual if not pd.isna(atr.iloc[i]) and precio_actual > 0 else 0,
                    'vol_ratio': volume.iloc[i] / volume.iloc[i-20:i].mean() if i >= 20 and volume.iloc[i-20:i].mean() > 0 else 1,
                    'adx': adx.iloc[i] / 100.0 if not pd.isna(adx.iloc[i]) else 0.5,
                    'stochastic': stochastic.iloc[i] / 100.0 if not pd.isna(stochastic.iloc[i]) else 0.5,
                    'vwap_desviacion': (precio_actual - vwap.iloc[i]) / precio_actual if not pd.isna(vwap.iloc[i]) and precio_actual > 0 else 0,
                    
                    # Features de Price Action
                    'sr_confluencia': sr_levels['zona_confluencia'],
                    'patron_fuerza': patron_velas['fuerza'],
                    'tendencia_fuerza': lineas_tendencia['fuerza'],
                    'precio_vs_sr_resistencia': (precio_actual - sr_levels['resistencia']) / precio_actual if sr_levels['resistencia'] > 0 else 0,
                    'precio_vs_sr_soporte': (precio_actual - sr_levels['soporte']) / precio_actual if sr_levels['soporte'] > 0 else 0,
                    
                    # Features Cuánticas/Geométricas
                    'alma_vs_precio': (alma.iloc[i] - precio_actual) / precio_actual if not pd.isna(alma.iloc[i]) else 0,
                    'kernel_vs_precio': (kernel_smooth.iloc[i] - precio_actual) / precio_actual if not pd.isna(kernel_smooth.iloc[i]) else 0,
                    'alma_kernel_distancia': cruce_alma_kernel['distancia'],
                    'bloque_consolidacion': 1 if bloques_geometricos['consolidacion'] else 0,
                    'bloque_momentum': bloques_geometricos['momentum'],
                    'bloque_rango_relativo': bloques_geometricos['rango_relativo'],
                    
                    # Features temporales
                    'hora': datetime.fromtimestamp(data.index[i].timestamp()).hour / 24.0,
                    'dia_semana': datetime.fromtimestamp(data.index[i].timestamp()).weekday() / 7.0,
                    **self.extraer_features_sr(precio_actual, sr_niveles)
                }
                
                precio_futuro = close.iloc[i + horizonte_prediccion]
                if pd.isna(precio_futuro) or precio_futuro <= 0 or pd.isna(precio_actual):
                    continue
                cambio_porcentual = (precio_futuro - precio_actual) / precio_actual * 100
                atr_val = atr.iloc[i]
                umbral_dinamico = max(0.05, (atr_val / precio_actual * 100) * 0.8) if not pd.isna(atr_val) and precio_actual > 0 else 0.05
                if cambio_porcentual > umbral_dinamico:
                    resultado = 1
                elif cambio_porcentual < -umbral_dinamico:
                    resultado = -1
                else:
                    resultado = 0
                features['resultado'] = resultado
                features['par'] = ticker
                features['cambio_real'] = cambio_porcentual
                features['umbral_usado'] = umbral_dinamico
                is_valid = True
                for k, v in features.items():
                    if isinstance(v, (int, float)):
                        if pd.isna(v) or np.isinf(v) or abs(v) > 10000:
                            is_valid = False
                            break
                if is_valid:
                    muestras.append(features)
            except Exception as e:
                self.logger.error(f"Error procesando muestra: {e}")
                continue
        return muestras

    def entrenar_modelo(self):
        self.logger.info("Iniciando entrenamiento del modelo IA...")
        datos = self.obtener_datos_multiples_pares()
        if len(datos) < self.min_datos_entrenamiento:
            self.logger.error(f"Datos insuficientes para entrenar: {len(datos)} < {self.min_datos_entrenamiento}")
            return False
        df = pd.DataFrame(datos)
        self.logger.info(f"Dataset creado: {len(df)} muestras")
        if len(df) == 0:
             self.logger.error("El DataFrame de datos está vacío.")
             return False
        self.logger.info("Distribución de clases (resultado):")
        self.logger.info(df['resultado'].value_counts())
        clases_unicas = df['resultado'].unique()
        if len(clases_unicas) < 2:
            self.logger.error("No hay suficientes clases para entrenar.")
            return False
        self.features_names = [col for col in df.columns if col not in ['resultado', 'par', 'cambio_real', 'umbral_usado']]
        if not self.features_names:
            self.logger.error("No se encontraron columnas de features.")
            return False
        X = df[self.features_names].fillna(0)
        y = df['resultado']
        
        rf_selector = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_selector.fit(X, y)
        importances = rf_selector.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_features = 25  # Aumentado para incluir nuevas features
        selected_features = [self.features_names[i] for i in indices[:top_features]]
        
        self.logger.info(f"Features seleccionadas: {', '.join(selected_features)}")
        self.features_names = selected_features
        X = X[selected_features]
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.logger.info("Comparando modelos con validación cruzada...")
        mejor_modelo = None
        mejor_precision = 0
        mejor_nombre = ""
        for nombre, modelo in self.modelos_a_probar.items():
            try:
                scores = cross_val_score(modelo, X_train_scaled, y_train, cv=5, scoring='accuracy')
                precision = scores.mean()
                self.logger.info(f"{nombre} - Precisión CV: {precision:.3f} (±{scores.std():.3f})")
                if precision > mejor_precision:
                    mejor_precision = precision
                    mejor_modelo = modelo
                    mejor_nombre = nombre
            except Exception as e:
                self.logger.error(f"Error con {nombre}: {e}")
                continue
        if mejor_modelo is None:
            self.logger.error("No se pudo entrenar ningún modelo.")
            return False
        mejor_modelo.fit(X_train_scaled, y_train)
        y_pred = mejor_modelo.predict(X_test_scaled)
        precision_test = accuracy_score(y_test, y_pred)
        self.logger.info(f"Mejor modelo: {mejor_nombre}")
        self.logger.info(f"Precisión en test: {precision_test:.3f}")
        self.model = mejor_modelo
        joblib.dump(self.model, ARCHIVO_MODELO)
        joblib.dump(self.scaler, ARCHIVO_SCALER)
        with open(ARCHIVO_DATOS_ENTRENAMIENTO, 'w') as f:
            json.dump({
                'features': self.features_names,
                'modelo_seleccionado': mejor_nombre,
                'accuracy': precision_test,
                'timestamp': datetime.now().isoformat(),
                'feature_importances': {selected_features[i]: importances[indices[i]] for i in range(top_features)}
            }, f, indent=2, default=str)
        self.logger.info(f"Modelo ({mejor_nombre}) y scaler guardados.")
        return True

    def cargar_modelo(self):
        try:
            if os.path.exists(ARCHIVO_MODELO) and os.path.exists(ARCHIVO_SCALER):
                self.model = joblib.load(ARCHIVO_MODELO)
                self.scaler = joblib.load(ARCHIVO_SCALER)
                if os.path.exists(ARCHIVO_DATOS_ENTRENAMIENTO):
                    with open(ARCHIVO_DATOS_ENTRENAMIENTO, 'r') as f:
                        info = json.load(f)
                    self.features_names = info.get('features', [])
                    modelo_nombre = info.get('modelo_seleccionado', 'Desconocido')
                    self.logger.info(f"Modelo cargado ({modelo_nombre}, Precisión guardada: {info.get('accuracy', 0):.3f})")
                return True
            else:
                self.logger.warning("Modelo no encontrado. Entrenando...")
                return self.entrenar_modelo()
        except Exception as e:
            self.logger.error(f"Error cargando modelo: {e}")
            return False

    def predecir_senal(self, data, ticker_info):
        if self.model is None:
            self.logger.error("Modelo no cargado.")
            return None, "Modelo no disponible"
        ticker = ticker_info['ticker']
        try:
            # Análisis de Price Action
            sr_levels = self.price_action.detectar_soportes_resistencias(data)
            patron_velas = self.price_action.detectar_patrones_velas(data)
            lineas_tendencia = self.price_action.detectar_lineas_tendencia(data)
            
            # Análisis Cuántico/Geométrico
            bloques_geometricos = self.quantum_geometric.detectar_bloques_geometricos(data)
            cruce_alma_kernel = self.quantum_geometric.detectar_cruce_alma_kernel(data)
            
            sr_niveles = self.calcular_sr_multitimeframe(ticker)
            data_formateada = self.asegurar_formato_datos(data)
            if data_formateada is None:
                 return None, "Error al formatear datos para predicción"
            close = data_formateada['Close']
            high = data_formateada['High']
            low = data_formateada['Low']
            volume = data_formateada['Volume']
            if len(data) < 60:
                return None, "Datos insuficientes para predicción"
            i = len(data) - 1
            precio_actual = close.iloc[i]
            if pd.isna(precio_actual) or precio_actual <= 0:
                 return None, "Precio actual inválido"
            rsi_val = RSIIndicator(close, window=14).rsi().iloc[i]
            macd_obj = MACD(close)
            macd_line_val = macd_obj.macd().iloc[i]
            macd_signal_val = macd_obj.macd_signal().iloc[i]
            bb = BollingerBands(close)
            bb_upper_val = bb.bollinger_hband().iloc[i]
            bb_lower_val = bb.bollinger_lband().iloc[i]
            ema_20_val = EMAIndicator(close, window=20).ema_indicator().iloc[i]
            ema_50_val = EMAIndicator(close, window=50).ema_indicator().iloc[i]
            atr_val = AverageTrueRange(high, low, close).average_true_range().iloc[i]
            adx_val = ADXIndicator(high, low, close).adx().iloc[i]
            stochastic_val = StochasticOscillator(high, low, close).stoch().iloc[i]
            vwap_val = VolumeWeightedAveragePrice(high, low, close, volume).volume_weighted_average_price().iloc[i]
            
            # Nuevos indicadores
            alma_val = self.quantum_geometric.calcular_alma(close).iloc[i]
            kernel_val = self.quantum_geometric.calcular_kernel_smooth(close).iloc[i]
            
            features = {
                'precio_vs_ema20': (precio_actual - ema_20_val) / precio_actual if not pd.isna(ema_20_val) and precio_actual > 0 else 0,
                'precio_vs_ema50': (precio_actual - ema_50_val) / precio_actual if not pd.isna(ema_50_val) and precio_actual > 0 else 0,
                'ema20_vs_ema50': 1 if not pd.isna(ema_20_val) and not pd.isna(ema_50_val) and ema_20_val > ema_50_val else 0,
                'precio_vs_bb_upper': (precio_actual - bb_upper_val) / precio_actual if not pd.isna(bb_upper_val) and precio_actual > 0 else 0,
                'precio_vs_bb_lower': (precio_actual - bb_lower_val) / precio_actual if not pd.isna(bb_lower_val) and precio_actual > 0 else 0,
                'bb_width': (bb_upper_val - bb_lower_val) / precio_actual if not pd.isna(bb_upper_val) and not pd.isna(bb_lower_val) and precio_actual > 0 else 0,
                'rsi': rsi_val / 100.0 if not pd.isna(rsi_val) else 0.5,
                'macd_histogram': (macd_line_val - macd_signal_val) if not pd.isna(macd_line_val) and not pd.isna(macd_signal_val) else 0,
                'momentum_3': (precio_actual / close.iloc[i-3] - 1) if i >= 3 and not pd.isna(close.iloc[i-3]) and close.iloc[i-3] > 0 else 0,
                'momentum_7': (precio_actual / close.iloc[i-7] - 1) if i >= 7 and not pd.isna(close.iloc[i-7]) and close.iloc[i-7] > 0 else 0,
                'atr_normalizado': atr_val / precio_actual if not pd.isna(atr_val) and precio_actual > 0 else 0,
                'vol_ratio': volume.iloc[i] / volume.iloc[i-20:i].mean() if i >= 20 and volume.iloc[i-20:i].mean() > 0 else 1,
                'adx': adx_val / 100.0 if not pd.isna(adx_val) else 0.5,
                'stochastic': stochastic_val / 100.0 if not pd.isna(stochastic_val) else 0.5,
                'vwap_desviacion': (precio_actual - vwap_val) / precio_actual if not pd.isna(vwap_val) and precio_actual > 0 else 0,
                
                # Features de Price Action
                'sr_confluencia': sr_levels['zona_confluencia'],
                'patron_fuerza': patron_velas['fuerza'],
                'tendencia_fuerza': lineas_tendencia['fuerza'],
                'precio_vs_sr_resistencia': (precio_actual - sr_levels['resistencia']) / precio_actual if sr_levels['resistencia'] > 0 else 0,
                'precio_vs_sr_soporte': (precio_actual - sr_levels['soporte']) / precio_actual if sr_levels['soporte'] > 0 else 0,
                
                # Features Cuánticas/Geométricas
                'alma_vs_precio': (alma_val - precio_actual) / precio_actual if not pd.isna(alma_val) else 0,
                'kernel_vs_precio': (kernel_val - precio_actual) / precio_actual if not pd.isna(kernel_val) else 0,
                'alma_kernel_distancia': cruce_alma_kernel['distancia'],
                'bloque_consolidacion': 1 if bloques_geometricos['consolidacion'] else 0,
                'bloque_momentum': bloques_geometricos['momentum'],
                'bloque_rango_relativo': bloques_geometricos['rango_relativo'],
                
                # Features temporales
                'hora': datetime.fromtimestamp(data.index[i].timestamp()).hour / 24.0,
                'dia_semana': datetime.fromtimestamp(data.index[i].timestamp()).weekday() / 7.0,
                **self.extraer_features_sr(precio_actual, sr_niveles)
            }
            
            feature_vector = np.array([features.get(fn, 0) for fn in self.features_names]).reshape(1, -1)
            feature_vector_scaled = self.scaler.transform(feature_vector)
            prediccion = self.model.predict(feature_vector_scaled)[0]
            probabilidades = self.model.predict_proba(feature_vector_scaled)[0]
            confianza = np.max(probabilidades)
            
            # Añadir análisis cualitativo a la señal
            analisis_cualitativo = {
                'price_action': {
                    'patron': patron_velas['patron'],
                    'direccion_patron': patron_velas['direccion'],
                    'tendencia': lineas_tendencia['tendencia'],
                    'sr_confluencia': sr_levels['zona_confluencia'] < 0.02  # Zona de confluencia ajustada
                },
                'quantum_geometric': {
                    'señal_cruce': cruce_alma_kernel['señal'],
                    'fuerza_cruce': cruce_alma_kernel['fuerza'],
                    'bloque_consolidacion': bloques_geometricos['consolidacion'],
                    'momentum_bloque': bloques_geometricos['momentum']
                }
            }
            
            return {
                'prediccion': prediccion,
                'confianza': confianza,
                'probabilidades': dict(zip(self.model.classes_, probabilidades)),
                'precio_actual': precio_actual,
                'ticker': ticker,
                'nombre': ticker_info['nombre'],
                'tipo': ticker_info['tipo'],
                'analisis_cualitativo': analisis_cualitativo
            }, "Predicción completada"
        except Exception as e:
            self.logger.error(f"Error en predicción: {e}")
            return None, f"Error en predicción: {e}"

    def filtrar_señal_mercado(self, data, prediccion, ticker, analisis_cualitativo):
        """Filtrado mejorado usando Price Action y análisis cuántico"""
        try:
            data_formateada = self.asegurar_formato_datos(data)
            close = data_formateada['Close']
            volatilidad = close.pct_change().rolling(20).std().iloc[-1]
            
            # 1. Filtrar por volatilidad
            if volatilidad < 0.0005:
                return False, "Volatilidad demasiado baja"
            
            # 2. Filtrar por tendencia principal (usando Price Action)
            tendencia_principal = analisis_cualitativo['price_action']['tendencia']
            if prediccion == 1 and tendencia_principal == 'bajista':
                return False, "Señal de compra contra tendencia principal"
            if prediccion == -1 and tendencia_principal == 'alcista':
                return False, "Señal de venta contra tendencia principal"
            
            # 3. Filtrar por patrones de velas
            patron = analisis_cualitativo['price_action']['patron']
            if 'engulfing' in patron:
                # Validar que la dirección del patrón coincida con la señal
                if (prediccion == 1 and 'bajista' in patron) or (prediccion == -1 and 'alcista' in patron):
                    return False, "Señal contradice patrón de vela"
            
            # 4. Filtrar por zonas de confluencia
            if analisis_cualitativo['price_action']['sr_confluencia']:
                precio_actual = close.iloc[-1]
                sr_resistencia = analisis_cualitativo['price_action']['precio_vs_sr_resistencia']
                sr_soporte = analisis_cualitativo['price_action']['precio_vs_sr_soporte']
                
                # Evitar operar muy cerca de SR/Resistencia sin confirmación
                if prediccion == 1 and abs(sr_resistencia) < 0.001:
                    return False, "Demasiado cerca de resistencia sin confirmación"
                if prediccion == -1 and abs(sr_soporte) < 0.001:
                    return False, "Demasiado cerca de soporte sin confirmación"
            
            # 5. Filtrar por análisis cuántico
            señal_cruce = analisis_cualitativo['quantum_geometric']['señal_cruce']
            if señal_cruce != 'neutral':
                # Validar que la señal del cruce coincida con la predicción
                if (prediccion == 1 and señal_cruce == 'venta') or (prediccion == -1 and señal_cruce == 'compra'):
                    return False, "Señal contradice cruce ALMA/Kernel"
            
            # 6. Filtrar por consolidación
            if analisis_cualitativo['quantum_geometric']['bloque_consolidacion']:
                return False, "Mercado en consolidación, evitar operar"
            
            return True, "Señal válida"
        except Exception as e:
            self.logger.error(f"Error en filtrado de señal: {e}")
            return False, "Error en filtrado"

    def calcular_sl_tp(self, data, prediccion, precio_entrada):
        try:
            data_formateada = self.asegurar_formato_datos(data)
            high = data_formateada['High']
            low = data_formateada['Low']
            close = data_formateada['Close']
            
            # Calcular ATR múltiple
            atr_14 = AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
            atr_7 = AverageTrueRange(high, low, close, window=7).average_true_range().iloc[-1]
            atr = (atr_14 + atr_7) / 2
            
            # Ajustar SL/TP según volatilidad y estructura de precio
            volatilidad = close.pct_change().rolling(20).std().iloc[-1]
            
            # Usar SR cercanos si existen
            precio_actual = close.iloc[-1]
            sr_analysis = self.price_action.detectar_soportes_resistencias(data)
            
            if volatilidad > 0.002:  # Alta volatilidad
                sl_multiplier = 2.0
                tp_multiplier = 3.0
            elif volatilidad < 0.0008:  # Baja volatilidad
                sl_multiplier = 1.2
                tp_multiplier = 2.5
            else:  # Volatilidad normal
                sl_multiplier = 1.5
                tp_multiplier = 2.5
            
            if prediccion == 1:  # COMPRA
                stop_loss = precio_actual - (atr * sl_multiplier)
                take_profit = precio_actual + (atr * tp_multiplier)
                
                # Ajustar TP si hay resistencia cercana
                if sr_analysis['resistencia'] > precio_actual and sr_analysis['resistencia'] < take_profit:
                    take_profit = sr_analysis['resistencia']
            elif prediccion == -1:  # VENTA
                stop_loss = precio_actual + (atr * sl_multiplier)
                take_profit = precio_actual - (atr * tp_multiplier)
                
                # Ajustar TP si hay soporte cercano
                if sr_analysis['soporte'] < precio_actual and sr_analysis['soporte'] > take_profit:
                    take_profit = sr_analysis['soporte']
            else:
                return None
            
            return {
                'stop_loss': round(stop_loss, 5),
                'take_profit': round(take_profit, 5),
                'risk_reward': round(abs(take_profit - precio_actual) / abs(stop_loss - precio_actual), 2)
            }
        except Exception as e:
            self.logger.error(f"Error en SL/TP: {e}")
            return None

    def enviar_telegram(self, mensaje):
        try:
            url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
            response = requests.post(url, data={'chat_id': CHAT_ID, 'text': mensaje, 'parse_mode': 'HTML'})
            if response.status_code == 200:
                self.logger.info("Alerta enviada a Telegram")
                return True
            else:
                self.logger.error(f"Error enviando a Telegram: {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"Excepción al enviar Telegram: {e}")
            return False

    def crear_mensaje_alerta(self, senal, sl_tp, tamano_operacion):
        direccion_emoji = "🟢" if senal['prediccion'] == 1 else "🔴"
        direccion_texto = "COMPRA" if senal['prediccion'] == 1 else "VENTA"
        
        # Añadir información cualitativa al mensaje
        patron = senal['analisis_cualitativo']['price_action']['patron']
        tendencia = senal['analisis_cualitativo']['price_action']['tendencia']
        cruce = senal['analisis_cualitativo']['quantum_geometric']['señal_cruce']
        
        mensaje = f"""
{direccion_emoji} <b>SEÑAL IA DETECTADA</b>
{'='*30}
<b>📊 PAR:</b> {senal['nombre']} ({senal['tipo'].upper()})
<b>📈 DIRECCIÓN:</b> {direccion_texto}
<b>💰 PRECIO:</b> {senal['precio_actual']:.5f}
<b>🛑 STOP LOSS:</b> {sl_tp['stop_loss']:.5f}
<b>🎯 TAKE PROFIT:</b> {sl_tp['take_profit']:.5f}
<b>📊 TAMAÑO:</b> {tamano_operacion:.2f} unidades
<b>🤖 CONFIANZA:</b> {senal['confianza']*100:.1f}%
<b>⚖️ R/R:</b> 1:{sl_tp['risk_reward']:.1f}

<b>📈 ANÁLISIS TÉCNICO:</b>
• Patrón: {patron.replace('_', ' ').title()}
• Tendencia: {tendencia.upper()}
• Cruce ALMA/Kernel: {cruce.upper()}

<b>⏰</b> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} UTC
        """
        return mensaje

    def calcular_capital_actual(self):
        stats = self.gestion_operaciones.obtener_estadisticas()
        if stats:
            return self.capital_inicial + stats['net_profit']
        return self.capital_inicial

    def puede_operar(self):
        operaciones_hoy = len([op for op in self.gestion_operaciones.operaciones_abiertas 
                              if datetime.fromisoformat(op['timestamp_apertura']).date() == datetime.now().date()])
        
        if operaciones_hoy >= OPERACIONES_DIA:
            return False
        
        riesgo_acumulado = sum(op['riesgo'] for op in self.gestion_operaciones.operaciones_abiertas)
        riesgo_maximo_diario = self.capital_inicial * RIESGO_MAXIMO_DIARIO
        
        if riesgo_acumulado >= riesgo_maximo_diario:
            self.logger.info(f"Riesgo máximo diario alcanzado: {riesgo_acumulado:.2f}/{riesgo_maximo_diario:.2f}")
            return False
        
        return True

    def es_horario_optimo(self, ticker):
        hora_actual = datetime.now().hour
        
        horarios_optimos = {
            'major': {
                'inicio': 8,
                'fin': 16
            },
            'commodity': {
                'inicio': 13,
                'fin': 20
            }
        }
        
        tipo_activo = PARES_FOREX[ticker]['tipo']
        if tipo_activo in horarios_optimos:
            return horarios_optimos[tipo_activo]['inicio'] <= hora_actual < horarios_optimos[tipo_activo]['fin']
        
        return True

    def ejecutar_robot_continuo(self):
        self.logger.info("ROBOT IA FINAL INICIANDO...")
        self.logger.info("="*50)
        
        if not self.cargar_modelo():
            self.logger.error("Error crítico: No se puede cargar/entrenar el modelo.")
            return
        
        self.logger.info(f"Pares objetivo: {len(PARES_FOREX)}")
        self.logger.info(f"Operaciones/día: {OPERACIONES_DIA}")
        self.logger.info(f"Confianza mínima: {CONFIANZA_MINIMA*100}%")
        self.logger.info("="*50)
        
        ultimo_reentrenamiento = datetime.now()
        intervalo_reentrenamiento = 7
        
        ciclo = 0
        while True:
            try:
                ciclo += 1
                ahora = datetime.now()
                self.logger.info(f"\nCICLO {ciclo} - {ahora.strftime('%H:%M:%S')}")
                
                # Reentrenamiento periódico
                dias_desde_reentrenamiento = (ahora - ultimo_reentrenamiento).days
                if dias_desde_reentrenamiento >= intervalo_reentrenamiento:
                    self.logger.info(f"Reentrenando modelo (hace {dias_desde_reentrenamiento} días)...")
                    if self.entrenar_modelo():
                        ultimo_reentrenamiento = ahora
                    else:
                        self.logger.warning("Falló reentrenamiento, continuando con modelo actual")
                
                # Obtener datos del mercado
                datos_mercado = {}
                for ticker, info in PARES_FOREX.items():
                    try:
                        data = self.data_provider.get_latest_data(ticker, "1h", 100)
                        if data is not None and len(data) >= 60:
                            data_formateada = self.asegurar_formato_datos(data)
                            precio_actual = data_formateada['Close'].iloc[-1]
                            datos_mercado[ticker] = {
                                'data': data,
                                'precio_actual': precio_actual,
                                'info': info
                            }
                    except Exception as e:
                        self.logger.error(f"Error obteniendo datos de {info['nombre']}: {e}")
                
                # Verificar operaciones abiertas
                self.gestion_operaciones.verificar_operaciones_abiertas(datos_mercado)
                
                # Mostrar estadísticas cada 10 ciclos
                if ciclo % 10 == 0:
                    stats = self.gestion_operaciones.obtener_estadisticas()
                    if stats:
                        self.logger.info(f"Estadísticas: Win Rate: {stats['win_rate']:.1f}%, "
                                      f"Profit Factor: {stats['profit_factor']:.2f}, "
                                      f"Drawdown: ${stats['max_drawdown']:.2f}")
                
                # Verificar si se puede operar
                if not self.puede_operar():
                    self.logger.info("Límite de operaciones diarias alcanzado.")
                    self.logger.info("Durmiendo 1 hora...")
                    time.sleep(3600)
                    continue
                
                # Buscar señales
                self.logger.info("Buscando señales...")
                mejor_senal = None
                mejor_confianza = 0
                
                for ticker, datos in datos_mercado.items():
                    info = datos['info']
                    data = datos['data']
                    
                    self.logger.info(f"Analizando {info['nombre']}...")
                    senal, mensaje = self.predecir_senal(data, info)
                    
                    if senal and senal['confianza'] >= CONFIANZA_MINIMA and senal['prediccion'] != 0:
                        # Filtrar por horario óptimo
                        if not self.es_horario_optimo(ticker):
                            self.logger.info(f"Fuera de horario óptimo para {info['nombre']}")
                            continue
                        
                        # Filtrar por condiciones de mercado mejoradas
                        filtro_valido, motivo_filtro = self.filtrar_señal_mercado(
                            data, senal['prediccion'], ticker, senal['analisis_cualitativo']
                        )
                        
                        if filtro_valido:
                            self.logger.info(f"Señal válida: {info['nombre']} (Conf: {senal['confianza']:.3f})")
                            if senal['confianza'] > mejor_confianza:
                                mejor_confianza = senal['confianza']
                                mejor_senal = senal
                        else:
                            self.logger.info(f"Señal filtrada: {info['nombre']} - {motivo_filtro}")
                    elif senal:
                        self.logger.info(f"Señal descartada: {info['nombre']} (Conf: {senal['confianza']:.3f} < {CONFIANZA_MINIMA})")
                
                # Procesar la mejor señal
                if mejor_senal:
                    self.logger.info(f"\nMEJOR SEÑAL: {mejor_senal['nombre']} - Conf: {mejor_senal['confianza']:.3f}")
                    
                    # Calcular SL/TP mejorado
                    sl_tp = self.calcular_sl_tp(datos_mercado[mejor_senal['ticker']]['data'], 
                                              mejor_senal['prediccion'], 
                                              mejor_senal['precio_actual'])
                    
                    if sl_tp:
                        capital_actual = self.calcular_capital_actual()
                        riesgo_operacion = capital_actual * RIESGO_POR_OPERACION
                        riesgo_por_unidad = abs(mejor_senal['precio_actual'] - sl_tp['stop_loss'])
                        tamano_operacion = riesgo_operacion / riesgo_por_unidad if riesgo_por_unidad > 0 else 0
                        
                        operacion = {
                            'par': mejor_senal['nombre'],
                            'ticker': mejor_senal['ticker'],
                            'direccion': "COMPRA" if mejor_senal['prediccion'] == 1 else "VENTA",
                            'precio_entrada': mejor_senal['precio_actual'],
                            'stop_loss': sl_tp['stop_loss'],
                            'take_profit': sl_tp['take_profit'],
                            'confianza': mejor_senal['confianza'],
                            'tamano': tamano_operacion,
                            'riesgo': riesgo_operacion,
                            'risk_reward': sl_tp['risk_reward']
                        }
                        
                        id_operacion = self.gestion_operaciones.agregar_operacion(operacion)
                        
                        # Crear y enviar alerta mejorada
                        mensaje_alerta = self.crear_mensaje_alerta(mejor_senal, sl_tp, tamano_operacion)
                        if self.enviar_telegram(mensaje_alerta):
                            self.logger.info("Alerta enviada a Telegram")
                        else:
                            self.logger.warning("Error enviando alerta a Telegram")
                    else:
                        self.logger.error("Error calculando SL/TP.")
                else:
                    self.logger.info("No se encontraron señales válidas.")
                
                self.logger.info("Pausa de 5 minutos...")
                time.sleep(300)
                
            except KeyboardInterrupt:
                self.logger.info("\nRobot detenido por el usuario.")
                break
            except Exception as e:
                self.logger.error(f"Error crítico en el ciclo: {e}")
                self.logger.info("Pausa de 2 minutos por error...")
                time.sleep(120)

# === PUNTO DE ENTRADA ===
if __name__ == "__main__":
    print("🚀 Iniciando Robot de Trading Mejorado")
    robot = RobotTradingFinal()
    
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "entrenar":
            print("🛠️ Modo: Entrenamiento")
            robot.entrenar_modelo()
        elif sys.argv[1] == "backtest":
            print("🔍 Modo: Backtesting")
            print("Función de backtesting no implementada en esta versión")
        else:
            print("🔄 Modo: Ejecución Continua")
            robot.ejecutar_robot_continuo()
    else:
        print("🔄 Modo: Ejecución Continua")
        robot.ejecutar_robot_continuo()
    
    print("🏁 Finalizando Robot.")