#!/usr/bin/env python3
"""
🚀 PROMPT TRAINER PRO - Sistema Avanzado de Entrenamiento y Mejora de Prompts
Versión optimizada con configuración externa y estructura mejorada
"""

import sys
import json
import os
import time
import requests
from pathlib import Path
from typing import List, Dict, Any
import pickle
import numpy as np
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

# Qt6 imports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QTextEdit, QPushButton, QLabel, QProgressBar,
                            QMessageBox, QSplitter, QFrame, QScrollArea, QTextBrowser,
                            QLineEdit, QGroupBox, QGridLayout, QCheckBox, QComboBox,
                            QStackedWidget, QSizePolicy)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QMutex, QWaitCondition, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QTextCursor, QPalette, QColor, QIcon

# Importar configuración
try:
    from config import LM_STUDIO_CONFIG, APP_CONFIG, STYLE_CONFIG, get_api_url, get_chat_url, update_config
except ImportError:
    # Configuración por defecto si no existe config.py
    LM_STUDIO_CONFIG = {
        "host": "127.0.0.1",
        "port": 1234,
        "api_url": "http://127.0.0.1:1234/v1",
        "chat_endpoint": "http://127.0.0.1:1234/v1/chat/completions",
        "model": "llama-3.2-3b-instruct"
    }
    APP_CONFIG = {
        "window_title": "Prompt Trainer Pro - Sistema Avanzado",
        "window_width": 1400,
        "window_height": 900,
        "prompts_directory": "@/PROMPTS", 
        "max_threads": 6,
        "chunk_size": 1000,
        "timeout": 30
    }
    STYLE_CONFIG = {
        "main_background": "#0a0a0a",
        "panel_background": "#1a1a1a",
        "card_background": "#2a2a2a",
        "text_color": "#ffffff",
        "text_secondary": "#b0b0b0",
        "accent_color": "#6366f1",
        "accent_hover": "#4f46e5",
        "success_color": "#10b981",
        "warning_color": "#f59e0b",
        "error_color": "#ef4444",
        "border_color": "#404040"
    }
    
    def get_api_url():
        return f"http://{LM_STUDIO_CONFIG['host']}:{LM_STUDIO_CONFIG['port']}/v1"
    
    def get_chat_url():
        return f"http://{LM_STUDIO_CONFIG['host']}:{LM_STUDIO_CONFIG['port']}/v1/chat/completions"
    
    def update_config(host=None, port=None):
        if host:
            LM_STUDIO_CONFIG["host"] = host
        if port:
            LM_STUDIO_CONFIG["port"] = port
        LM_STUDIO_CONFIG["api_url"] = get_api_url()
        LM_STUDIO_CONFIG["chat_endpoint"] = get_chat_url()

class AdvancedPromptTrainer:
    def __init__(self):
        self.api_url = get_api_url()
        self.chat_url = get_chat_url()
        self.model = LM_STUDIO_CONFIG["model"]
        self.training_data = []
        self.prompt_patterns = {}
        self.category_insights = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=APP_CONFIG["max_threads"])
        self.cache = {}
        self.prompts_directory = APP_CONFIG["prompts_directory"]
        
    def extract_all_prompts_parallel(self) -> List[Dict[str, Any]]:
        """Extrae prompts usando múltiples threads desde la nueva ruta."""
        print("📂 Extrayendo prompts con múltiples threads...")
        
        # Buscar la carpeta de prompts
        prompts_path = Path(self.prompts_directory)
        if not prompts_path.exists():
            # Intentar con rutas alternativas
            possible_paths = [
                Path("PROMPTS"),
                Path("../PROMPTS"),
                Path("../../PROMPTS"),
                Path("@/PROMPTS")
            ]
            for path in possible_paths:
                if path.exists():
                    prompts_path = path
                    break
        
        if not prompts_path.exists():
            print(f"❌ No se encontró la carpeta de prompts en: {self.prompts_directory}")
            return []
        
        print(f"📁 Carpeta de prompts encontrada: {prompts_path}")
        
        # Obtener todas las subcarpetas
        prompt_dirs = [d.name for d in prompts_path.iterdir() if d.is_dir()]
        print(f"📂 Carpetas encontradas: {prompt_dirs}")
        
        all_prompts = []
        
        def process_directory(dir_name):
            """Procesa una carpeta en un thread separado."""
            dir_prompts = []
            dir_path = prompts_path / dir_name
            
            if dir_path.exists():
                print(f"📁 Procesando: {dir_name}")
                
                # Procesar archivos .txt
                for file_path in dir_path.glob("*.txt"):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if content and len(content) > 50:
                                prompt_data = {
                                    "source": str(file_path),
                                    "content": content,
                                    "length": len(content),
                                    "category": dir_name.lower().replace(" ", "_"),
                                    "filename": file_path.name
                                }
                                dir_prompts.append(prompt_data)
                    except Exception as e:
                        print(f"Error leyendo {file_path}: {e}")
                
                # Procesar archivos .json
                for file_path in dir_path.glob("*.json"):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, dict):
                                for key, value in data.items():
                                    if isinstance(value, str) and len(value) > 50:
                                        prompt_data = {
                                            "source": f"{file_path}#{key}",
                                            "content": value,
                                            "length": len(value),
                                            "category": dir_name.lower().replace(" ", "_"),
                                            "filename": f"{file_path.name}#{key}"
                                        }
                                        dir_prompts.append(prompt_data)
                    except Exception as e:
                        print(f"Error leyendo JSON {file_path}: {e}")
            
            return dir_prompts
        
        # Ejecutar procesamiento en paralelo
        futures = [self.thread_pool.submit(process_directory, dir_name) for dir_name in prompt_dirs]
        
        for future in as_completed(futures):
            try:
                dir_prompts = future.result()
                all_prompts.extend(dir_prompts)
            except Exception as e:
                print(f"Error en procesamiento paralelo: {e}")
        
        print(f"✅ Extraídos {len(all_prompts)} prompts de {len(prompt_dirs)} carpetas")
        return all_prompts
    
    def analyze_prompt_patterns_parallel(self, prompts: List[Dict[str, Any]]):
        """Analiza patrones usando múltiples threads."""
        print("🔍 Analizando patrones con múltiples threads...")
        
        # Agrupar por categoría
        categories = defaultdict(list)
        for prompt in prompts:
            categories[prompt['category']].append(prompt)
        
        def analyze_category(category, category_prompts):
            """Analiza una categoría en un thread separado."""
            print(f"📊 Analizando categoría: {category}")
            
            common_elements = self.extract_common_elements(category_prompts)
            structure_patterns = self.analyze_structure(category_prompts)
            tone_analysis = self.analyze_tone(category_prompts)
            
            return category, {
                "common_elements": common_elements,
                "structure_patterns": structure_patterns,
                "tone_analysis": tone_analysis,
                "sample_count": len(category_prompts),
                "avg_length": np.mean([p['length'] for p in category_prompts])
            }
        
        # Ejecutar análisis en paralelo
        futures = [self.thread_pool.submit(analyze_category, cat, prompts_list) 
                  for cat, prompts_list in categories.items()]
        
        for future in as_completed(futures):
            try:
                category, insights = future.result()
                self.category_insights[category] = insights
            except Exception as e:
                print(f"Error analizando categoría: {e}")
    
    def extract_common_elements(self, prompts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extrae elementos comunes de los prompts (optimizado)."""
        elements = {
            "keywords": [],
            "phrases": [],
            "structure_markers": []
        }
        
        # Optimización: procesar en chunks
        chunk_size = APP_CONFIG["chunk_size"]
        all_text = ""
        
        for i in range(0, len(prompts), chunk_size):
            chunk = prompts[i:i+chunk_size]
            chunk_text = " ".join([p['content'].lower() for p in chunk])
            all_text += chunk_text + " "
        
        words = all_text.split()
        
        # Filtrar palabras comunes
        common_words = {'you', 'are', 'the', 'and', 'to', 'in', 'of', 'a', 'is', 'that', 'it', 'with', 'as', 'for', 'on', 'be', 'at', 'this', 'by', 'i', 'have', 'or', 'an', 'they', 'which', 'one', 'had', 'word', 'but', 'not', 'what', 'all', 'were', 'we', 'when', 'your', 'can', 'said', 'there', 'use', 'each', 'she', 'do', 'how', 'their', 'if', 'will', 'up', 'other', 'about', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more', 'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call', 'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part'}
        
        word_freq = defaultdict(int)
        for word in words:
            if len(word) > 3 and word not in common_words:
                word_freq[word] += 1
        
        elements["keywords"] = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Buscar frases comunes
        phrases = []
        for prompt in prompts[:50]:  # Limitar para rendimiento
            content = prompt['content'].lower()
            import re
            quoted = re.findall(r'"([^"]+)"', content)
            phrases.extend(quoted)
        
        phrase_freq = defaultdict(int)
        for phrase in phrases:
            if len(phrase) > 10:
                phrase_freq[phrase] += 1
        
        elements["phrases"] = sorted(phrase_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return elements
    
    def analyze_structure(self, prompts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analiza la estructura de los prompts."""
        structure = {
            "has_sections": 0,
            "has_examples": 0,
            "has_rules": 0,
            "has_formatting": 0,
            "avg_sections": 0
        }
        
        section_count = 0
        for prompt in prompts:
            content = prompt['content']
            
            # Contar secciones
            sections = content.count('\n\n') + content.count('\n#') + content.count('\n##')
            section_count += sections
            
            # Detectar elementos estructurales
            if '#' in content or '##' in content:
                structure["has_sections"] += 1
            if 'example' in content.lower() or 'ejemplo' in content.lower():
                structure["has_examples"] += 1
            if 'rule' in content.lower() or 'regla' in content.lower():
                structure["has_rules"] += 1
            if '```' in content or '`' in content:
                structure["has_formatting"] += 1
        
        if prompts:
            structure["avg_sections"] = section_count / len(prompts)
        
        return structure
    
    def analyze_tone(self, prompts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analiza el tono de los prompts."""
        tone = {
            "formal": 0,
            "casual": 0,
            "technical": 0,
            "instructional": 0
        }
        
        formal_words = {'shall', 'must', 'should', 'will', 'shall', 'hereby', 'therefore', 'thus', 'hence'}
        casual_words = {'hey', 'hi', 'hello', 'cool', 'awesome', 'great', 'nice', 'thanks', 'thank you'}
        technical_words = {'function', 'class', 'method', 'api', 'endpoint', 'parameter', 'variable', 'algorithm'}
        instructional_words = {'step', 'instruction', 'guide', 'tutorial', 'how to', 'follow', 'process'}
        
        for prompt in prompts:
            content = prompt['content'].lower()
            words = set(content.split())
            
            if formal_words.intersection(words):
                tone["formal"] += 1
            if casual_words.intersection(words):
                tone["casual"] += 1
            if technical_words.intersection(words):
                tone["technical"] += 1
            if instructional_words.intersection(words):
                tone["instructional"] += 1
        
        return tone
    
    def train_model(self, prompts: List[Dict[str, Any]] = None):
        """Entrena el modelo con los prompts extraídos."""
        if prompts is None:
            prompts = self.extract_all_prompts_parallel()
        
        if not prompts:
            raise Exception("No se encontraron prompts para entrenar")
        
        print(f"🧠 Entrenando modelo con {len(prompts)} prompts...")
        
        # Analizar patrones
        self.analyze_prompt_patterns_parallel(prompts)
        
        # Crear datos de entrenamiento
        self.training_data = prompts
        
        # Guardar modelo entrenado
        model_data = {
            "prompts": prompts,
            "patterns": self.prompt_patterns,
            "insights": self.category_insights,
            "timestamp": time.time(),
            "total_prompts": len(prompts),
            "categories": list(self.category_insights.keys())
        }
        
        with open("trained_prompt_model.pkl", "wb") as f:
            pickle.dump(model_data, f)
        
        print("✅ Modelo entrenado y guardado")
        return model_data
    
    def create_training_prompt(self) -> str:
        """Crea un prompt de entrenamiento basado en los insights."""
        if not self.category_insights:
            return "Eres un experto en mejorar prompts de IA."
        
        prompt = "Eres un experto en mejorar prompts de IA. Basándote en el análisis de miles de prompts profesionales, debes:\n\n"
        
        # Agregar insights de cada categoría
        for category, insights in self.category_insights.items():
            prompt += f"📊 {category.upper()}:\n"
            prompt += f"- Prompts analizados: {insights['sample_count']}\n"
            prompt += f"- Longitud promedio: {insights['avg_length']:.0f} caracteres\n"
            
            if insights['common_elements']['keywords']:
                top_keywords = [kw[0] for kw in insights['common_elements']['keywords'][:5]]
                prompt += f"- Palabras clave: {', '.join(top_keywords)}\n"
            
            prompt += "\n"
        
        prompt += "Mejora el siguiente prompt aplicando estos patrones y mejores prácticas:\n\n"
        return prompt
    
    def improve_prompt_streaming(self, original_prompt: str, mode: str = "text", progress_callback=None):
        """Mejora un prompt usando streaming."""
        
        # Prompt base para mejorar (NO responder)
        base_prompt = """Eres un experto en mejorar prompts de IA. Tu tarea es MEJORAR el prompt del usuario, NO responder a él.

INSTRUCCIONES:
- Analiza el prompt del usuario
- Identifica áreas de mejora
- Crea una versión mejorada más clara, específica y efectiva
- Mantén la intención original pero optimiza la estructura y claridad
- NO respondas al prompt, solo mejóralo

MODO: {mode}

PROMPT A MEJORAR:
{original_prompt}

MEJORA EL PROMPT ANTERIOR:"""

        # Personalizar según el modo
        if mode == "code":
            base_prompt = base_prompt.replace("{mode}", "CÓDIGO - El prompt mejorado debe generar código cuando sea apropiado")
        else:
            base_prompt = base_prompt.replace("{mode}", "TEXTO - El prompt mejorado debe generar solo texto, sin código")
        
        full_prompt = base_prompt.replace("{original_prompt}", original_prompt)
        
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": full_prompt}
            ],
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(
                self.chat_url,
                headers=headers,
                json=data,
                stream=True,
                timeout=APP_CONFIG["timeout"]
            )
            
            if response.status_code != 200:
                raise Exception(f"Error en la API: {response.status_code}")
            
            full_response = ""
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str == '[DONE]':
                            break
                        
                        try:
                            chunk = json.loads(data_str)
                            if 'choices' in chunk and chunk['choices']:
                                delta = chunk['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    content = delta['content']
                                    full_response += content
                                    if progress_callback:
                                        progress_callback(full_response)
                        except json.JSONDecodeError:
                            continue
            
            return full_response
            
        except Exception as e:
            raise Exception(f"Error en streaming: {str(e)}")

class TrainingThread(QThread):
    """Thread para entrenamiento en background."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, trainer: AdvancedPromptTrainer):
        super().__init__()
        self.trainer = trainer
    
    def run(self):
        try:
            self.progress.emit("🔄 Extrayendo prompts...")
            model_data = self.trainer.train_model()
            self.finished.emit(model_data)
        except Exception as e:
            self.error.emit(str(e))

class StreamingThread(QThread):
    """Thread para streaming de respuestas."""
    content_update = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, trainer: AdvancedPromptTrainer, original_prompt: str, mode: str):
        super().__init__()
        self.trainer = trainer
        self.original_prompt = original_prompt
        self.mode = mode
    
    def run(self):
        try:
            def progress_callback(content):
                self.content_update.emit(content)
            
            result = self.trainer.improve_prompt_streaming(
                self.original_prompt, 
                self.mode,
                progress_callback
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

class ConfigSidebar(QWidget):
    """Sidebar de configuración."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()
        
    def init_ui(self):
        """Inicializa la interfaz del sidebar."""
        self.setFixedWidth(320)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {STYLE_CONFIG['card_background']};
                color: {STYLE_CONFIG['text_color']};
                border-left: 2px solid {STYLE_CONFIG['accent_color']};
            }}
            QGroupBox {{
                background-color: {STYLE_CONFIG['panel_background']};
                border: 1px solid {STYLE_CONFIG['border_color']};
                border-radius: 8px;
                padding: 16px;
                margin-top: 8px;
                font-weight: 600;
            }}
            QGroupBox::title {{
                color: {STYLE_CONFIG['text_color']};
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px 0 8px;
            }}
            QPushButton {{
                background-color: {STYLE_CONFIG['accent_color']};
                color: white;
                border: none;
                padding: 10px 16px;
                font-size: 13px;
                font-weight: 600;
                border-radius: 6px;
                min-width: 100px;
            }}
            QPushButton:hover {{
                background-color: {STYLE_CONFIG['accent_hover']};
            }}
            QLineEdit {{
                background-color: {STYLE_CONFIG['main_background']};
                color: {STYLE_CONFIG['text_color']};
                border: 1px solid {STYLE_CONFIG['border_color']};
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 13px;
            }}
            QLineEdit:focus {{
                border: 2px solid {STYLE_CONFIG['accent_color']};
            }}
            QComboBox {{
                background-color: {STYLE_CONFIG['main_background']};
                color: {STYLE_CONFIG['text_color']};
                border: 1px solid {STYLE_CONFIG['border_color']};
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 13px;
                min-width: 120px;
            }}
            QComboBox:focus {{
                border: 2px solid {STYLE_CONFIG['accent_color']};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid {STYLE_CONFIG['text_color']};
            }}
            QComboBox QAbstractItemView {{
                background-color: {STYLE_CONFIG['main_background']};
                color: {STYLE_CONFIG['text_color']};
                border: 1px solid {STYLE_CONFIG['border_color']};
                border-radius: 6px;
                selection-background-color: {STYLE_CONFIG['accent_color']};
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Título del sidebar
        title = QLabel("⚙️ Configuración")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {STYLE_CONFIG['text_color']}; margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid {STYLE_CONFIG['border_color']};")
        layout.addWidget(title)
        
        # Configuración del servidor
        server_group = QGroupBox("🌐 Servidor LM Studio")
        server_layout = QVBoxLayout(server_group)
        server_layout.setSpacing(12)
        
        # Host
        host_layout = QVBoxLayout()
        host_label = QLabel("Host")
        host_label.setStyleSheet(f"color: {STYLE_CONFIG['text_secondary']}; font-size: 12px; font-weight: 600;")
        self.host_input = QLineEdit(LM_STUDIO_CONFIG["host"])
        self.host_input.setPlaceholderText("127.0.0.1")
        host_layout.addWidget(host_label)
        host_layout.addWidget(self.host_input)
        server_layout.addLayout(host_layout)
        
        # Puerto
        port_layout = QVBoxLayout()
        port_label = QLabel("Puerto")
        port_label.setStyleSheet(f"color: {STYLE_CONFIG['text_secondary']}; font-size: 12px; font-weight: 600;")
        self.port_input = QLineEdit(str(LM_STUDIO_CONFIG["port"]))
        self.port_input.setPlaceholderText("1234")
        port_layout.addWidget(port_label)
        port_layout.addWidget(self.port_input)
        server_layout.addLayout(port_layout)
        
        # Botón actualizar servidor
        self.update_server_btn = QPushButton("🔄 Actualizar Servidor")
        self.update_server_btn.clicked.connect(self.update_server_config)
        server_layout.addWidget(self.update_server_btn)
        
        layout.addWidget(server_group)
        
        # Configuración de la aplicación
        app_group = QGroupBox("📱 Aplicación")
        app_layout = QVBoxLayout(app_group)
        app_layout.setSpacing(12)
        
        # Modo
        mode_layout = QVBoxLayout()
        mode_label = QLabel("Modo por defecto")
        mode_label.setStyleSheet(f"color: {STYLE_CONFIG['text_secondary']}; font-size: 12px; font-weight: 600;")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Texto", "Código"])
        self.mode_combo.setCurrentText("Texto")
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        app_layout.addLayout(mode_layout)
        
        # Timeout
        timeout_layout = QVBoxLayout()
        timeout_label = QLabel("Timeout (segundos)")
        timeout_label.setStyleSheet(f"color: {STYLE_CONFIG['text_secondary']}; font-size: 12px; font-weight: 600;")
        self.timeout_input = QLineEdit(str(APP_CONFIG["timeout"]))
        self.timeout_input.setPlaceholderText("30")
        timeout_layout.addWidget(timeout_label)
        timeout_layout.addWidget(self.timeout_input)
        app_layout.addLayout(timeout_layout)
        
        # Botón actualizar aplicación
        self.update_app_btn = QPushButton("🔄 Actualizar Aplicación")
        self.update_app_btn.clicked.connect(self.update_app_config)
        app_layout.addWidget(self.update_app_btn)
        
        layout.addWidget(app_group)
        
        # Información del sistema
        info_group = QGroupBox("ℹ️ Información")
        info_layout = QVBoxLayout(info_group)
        info_layout.setSpacing(8)
        
        # Estado del servidor
        self.server_status = QLabel("🔴 Servidor desconectado")
        self.server_status.setStyleSheet(f"color: {STYLE_CONFIG['warning_color']}; font-size: 12px;")
        info_layout.addWidget(self.server_status)
        
        # Estado del modelo
        self.model_status = QLabel("⚠️ Modelo no entrenado")
        self.model_status.setStyleSheet(f"color: {STYLE_CONFIG['warning_color']}; font-size: 12px;")
        info_layout.addWidget(self.model_status)
        
        layout.addWidget(info_group)
        
        # Botón cerrar
        self.close_btn = QPushButton("✕ Cerrar")
        self.close_btn.clicked.connect(self.close_sidebar)
        self.close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {STYLE_CONFIG['error_color']};
                color: white;
                border: none;
                padding: 12px;
                font-size: 14px;
                font-weight: 600;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: #dc2626;
            }}
        """)
        layout.addWidget(self.close_btn)
        
        layout.addStretch()
        
        # Actualizar información inicial
        self.update_info()
    
    def update_info(self):
        """Actualiza la información del sistema."""
        # Verificar modelo entrenado
        if os.path.exists("trained_prompt_model.pkl"):
            self.model_status.setText("✅ Modelo entrenado")
            self.model_status.setStyleSheet(f"color: {STYLE_CONFIG['success_color']}; font-size: 12px;")
        else:
            self.model_status.setText("⚠️ Modelo no entrenado")
            self.model_status.setStyleSheet(f"color: {STYLE_CONFIG['warning_color']}; font-size: 12px;")
        
        # Verificar servidor (simulación)
        self.server_status.setText("🟡 Verificando servidor...")
        self.server_status.setStyleSheet(f"color: {STYLE_CONFIG['warning_color']}; font-size: 12px;")
    
    def update_server_config(self):
        """Actualiza la configuración del servidor."""
        try:
            host = self.host_input.text()
            port = int(self.port_input.text())
            
            update_config(host, port)
            if self.parent:
                self.parent.trainer.api_url = get_api_url()
                self.parent.trainer.chat_url = get_chat_url()
                self.parent.update_status("✅ Configuración del servidor actualizada", "success")
                
                # Actualizar estado del servidor
                self.server_status.setText("✅ Servidor configurado")
                self.server_status.setStyleSheet(f"color: {STYLE_CONFIG['success_color']}; font-size: 12px;")
            
        except ValueError:
            QMessageBox.warning(self, "Error", "El puerto debe ser un número válido")
    
    def update_app_config(self):
        """Actualiza la configuración de la aplicación."""
        try:
            timeout = int(self.timeout_input.text())
            APP_CONFIG["timeout"] = timeout
            
            if self.parent:
                self.parent.update_status("✅ Configuración de la aplicación actualizada", "success")
            
        except ValueError:
            QMessageBox.warning(self, "Error", "El timeout debe ser un número válido")
    
    def close_sidebar(self):
        """Cierra el sidebar."""
        if self.parent:
            self.parent.toggle_config_sidebar()

class PromptTrainerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.trainer = AdvancedPromptTrainer()
        self.streaming_thread = None
        self.config_sidebar = None
        self.sidebar_visible = False
        self.init_ui()
        
    def init_ui(self):
        """Inicializa la interfaz de usuario minimalista con tema oscuro."""
        self.setWindowTitle(APP_CONFIG["window_title"])
        self.setGeometry(100, 100, APP_CONFIG["window_width"], APP_CONFIG["window_height"])
        
        # Estilo minimalista con tema oscuro
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {STYLE_CONFIG['main_background']};
                color: {STYLE_CONFIG['text_color']};
            }}
            QWidget {{
                background-color: transparent;
                color: {STYLE_CONFIG['text_color']};
            }}
            QLabel {{
                color: {STYLE_CONFIG['text_color']};
                font-weight: 500;
            }}
            QPushButton {{
                background-color: {STYLE_CONFIG['accent_color']};
                color: white;
                border: none;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
                border-radius: 8px;
                min-width: 120px;
            }}
            QPushButton:hover {{
                background-color: {STYLE_CONFIG['accent_hover']};
            }}
            QPushButton:pressed {{
                background-color: #4338ca;
            }}
            QPushButton:disabled {{
                background-color: #374151;
                color: #6b7280;
            }}
            QTextEdit, QTextBrowser {{
                background-color: {STYLE_CONFIG['card_background']};
                color: {STYLE_CONFIG['text_color']};
                border: 1px solid {STYLE_CONFIG['border_color']};
                border-radius: 8px;
                padding: 16px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
                line-height: 1.5;
            }}
            QTextEdit:focus, QTextBrowser:focus {{
                border: 2px solid {STYLE_CONFIG['accent_color']};
            }}
            QProgressBar {{
                border: 1px solid {STYLE_CONFIG['border_color']};
                border-radius: 6px;
                text-align: center;
                font-weight: 600;
                background-color: {STYLE_CONFIG['card_background']};
                color: {STYLE_CONFIG['text_color']};
                height: 8px;
            }}
            QProgressBar::chunk {{
                background-color: {STYLE_CONFIG['accent_color']};
                border-radius: 5px;
            }}
            QFrame {{
                background-color: {STYLE_CONFIG['panel_background']};
                border: 1px solid {STYLE_CONFIG['border_color']};
                border-radius: 12px;
                padding: 20px;
            }}
            QGroupBox {{
                background-color: {STYLE_CONFIG['card_background']};
                border: 1px solid {STYLE_CONFIG['border_color']};
                border-radius: 8px;
                padding: 16px;
                margin-top: 8px;
                font-weight: 600;
            }}
            QGroupBox::title {{
                color: {STYLE_CONFIG['text_color']};
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px 0 8px;
            }}
            QLineEdit {{
                background-color: {STYLE_CONFIG['card_background']};
                color: {STYLE_CONFIG['text_color']};
                border: 1px solid {STYLE_CONFIG['border_color']};
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 13px;
            }}
            QLineEdit:focus {{
                border: 2px solid {STYLE_CONFIG['accent_color']};
            }}
            QComboBox {{
                background-color: {STYLE_CONFIG['card_background']};
                color: {STYLE_CONFIG['text_color']};
                border: 1px solid {STYLE_CONFIG['border_color']};
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 13px;
                min-width: 120px;
            }}
            QComboBox:focus {{
                border: 2px solid {STYLE_CONFIG['accent_color']};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid {STYLE_CONFIG['text_color']};
            }}
            QComboBox QAbstractItemView {{
                background-color: {STYLE_CONFIG['card_background']};
                color: {STYLE_CONFIG['text_color']};
                border: 1px solid {STYLE_CONFIG['border_color']};
                border-radius: 6px;
                selection-background-color: {STYLE_CONFIG['accent_color']};
            }}
        """)
        
        # Widget central principal
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal horizontal (para incluir sidebar)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Contenedor principal de la aplicación
        app_container = QWidget()
        app_layout = QVBoxLayout(app_container)
        app_layout.setSpacing(20)
        app_layout.setContentsMargins(24, 24, 24, 24)
        
        # Header con botón de configuración
        header_frame = QFrame()
        header_layout = QHBoxLayout(header_frame)
        header_layout.setSpacing(16)
        
        # Título
        title_layout = QVBoxLayout()
        title_layout.setSpacing(4)
        
        title = QLabel("Prompt Trainer Pro")
        title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {STYLE_CONFIG['text_color']};")
        title_layout.addWidget(title)
        
        subtitle = QLabel("Sistema de Mejora de Prompts con IA")
        subtitle.setFont(QFont("Segoe UI", 14))
        subtitle.setStyleSheet(f"color: {STYLE_CONFIG['text_secondary']};")
        title_layout.addWidget(subtitle)
        
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        # Botón de configuración
        self.config_btn = QPushButton("⚙️")
        self.config_btn.setFixedSize(48, 48)
        self.config_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {STYLE_CONFIG['card_background']};
                color: {STYLE_CONFIG['text_color']};
                border: 1px solid {STYLE_CONFIG['border_color']};
                border-radius: 24px;
                font-size: 18px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {STYLE_CONFIG['accent_color']};
                color: white;
            }}
        """)
        self.config_btn.clicked.connect(self.toggle_config_sidebar)
        header_layout.addWidget(self.config_btn)
        
        app_layout.addWidget(header_frame)
        
        # Área principal
        main_area = QHBoxLayout()
        main_area.setSpacing(20)
        
        # Panel izquierdo - Entrada
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(16)
        
        # Área de entrada
        input_label = QLabel("Prompt Original")
        input_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        left_layout.addWidget(input_label)
        
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Escribe tu prompt aquí...")
        self.input_text.setMaximumHeight(150)
        left_layout.addWidget(self.input_text)
        
        # Controles
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(12)
        
        self.improve_btn = QPushButton("✨ Mejorar")
        self.improve_btn.clicked.connect(self.improve_prompt)
        controls_layout.addWidget(self.improve_btn)
        
        self.train_btn = QPushButton("🧠 Entrenar")
        self.train_btn.clicked.connect(self.start_training)
        controls_layout.addWidget(self.train_btn)
        
        left_layout.addLayout(controls_layout)
        
        # Barra de progreso
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        # Status
        self.status_label = QLabel("Listo para mejorar prompts")
        self.status_label.setStyleSheet(f"color: {STYLE_CONFIG['success_color']}; font-weight: 600; padding: 12px; background-color: {STYLE_CONFIG['card_background']}; border-radius: 6px;")
        left_layout.addWidget(self.status_label)
        
        left_layout.addStretch()
        
        # Panel derecho - Salida
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(16)
        
        output_label = QLabel("Prompt Mejorado")
        output_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        right_layout.addWidget(output_label)
        
        self.output_text = QTextBrowser()
        right_layout.addWidget(self.output_text)
        
        # Agregar paneles al área principal
        main_area.addWidget(left_panel, 1)
        main_area.addWidget(right_panel, 2)
        
        app_layout.addLayout(main_area)
        
        # Agregar contenedor principal al layout principal
        main_layout.addWidget(app_container)
        
        # Sidebar de configuración (inicialmente oculto)
        self.config_sidebar = ConfigSidebar(self)
        self.config_sidebar.hide()
        main_layout.addWidget(self.config_sidebar)
        
        # Verificar modelo entrenado
        self.check_trained_model()
    
    def toggle_config_sidebar(self):
        """Alterna la visibilidad del sidebar de configuración."""
        if not self.sidebar_visible:
            # Mostrar sidebar
            self.config_sidebar.show()
            self.sidebar_visible = True
            print("🔧 Sidebar de configuración abierto")
        else:
            # Ocultar sidebar
            self.config_sidebar.hide()
            self.sidebar_visible = False
            print("🔧 Sidebar de configuración cerrado")
    
    def update_status(self, message: str, status_type: str = "info"):
        """Actualiza el mensaje de estado."""
        self.status_label.setText(message)
        
        if status_type == "success":
            color = STYLE_CONFIG['success_color']
        elif status_type == "warning":
            color = STYLE_CONFIG['warning_color']
        elif status_type == "error":
            color = STYLE_CONFIG['error_color']
        else:
            color = STYLE_CONFIG['accent_color']
        
        self.status_label.setStyleSheet(f"color: {color}; font-weight: 600; padding: 12px; background-color: {STYLE_CONFIG['card_background']}; border-radius: 6px;")
    
    def check_trained_model(self):
        """Verifica si existe un modelo entrenado."""
        if os.path.exists("trained_prompt_model.pkl"):
            self.update_status("✅ Modelo entrenado disponible", "success")
        else:
            self.update_status("⚠️ Modelo no entrenado - Haz clic en 'Entrenar'", "warning")
    
    def start_training(self):
        """Inicia el entrenamiento del modelo."""
        self.train_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminado
        self.update_status("🔄 Entrenando modelo...", "info")
        
        # Crear y ejecutar thread de entrenamiento
        self.training_thread = TrainingThread(self.trainer)
        self.training_thread.progress.connect(self.update_status)
        self.training_thread.finished.connect(self.training_finished)
        self.training_thread.error.connect(self.training_error)
        self.training_thread.start()
    
    def training_finished(self, model_data: dict):
        """Se ejecuta cuando termina el entrenamiento."""
        self.train_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.update_status("✅ Modelo entrenado exitosamente", "success")
        
        # Actualizar información del sidebar si está visible
        if self.sidebar_visible and self.config_sidebar:
            self.config_sidebar.update_info()
        
        # Mostrar estadísticas
        stats = f"📊 Entrenamiento completado:\n"
        stats += f"• Prompts procesados: {model_data['total_prompts']}\n"
        stats += f"• Categorías analizadas: {len(model_data['categories'])}\n"
        stats += f"• Tiempo: {time.strftime('%H:%M:%S')}"
        
        QMessageBox.information(self, "✅ Entrenamiento Completado", stats)
    
    def training_error(self, error: str):
        """Se ejecuta cuando hay un error en el entrenamiento."""
        self.train_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.update_status("❌ Error en el entrenamiento", "error")
        
        QMessageBox.critical(self, "❌ Error", f"Error durante el entrenamiento:\n{error}")
    
    def improve_prompt(self):
        """Mejora el prompt ingresado."""
        original_prompt = self.input_text.toPlainText().strip()
        
        if not original_prompt:
            QMessageBox.warning(self, "Advertencia", "Por favor ingresa un prompt para mejorar")
            return
        
        # Obtener modo del sidebar o por defecto
        mode = "text"
        if self.sidebar_visible and self.config_sidebar and self.config_sidebar.mode_combo:
            mode = "code" if self.config_sidebar.mode_combo.currentText() == "Código" else "text"
        
        self.improve_btn.setEnabled(False)
        self.output_text.clear()
        self.update_status("🔄 Mejorando prompt...", "info")
        
        # Crear y ejecutar thread de streaming
        self.streaming_thread = StreamingThread(self.trainer, original_prompt, mode)
        self.streaming_thread.content_update.connect(self.update_streaming_content)
        self.streaming_thread.finished.connect(self.streaming_finished)
        self.streaming_thread.error.connect(self.streaming_error)
        self.streaming_thread.start()
    
    def update_streaming_content(self, content: str):
        """Actualiza el contenido en streaming."""
        self.output_text.setPlainText(content)
        
        # Auto-scroll al final
        cursor = self.output_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.output_text.setTextCursor(cursor)
    
    def streaming_finished(self, final_content: str):
        """Se ejecuta cuando termina el streaming."""
        self.improve_btn.setEnabled(True)
        self.update_status("✅ Prompt mejorado completado", "success")
    
    def streaming_error(self, error: str):
        """Se ejecuta cuando hay un error en el streaming."""
        self.improve_btn.setEnabled(True)
        self.update_status("❌ Error al mejorar prompt", "error")
        
        QMessageBox.critical(self, "❌ Error", f"Error al mejorar el prompt:\n{error}")

def main():
    """Función principal."""
    app = QApplication(sys.argv)
    
    # Configurar aplicación
    app.setApplicationName("Prompt Trainer Pro")
    app.setApplicationVersion("2.0")
    
    # Crear y mostrar ventana principal
    window = PromptTrainerGUI()
    window.show()
    
    # Ejecutar aplicación
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 