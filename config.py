#!/usr/bin/env python3
"""
⚙️ CONFIGURACIÓN DEL SISTEMA PROMPT TRAINER
Archivo de configuración para el servidor y puerto
"""

# Configuración del servidor LM Studio
LM_STUDIO_CONFIG = {
    "host": "127.0.0.1",
    "port": 1234,
    "api_url": "http://127.0.0.1:1234/v1",
    "chat_endpoint": "http://127.0.0.1:1234/v1/chat/completions",
    "model": "llama-3.2-3b-instruct"
}

# Configuración de la aplicación
APP_CONFIG = {
    "window_title": "Prompt Trainer Pro - Sistema Avanzado",
    "window_width": 1400,
    "window_height": 900,
    "prompts_directory": "@/PROMPTS", 
    "max_threads": 6,
    "chunk_size": 1000,
    "timeout": 30
}

# Configuración de estilos (tema oscuro minimalista)
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
    """Obtiene la URL de la API."""
    return f"http://{LM_STUDIO_CONFIG['host']}:{LM_STUDIO_CONFIG['port']}/v1"

def get_chat_url():
    """Obtiene la URL del chat."""
    return f"http://{LM_STUDIO_CONFIG['host']}:{LM_STUDIO_CONFIG['port']}/v1/chat/completions"

def update_config(host=None, port=None):
    """Actualiza la configuración del servidor."""
    if host:
        LM_STUDIO_CONFIG["host"] = host
    if port:
        LM_STUDIO_CONFIG["port"] = port
    LM_STUDIO_CONFIG["api_url"] = get_api_url()
    LM_STUDIO_CONFIG["chat_endpoint"] = get_chat_url() 