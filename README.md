# 🚀 Prompt Trainer

Sistema avanzado de entrenamiento y mejora de prompts de IA con interfaz gráfica moderna.

## ✨ Características

- **🎯 Extracción Automática**: Lee todos los prompts de la carpeta `@/PROMPTS`
- **🧠 Análisis Inteligente**: Analiza patrones, estructura y tono de los prompts
- **⚡ Streaming en Tiempo Real**: Ve cómo se mejora tu prompt en vivo
- **🔄 Múltiples Threads**: Procesamiento paralelo para mejor rendimiento
- **⚙️ Configuración Flexible**: Cambia servidor y puerto fácilmente
- **🎨 Interfaz Moderna**: Diseño limpio y profesional

## 📋 Requisitos

- Python 3.8 o superior
- LM Studio con modelo Llama 3.2 3B
- Conexión a internet (para instalar dependencias)

## Screenshots
![2025-06-21 18_50_56-](https://github.com/user-attachments/assets/cb1211f8-9a5e-41e6-a68c-6bc291701403)
![2025-06-21 18_50_46-LM Studio](https://github.com/user-attachments/assets/d037032a-a28d-4728-bbe4-39885410d0ff)


## 🚀 Instalación Rápida

1. **Clona o descarga el proyecto**
2. **Instala dependencias**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configura LM Studio**:
   - Abre LM Studio
   - Carga el modelo "Llama 3.2 3B Instruct"
   - Activa el servidor local en puerto 1234
4. **Ejecuta el sistema**:
   ```bash
   python prompt_trainer_qt.py
   ```

## 📁 Estructura del Proyecto

```
prompt-trainer-pro/
├── prompt_trainer_qt.py    # Sistema principal
├── config.py              # Configuración del servidor
├── requirements.txt       # Dependencias
├── tutorial.txt          # Guía completa
├── README.md            # Este archivo
└── @/PROMPTS/           # Carpeta con prompts (crear manualmente)
    ├── Cursor Prompts/
    ├── Devin AI/
    ├── Dia/
    ├── Junie/
    ├── Manus Agent Tools & Prompt/
    ├── Replit/
    └── Windsurf/
```

## 🎯 Cómo Usar

### 1. Preparación
- Asegúrate de que LM Studio esté ejecutándose
- Coloca tus prompts en la carpeta `@/PROMPTS`

### 2. Entrenamiento
- Haz clic en "🧠 Entrenar Modelo"
- El sistema analizará todos los prompts automáticamente
- Espera a que termine el proceso

### 3. Mejora de Prompts
- Escribe tu prompt en el área de texto
- Haz clic en "✨ Mejorar Prompt"
- Observa cómo se mejora en tiempo real

## ⚙️ Configuración

### Cambiar Servidor/Puerto
1. Edita `config.py`:
   ```python
   LM_STUDIO_CONFIG = {
       "host": "127.0.0.1",  # Cambia aquí
       "port": 1234,         # Cambia aquí
       # ...
   }
   ```

### Desde la Interfaz
- Usa los campos "Host" y "Puerto" en la aplicación
- Haz clic en "🔄 Actualizar Configuración"

## 🔧 Solución de Problemas

### Error de Conexión
- Verifica que LM Studio esté ejecutándose
- Confirma que el puerto 1234 esté disponible
- Revisa la configuración en `config.py`

### Prompts No Encontrados
- Verifica que la carpeta `@/PROMPTS` exista
- Asegúrate de que contenga archivos `.txt` o `.json`

### Error de Dependencias
```bash
pip install -r requirements.txt
# Si persiste:
pip install PyQt6 requests numpy pathlib2
```

## 📊 Características Técnicas

- **Procesamiento Paralelo**: Hasta 6 threads simultáneos
- **Caché Inteligente**: Mejora el rendimiento
- **Streaming API**: Respuestas en tiempo real
- **Análisis de Patrones**: Detecta elementos comunes
- **Interfaz Responsiva**: Se adapta al contenido

## 🎨 Personalización

### Cambiar Colores
Edita `config.py`:
```python
STYLE_CONFIG = {
    "main_background": "#1a1a2e",
    "accent_color": "#4fc3f7",
    "success_color": "#4CAF50",
    # ...
}
```

### Ajustar Rendimiento
```python
APP_CONFIG = {
    "max_threads": 6,      # Más threads = más velocidad
    "chunk_size": 1000,    # Tamaño de procesamiento
    "timeout": 30,         # Timeout de conexión
}
```

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🆘 Soporte

- 📖 Lee `tutorial.txt` para instrucciones detalladas
- 🐛 Reporta bugs en Issues
- 💡 Sugiere features en Discussions

---

**¡Disfruta usando Prompt Trainer Pro! 🚀**
