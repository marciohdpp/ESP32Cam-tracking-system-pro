"""
Sistema Avançado de Rastreamento ESP32-CAM com IA
Integra YOLO v8, DeepSORT, Face Recognition e técnicas modernas
Arquitetura profissional com separação de responsabilidades
"""

import cv2
import time
import threading
import tkinter as tk
from tkinter import messagebox, ttk, filedialog, simpledialog
from PIL import Image, ImageTk
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import numpy as np
from queue import Queue, Empty
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

# Configuração avançada de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tracking_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SystemConfig:
    """Configurações centralizadas do sistema"""
    # Conexão ESP32-CAM
    ESP32_STREAM_URL: str = "http://192.168.0.100:81/stream"
    esp32_url: str = "http://192.168.0.100:81/stream"
    backup_urls: List[str] = field(default_factory=lambda: [
        "http://192.168.1.100:81/stream",
        "http://192.168.0.101:81/stream"
    ])
    
    # Modelos AI
    yolo_model: str = "yolov8n.pt"
    face_recognition_model: str = "dlib"  # ou "mtcnn", "insightface"
    
    # Tracking parameters
    detection_threshold: float = 0.5
    tracking_max_age: int = 30
    tracking_max_iou: float = 0.7
    
    # Interface
    display_width: int = 800
    display_height: int = 600
    fps_limit: int = 30
    
    # Storage
    faces_path: str = "known_faces"
    output_path: str = "tracking_outputs"
    database_path: str = "tracking_data.db"
    
    # Performance
    frame_skip: int = 1  # Process every nth frame
    face_recognition_interval: int = 5  # Process faces every nth frame
    max_concurrent_faces: int = 10

class DatabaseManager:
    """Gerenciador de banco de dados para logs e estatísticas"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Inicializa o banco de dados"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS tracking_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    total_detections INTEGER,
                    unique_persons INTEGER,
                    avg_fps REAL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS face_identifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    timestamp TIMESTAMP,
                    person_name TEXT,
                    confidence REAL,
                    bbox_x INTEGER,
                    bbox_y INTEGER,
                    bbox_w INTEGER,
                    bbox_h INTEGER,
                    FOREIGN KEY (session_id) REFERENCES tracking_sessions (id)
                )
            ''')
    
    def start_session(self) -> int:
        """Inicia nova sessão de tracking"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'INSERT INTO tracking_sessions (start_time) VALUES (?)',
                (datetime.now(),)
            )
            return cursor.lastrowid
    
    def end_session(self, session_id: int, stats: Dict):
        """Finaliza sessão de tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE tracking_sessions 
                SET end_time = ?, total_detections = ?, unique_persons = ?, avg_fps = ?
                WHERE id = ?
            ''', (datetime.now(), stats.get('total_detections', 0),
                  stats.get('unique_persons', 0), stats.get('avg_fps', 0), session_id))
    
    def log_face_identification(self, session_id: int, person_name: str, 
                              confidence: float, bbox: Tuple[int, int, int, int]):
        """Registra identificação facial"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO face_identifications 
                (session_id, timestamp, person_name, confidence, bbox_x, bbox_y, bbox_w, bbox_h)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (session_id, datetime.now(), person_name, confidence, *bbox))

class AIModelManager:
    """Gerenciador centralizado dos modelos de IA"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.yolo_model = None
        self.tracker = None
        self.face_recognition_enabled = False
        self.face_encodings = {}
        self.face_names = []
        
        self._init_models()
    
    def _init_models(self):
        """Inicializa todos os modelos"""
        try:
            # YOLO v8
            logger.info("Carregando modelo YOLO...")
            self.yolo_model = YOLO(self.config.yolo_model)
            logger.info("✅ YOLO carregado com sucesso")
            
            # DeepSORT Tracker
            logger.info("Inicializando tracker DeepSORT...")
            self.tracker = DeepSort(
                max_age=self.config.tracking_max_age,
                max_iou_distance=self.config.tracking_max_iou,
                n_init=3
            )
            logger.info("✅ DeepSORT inicializado")
            
            # Face Recognition
            self._init_face_recognition()
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelos: {e}")
            raise
    
    def _init_face_recognition(self):
        """Inicializa sistema de reconhecimento facial"""
        try:
            import face_recognition
            import dlib
            
            self.face_recognition = face_recognition
            self.face_recognition_enabled = True
            self._load_known_faces()
            logger.info("✅ Reconhecimento facial habilitado")
            
        except ImportError as e:
            logger.warning(f"⚠️ Reconhecimento facial desabilitado: {e}")
            self.face_recognition_enabled = False
    
    def _load_known_faces(self):
        """Carrega faces conhecidas otimizado"""
        faces_path = Path(self.config.faces_path)
        faces_path.mkdir(exist_ok=True)
        
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
        face_files = [f for f in faces_path.iterdir() 
                     if f.suffix.lower() in supported_formats]
        
        logger.info(f"Carregando {len(face_files)} imagens de faces...")
        
        for face_file in face_files:
            try:
                image = self.face_recognition.load_image_file(str(face_file))
                encodings = self.face_recognition.face_encodings(image, model="large")
                
                if encodings:
                    name = face_file.stem
                    self.face_encodings[name] = encodings[0]
                    self.face_names.append(name)
                    logger.info(f"✅ Face carregada: {name}")
                else:
                    logger.warning(f"⚠️ Nenhuma face em: {face_file.name}")
                    
            except Exception as e:
                logger.error(f"❌ Erro ao carregar {face_file.name}: {e}")
        
        logger.info(f"Total de faces conhecidas: {len(self.face_names)}")
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detecta objetos usando YOLO"""
        results = self.yolo_model(frame, verbose=False, conf=self.config.detection_threshold)
        
        detections = []
        if results and len(results) > 0:
            for result in results[0].boxes.data.tolist():
                x1, y1, x2, y2, score, cls = result
                if int(cls) == 0:  # Person class
                    bbox = [x1, y1, x2 - x1, y2 - y1]  # Convert to [x, y, w, h]
                    detections.append({
                        'bbox': bbox,
                        'confidence': score,
                        'class': 'person',
                        'class_id': int(cls)
                    })
        
        return detections
    
    def update_tracks(self, detections: List[Dict], frame: np.ndarray):
        """Atualiza tracking com DeepSORT"""
        detection_list = [(det['bbox'], det['confidence'], det['class']) 
                         for det in detections]
        
        return self.tracker.update_tracks(detection_list, frame=frame)
    
    def identify_faces(self, frame: np.ndarray, face_locations: List) -> List[Dict]:
        """Identifica faces no frame"""
        if not self.face_recognition_enabled or not self.face_encodings:
            return []
        
        try:
            # Limitar número de faces processadas
            face_locations = face_locations[:self.config.max_concurrent_faces]
            
            face_encodings = self.face_recognition.face_encodings(
                frame, face_locations, model="large"
            )
            
            identifications = []
            known_encodings = list(self.face_encodings.values())
            known_names = list(self.face_encodings.keys())
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = self.face_recognition.compare_faces(
                    known_encodings, face_encoding, tolerance=0.4
                )
                
                name = "Desconhecido"
                confidence = 0.0
                
                face_distances = self.face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]
                
                identifications.append({
                    'name': name,
                    'confidence': confidence,
                    'location': (top, right, bottom, left),
                    'bbox': (left, top, right - left, bottom - top)
                })
            
            return identifications
            
        except Exception as e:
            logger.error(f"Erro na identificação facial: {e}")
            return []

class VideoProcessor:
    """Processador de vídeo otimizado"""
    
    def __init__(self, config: SystemConfig, ai_manager: AIModelManager, db_manager: DatabaseManager):
        self.config = config
        self.ai_manager = ai_manager
        self.db_manager = db_manager
        self.frame_count = 0
        self.session_id = None
        
        # Métricas
        self.fps_history = []
        self.detection_history = []
        self.face_history = []
        
        # Estado
        self.active_tracks = {}
        self.total_detections = 0
        self.unique_persons = set()
    
    def start_session(self):
        """Inicia nova sessão de processamento"""
        self.session_id = self.db_manager.start_session()
        self.frame_count = 0
        self.total_detections = 0
        self.unique_persons.clear()
        logger.info(f"Nova sessão iniciada: {self.session_id}")
    
    def end_session(self):
        """Finaliza sessão atual"""
        if self.session_id:
            stats = {
                'total_detections': self.total_detections,
                'unique_persons': len(self.unique_persons),
                'avg_fps': np.mean(self.fps_history) if self.fps_history else 0
            }
            self.db_manager.end_session(self.session_id, stats)
            logger.info(f"Sessão finalizada: {self.session_id}")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Processa um frame completo"""
        start_time = time.time()
        self.frame_count += 1
        
        # Skip frames para otimização
        if self.frame_count % self.config.frame_skip != 0:
            return frame, self._get_current_stats()
        
        try:
            # 1. Detecção YOLO
            detections = self.ai_manager.detect_objects(frame)
            self.total_detections += len(detections)
            
            # 2. Tracking DeepSORT
            tracks = self.ai_manager.update_tracks(detections, frame)
            
            # 3. Processar tracks ativos
            active_tracks = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                self.unique_persons.add(track_id)
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                
                active_tracks.append({
                    'id': track_id,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': getattr(track, 'confidence', 0.0)
                })
            
            # 4. Reconhecimento facial (intervalos)
            face_identifications = []
            if (self.ai_manager.face_recognition_enabled and 
                self.frame_count % self.config.face_recognition_interval == 0):
                
                # Redimensionar para acelerar processamento
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                face_locations = self.ai_manager.face_recognition.face_locations(
                    small_frame, model="hog"
                )
                
                # Escalar coordenadas de volta
                face_locations = [(top*2, right*2, bottom*2, left*2) 
                                for (top, right, bottom, left) in face_locations]
                
                face_identifications = self.ai_manager.identify_faces(frame, face_locations)
                
                # Log identificações no banco
                for identification in face_identifications:
                    if identification['confidence'] > 0.6:  # Threshold de confiança
                        self.db_manager.log_face_identification(
                            self.session_id,
                            identification['name'],
                            identification['confidence'],
                            identification['bbox']
                        )
            
            # 5. Anotar frame
            annotated_frame = self._annotate_frame(frame, active_tracks, face_identifications)
            
            # 6. Atualizar métricas
            processing_time = time.time() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0
            self.fps_history.append(fps)
            self.detection_history.append(len(active_tracks))
            self.face_history.append(len(face_identifications))
            
            # Manter histórico limitado
            max_history = 300  # 10 segundos a 30fps
            self.fps_history = self.fps_history[-max_history:]
            self.detection_history = self.detection_history[-max_history:]
            self.face_history = self.face_history[-max_history:]
            
            return annotated_frame, self._get_current_stats()
            
        except Exception as e:
            logger.error(f"Erro no processamento do frame: {e}")
            return frame, self._get_current_stats()
    
    def _annotate_frame(self, frame: np.ndarray, tracks: List[Dict], faces: List[Dict]) -> np.ndarray:
        """Anota frame com informações visuais"""
        annotated = frame.copy()
        
        # Desenhar tracks de pessoas
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            
            # Caixa de tracking
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # ID e confiança
            label = f"ID: {track['id']}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Desenhar identificações faciais
        for face in faces:
            top, right, bottom, left = face['location']
            name = face['name']
            confidence = face['confidence']
            
            # Cor baseada na confiança
            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
            
            # Caixa facial
            cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
            
            # Nome e confiança
            text = f"{name} ({confidence:.2f})"
            cv2.putText(annotated, text, (left, top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Informações do sistema
        self._draw_info_panel(annotated)
        
        return annotated
    
    def _draw_info_panel(self, frame: np.ndarray):
        """Desenha painel de informações"""
        h, w = frame.shape[:2]
        
        # Painel semi-transparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Informações
        stats = self._get_current_stats()
        info_lines = [
            f"FPS: {stats['current_fps']:.1f} (avg: {stats['avg_fps']:.1f})",
            f"Frame: {self.frame_count}",
            f"Pessoas Ativas: {stats['active_persons']}",
            f"Pessoas Únicas: {stats['unique_persons']}",
            f"Faces Identificadas: {stats['faces_detected']}",
            f"Total Detecções: {self.total_detections}"
        ]
        
        for i, line in enumerate(info_lines):
            y = 30 + (i * 18)
            cv2.putText(frame, line, (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _get_current_stats(self) -> Dict:
        """Retorna estatísticas atuais"""
        return {
            'current_fps': self.fps_history[-1] if self.fps_history else 0,
            'avg_fps': np.mean(self.fps_history) if self.fps_history else 0,
            'active_persons': self.detection_history[-1] if self.detection_history else 0,
            'unique_persons': len(self.unique_persons),
            'faces_detected': self.face_history[-1] if self.face_history else 0,
            'total_detections': self.total_detections,
            'frame_count': self.frame_count
        }

class AdvancedTrackingGUI:
    """Interface gráfica avançada do sistema"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.db_manager = DatabaseManager(config.database_path)
        self.ai_manager = AIModelManager(config)
        self.video_processor = VideoProcessor(config, self.ai_manager, self.db_manager)
        
        # Estado da aplicação
        self.is_running = False
        self.is_paused = False
        self.frame_queue = Queue(maxsize=3)
        self.current_frame = None
        self.video_thread = None
        
        # Conexão ESP32
        self.current_url_index = 0
        self.all_urls = [config.ESP32_STREAM_URL] + config.backup_urls
        
        # Inicializar interface
        self._init_gui()
        
        # Configurar paths
        Path(config.faces_path).mkdir(exist_ok=True)
        Path(config.output_path).mkdir(exist_ok=True)
    
    def _init_gui(self):
        """Inicializa interface gráfica moderna"""
        self.root = tk.Tk()
        self.root.title("🛰️ ESP32-CAM AI Tracking System Pro")
        self.root.geometry("1200x900")
        self.root.configure(bg='#2b2b2b')
        
        # Estilo moderno
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        
        self._create_main_layout()
        self._create_menu()
        
    def _create_main_layout(self):
        """Cria layout principal da interface"""
        # Frame principal dividido
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg='#2b2b2b')
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Painel esquerdo - Vídeo
        left_frame = tk.Frame(main_paned, bg='#2b2b2b')
        main_paned.add(left_frame, width=800)
        
        # Painel direito - Controles e estatísticas
        right_frame = tk.Frame(main_paned, bg='#3b3b3b', width=400)
        main_paned.add(right_frame)
        
        self._create_video_panel(left_frame)
        self._create_control_panel(right_frame)
        self._create_stats_panel(right_frame)
    
    def _create_video_panel(self, parent):
        """Cria painel de vídeo"""
        video_frame = tk.LabelFrame(parent, text="📹 Vídeo Stream", 
                                   font=('Arial', 12, 'bold'), bg='#2b2b2b', fg='white')
        video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas para vídeo com scroll se necessário
        self.video_canvas = tk.Canvas(
            video_frame,
            width=self.config.display_width,
            height=self.config.display_height,
            bg='black'
        )
        self.video_canvas.pack(pady=10)
        
        # Label para o vídeo
        self.video_label = tk.Label(
            self.video_canvas,
            text="Pressione 'Iniciar' para começar",
            font=('Arial', 14),
            bg='black',
            fg='white'
        )
        self.video_label.pack(expand=True)
    
    def _create_control_panel(self, parent):
        """Cria painel de controles"""
        control_frame = tk.LabelFrame(parent, text="🎮 Controles", 
                                     font=('Arial', 12, 'bold'), bg='#3b3b3b', fg='white')
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Botões principais
        btn_frame = tk.Frame(control_frame, bg='#3b3b3b')
        btn_frame.pack(pady=10)
        
        self.btn_start = tk.Button(
            btn_frame, text="▶️ Iniciar", width=12, height=2,
            command=self.start_tracking, 
            bg='#4CAF50', fg='white', font=('Arial', 10, 'bold')
        )
        self.btn_start.grid(row=0, column=0, padx=5, pady=2)
        
        self.btn_pause = tk.Button(
            btn_frame, text="⏸️ Pausar", width=12, height=2,
            command=self.pause_tracking, state='disabled',
            bg='#FF9800', fg='white', font=('Arial', 10, 'bold')
        )
        self.btn_pause.grid(row=0, column=1, padx=5, pady=2)
        
        self.btn_stop = tk.Button(
            btn_frame, text="⏹️ Parar", width=12, height=2,
            command=self.stop_tracking, state='disabled',
            bg='#f44336', fg='white', font=('Arial', 10, 'bold')
        )
        self.btn_stop.grid(row=1, column=0, padx=5, pady=2)
        
        self.btn_save = tk.Button(
            btn_frame, text="💾 Salvar", width=12, height=2,
            command=self.save_frame,
            bg='#2196F3', fg='white', font=('Arial', 10, 'bold')
        )
        self.btn_save.grid(row=1, column=1, padx=5, pady=2)
        
        # Configurações
        config_frame = tk.LabelFrame(control_frame, text="⚙️ Configurações", 
                                    bg='#3b3b3b', fg='white')
        config_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # URL ESP32
        tk.Label(config_frame, text="ESP32-CAM URL:", bg='#3b3b3b', fg='white').pack(anchor='w')
        self.url_var = tk.StringVar(value=self.config.ESP32_STREAM_URL)
        url_entry = tk.Entry(config_frame, textvariable=self.url_var, width=40)
        url_entry.pack(fill='x', pady=2)
        
        # Threshold de detecção
        tk.Label(config_frame, text="Threshold Detecção:", bg='#3b3b3b', fg='white').pack(anchor='w')
        self.threshold_var = tk.DoubleVar(value=self.config.detection_threshold)
        threshold_scale = tk.Scale(
            config_frame, from_=0.1, to=0.9, resolution=0.1,
            orient='horizontal', variable=self.threshold_var,
            bg='#3b3b3b', fg='white', highlightbackground='#3b3b3b'
        )
        threshold_scale.pack(fill='x', pady=2)
        
        # Botões de utilitários
        util_frame = tk.Frame(control_frame, bg='#3b3b3b')
        util_frame.pack(fill='x', pady=10)
        
        tk.Button(
            util_frame, text="📊 Estatísticas", width=15,
            command=self.show_statistics,
            bg='#9C27B0', fg='white', font=('Arial', 9, 'bold')
        ).pack(side='left', padx=5)
        
        tk.Button(
            util_frame, text="👥 Gerenciar Faces", width=15,
            command=self.manage_faces,
            bg='#607D8B', fg='white', font=('Arial', 9, 'bold')
        ).pack(side='right', padx=5)
    
    def _create_stats_panel(self, parent):
        """Cria painel de estatísticas em tempo real"""
        stats_frame = tk.LabelFrame(parent, text="📈 Estatísticas em Tempo Real", 
                                   font=('Arial', 12, 'bold'), bg='#3b3b3b', fg='white')
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Estatísticas textuais
        self.stats_text = tk.Text(
            stats_frame, height=8, width=35,
            bg='#2b2b2b', fg='#00ff00', font=('Courier', 10)
        )
        self.stats_text.pack(fill='x', padx=10, pady=10)
        
        # Gráfico de FPS (placeholder)
        self.stats_canvas = tk.Canvas(stats_frame, height=200, bg='#2b2b2b')
        self.stats_canvas.pack(fill='x', padx=10, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Sistema pronto")
        status_bar = tk.Label(
            self.root, textvariable=self.status_var,
            relief='sunken', anchor='w', bg='#2b2b2b', fg='white'
        )
        status_bar.pack(side='bottom', fill='x')
    
    def _create_menu(self):
        """Cria menu da aplicação"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Menu Arquivo
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Arquivo", menu=file_menu)
        file_menu.add_command(label="Carregar Configuração", command=self.load_config)
        file_menu.add_command(label="Salvar Configuração", command=self.save_config)
        file_menu.add_separator()
        file_menu.add_command(label="Exportar Dados", command=self.export_data)
        file_menu.add_separator()
        file_menu.add_command(label="Sair", command=self.on_closing)
        
        # Menu Ferramentas
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Ferramentas", menu=tools_menu)
        tools_menu.add_command(label="Calibrar Câmera", command=self.calibrate_camera)
        tools_menu.add_command(label="Testar Conexão", command=self.test_connection)
        tools_menu.add_command(label="Limpar Banco de Dados", command=self.clear_database)
        
        # Menu Ajuda
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Ajuda", menu=help_menu)
        help_menu.add_command(label="Sobre", command=self.show_about)
        help_menu.add_command(label="Manual", command=self.show_manual)
    
    def start_tracking(self):
        """Inicia o sistema de tracking"""
        if self.is_running:
            return
        
        try:
            # Atualizar configurações
            self.config.ESP32_STREAM_URL = self.url_var.get()
            self.config.esp32_url = self.url_var.get()
            self.config.detection_threshold = self.threshold_var.get()
            self.ai_manager.config = self.config
            
            self.is_running = True
            self.is_paused = False
            
            # Iniciar sessão de processamento
            self.video_processor.start_session()
            
            # Iniciar thread de captura
            self.video_thread = threading.Thread(target=self._video_capture_thread, daemon=True)
            self.video_thread.start()
            
            # Iniciar atualização da interface
            self._update_display()
            
            # Atualizar botões
            self._update_button_states()
            
            self.status_var.set("Sistema iniciado - Processando...")
            logger.info("🚀 Sistema de tracking iniciado")
            
        except Exception as e:
            self.status_var.set(f"Erro ao iniciar: {str(e)}")
            messagebox.showerror("Erro", f"Falha ao iniciar o sistema:\n{e}")
            logger.error(f"Erro ao iniciar tracking: {e}")
    
    def pause_tracking(self):
        """Pausa/resume o tracking"""
        if not self.is_running:
            return
        
        self.is_paused = not self.is_paused
        
        status_text = "Sistema pausado" if self.is_paused else "Sistema ativo"
        self.status_var.set(status_text)
        
        button_text = "▶️ Continuar" if self.is_paused else "⏸️ Pausar"
        self.btn_pause.config(text=button_text)
        
        logger.info(f"Sistema {'pausado' if self.is_paused else 'retomado'}")
    
    def stop_tracking(self):
        """Para o sistema de tracking"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.is_paused = False
        
        # Finalizar sessão
        self.video_processor.end_session()
        
        # Atualizar interface
        self._update_button_states()
        self.status_var.set("Sistema parado")
        
        # Limpar display
        self.video_label.config(text="Pressione 'Iniciar' para começar")
        
        logger.info("🛑 Sistema de tracking parado")
    
    def _video_capture_thread(self):
        """Thread de captura de vídeo otimizada"""
        cap = None
        retry_count = 0
        max_retries = 3
        
        while self.is_running and retry_count < max_retries:
            try:
                # Tentar conectar com URLs alternativas
                current_url = self.all_urls[self.current_url_index % len(self.all_urls)]
                logger.info(f"Tentando conectar: {current_url}")
                
                cap = cv2.VideoCapture(current_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduzir buffer
                cap.set(cv2.CAP_PROP_FPS, self.config.fps_limit)
                
                if not cap.isOpened():
                    raise ConnectionError(f"Falha ao conectar com {current_url}")
                
                logger.info(f"✅ Conectado com sucesso: {current_url}")
                retry_count = 0  # Reset contador em caso de sucesso
                
                # Loop principal de captura
                while self.is_running:
                    if self.is_paused:
                        time.sleep(0.1)
                        continue
                    
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning("Falha na captura do frame")
                        break
                    
                    # Processar frame
                    processed_frame, stats = self.video_processor.process_frame(frame)
                    
                    # Enviar para interface (não-bloqueante)
                    if not self.frame_queue.full():
                        try:
                            self.frame_queue.put_nowait((processed_frame, stats))
                        except:
                            pass  # Queue cheia, ignorar frame
                
            except Exception as e:
                retry_count += 1
                logger.error(f"Erro na captura (tentativa {retry_count}): {e}")
                
                if retry_count < max_retries:
                    # Tentar próxima URL
                    self.current_url_index += 1
                    time.sleep(2)
                else:
                    # Mostrar erro na interface
                    self.root.after(0, lambda: messagebox.showerror(
                        "Erro de Conexão", 
                        f"Não foi possível conectar após {max_retries} tentativas.\n"
                        f"Verifique as URLs: {self.all_urls}"
                    ))
                    break
            
            finally:
                if cap:
                    cap.release()
        
        logger.info("Thread de captura finalizada")
    
    def _update_display(self):
        """Atualiza display da interface"""
        try:
            if not self.frame_queue.empty():
                frame, stats = self.frame_queue.get_nowait()
                
                # Redimensionar frame para o display
                display_frame = self._resize_frame_for_display(frame)
                
                # Converter para formato Tkinter
                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Atualizar label
                self.video_label.config(image=photo, text="")
                self.video_label.image = photo  # Manter referência
                
                # Salvar frame atual
                self.current_frame = frame
                
                # Atualizar estatísticas
                self._update_stats_display(stats)
        
        except Empty:
            pass
        except Exception as e:
            logger.error(f"Erro na atualização do display: {e}")
        
        # Agendar próxima atualização
        if self.is_running:
            self.root.after(33, self._update_display)  # ~30 FPS
    
    def _resize_frame_for_display(self, frame: np.ndarray) -> np.ndarray:
        """Redimensiona frame mantendo proporção"""
        h, w = frame.shape[:2]
        
        # Calcular nova dimensão mantendo proporção
        aspect_ratio = w / h
        if aspect_ratio > self.config.display_width / self.config.display_height:
            new_w = self.config.display_width
            new_h = int(new_w / aspect_ratio)
        else:
            new_h = self.config.display_height
            new_w = int(new_h * aspect_ratio)
        
        return cv2.resize(frame, (new_w, new_h))
    
    def _update_stats_display(self, stats: Dict):
        """Atualiza display de estatísticas"""
        self.stats_text.delete(1.0, tk.END)
        
        stats_text = f"""
╔══════════════════════════╗
║      ESTATÍSTICAS        ║
╠══════════════════════════╣
║ FPS Atual: {stats['current_fps']:>6.1f}     ║
║ FPS Médio: {stats['avg_fps']:>6.1f}     ║
║ Frame: {stats['frame_count']:>10d}     ║
║ Pessoas Ativas: {stats['active_persons']:>4d}   ║
║ Pessoas Únicas: {stats['unique_persons']:>4d}   ║
║ Faces Detectadas: {stats['faces_detected']:>3d}   ║
║ Total Detecções: {stats['total_detections']:>5d}  ║
╚══════════════════════════╝
        """
        
        self.stats_text.insert(1.0, stats_text.strip())
        
        # Atualizar status bar
        status = f"FPS: {stats['current_fps']:.1f} | Pessoas: {stats['active_persons']} | Faces: {stats['faces_detected']}"
        self.status_var.set(status)
    
    def _update_button_states(self):
        """Atualiza estado dos botões baseado no status"""
        if self.is_running:
            self.btn_start.config(state='disabled')
            self.btn_pause.config(state='normal')
            self.btn_stop.config(state='normal')
        else:
            self.btn_start.config(state='normal')
            self.btn_pause.config(state='disabled')
            self.btn_stop.config(state='disabled')
    
    def save_frame(self):
        """Salva o frame atual"""
        if self.current_frame is None:
            messagebox.showwarning("Aviso", "Nenhum frame disponível para salvar")
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.output_path}/frame_{timestamp}.jpg"
            
            cv2.imwrite(filename, self.current_frame)
            messagebox.showinfo("Sucesso", f"Frame salvo como:\n{filename}")
            logger.info(f"💾 Frame salvo: {filename}")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao salvar frame:\n{e}")
            logger.error(f"Erro ao salvar frame: {e}")
    
    def show_statistics(self):
        """Mostra janela de estatísticas avançadas"""
        try:
            stats_window = tk.Toplevel(self.root)
            stats_window.title("📊 Estatísticas Avançadas")
            stats_window.geometry("800x600")
            stats_window.configure(bg='#2b2b2b')
            
            # Criar gráficos com matplotlib
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8), facecolor='#2b2b2b')
            
            # FPS ao longo do tempo
            fps_data = self.video_processor.fps_history[-100:]
            ax1.plot(fps_data, color='#00ff00', linewidth=2)
            ax1.set_title('FPS em Tempo Real', color='white')
            ax1.set_ylabel('FPS', color='white')
            ax1.set_facecolor('#3b3b3b')
            ax1.tick_params(colors='white')
            
            # Detecções ao longo do tempo
            detection_data = self.video_processor.detection_history[-100:]
            ax2.plot(detection_data, color='#ff6600', linewidth=2)
            ax2.set_title('Pessoas Detectadas', color='white')
            ax2.set_ylabel('Pessoas', color='white')
            ax2.set_facecolor('#3b3b3b')
            ax2.tick_params(colors='white')
            
            # Histograma de FPS
            ax3.hist(fps_data, bins=20, color='#00ff00', alpha=0.7, edgecolor='white')
            ax3.set_title('Distribuição de FPS', color='white')
            ax3.set_xlabel('FPS', color='white')
            ax3.set_ylabel('Frequência', color='white')
            ax3.set_facecolor('#3b3b3b')
            ax3.tick_params(colors='white')
            
            # Faces detectadas
            face_data = self.video_processor.face_history[-100:]
            ax4.plot(face_data, color='#ff00ff', linewidth=2)
            ax4.set_title('Faces Identificadas', color='white')
            ax4.set_ylabel('Faces', color='white')
            ax4.set_facecolor('#3b3b3b')
            ax4.tick_params(colors='white')
            
            plt.tight_layout()
            
            # Incorporar gráfico no Tkinter
            canvas = FigureCanvasTkAgg(fig, stats_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar estatísticas:\n{e}")
            logger.error(f"Erro nas estatísticas: {e}")
    
    def manage_faces(self):
        """Abre gerenciador de faces conhecidas"""
        faces_window = tk.Toplevel(self.root)
        faces_window.title("👥 Gerenciar Faces Conhecidas")
        faces_window.geometry("600x500")
        faces_window.configure(bg='#2b2b2b')
        
        # Lista de faces
        list_frame = tk.LabelFrame(faces_window, text="Faces Conhecidas", 
                                  bg='#2b2b2b', fg='white', font=('Arial', 12, 'bold'))
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Listbox com scrollbar
        list_container = tk.Frame(list_frame, bg='#2b2b2b')
        list_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = tk.Scrollbar(list_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.faces_listbox = tk.Listbox(
            list_container, yscrollcommand=scrollbar.set,
            bg='#3b3b3b', fg='white', font=('Arial', 11)
        )
        self.faces_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.faces_listbox.yview)
        
        # Carregar lista de faces
        self._refresh_faces_list()
        
        # Botões
        btn_frame = tk.Frame(faces_window, bg='#2b2b2b')
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(
            btn_frame, text="➕ Adicionar Face", width=15,
            command=self.add_face, bg='#4CAF50', fg='white', font=('Arial', 10, 'bold')
        ).pack(side='left', padx=5)
        
        tk.Button(
            btn_frame, text="🗑️ Remover Face", width=15,
            command=self.remove_face, bg='#f44336', fg='white', font=('Arial', 10, 'bold')
        ).pack(side='left', padx=5)
        
        tk.Button(
            btn_frame, text="🔄 Recarregar", width=15,
            command=self.reload_faces, bg='#2196F3', fg='white', font=('Arial', 10, 'bold')
        ).pack(side='right', padx=5)
    
    def _refresh_faces_list(self):
        """Atualiza lista de faces na interface"""
        self.faces_listbox.delete(0, tk.END)
        for name in self.ai_manager.face_names:
            self.faces_listbox.insert(tk.END, name)
    
    def add_face(self):
        """Adiciona nova face conhecida"""
        file_path = filedialog.askopenfilename(
            title="Selecionar Imagem da Face",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not file_path:
            return
        
        # Solicitar nome
        name = simpledialog.askstring("Nome da Pessoa", "Digite o nome da pessoa:")
        if not name:
            return
        
        try:
            # Copiar arquivo para pasta de faces
            src_path = Path(file_path)
            dst_path = Path(self.config.faces_path) / f"{name}{src_path.suffix}"
            
            import shutil
            shutil.copy2(file_path, dst_path)
            
            messagebox.showinfo("Sucesso", f"Face adicionada: {name}")
            logger.info(f"Nova face adicionada: {name}")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao adicionar face:\n{e}")
            logger.error(f"Erro ao adicionar face: {e}")
    
    def remove_face(self):
        """Remove face selecionada"""
        selection = self.faces_listbox.curselection()
        if not selection:
            messagebox.showwarning("Aviso", "Selecione uma face para remover")
            return
        
        name = self.faces_listbox.get(selection[0])
        
        if messagebox.askyesno("Confirmar", f"Remover a face '{name}'?"):
            try:
                # Encontrar e remover arquivo
                faces_path = Path(self.config.faces_path)
                for file_path in faces_path.glob(f"{name}.*"):
                    file_path.unlink()
                    break
                
                messagebox.showinfo("Sucesso", f"Face removida: {name}")
                logger.info(f"Face removida: {name}")
                
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao remover face:\n{e}")
                logger.error(f"Erro ao remover face: {e}")
    
    def reload_faces(self):
        """Recarrega faces conhecidas"""
        try:
            self.ai_manager._load_known_faces()
            self._refresh_faces_list()
            messagebox.showinfo("Sucesso", "Faces recarregadas com sucesso!")
            logger.info("Faces recarregadas")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao recarregar faces:\n{e}")
    
    def test_connection(self):
        """Testa conexão com ESP32-CAM"""
        test_window = tk.Toplevel(self.root)
        test_window.title("🔗 Teste de Conexão")
        test_window.geometry("400x300")
        test_window.configure(bg='#2b2b2b')
        
        result_text = tk.Text(
            test_window, height=15, width=50,
            bg='#3b3b3b', fg='white', font=('Courier', 10)
        )
        result_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        def run_test():
            result_text.insert(tk.END, "Iniciando teste de conexão...\n\n")
            
            for i, url in enumerate(self.all_urls):
                result_text.insert(tk.END, f"Testando URL {i+1}: {url}\n")
                result_text.update()
                
                try:
                    cap = cv2.VideoCapture(url)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            result_text.insert(tk.END, "✅ Conexão OK - Frame capturado\n\n")
                        else:
                            result_text.insert(tk.END, "⚠️ Conexão OK - Erro na captura\n\n")
                    else:
                        result_text.insert(tk.END, "❌ Falha na conexão\n\n")
                    cap.release()
                    
                except Exception as e:
                    result_text.insert(tk.END, f"❌ Erro: {e}\n\n")
                
                result_text.see(tk.END)
                result_text.update()
        
        threading.Thread(target=run_test, daemon=True).start()
    
    def load_config(self):
        """Carrega configuração de arquivo JSON"""
        file_path = filedialog.askopenfilename(
            title="Carregar Configuração",
            filetypes=[("JSON files", "*.json")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config_data = json.load(f)
                
                # Atualizar configurações
                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                
                # Atualizar interface
                self.url_var.set(self.config.ESP32_STREAM_URL)
                self.threshold_var.set(self.config.detection_threshold)
                
                messagebox.showinfo("Sucesso", "Configuração carregada!")
                
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao carregar configuração:\n{e}")
    
    def save_config(self):
        """Salva configuração atual em arquivo JSON"""
        file_path = filedialog.asksaveasfilename(
            title="Salvar Configuração",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        
        if file_path:
            try:
                config_data = {
                    'ESP32_STREAM_URL': self.config.ESP32_STREAM_URL,
                    'esp32_url': self.config.esp32_url,
                    'detection_threshold': self.config.detection_threshold,
                    'tracking_max_age': self.config.tracking_max_age,
                    'display_width': self.config.display_width,
                    'display_height': self.config.display_height,
                    'fps_limit': self.config.fps_limit
                }
                
                with open(file_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
                
                messagebox.showinfo("Sucesso", "Configuração salva!")
                
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao salvar configuração:\n{e}")
    
    def export_data(self):
        """Exporta dados do banco para CSV"""
        file_path = filedialog.asksaveasfilename(
            title="Exportar Dados",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if file_path:
            try:
                import pandas as pd
                
                # Conectar ao banco
                conn = sqlite3.connect(self.config.database_path)
                
                # Exportar sessões
                sessions_df = pd.read_sql_query(
                    "SELECT * FROM tracking_sessions", conn
                )
                
                # Exportar identificações faciais
                faces_df = pd.read_sql_query(
                    "SELECT * FROM face_identifications", conn
                )
                
                conn.close()
                
                # Salvar CSVs
                base_path = Path(file_path).stem
                sessions_df.to_csv(f"{base_path}_sessions.csv", index=False)
                faces_df.to_csv(f"{base_path}_faces.csv", index=False)
                
                messagebox.showinfo("Sucesso", f"Dados exportados:\n{base_path}_sessions.csv\n{base_path}_faces.csv")
                
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao exportar dados:\n{e}")
    
    def clear_database(self):
        """Limpa dados do banco de dados"""
        if messagebox.askyesno("Confirmar", "Limpar todos os dados do banco de dados?"):
            try:
                conn = sqlite3.connect(self.config.database_path)
                conn.execute("DELETE FROM tracking_sessions")
                conn.execute("DELETE FROM face_identifications")
                conn.commit()
                conn.close()
                
                messagebox.showinfo("Sucesso", "Banco de dados limpo!")
                
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao limpar banco:\n{e}")
    
    def calibrate_camera(self):
        """Abre utilitário de calibração da câmera"""
        messagebox.showinfo("Calibração", "Funcionalidade de calibração será implementada em versão futura")
    
    def show_about(self):
        """Mostra informações sobre o sistema"""
        about_text = """
🛰️ ESP32-CAM AI Tracking System Pro
Versão 2.0

Sistema avançado de rastreamento com:
• YOLO v8 para detecção de pessoas
• DeepSORT para tracking robusto  
• Reconhecimento facial com dlib
• Interface moderna e intuitiva
• Banco de dados SQLite
• Estatísticas em tempo real
• Exportação de dados

Desenvolvido com tecnologias de ponta em AI/ML
        """
        messagebox.showinfo("Sobre", about_text.strip())
    
    def show_manual(self):
        """Mostra manual do usuário"""
        manual_window = tk.Toplevel(self.root)
        manual_window.title("📖 Manual do Usuário")
        manual_window.geometry("700x600")
        manual_window.configure(bg='#2b2b2b')
        
        manual_text = tk.Text(
            manual_window, wrap=tk.WORD,
            bg='#3b3b3b', fg='white', font=('Arial', 11),
            padx=20, pady=20
        )
        manual_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        manual_content = """
📖 MANUAL DO USUÁRIO - ESP32-CAM AI TRACKING SYSTEM PRO

🚀 COMO USAR:

1. CONFIGURAÇÃO INICIAL
   • Configure a URL da ESP32-CAM no campo correspondente
   • Ajuste o threshold de detecção (0.1 a 0.9)
   • Adicione faces conhecidas pelo menu "Gerenciar Faces"

2. INICIANDO O TRACKING
   • Clique em "▶️ Iniciar" para começar
   • Use "⏸️ Pausar" para pausar temporariamente
   • Use "⏹️ Parar" para finalizar completamente

3. SALVANDO FRAMES
   • Clique em "💾 Salvar" para capturar o frame atual
   • Os frames são salvos na pasta "tracking_outputs"

4. GERENCIAMENTO DE FACES
   • Menu "Ferramentas" → "Gerenciar Faces"
   • Adicione fotos das pessoas conhecidas
   • O sistema reconhecerá automaticamente

5. ESTATÍSTICAS
   • Clique em "📊 Estatísticas" para ver gráficos detalhados
   • Dados são salvos automaticamente no banco SQLite

6. EXPORTAÇÃO DE DADOS
   • Menu "Arquivo" → "Exportar Dados"
   • Gera arquivos CSV com todas as detecções

🔧 CONFIGURAÇÕES AVANÇADAS:

• URLs de Backup: O sistema tenta múltiplas URLs automaticamente
• Banco de Dados: SQLite local para persistência
• Performance: Ajuste automático baseado na capacidade do sistema

⚠️ TROUBLESHOOTING:

• Conexão ESP32: Use "Ferramentas" → "Testar Conexão"
• Performance baixa: Reduza resolução ou aumente frame skip
• Reconhecimento facial: Certifique-se que face_recognition está instalado

📞 SUPORTE:
Sistema desenvolvido com tecnologias de ponta em AI/ML
        """
        
        manual_text.insert(1.0, manual_content.strip())
        manual_text.config(state='disabled')  # Somente leitura
        
        # Scrollbar para o manual
        scrollbar_manual = tk.Scrollbar(manual_window)
        scrollbar_manual.pack(side=tk.RIGHT, fill=tk.Y)
        manual_text.config(yscrollcommand=scrollbar_manual.set)
        scrollbar_manual.config(command=manual_text.yview)
    
    def on_closing(self):
        """Callback para fechamento da aplicação"""
        if self.is_running:
            if messagebox.askyesno("Fechar", "Sistema em execução. Deseja realmente sair?"):
                self.stop_tracking()
                self.root.after(1000, self.root.destroy)  # Aguardar 1s para finalizar threads
        else:
            self.root.destroy()
    
    def run(self):
        """Inicia a aplicação"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        logger.info("🖥️ Interface gráfica iniciada")
        self.root.mainloop()

def main():
    """Função principal do sistema"""
    try:
        # Configuração do sistema
        config = SystemConfig(
            ESP32_STREAM_URL="http://192.168.0.100:81/stream",
            esp32_url="http://192.168.0.100:81/stream",
            backup_urls=[
                "http://192.168.1.100:81/stream",
                "http://192.168.0.101:81/stream"
            ],
            yolo_model="yolov8n.pt",
            detection_threshold=0.5,
            display_width=800,
            display_height=600,
            fps_limit=30
        )
        
        logger.info("🚀 Iniciando ESP32-CAM AI Tracking System Pro")
        logger.info(f"Configuração: {config}")
        
        # Criar e executar aplicação
        app = AdvancedTrackingGUI(config)
        app.run()
        
    except KeyboardInterrupt:
        logger.info("Sistema interrompido pelo usuário")
    except Exception as e:
        logger.error(f"Erro crítico no sistema: {e}")
        messagebox.showerror("Erro Crítico", f"Erro no sistema:\n{e}")
    finally:
        logger.info("Sistema finalizado")

if __name__ == "__main__":
    main()
