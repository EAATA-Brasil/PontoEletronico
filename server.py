
#!/usr/bin/env python3
"""
Servidor Flask para Sistema de Reconhecimento Facial + Ponto Eletrônico
Com suporte para armazenamento local, banco de dados e relatórios CLT
"""

from flask import Flask, render_template, request, jsonify, Response, send_file
import cv2
import numpy as np
import base64
import json
import os
from datetime import datetime, timezone, timedelta, time as dt_time
import threading
import time
from deepface import DeepFace
import logging
import csv
import sqlite3
import pandas as pd
from io import BytesIO
import pickle

# Configurar logging para registrar informações, avisos e erros
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Inicializa a aplicação Flask e define o diretório de templates
app = Flask(__name__, template_folder='templates')

class DatabaseManager:
    """Gerencia a conexão e operações com diferentes tipos de bancos de dados (SQLite, MySQL, PostgreSQL)
    para armazenar dados de faces conhecidas e registros de ponto.
    """
    
    def __init__(self):
        self.connection = None  # Objeto de conexão com o banco de dados
        self.config = None      # Configurações do banco de dados (tipo, host, etc.)
        
    def connect(self, config):
        """Estabelece a conexão com o banco de dados com base nas configurações fornecidas.
        
        Args:
            config (dict): Dicionário contendo as configurações do banco de dados.
            
        Returns:
            bool: True se a conexão for bem-sucedida, False caso contrário.
        """
        try:
            self.config = config
            db_type = config.get('type', 'sqlite')
            
            if db_type == 'sqlite':
                db_path = config.get('database', 'ponto_eletronico.db')
                # Conecta ao SQLite, permitindo acesso de threads diferentes
                self.connection = sqlite3.connect(db_path, check_same_thread=False)
                self._init_sqlite_tables()
                
            elif db_type == 'mysql':
                import mysql.connector # Importa o conector MySQL apenas se necessário
                self.connection = mysql.connector.connect(
                    host=config.get('host', 'localhost'),
                    port=config.get('port', 3306),
                    user=config.get('user', 'root'),
                    password=config.get('password', ''),
                    database=config.get('database', 'ponto_eletronico')
                )
                self._init_mysql_tables()
                
            elif db_type == 'postgresql':
                import psycopg2 # Importa o conector PostgreSQL apenas se necessário
                self.connection = psycopg2.connect(
                    host=config.get('host', 'localhost'),
                    port=config.get('port', 5432),
                    user=config.get('user', 'postgres'),
                    password=config.get('password', ''),
                    database=config.get('database', 'ponto_eletronico')
                )
                self._init_postgresql_tables()
                
            logger.info(f"Conectado ao banco de dados: {db_type}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao conectar ao banco de dados: {e}")
            return False
    
    def _init_sqlite_tables(self):
        """Inicializa as tabelas `attendance` e `known_faces` para SQLite.
        Cria as tabelas se não existirem.
        """
        cursor = self.connection.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS known_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                image_path TEXT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name)
            )
        ''')
        
        self.connection.commit()
    
    def _init_mysql_tables(self):
        """Inicializa as tabelas `attendance` e `known_faces` para MySQL.
        Cria as tabelas se não existirem.
        """
        cursor = self.connection.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                timestamp TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS known_faces (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                embedding LONGBLOB NOT NULL,
                image_path TEXT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name)
            )
        ''')
        
        self.connection.commit()
    
    def _init_postgresql_tables(self):
        """Inicializa as tabelas `attendance` e `known_faces` para PostgreSQL.
        Cria as tabelas se não existirem.
        """
        cursor = self.connection.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                timestamp TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS known_faces (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                embedding BYTEA NOT NULL,
                image_path TEXT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name)
            )
        ''')
        
        self.connection.commit()
    
    # ========== MÉTODOS PARA FACES CONHECIDAS ==========
    
    def save_face(self, name, embedding, image_path=None):
        """Salva uma face conhecida (nome, embedding e caminho da imagem) no banco de dados.
        Se a face já existir, atualiza o embedding e o caminho da imagem.
        
        Args:
            name (str): Nome da pessoa.
            embedding (list): Vetor de características faciais (embedding).
            image_path (str, optional): Caminho da imagem da face. Defaults to None.
            
        Returns:
            bool: True se a operação for bem-sucedida, False caso contrário.
        """
        try:
            cursor = self.connection.cursor()
            embedding_bytes = pickle.dumps(embedding) # Serializa o embedding para BLOB
            
            if self.config.get('type') == 'mysql':
                cursor.execute(
                    """INSERT INTO known_faces (name, embedding, image_path) \n                       VALUES (%s, %s, %s) \n                       ON DUPLICATE KEY UPDATE embedding=%s, image_path=%s""",
                    (name, embedding_bytes, image_path, embedding_bytes, image_path)
                )
            elif self.config.get('type') == 'postgresql':
                cursor.execute(
                    """INSERT INTO known_faces (name, embedding, image_path) \n                       VALUES (%s, %s, %s) \n                       ON CONFLICT (name) DO UPDATE SET embedding=%s, image_path=%s""",
                    (name, embedding_bytes, image_path, embedding_bytes, image_path)
                )
            else:  # SQLite
                cursor.execute(
                    """INSERT OR REPLACE INTO known_faces (name, embedding, image_path) \n                       VALUES (?, ?, ?)""",
                    (name, embedding_bytes, image_path)
                )
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar face no banco: {e}")
            return False
    
    def get_all_faces(self):
        """Obtém todas as faces conhecidas armazenadas no banco de dados.
        
        Returns:
            list: Uma lista de dicionários, onde cada dicionário representa uma face conhecida.
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT name, embedding, image_path, added_at FROM known_faces")
            results = cursor.fetchall()
            
            faces = []
            for row in results:
                try:
                    embedding = pickle.loads(row[1]) # Desserializa o embedding
                    faces.append({
                        'name': row[0],
                        'embedding': embedding,
                        'image_path': row[2],
                        'added_at': row[3]
                    })
                except Exception as e:
                    logger.warning(f"Erro ao carregar embedding para {row[0]}: {e}")
                    continue # Continua processando as outras faces mesmo com erro em uma
            
            return faces
            
        except Exception as e:
            logger.error(f"Erro ao obter faces do banco: {e}")
            return []
    
    def delete_face(self, name):
        """Deleta uma face do banco de dados pelo nome.
        
        Args:
            name (str): Nome da pessoa a ser deletada.
            
        Returns:
            bool: True se a face foi deletada, False caso contrário.
        """
        try:
            cursor = self.connection.cursor()
            
            if self.config.get('type') in ['mysql', 'postgresql']:
                cursor.execute("DELETE FROM known_faces WHERE name = %s", (name,))
            else:  # SQLite
                cursor.execute("DELETE FROM known_faces WHERE name = ?", (name,))
            
            self.connection.commit()
            return cursor.rowcount > 0 # Retorna True se alguma linha foi afetada (deletada)
            
        except Exception as e:
            logger.error(f"Erro ao deletar face do banco: {e}")
            return False
    
    def get_face_count(self):
        """Retorna a quantidade total de faces cadastradas no banco de dados.
        
        Returns:
            int: Número de faces cadastradas.
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM known_faces")
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Erro ao contar faces: {e}")
            return 0
    
    # ========== MÉTODOS PARA ATTENDANCE ==========
    
    def save_attendance(self, name, timestamp):
        """Salva um registro de ponto (attendance) no banco de dados.
        
        Args:
            name (str): Nome da pessoa que registrou o ponto.
            timestamp (str): Timestamp ISO 8601 do registro de ponto.
            
        Returns:
            bool: True se o registro foi salvo, False caso contrário.
        """
        try:
            cursor = self.connection.cursor()
            
            if self.config.get('type') == 'mysql':
                cursor.execute(
                    "INSERT INTO attendance (name, timestamp) VALUES (%s, %s)",
                    (name, timestamp)
                )
            elif self.config.get('type') == 'postgresql':
                cursor.execute(
                    "INSERT INTO attendance (name, timestamp) VALUES (%s, %s)",
                    (name, timestamp)
                )
            else:  # SQLite
                cursor.execute(
                    "INSERT INTO attendance (name, timestamp) VALUES (?, ?)",
                    (name, timestamp)
                )
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar attendance no banco: {e}")
            return False
    
    def get_attendance(self, limit=None):
        """Obtém registros de ponto do banco de dados.
        
        Args:
            limit (int, optional): Limita o número de registros retornados. Defaults to None.
            
        Returns:
            list: Uma lista de dicionários, onde cada dicionário representa um registro de ponto.
        """
        try:
            cursor = self.connection.cursor()
            
            query = "SELECT name, timestamp FROM attendance ORDER BY timestamp DESC"
            if limit:
                if self.config.get('type') in ['mysql', 'postgresql']:
                    query += f" LIMIT {limit}"
                else:  # SQLite
                    query += f" LIMIT {limit}"
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            return [{'name': row[0], 'timestamp': row[1]} for row in results]
            
        except Exception as e:
            logger.error(f"Erro ao obter attendance do banco: {e}")
            return []
    
    def get_attendance_by_name(self, name, limit=None):
        """Obtém registros de ponto para um funcionário específico.
        
        Args:
            name (str): Nome do funcionário.
            limit (int, optional): Limita o número de registros retornados. Defaults to None.
            
        Returns:
            list: Uma lista de dicionários com os registros de ponto do funcionário.
        """
        try:
            cursor = self.connection.cursor()
            
            query = "SELECT name, timestamp FROM attendance WHERE name = ? ORDER BY timestamp"
            if limit:
                query += f" LIMIT {limit}"
            
            if self.config.get('type') in ['mysql', 'postgresql']:
                cursor.execute(query.replace("?", "%s"), (name,))
            else:  # SQLite
                cursor.execute(query, (name,))
            
            results = cursor.fetchall()
            return [{'name': row[0], 'timestamp': row[1]} for row in results]
            
        except Exception as e:
            logger.error(f"Erro ao obter attendance por nome: {e}")
            return []
    
    def get_attendance_count(self):
        """Retorna a quantidade total de registros de ponto no banco de dados.
        
        Returns:
            int: Número de registros de ponto.
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM attendance")
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Erro ao contar registros: {e}")
            return 0
    
    def close(self):
        """Fecha a conexão com o banco de dados, se estiver aberta.
        """
        if self.connection:
            self.connection.close()

class FaceRecognitionServer:
    """Classe principal que gerencia o reconhecimento facial, registro de ponto e relatórios CLT.
    """
    
    def __init__(self):
        self.known_faces = {}  # Dicionário para armazenar embeddings de faces conhecidas
        # Carrega o classificador de faces Haar Cascade do OpenCV
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cap = None  # Objeto VideoCapture para a câmera
        self.is_running = False  # Flag para indicar se a câmera está ativa
        self.current_frame = None  # Último frame processado da câmera
        self.recognition_results = []  # Resultados do último reconhecimento facial
        self.faces_directory = "known_faces"  # Diretório para armazenar imagens de faces
        self.recognition_threshold = 0.6  # Limiar de distância para reconhecimento facial
        
        # Configurações do sistema, carregadas de um arquivo JSON
        self.confirmation_seconds = 3.0  # Tempo em segundos para confirmar um reconhecimento
        self.attendance_cooldown_seconds = 60  # Cooldown em segundos entre registros de ponto para a mesma pessoa
        self.storage_type = "local"  # Tipo de armazenamento (local ou database)
        self.database_config = {}  # Configurações do banco de dados
        
        # Configurações CLT (Consolidação das Leis do Trabalho)
        self.entrada_padrao = "09:00"  # Horário de entrada padrão
        self.saida_padrao = "18:00"    # Horário de saída padrão
        self.tempo_almoco_minutos = 60 # Tempo de almoço padrão em minutos
        
        # Nomes de arquivos para armazenamento local e relatórios
        self.attendance_file = "attendance.csv"
        self.report_file = "relatorio_ponto.csv"
        self.config_file = "system_config.json"
        
        # Gerenciadores de estado para o processo de ponto
        self.pending_confirmations = {} # Faces detectadas aguardando confirmação
        self.last_attendance = {}       # Último timestamp de ponto registrado por pessoa
        self._lock = threading.Lock()   # Lock para proteger acesso a recursos compartilhados
        self.attendance_log = []        # Log de todos os registros de ponto (em memória)
        self.db_manager = DatabaseManager() # Instância do gerenciador de banco de dados
        
        # Inicializa o sistema ao criar a instância do servidor
        self._initialize_system()
        
        logger.info(f"Sistema inicializado: Modo {self.storage_type}")
        
    def _initialize_system(self):
        """Inicializa diretórios, carrega configurações e dados iniciais do sistema.
        """
        # Cria o diretório para faces conhecidas se não existir
        if not os.path.exists(self.faces_directory):
            os.makedirs(self.faces_directory)
        
        self._load_config() # Carrega as configurações do arquivo
        
        # Tenta conectar ao banco de dados se o tipo de armazenamento for 'database'
        if self.storage_type == "database":
            if not self.db_manager.connect(self.database_config):
                logger.warning("Falha ao conectar ao banco de dados. Voltando para modo local.")
                self.storage_type = "local" # Volta para modo local em caso de falha
        
        # Inicializa arquivos locais se o tipo de armazenamento for 'local'
        if self.storage_type == "local":
            self._initialize_local_files()
        
        self.load_known_faces()      # Carrega as faces conhecidas
        self._load_attendance_history() # Carrega o histórico de registros de ponto
        
        logger.info(f"Faces carregadas: {len(self.known_faces)}")
        logger.info(f"Registros carregados: {len(self.attendance_log)}")
        
    def _load_config(self):
        """Carrega as configurações do sistema a partir do arquivo `system_config.json`.
        Define valores padrão se o arquivo não existir ou estiver incompleto.
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                self.storage_type = config.get('storage_type', 'local')
                self.confirmation_seconds = config.get('confirmation_seconds', 3.0)
                self.attendance_cooldown_seconds = config.get('attendance_cooldown_seconds', 60)
                self.database_config = config.get('database_config', {})
                self.entrada_padrao = config.get('entrada_padrao', '09:00')
                self.saida_padrao = config.get('saida_padrao', '18:00')
                self.tempo_almoco_minutos = config.get('tempo_almoco_minutos', 60)
                
                logger.info("Configurações carregadas do arquivo")
        except Exception as e:
            logger.warning(f"Erro ao carregar configurações: {e}")
    
    def save_config(self):
        """Salva as configurações atuais do sistema no arquivo `system_config.json`.
        
        Returns:
            bool: True se as configurações foram salvas, False caso contrário.
        """
        try:
            config = {
                'storage_type': self.storage_type,
                'confirmation_seconds': self.confirmation_seconds,
                'attendance_cooldown_seconds': self.attendance_cooldown_seconds,
                'database_config': self.database_config,
                'entrada_padrao': self.entrada_padrao,
                'saida_padrao': self.saida_padrao,
                'tempo_almoco_minutos': self.tempo_almoco_minutos
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                
            logger.info("Configurações salvas")
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar configurações: {e}")
            return False
    
    def _initialize_local_files(self):
        """Garante que o arquivo `attendance.csv` e o relatório CSV existam e estejam com cabeçalhos corretos.
        """
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['name', 'timestamp'])
        
        self._generate_report_file() # Gera o relatório inicial

    def _load_attendance_history(self):
        """Carrega o histórico de registros de ponto do sistema (seja de arquivo local ou banco de dados).
        Os registros são ordenados por timestamp.
        """
        self.attendance_log = []
        if self.storage_type == "local":
            if os.path.exists(self.attendance_file):
                with open(self.attendance_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None)  # Pular cabeçalho
                    for row in reader:
                        if len(row) == 2:
                            self.attendance_log.append({'name': row[0], 'timestamp': row[1]})
        else: # database
            self.attendance_log = self.db_manager.get_attendance() # Obtém todos os registros do DB
        
        # Ordenar os registros por timestamp para garantir consistência
        self.attendance_log.sort(key=lambda x: self._parse_timestamp(x['timestamp']))
        logger.info(f"Histórico de attendance carregado: {len(self.attendance_log)} registros")

    def load_known_faces(self):
        """Carrega as faces conhecidas do sistema, seja do diretório local ou do banco de dados.
        """
        logger.info("Carregando faces conhecidas...")
        
        if self.storage_type == "local":
            self._load_faces_from_directory()
        else:
            self._load_faces_from_database()
    
    def _load_faces_from_directory(self):
        """Carrega faces (embeddings) a partir de imagens no diretório local.
        """
        try:
            for filename in os.listdir(self.faces_directory):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    # Extrai o nome da pessoa do nome do arquivo (ex: 'nome_timestamp.jpg' -> 'nome')
                    name = os.path.splitext(filename)[0].rsplit('_', 1)[0]
                    image_path = os.path.join(self.faces_directory, filename)
                    self._add_face_to_memory(name, image_path)
            logger.info(f"Faces carregadas do diretório: {len(self.known_faces)}")
        except Exception as e:
            logger.error(f"Erro ao carregar faces do diretório: {e}")

    def _load_faces_from_database(self):
        """Carrega faces (embeddings) a partir do banco de dados.
        """
        try:
            faces = self.db_manager.get_all_faces()
            for face_data in faces:
                self.known_faces[face_data['name']] = {
                    'embedding': face_data['embedding'],
                    'image_path': face_data['image_path'],
                    'added_at': face_data['added_at']
                }
            logger.info(f"Faces carregadas do banco: {len(self.known_faces)}")
        except Exception as e:
            logger.error(f"Erro ao carregar faces do banco: {e}")

    def _parse_timestamp(self, timestamp_str):
        """Converte uma string de timestamp ISO 8601 para um timestamp Unix (float).
        Trata o formato ISO 8601 com ou sem 'Z' (UTC).
        
        Args:
            timestamp_str (str): String do timestamp no formato ISO 8601.
            
        Returns:
            float: Timestamp Unix correspondente.
        """
        try:
            if 'Z' in timestamp_str:
                timestamp_str = timestamp_str.replace('Z', '+00:00')
            dt = datetime.fromisoformat(timestamp_str)
            return dt.timestamp()
        except Exception as e:
            logger.error(f"Erro ao parsear timestamp {timestamp_str}: {e}")
            # Retorna o timestamp atual em caso de erro para evitar falhas
            return datetime.now(timezone.utc).timestamp()

    def _add_face_to_memory(self, name, image_path):
        """Extrai o embedding de uma imagem e o adiciona à memória do sistema e, se configurado, ao banco de dados.
        
        Args:
            name (str): Nome da pessoa.
            image_path (str): Caminho para a imagem da face.
            
        Returns:
            bool: True se a face foi adicionada com sucesso, False caso contrário.
        """
        try:
            # Usa DeepFace para extrair o embedding da face
            embedding_objs = DeepFace.represent(
                img_path=image_path, 
                model_name="VGG-Face", 
                enforce_detection=True # Garante que uma face seja detectada na imagem
            )
            if embedding_objs:
                embedding = embedding_objs[0]["embedding"]
                self.known_faces[name] = {
                    'embedding': embedding,
                    'image_path': image_path,
                    'added_at': datetime.now(timezone.utc).isoformat() # Armazena a data de adição em UTC
                }
                
                if self.storage_type == "database":
                    self.db_manager.save_face(name, embedding, image_path)
                
                logger.info(f"Face de '{name}' adicionada com sucesso.")
                return True
            else:
                logger.warning(f"Nenhuma face detectada em {image_path} para {name}.")
                return False
        except Exception as e:
            logger.error(f"Erro ao extrair embedding para {name} de {image_path}: {e}")
            return False
    
    def save_uploaded_face(self, name, image_data):
        """Salva uma imagem de face enviada via upload, extrai seu embedding e a adiciona ao sistema.
        
        Args:
            name (str): Nome da pessoa.
            image_data (str): Dados da imagem em base64.
            
        Returns:
            tuple: (bool, str) - True/False para sucesso e o caminho do arquivo salvo (ou None).
        """
        try:
            # Remove o prefixo de dados se presente (ex: 'data:image/jpeg;base64,')
            if image_data.startswith('data:image'):
                image_data = image_data.split(',',1)[1]
            image_bytes = base64.b64decode(image_data) # Decodifica a imagem base64
            
            # Gera um nome de arquivo único para a imagem
            filename = f"{name}_{int(time.time())}.jpg"
            filepath = os.path.join(self.faces_directory, filename)
            
            with open(filepath, 'wb') as f:
                f.write(image_bytes)
            
            success = self._add_face_to_memory(name, filepath)
            return success, filepath
            
        except Exception as e:
            logger.error(f"Erro ao salvar face enviada: {e}")
            return False, None
    
    def delete_face(self, name):
        """Deleta uma face do sistema, removendo-a da memória, do diretório de imagens e do banco de dados.
        
        Args:
            name (str): Nome da pessoa a ser deletada.
            
        Returns:
            bool: True se a face foi deletada, False caso contrário.
        """
        try:
            if name in self.known_faces:
                del self.known_faces[name]
            
            deleted_files = []
            # Remove todos os arquivos de imagem associados ao nome
            for filename in os.listdir(self.faces_directory):
                if filename.startswith(name + "_"):
                    filepath = os.path.join(self.faces_directory, filename)
                    try:
                        os.remove(filepath)
                        deleted_files.append(filepath)
                    except Exception as e:
                        logger.error(f"Erro ao deletar {filepath}: {e}")
            
            if self.storage_type == "database":
                self.db_manager.delete_face(name)
            
            logger.info(f"Face '{name}' deletada. Arquivos removidos: {len(deleted_files)}")
            # Retorna True se algum arquivo foi deletado ou se estava no modo banco de dados
            return len(deleted_files) > 0 or self.storage_type == "database"
            
        except Exception as e:
            logger.error(f"Erro ao deletar face: {e}")
            return False

    def recognize_face_in_frame(self, frame):
        """Detecta e reconhece faces em um frame de vídeo.
        
        Args:
            frame (numpy.array): O frame de imagem da câmera.
            
        Returns:
            tuple: (list, numpy.array) - Lista de resultados de reconhecimento e o frame processado com marcações.
        """
        results = []
        processed_frame = frame.copy()
        detected_names = set() # Nomes das faces detectadas no frame atual
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Converte para escala de cinza para detecção mais rápida
            # Detecta faces usando Haar Cascade
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w] # Região de interesse da face
                recognized_name = "Unknown"
                min_distance = None
                
                if self.known_faces: # Se houver faces cadastradas para comparação
                    try:
                        # Extrai o embedding da face detectada
                        face_embedding_objs = DeepFace.represent(
                            img_path=face_roi, 
                            model_name="VGG-Face", 
                            enforce_detection=False # Já detectamos a face, não precisa forçar novamente
                        )
                        if face_embedding_objs:
                            face_embedding = face_embedding_objs[0]["embedding"]
                            min_distance = float('inf') # Inicializa com distância infinita
                            temp_recognized_name = "Unknown"
                            
                            # Compara com todas as faces conhecidas
                            for name, face_data in self.known_faces.items():
                                known_embedding = face_data['embedding']
                                # Calcula a distância euclidiana entre os embeddings
                                distance = np.linalg.norm(np.array(face_embedding) - np.array(known_embedding))
                                if distance < min_distance: # Encontra a menor distância
                                    min_distance = distance
                                    temp_recognized_name = name
                            
                            # Se a menor distância estiver abaixo do limiar, a face é reconhecida
                            if min_distance < self.recognition_threshold:
                                recognized_name = temp_recognized_name
                                detected_names.add(recognized_name)
                    except Exception as e:
                        logger.error(f"Erro durante reconhecimento DeepFace: {e}")
                
                # Desenha um retângulo e o nome na face detectada no frame
                color = (0, 255, 0) if recognized_name != "Unknown" else (0, 0, 255) # Verde para conhecido, Vermelho para desconhecido
                cv2.rectangle(processed_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(processed_frame, recognized_name, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
                
                results.append({
                    'name': recognized_name,
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'distance': round(min_distance, 2) if min_distance is not None else None
                })
                
                # Se a face foi reconhecida, trata como candidato a registro de ponto
                if recognized_name != "Unknown":
                    self._handle_attendance_candidate(recognized_name)
            
            # Remove da lista de confirmações pendentes as faces que não estão mais no frame
            with self._lock:
                pending_names = list(self.pending_confirmations.keys())
                for name in pending_names:
                    if name not in detected_names:
                        del self.pending_confirmations[name]
                        
            return results, processed_frame
            
        except Exception as e:
            logger.error(f"Erro geral no reconhecimento de faces: {e}")
            return [], processed_frame

    def _handle_attendance_candidate(self, name):
        """Gerencia o processo de confirmação e registro de ponto para uma face reconhecida.
        Implementa cooldown para evitar múltiplos registros rápidos e um tempo de confirmação.
        
        Args:
            name (str): Nome da pessoa reconhecida.
        """
        now = time.time() # Tempo atual em segundos desde a época
        with self._lock: # Protege o acesso a `last_attendance` e `pending_confirmations`
            last_ts = self.last_attendance.get(name) # Último registro de ponto para esta pessoa
            
            # Verifica o cooldown: se o último registro foi muito recente, ignora
            if last_ts and (now - last_ts) < self.attendance_cooldown_seconds:
                remaining = int(self.attendance_cooldown_seconds - (now - last_ts))
                self.last_event = {"type": "cooldown", "name": name, "remaining": remaining}
                self.stop_camera() # Adicionado para parar a câmera após o cooldown
                return
            
            pending = self.pending_confirmations.get(name) # Verifica se a pessoa já está em processo de confirmação
            if not pending:
                # Se não estiver, inicia o processo de confirmação
                self.pending_confirmations[name] = {'first_seen': now}
            else:
                # Se já estiver, verifica se o tempo de confirmação foi atingido
                if (now - pending['first_seen']) >= self.confirmation_seconds:
                    ts_iso = datetime.now(timezone.utc).isoformat() # Timestamp atual em ISO 8601 UTC
                    self._log_attendance(name, ts_iso) # Registra o ponto
                    self.last_attendance[name] = now # Atualiza o último timestamp de ponto
                    if name in self.pending_confirmations:
                        del self.pending_confirmations[name] # Remove da lista de confirmações pendentes
                    self.last_event = {"type": "success", "name": name, "timestamp": ts_iso}
                    self.stop_camera() # Adicionado para parar a câmera após o registro de ponto

    def _log_attendance(self, name, timestamp_iso):
        """Registra um evento de ponto no sistema (arquivo local ou banco de dados).
        Atualiza o log em memória e gera o arquivo de relatório.
        
        Args:
            name (str): Nome da pessoa.
            timestamp_iso (str): Timestamp ISO 8601 do registro.
        """
        try:
            if self.storage_type == "local":
                with open(self.attendance_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([name, timestamp_iso])
            else:
                if not self.db_manager.save_attendance(name, timestamp_iso):
                    logger.error("Falha ao salvar no banco de dados")
                    return
            
            # Adiciona o registro ao log em memória e o mantém ordenado
            self.attendance_log.append({'name': name, 'timestamp': timestamp_iso})
            self.attendance_log.sort(key=lambda x: self._parse_timestamp(x['timestamp']))
            
            ts_timestamp = self._parse_timestamp(timestamp_iso)
            self.last_attendance[name] = ts_timestamp
            
            self._generate_report_file() # Regenera o arquivo de relatório
            
            logger.info(f"Ponto registrado para {name} às {timestamp_iso}")
            
        except Exception as e:
            logger.error(f"Erro ao gravar attendance: {e}")

    def _generate_report_file(self):
        """Gera/atualiza o arquivo de relatório CSV com todos os registros de ponto.
        Formata os timestamps para o fuso horário local e adiciona o dia da semana em português.
        """
        try:
            report_data = []
            
            # Garante que a attendance_log esteja atualizada antes de gerar o relatório
            self._load_attendance_history()

            for record in self.attendance_log:
                try:
                    ts_str = record['timestamp']
                    if 'Z' in ts_str:
                        ts_str = ts_str.replace('Z', '+00:00')
                    dt_utc = datetime.fromisoformat(ts_str)
                    dt_local = dt_utc.astimezone() # Converte para o fuso horário local
                    
                    # Mapeamento de dias da semana para português
                    days_translation = {
                        'Monday': 'Segunda', 'Tuesday': 'Terça', 'Wednesday': 'Quarta',
                        'Thursday': 'Quinta', 'Friday': 'Sexta', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
                    }
                    day_name_en = dt_local.strftime("%A")
                    day_name_pt = days_translation.get(day_name_en, day_name_en)
                    date_formatted = dt_local.strftime(f"%d/%m - {day_name_pt}")
                    
                    time_formatted = dt_local.strftime("%H:%M")
                    
                    report_data.append({
                        'date': date_formatted,
                        'name': record['name'],
                        'time': time_formatted,
                        'timestamp': record['timestamp']
                    })
                    
                except Exception as e:
                    logger.warning(f"Erro ao processar registro {record} para relatório CSV: {e}")
                    continue
            
            # Ordena os dados do relatório por timestamp (mais recente primeiro)
            report_data.sort(key=lambda x: x['timestamp'], reverse=True)
            
            with open(self.report_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(['Data - Dia da Semana', 'Nome', 'Horário'])
                for row in report_data:
                    writer.writerow([row['date'], row['name'], row['time']])
            
            logger.info(f"Relatório atualizado: {len(report_data)} registros")
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {e}")

    def get_attendance_data(self, limit=None):
        """Obtém dados de attendance formatados para exibição em relatórios.
        
        Args:
            limit (int, optional): Limita o número de registros retornados. Defaults to None.
            
        Returns:
            list: Uma lista de dicionários com os dados de attendance formatados.
        """
        try:
            if self.storage_type == "local":
                # Garante que os logs locais estejam atualizados e ordenados
                self._load_attendance_history() 
                records = self.attendance_log
            else:
                # Se for banco de dados, o limit já é aplicado na consulta
                records = self.db_manager.get_attendance(limit)
            
            report_data = []
            for record in records:
                try:
                    ts_str = record['timestamp']
                    if 'Z' in ts_str:
                        ts_str = ts_str.replace('Z', '+00:00')
                    dt_utc = datetime.fromisoformat(ts_str)
                    dt_local = dt_utc.astimezone() # Converte para o fuso horário local
                    
                    days_translation = {
                        'Monday': 'Segunda', 'Tuesday': 'Terça', 'Wednesday': 'Quarta',
                        'Thursday': 'Quinta', 'Friday': 'Sexta', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
                    }
                    day_name_pt = days_translation.get(dt_local.strftime("%A"), dt_local.strftime("%A"))
                    date_formatted = dt_local.strftime(f"%d/%m - {day_name_pt}")
                    
                    report_data.append({
                        'date': date_formatted,
                        'name': record['name'],
                        'time': dt_local.strftime("%H:%M"),
                        'timestamp': record['timestamp']
                    })
                except Exception as e:
                    logger.warning(f"Erro ao formatar registro para relatório: {record} - {e}")
                    continue
            
            # A ordenação já deve ter sido feita ao carregar do DB ou no _load_attendance_history
            # Se for local e o limit for aplicado aqui, deve ser feito após a ordenação
            if self.storage_type == "local" and limit:
                report_data = report_data[:limit]
                
            return report_data
            
        except Exception as e:
            logger.error(f"Erro ao obter dados para relatório: {e}")
            return []

    # ========== MÉTODOS CLT ==========

    def _calcular_diferenca_minutos(self, hora1_str, hora2_str):
        """Calcula a diferença em minutos entre duas strings de hora (HH:MM).
        Considera a passagem de meia-noite se a segunda hora for menor que a primeira.
        
        Args:
            hora1_str (str): Primeira hora no formato HH:MM.
            hora2_str (str): Segunda hora no formato HH:MM.
            
        Returns:
            int: Diferença em minutos entre as duas horas.
        """
        try:
            t1 = datetime.strptime(hora1_str, '%H:%M').time()
            t2 = datetime.strptime(hora2_str, '%H:%M').time()
            
            # Cria objetos datetime para calcular a diferença, assumindo a mesma data
            dummy_date = datetime(2000, 1, 1) # Data arbitrária para cálculo
            dt1 = datetime.combine(dummy_date, t1)
            dt2 = datetime.combine(dummy_date, t2)
            
            # Se t2 for menor que t1 (ex: 23:00 - 01:00), significa que passou para o dia seguinte
            if dt2 < dt1:
                dt2 += timedelta(days=1)
                
            diferenca = dt2 - dt1
            return int(diferenca.total_seconds() / 60)
        except Exception as e:
            logger.error(f"Erro ao calcular diferença de minutos entre {hora1_str} e {hora2_str}: {e}")
            return 0

    def _formatar_minutos(self, minutos):
        """Formata um total de minutos no formato HH:MM.
        Adiciona um sinal negativo se os minutos forem negativos.
        
        Args:
            minutos (int): Total de minutos.
            
        Returns:
            str: Horas e minutos formatados (HH:MM ou -HH:MM).
        """
        if minutos is None:
            return "00:00"
        # Garante que minutos não seja negativo para formatação, o sinal é tratado separadamente
        abs_minutos = abs(minutos)
        horas = abs_minutos // 60
        mins = abs_minutos % 60
        sinal = "-" if minutos < 0 else ""
        return f"{sinal}{horas:02d}:{mins:02d}"

    def _calcular_saldo_dia(self, registros_dia, entrada_padrao_str, saida_padrao_str, tempo_almoco_minutos):
        """Calcula o saldo de horas para um dia específico, considerando registros flexíveis.
        
        Args:
            registros_dia (list): Lista de registros de ponto para o dia, já ordenados por hora.
            entrada_padrao_str (str): Horário de entrada padrão (HH:MM).
            saida_padrao_str (str): Horário de saída padrão (HH:MM).
            tempo_almoco_minutos (int): Tempo de almoço padrão em minutos.
            
        Returns:
            dict: Contendo entrada, saída, almoço (ida/volta), horas trabalhadas, horas previstas e saldo.
        """
        
        # Inicializa valores padrão para exibição
        entrada = "--:--"
        saida = "--:--"
        almoco_ida = "--:--"
        almoco_volta = "--:--"
        almocou = "Não"
        horas_trabalhadas_minutos = 0
        
        # Extrai apenas as horas dos registros para facilitar a lógica
        horas_registradas = [r['hora'] for r in registros_dia]
        
        # Lógica para identificar os pontos principais e calcular as horas trabalhadas com base no número de registros
        if len(horas_registradas) > 0:
            entrada = horas_registradas[0] # Primeiro ponto do dia é a entrada
            saida = horas_registradas[-1]  # Último ponto do dia é a saída
            
            # Inicializa horas trabalhadas com 0
            horas_trabalhadas_minutos = 0

            # Cenário 1: 2 batidas (Entrada, Saída)
            if len(horas_registradas) == 2:
                # Não houve almoço registrado, calcula a diferença total e subtrai o tempo de almoço padrão
                horas_trabalhadas_minutos = self._calcular_diferenca_minutos(entrada, saida) - tempo_almoco_minutos
                almocou = "Não (2 batidas)"
            
            # Cenário 2: 3 batidas (Entrada, Saída Almoço, Volta Almoço/Saída)
            elif len(horas_registradas) == 3:
                almoco_ida = horas_registradas[1]
                # A volta do almoço é calculada adicionando o tempo de almoço configurado à saída para almoço
                # Isso simula o tempo de almoço e considera o terceiro ponto como o fim da jornada.
                # Para fins de exibição, almoco_volta será a hora calculada.
                almoco_volta_dt = datetime.strptime(almoco_ida, '%H:%M') + timedelta(minutes=tempo_almoco_minutos)
                almoco_volta = almoco_volta_dt.strftime('%H:%M')
                
                # Horas trabalhadas = (Saída Almoço - Entrada) + (Saída - Volta Almoço Calculada)
                manha_minutos = self._calcular_diferenca_minutos(entrada, almoco_ida)
                tarde_minutos = self._calcular_diferenca_minutos(almoco_volta, saida)
                horas_trabalhadas_minutos = manha_minutos + tarde_minutos
                almocou = "Sim (3 batidas)"

            # Cenário 3: 4 ou mais batidas (Entrada, Saída Almoço, Volta Almoço, Saída)
            elif len(horas_registradas) >= 4:
                almoco_ida = horas_registradas[1]
                almoco_volta = horas_registradas[2]
                
                # Horas trabalhadas = (Saída Almoço - Entrada) + (Saída - Volta Almoço)
                manha_minutos = self._calcular_diferenca_minutos(entrada, almoco_ida)
                tarde_minutos = self._calcular_diferenca_minutos(almoco_volta, saida)
                horas_trabalhadas_minutos = manha_minutos + tarde_minutos
                almocou = "Sim (4+ batidas)"
            
            # Garante que horas trabalhadas não seja negativo se o período for muito curto
            horas_trabalhadas_minutos = max(0, horas_trabalhadas_minutos)
        
        # Calcular horas previstas com base nos horários padrão e tempo de almoço
        # A carga horária prevista é sempre a diferença entre entrada e saída padrão, menos o tempo de almoço
        horas_previstas_minutos = self._calcular_diferenca_minutos(entrada_padrao_str, saida_padrao_str) - tempo_almoco_minutos
        horas_previstas_minutos = max(0, horas_previstas_minutos) # Garante que não seja negativo
        
        # Calcular saldo do dia
        saldo_minutos = horas_trabalhadas_minutos - horas_previstas_minutos
        
        # Calcular horas previstas com base nos horários padrão e tempo de almoço
        horas_previstas_minutos = self._calcular_diferenca_minutos(entrada_padrao_str, saida_padrao_str) - tempo_almoco_minutos
        horas_previstas_minutos = max(0, horas_previstas_minutos) # Garante que não seja negativo
        
        # Calcular saldo do dia
        saldo_minutos = horas_trabalhadas_minutos - horas_previstas_minutos
        
        return {
            'entrada': entrada,
            'almoco_ida': almoco_ida,
            'almoco_volta': almoco_volta,
            'saida': saida,
            'almocou': almocou,
            'horas_trabalhadas_minutos': horas_trabalhadas_minutos,
            'horas_previstas_minutos': horas_previstas_minutos,
            'saldo_minutos': saldo_minutos,
            'horas_trabalhadas': self._formatar_minutos(horas_trabalhadas_minutos),
            'horas_previstas': self._formatar_minutos(horas_previstas_minutos),
            'saldo': self._formatar_minutos(saldo_minutos)
        }

    def gerar_relatorio_clt(self, nome, mes, ano, tipo='mensal'):
        """Gera um relatório CLT (Consolidação das Leis do Trabalho) para um funcionário específico.
        Pode gerar relatórios mensais ou anuais.
        
        Args:
            nome (str): Nome do funcionário.
            mes (int): Mês para o relatório (1-12). Relevante para tipo 'mensal'.
            ano (int): Ano para o relatório.
            tipo (str, optional): Tipo de relatório ('mensal' ou 'anual'). Defaults to 'mensal'.
            
        Returns:
            dict: Um dicionário contendo os dados do relatório CLT formatados.
        """
        
        # Carregar todos os registros de ponto para o funcionário
        # Se for DB, busca diretamente. Se for local, filtra o log em memória.
        all_attendance = self.db_manager.get_attendance_by_name(nome) if self.storage_type == "database" else \
                         [r for r in self.attendance_log if r['name'] == nome]
        
        # Converter timestamps para objetos datetime locais e adicionar ao dicionário
        processed_attendance = []
        for record in all_attendance:
            try:
                ts_str = record['timestamp']
                if 'Z' in ts_str:
                    ts_str = ts_str.replace('Z', '+00:00')
                dt_utc = datetime.fromisoformat(ts_str)
                dt_local = dt_utc.astimezone() # Converte para o fuso horário local do servidor
                processed_attendance.append({
                    'name': record['name'],
                    'datetime': dt_local,
                    'data': dt_local.strftime("%Y-%m-%d"),
                    'hora': dt_local.strftime("%H:%M")
                })
            except Exception as e:
                logger.warning(f"Erro ao processar registro para relatório CLT: {record} - {e}")
                continue
        
        # Agrupar registros por dia para facilitar o cálculo diário
        attendance_by_day = {}
        for record in processed_attendance:
            day_key = record['data']
            if day_key not in attendance_by_day:
                attendance_by_day[day_key] = []
            attendance_by_day[day_key].append(record)
        
        # Ordenar registros dentro de cada dia por hora para garantir a sequência correta dos pontos
        for day_key in attendance_by_day:
            attendance_by_day[day_key].sort(key=lambda x: x['datetime'])
            
        if tipo == 'mensal':
            return self._gerar_relatorio_mensal(nome, mes, ano, attendance_by_day)
        elif tipo == 'anual':
            return self._gerar_relatorio_anual(nome, ano, processed_attendance)
        else:
            raise ValueError("Tipo de relatório inválido. Use 'mensal' ou 'anual'.")

    def _gerar_relatorio_mensal(self, nome, mes, ano, attendance_by_day):
        """Gera um relatório mensal detalhado para um funcionário.
        Calcula horas trabalhadas, previstas e saldo para cada dia do mês.
        
        Args:
            nome (str): Nome do funcionário.
            mes (int): Mês para o relatório (1-12).
            ano (int): Ano para o relatório.
            attendance_by_day (dict): Registros de ponto agrupados por dia.
            
        Returns:
            dict: Relatório mensal formatado com detalhes diários e totais.
        """
        
        dias_relatorio = []
        total_horas_trabalhadas_minutos = 0
        total_horas_previstas_minutos = 0
        
        # Obter configurações CLT do servidor para o cálculo
        entrada_padrao_str = self.entrada_padrao
        saida_padrao_str = self.saida_padrao
        tempo_almoco_minutos = self.tempo_almoco_minutos

        # Iterar por todos os dias do mês para garantir que dias sem ponto também apareçam
        for day in range(1, 32):
            try:
                current_date = datetime(ano, mes, day)
            except ValueError: # Dia inválido para o mês (ex: 31 de fev, 31 de abr)
                break # Sai do loop quando o dia é inválido para o mês
            
            day_key = current_date.strftime("%Y-%m-%d")
            registros_dia = attendance_by_day.get(day_key, []) # Pega registros para o dia, ou lista vazia
            
            # Mapeamento de dias da semana para português
            days_translation = {
                'Monday': 'Segunda', 'Tuesday': 'Terça', 'Wednesday': 'Quarta',
                'Thursday': 'Quinta', 'Friday': 'Sexta', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
            }
            day_name_en = current_date.strftime("%A")
            dia_semana_pt = days_translation.get(day_name_en, day_name_en)
            
            # Calcula o saldo do dia usando a função auxiliar
            saldo_dia_info = self._calcular_saldo_dia(
                registros_dia, entrada_padrao_str, saida_padrao_str, tempo_almoco_minutos
            )
            
            dias_relatorio.append({
                'data': current_date.strftime("%d/%m/%Y"),
                'dia_semana': dia_semana_pt,
                'entrada': saldo_dia_info['entrada'],
                'almoco_ida': saldo_dia_info['almoco_ida'],
                'almoco_volta': saldo_dia_info['almoco_volta'],
                'saida': saldo_dia_info['saida'],
                'almocou': saldo_dia_info['almocou'],
                'horas_trabalhadas': saldo_dia_info['horas_trabalhadas'],
                'horas_previstas': saldo_dia_info['horas_previstas'],
                'saldo': saldo_dia_info['saldo'],
                'saldo_minutos': saldo_dia_info['saldo_minutos']
            })
            
            total_horas_trabalhadas_minutos += saldo_dia_info['horas_trabalhadas_minutos']
            total_horas_previstas_minutos += saldo_dia_info['horas_previstas_minutos']
            
        saldo_final_minutos = total_horas_trabalhadas_minutos - total_horas_previstas_minutos
        
        return {
            'nome': nome,
            'mes': mes,
            'ano': ano,
            'dias': dias_relatorio,
            'total_horas_trabalhadas': self._formatar_minutos(total_horas_trabalhadas_minutos),
            'total_horas_previstas': self._formatar_minutos(total_horas_previstas_minutos),
            'saldo_final': self._formatar_minutos(saldo_final_minutos),
            'saldo_final_minutos': saldo_final_minutos,
            'configuracoes': {
                'entrada_padrao': entrada_padrao_str,
                'saida_padrao': saida_padrao_str,
                'tempo_almoco': self._formatar_minutos(tempo_almoco_minutos)
            },
            'tipo': 'mensal'
        }

    def _gerar_relatorio_anual(self, nome, ano, processed_attendance):
        """Gera um relatório anual consolidado por mês para um funcionário.
        
        Args:
            nome (str): Nome do funcionário.
            ano (int): Ano para o relatório.
            processed_attendance (list): Lista de todos os registros de ponto processados para o funcionário.
            
        Returns:
            dict: Relatório anual formatado com relatórios mensais aninhados.
        """
        meses_relatorio = {}
        
        # Agrupar registros por mês e dia para passar para o gerador mensal
        attendance_by_month_and_day = {}
        for record in processed_attendance:
            if record['datetime'].year == ano:
                month_key = record['datetime'].month
                day_key = record['data']
                if month_key not in attendance_by_month_and_day:
                    attendance_by_month_and_day[month_key] = {}
                if day_key not in attendance_by_month_and_day[month_key]:
                    attendance_by_month_and_day[month_key][day_key] = []
                attendance_by_month_and_day[month_key][day_key].append(record)

        # Itera sobre os meses que possuem registros e gera o relatório mensal para cada um
        for mes_num in sorted(attendance_by_month_and_day.keys()):
            relatorio_mes = self._gerar_relatorio_mensal(nome, mes_num, ano, attendance_by_month_and_day[mes_num])
            meses_relatorio[mes_num] = relatorio_mes
        
        return {
            'nome': nome,
            'ano': ano,
            'meses': meses_relatorio,
            'tipo': 'anual'
        }

    # ========== MÉTODOS DA CÂMERA ==========
    
    def start_camera(self):
        """Inicia a câmera e define a flag `is_running` como True.
        Retorna True se a câmera foi iniciada com sucesso, False caso contrário.
        """
        try:
            # Tenta abrir a câmera, se já estiver aberta, não faz nada e retorna True
            if self.cap and self.cap.isOpened():
                logger.info("Câmera já está ativa.")
                self.is_running = True
                return True

            self.cap = cv2.VideoCapture(0) # Tenta abrir a câmera padrão (índice 0)
            if not self.cap.isOpened():
                logger.error("Não foi possível abrir a câmera.")
                return False
            self.is_running = True
            logger.info("Câmera iniciada com sucesso.")
            return True
        except Exception as e:
            logger.error(f"Erro ao iniciar câmera: {e}")
            return False
    
    def stop_camera(self):
        """Para a câmera, libera os recursos e define a flag `is_running` como False.
        """
        self.is_running = False
        if self.cap:
            self.cap.release() # Libera o hardware da câmera
            self.cap = None # Define o objeto da câmera como None
        logger.info("Câmera parada.")
    
    def get_frame(self):
        """Obtém um frame da câmera, realiza o reconhecimento facial e retorna o frame processado.
        
        Returns:
            numpy.array: O frame da câmera com as detecções e reconhecimentos, ou None se a câmera não estiver ativa.
        """
        if not self.cap or not self.is_running:
            return None
        ret, frame = self.cap.read() # Lê um frame da câmera
        if not ret:
            logger.warning("Falha ao ler frame da câmera.")
            return None
        results, processed_frame = self.recognize_face_in_frame(frame) # Processa o frame para reconhecimento
        self.recognition_results = results # Armazena os resultados do reconhecimento
        self.current_frame = processed_frame # Armazena o frame processado
        return processed_frame
    
    def get_known_faces_list(self):
        """Retorna uma lista de faces conhecidas com seus nomes, datas de adição e caminhos de imagem.
        
        Returns:
            list: Lista de dicionários, cada um representando uma face conhecida.
        """
        faces_list = []
        for name, data in self.known_faces.items():
            faces_list.append({
                'name': name,
                'added_at': data['added_at'],
                'image_path': data['image_path']
            })
        return faces_list

# Instância global do servidor de reconhecimento facial e ponto eletrônico
face_server = FaceRecognitionServer()

# ========== ROTAS DA API FLASK ==========

@app.route('/')
def index():
    """Rota principal que renderiza a página HTML da interface do usuário.
    """
    return render_template('index.html')

@app.route('/api/start_camera', methods=['POST'])
def start_camera_api():
    """API para iniciar a câmera.
    Retorna um JSON indicando sucesso ou falha.
    """
    success = face_server.start_camera()
    return jsonify({'success': success})

@app.route('/api/stop_camera', methods=['POST'])
def stop_camera_api():
    """API para parar a câmera.
    Retorna um JSON indicando sucesso.
    """
    face_server.stop_camera()
    return jsonify({'success': True})

@app.route('/api/add_face', methods=['POST'])
def add_face_api():
    """API para adicionar uma nova face via upload de imagem.
    Recebe o nome e os dados da imagem em base64.
    """
    try:
        data = request.get_json()
        name = data.get('name')
        image_data = data.get('image')
        if not name or not image_data:
            return jsonify({'success': False, 'error': 'Nome e imagem são obrigatórios'}) # Validação de entrada
        
        success, filepath = face_server.save_uploaded_face(name, image_data)
        if success:
            return jsonify({'success': True, 'message': f'Face de {name} adicionada com sucesso'}) # Mensagem de sucesso
        else:
            return jsonify({'success': False, 'error': 'Erro ao processar imagem ou nenhuma face detectada'}) # Mensagem de erro
    except Exception as e:
        logger.error(f"Erro na API add_face: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/capture_face', methods=['POST'])
def capture_face_api():
    """API para capturar uma face diretamente do feed da câmera e adicioná-la ao sistema.
    """
    try:
        data = request.get_json()
        name = data.get('name')
        if not name:
            return jsonify({'success': False, 'error': 'Nome é obrigatório'}) # Validação de entrada
        if face_server.current_frame is None:
            return jsonify({'success': False, 'error': 'Nenhum frame disponível da câmera'}) # Verifica se há frame para capturar
        
        # Codifica o frame atual para JPEG e depois para base64
        ret, buffer = cv2.imencode(".jpg", face_server.current_frame)
        if not ret:
            return jsonify({'success': False, 'error': 'Erro ao capturar imagem da câmera'}) # Erro na codificação
        
        image_data = base64.b64encode(buffer).decode('utf-8')
        image_data = "data:image/jpeg;base64," + image_data # Adiciona prefixo para formato de dados URL
        success, filepath = face_server.save_uploaded_face(name, image_data)
        
        if success:
            return jsonify({'success': True, 'message': f'Face de {name} adicionada com sucesso pela câmera'}) # Sucesso
        else:
            return jsonify({'success': False, 'error': 'Erro ao processar imagem capturada'}) # Erro
    except Exception as e:
        logger.error(f"Erro na API capture_face: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_faces', methods=['GET'])
def get_faces_api():
    """API que retorna a lista de todas as faces conhecidas cadastradas no sistema.
    """
    faces = face_server.get_known_faces_list()
    return jsonify({'faces': faces})

@app.route('/api/delete_face', methods=['POST'])
def delete_face_api():
    """API para deletar uma face do sistema pelo nome.
    """
    try:
        data = request.get_json()
        name = data.get('name')
        if not name:
            return jsonify({'success': False, 'error': 'Nome é obrigatório'}) # Validação de entrada
        
        success = face_server.delete_face(name)
        
        if success:
            return jsonify({'success': True, 'message': f'Face de {name} removida com sucesso'}) # Sucesso
        else:
            return jsonify({'success': False, 'error': 'Erro ao remover face'}) # Erro
            
    except Exception as e:
        logger.error(f"Erro na API delete_face: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/recognition_results', methods=['GET'])
def get_recognition_results_api():
    """API que retorna os resultados do último reconhecimento facial e eventos de ponto.
    """
    response = {
        'results': face_server.recognition_results,
        'event': None
    }
    with face_server._lock: # Protege o acesso a `last_event`
        if hasattr(face_server, 'last_event') and face_server.last_event:
            response['event'] = face_server.last_event
            face_server.last_event = None # Limpa o evento após ser lido
    return jsonify(response)

@app.route('/api/get_attendance', methods=['GET'])
def get_attendance_api():
    """API que retorna os registros de ponto, opcionalmente limitados.
    """
    try:
        limit = request.args.get('limit', default=None, type=int)
        
        # Garante que a attendance_log esteja atualizada antes de retornar
        face_server._load_attendance_history()

        logs = face_server.attendance_log
            
        if limit:
            # Retorna os registros mais recentes (do final da lista ordenada)
            logs = logs[-limit:]
        
        # Inverte a ordem para mostrar os mais recentes primeiro na API (UI espera ordem decrescente)
        logs_reversed = list(reversed(logs))

        return jsonify({'success': True, 'attendance': logs_reversed})
    except Exception as e:
        logger.error(f"Erro em get_attendance: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/set_confirmation_seconds', methods=['POST'])
def set_confirmation_api():
    """API para configurar o tempo de confirmação e o cooldown entre registros de ponto.
    """
    try:
        data = request.get_json()
        sec = data.get('seconds')
        cooldown = data.get('cooldown_seconds')
        updated = {}
        with face_server._lock: # Protege o acesso às configurações
            if sec is not None:
                face_server.confirmation_seconds = float(sec)
                updated['confirmation_seconds'] = face_server.confirmation_seconds
            if cooldown is not None:
                face_server.attendance_cooldown_seconds = int(cooldown)
                updated['attendance_cooldown_seconds'] = face_server.attendance_cooldown_seconds
        
        face_server.save_config() # Salva as configurações atualizadas no arquivo
        return jsonify({'success': True, 'updated': updated})
    except Exception as e:
        logger.error(f"Erro em set_confirmation_seconds: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_config', methods=['GET'])
def get_config_api():
    """API que retorna as configurações gerais do sistema (tempo de confirmação, cooldown, threshold).
    """
    try:
        cfg = {
            'confirmation_seconds': face_server.confirmation_seconds,
            'attendance_cooldown_seconds': face_server.attendance_cooldown_seconds,
            'recognition_threshold': face_server.recognition_threshold
        }
        return jsonify({'success': True, 'config': cfg})
    except Exception as e:
        logger.error(f"Erro em get_config: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_storage_config', methods=['GET'])
def get_storage_config_api():
    """API que retorna as configurações de armazenamento (tipo e detalhes do banco de dados).
    """
    try:
        config = {
            'storage_type': face_server.storage_type,
            'confirmation_seconds': face_server.confirmation_seconds,
            'attendance_cooldown_seconds': face_server.attendance_cooldown_seconds,
            'database_config': face_server.database_config
        }
        return jsonify({'success': True, 'config': config})
    except Exception as e:
        logger.error(f"Erro em get_storage_config: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/set_storage_config', methods=['POST'])
def set_storage_config_api():
    """API para definir as configurações de armazenamento.
    Permite alternar entre armazenamento local e banco de dados, e configurar o banco.
    """
    try:
        data = request.get_json()
        
        face_server.storage_type = data.get('storage_type', 'local')
        face_server.confirmation_seconds = float(data.get('confirmation_seconds', 3.0))
        face_server.attendance_cooldown_seconds = int(data.get('attendance_cooldown_seconds', 60))
        face_server.database_config = data.get('database_config', {})
        
        # Se o tipo de armazenamento for banco de dados, tenta conectar
        if face_server.storage_type == "database":
            if not face_server.db_manager.connect(face_server.database_config):
                return jsonify({'success': False, 'error': 'Falha ao conectar ao banco de dados'}) # Erro na conexão
        
        if face_server.save_config():
            return jsonify({'success': True, 'message': 'Configurações salvas com sucesso'}) # Sucesso
        else:
            return jsonify({'success': False, 'error': 'Erro ao salvar configurações'}) # Erro ao salvar
            
    except Exception as e:
        logger.error(f"Erro em set_storage_config: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/test_database_connection', methods=['POST'])
def test_database_connection_api():
    """API para testar a conexão com um banco de dados usando as configurações fornecidas.
    """
    try:
        data = request.get_json()
        
        temp_db_manager = DatabaseManager() # Cria uma instância temporária para testar a conexão
        success = temp_db_manager.connect(data)
        temp_db_manager.close() # Fecha a conexão temporária
        
        if success:
            return jsonify({'success': True, 'message': 'Conexão bem-sucedida!'}) # Sucesso
        else:
            return jsonify({'success': False, 'error': 'Falha na conexão'}) # Falha
            
    except Exception as e:
        logger.error(f"Erro ao testar conexão com banco: {e}")
        return jsonify({'success': False, 'error': str(e)})

# ========== ROTAS CLT ==========

@app.route('/api/set_clt_config', methods=['POST'])
def set_clt_config_api():
    """API para definir as configurações CLT (horário padrão de entrada, saída, tempo de almoço).
    """
    try:
        data = request.get_json()
        
        face_server.entrada_padrao = data.get('entrada_padrao', '09:00')
        face_server.saida_padrao = data.get('saida_padrao', '18:00')
        face_server.tempo_almoco_minutos = data.get('tempo_almoco_minutos', 60)
        
        face_server.save_config() # Salva as configurações atualizadas no arquivo
        
        return jsonify({'success': True, 'message': 'Configurações CLT salvas'}) # Sucesso
        
    except Exception as e:
        logger.error(f"Erro em set_clt_config: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_clt_config', methods=['GET'])
def get_clt_config_api():
    """API que retorna as configurações CLT atuais.
    """
    try:
        config = {
            'entrada_padrao': face_server.entrada_padrao,
            'saida_padrao': face_server.saida_padrao,
            'tempo_almoco_minutos': face_server.tempo_almoco_minutos
        }
        return jsonify({'success': True, 'config': config})
    except Exception as e:
        logger.error(f"Erro em get_clt_config: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/relatorio_clt', methods=['GET'])
def relatorio_clt_api():
    """API para gerar um relatório CLT para um funcionário, mês e ano específicos.
    """
    try:
        nome = request.args.get('nome', '')
        mes = request.args.get('mes', type=int)
        ano = request.args.get('ano', type=int)
        tipo = request.args.get('tipo', 'mensal')
        
        if not nome or not mes or not ano:
            return jsonify({'success': False, 'error': 'Nome, mês e ano são obrigatórios'}) # Validação de entrada
        
        relatorio = face_server.gerar_relatorio_clt(nome, mes, ano, tipo)
        return jsonify({'success': True, 'relatorio': relatorio})
        
    except Exception as e:
        logger.error(f"Erro em relatorio_clt: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/export_relatorio_clt', methods=['GET'])
def export_relatorio_clt_api():
    """API para exportar o relatório CLT em formato Excel (XLSX).
    """
    try:
        nome = request.args.get('nome', '')
        mes = request.args.get('mes', type=int)
        ano = request.args.get('ano', type=int)
        tipo = request.args.get('tipo', 'mensal')
        
        if not nome or not mes or not ano:
            return jsonify({'success': False, 'error': 'Nome, mês e ano são obrigatórios'}) # Validação de entrada
        
        relatorio = face_server.gerar_relatorio_clt(nome, mes, ano, tipo)
        
        output = BytesIO() # Cria um buffer em memória para o arquivo Excel
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            if tipo == 'mensal':
                df = pd.DataFrame(relatorio['dias'])
                df.to_excel(writer, sheet_name=f"{mes}_{ano}", index=False)
                
                # Adiciona uma aba de resumo para o relatório mensal
                resumo_data = {
                    'Métrica': ['Horas Trabalhadas', 'Horas Previstas', 'Saldo'],
                    'Valor': [
                        relatorio['total_horas_trabalhadas'],
                        relatorio['total_horas_previstas'],
                        relatorio['saldo_final']
                    ]
                }
                pd.DataFrame(resumo_data).to_excel(writer, sheet_name='RESUMO', index=False)
            else: # Tipo anual
                # Itera sobre os meses do relatório anual e cria uma aba para cada mês
                for mes_num, dados_mes in relatorio['meses'].items():
                    df = pd.DataFrame(dados_mes['dias'])
                    df.to_excel(writer, sheet_name=f"MES_{mes_num}", index=False)
        
        output.seek(0) # Volta o ponteiro do buffer para o início
        
        filename = f"relatorio_clt_{nome}_{mes}_{ano}.xlsx"
        return send_file(
            output,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        logger.error(f"Erro ao exportar relatório CLT: {e}")
        return jsonify({'success': False, 'error': str(e)})

# ========== ROTAS DE RELATÓRIOS GERAIS ==========

@app.route('/api/download_report', methods=['GET'])
def download_report_api():
    """API para fazer download do relatório de ponto geral em formato CSV.
    """
    try:
        report_data = face_server.get_attendance_data()
        
        output = BytesIO() # Buffer em memória
        output.write('Data - Dia da Semana;Nome;Horário\\n'.encode('utf-8')) # Escreve o cabeçalho
        
        for row in report_data:
            line = f"{row['date']};{row['name']};{row['time']}\\n"
            output.write(line.encode('utf-8')) # Escreve cada linha de dados
        
        output.seek(0) # Volta o ponteiro para o início
        
        return send_file(
            output,
            as_attachment=True,
            download_name=f"relatorio_ponto_{datetime.now().strftime('%Y%m%d')}.csv",
            mimetype='text/csv'
        )
        
    except Exception as e:
        logger.error(f"Erro ao baixar relatório: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/download_excel', methods=['GET'])
def download_excel_api():
    """API para fazer download do relatório de ponto geral em formato Excel (XLSX).
    """
    try:
        report_data = face_server.get_attendance_data()
        
        df = pd.DataFrame(report_data) # Cria um DataFrame pandas a partir dos dados
        df = df[['date', 'name', 'time']] # Seleciona e reordena colunas
        df.columns = ['Data - Dia da Semana', 'Nome', 'Horário'] # Renomeia colunas
        
        output = BytesIO() # Buffer em memória
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Relatório Ponto', index=False) # Escreve o DataFrame na planilha
            
            # Ajusta a largura das colunas no Excel para melhor visualização
            worksheet = writer.sheets['Relatório Ponto']
            worksheet.column_dimensions['A'].width = 20
            worksheet.column_dimensions['B'].width = 15
            worksheet.column_dimensions['C'].width = 10
        
        output.seek(0) # Volta o ponteiro para o início
        
        return send_file(
            output,
            as_attachment=True,
            download_name=f"relatorio_ponto_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        logger.error(f"Erro ao baixar relatório Excel: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/video_feed')
def video_feed():
    """Rota para o feed de vídeo da câmera.
    Retorna um stream de frames JPEG para o navegador.
    """
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    """Função geradora que captura frames da câmera, processa-os e os codifica como JPEG.
    Usada para o streaming de vídeo no navegador.
    """
    while True:
        frame = face_server.get_frame()
        if frame is None:
            time.sleep(0.1) # Espera um pouco antes de tentar novamente se não houver frame
            continue
        
        ret, buffer = cv2.imencode(".jpg", frame) # Codifica o frame para JPEG
        if not ret:
            continue # Pula para o próximo frame se a codificação falhar
        
        frame_bytes = buffer.tobytes()
        # Retorna o frame no formato multipart/x-mixed-replace
        yield (b'--frame\n'b'Content-Type: image/jpeg\n\n' + frame_bytes + b'\n')

if __name__ == '__main__':
    # Inicia a aplicação Flask. Host 0.0.0.0 permite acesso externo.
    # debug=False para produção, threaded=True para lidar com múltiplas requisições.
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

