# 📖 Sistema de Ponto Eletrônico com Reconhecimento Facial

## 📝 Introdução
O **Sistema de Ponto Eletrônico** é uma aplicação desktop multiplataforma que utiliza **reconhecimento facial** para registro de entradas e saídas de funcionários.  
O sistema substitui o relógio de ponto convencional, oferecendo:
- Maior segurança
- Relatórios automáticos
- Configurações 
- Interface desktop nativa (via PyWebView)

---

## 🏗️ Arquitetura do Sistema
O sistema é dividido em dois principais componentes:

- **Frontend (`index.html`)**
  - Interface amigável para cadastro, relatórios e controle de câmera
  - Consome os endpoints REST do Flask
  - Permite exportação de relatórios em **CSV/Excel**

- **Backend (`server.py`)**
  - Servidor Flask que processa requisições
  - Reconhecimento facial com **DeepFace + OpenCV**
  - Armazenamento em **CSV local** ou **Banco de Dados** (SQLite/MySQL/PostgreSQL)
  - Geração de relatórios **Diário**

- **Camada Desktop (`app.py`)**
  - Lança Flask em segundo plano
  - Abre o frontend em uma janela desktop via **PyWebView**

---

## ⚙️ Instalação e Configuração

### 1. Requisitos
- Python **3.8+**
- Pip atualizado
- Câmera disponível
- Dependências:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Executar em modo desenvolvimento
```bash
python app.py
```
A aplicação abrirá em uma janela desktop automaticamente.

### 3. Configuração de Banco de Dados
- **Local (padrão)** → `attendance.csv` e `relatorio_ponto.csv` são gerados na raiz.  
- **Database (via Configurações)**:
  - Suporte a **SQLite, MySQL, PostgreSQL**
  - Configure host, porta, usuário, senha e nome do banco
  - Teste a conexão via interface gráfica

### 4. Estrutura de Arquivos
```
📂 projeto/
 ┣ 📜 index.html
 ┣ 📜 server.py
 ┣ 📜 app.py
 ┣ 📂 known_faces/
 ┣ 📜 attendance.csv
 ┣ 📜 relatorio_ponto.csv
 ┣ 📜 system_config.json
 ┗ 📂 templates/
```

---

## 👩‍💼 Uso do Sistema

### Cadastro de Funcionários
- Upload de foto ou captura da câmera  
- Faces armazenadas em `known_faces/` (modo local)  
- Reconhecimento baseado em embeddings DeepFace  

### Controle de Ponto
- Câmera reconhece automaticamente  
- Confirmação em **N segundos**  
- **Cooldown** evita registros duplicados  
- Registro salvo em CSV ou DB  

### Relatórios
- **Diário** → exportação em CSV/Excel  
- ** Mensal/Anual**:
  - Entrada/saída
  - Tempo de almoço
  - Total previsto x total trabalhado
  - Saldo positivo/negativo  

### Configurações
- Tempo de confirmação  
- Cooldown  
- Tipo de armazenamento  
- Banco de dados (opcional)  
- Entrada/Saída padrão ()  

---

## 🔌 Endpoints da API (internos)
- `GET /video_feed` → feed da câmera  
- `POST /api/add_face` → cadastro de face  
- `GET /api/get_faces` → lista de faces  
- `GET /api/attendance` → registros recentes  
- `GET /api/relatorio_clt` → relatório   

*(A interface chama esses endpoints automaticamente — útil para integrações futuras)*  

---

## 📦 Distribuição Desktop
Para gerar um executável standalone:
```bash
pip install pyinstaller
pyinstaller --onefile --noconsole app.py
```
Saída: `dist/app.exe` (Windows) ou binários equivalentes no Linux/Mac.  

---

## 🛡️ Segurança
- Embeddings faciais são salvos no banco/CSV, não as imagens originais (opcional).  
- Dados sensíveis de DB devem ser configurados via interface e `system_config.json`.  

---

## 📌 Roadmap Futuro
- 🔒 Autenticação de administrador  
- 🌐 Suporte a rede local  
- 📱 Versão mobile (via React Native)  
- ⏰ Notificações de atrasos e horas extras  
- Configurar melhor Docker
- Adicionar testes automatizados
---

## 👨‍💻 Autor
Sistema desenvolvido para uso corporativo, com foco em **simplicidade, segurança e compliance**.
