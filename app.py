import webview
import threading
import time
from server import app

def run_server():
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)

if __name__ == '__main__':
    # Inicia o servidor em uma thread separada
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Aguarda o servidor iniciar
    time.sleep(2)
    
    # Cria a janela do PyWebView
    window = webview.create_window(
        'Sistema de Ponto por Reconhecimento Facial',
        'http://127.0.0.1:5000',
        width=1200,
        height=800,
        resizable=True
    )
    
    webview.start()