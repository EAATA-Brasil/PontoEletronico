import threading
import webview
from server import app

def start_flask():
    # roda Flask em thread separada
    app.run(host="127.0.0.1", port=5000, debug=False)

if __name__ == "__main__":
    # inicia Flask
    threading.Thread(target=start_flask, daemon=True).start()

    # abre a janela desktop
    webview.create_window(
        title="Sistema de Ponto Eletrônico",
        url="http://127.0.0.1:5000",
        width=1200,
        height=800,
        resizable=True
    )
    webview.start()
