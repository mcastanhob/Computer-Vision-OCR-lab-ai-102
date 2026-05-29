import cv2
import numpy as np
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

ENDPOINT = ""
KEY = ""

print("Conectando com o motor OCR da Azure.")

#envia a chave de acesso no formato esperado pelo SDK Azure
client_azure = ImageAnalysisClient(
    endpoint=ENDPOINT
    credential=AzureKeyCredential(KEY)
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: Nao foi possivel acessar a camera.")
    exit()

print("Camera Ativada")
print("Mostre um texto/papel para a camera e pressione 'S' para escanear")
print("Pressione 'q' para sair.")

nome_janela = "Scanner de Documentos Azure"

while True:
    sucesso, frame = cap.read()
    if not sucesso:
        print("Nao foi possivel capturar a imagem da camera")

    cv2.imshow(nome_janela, frame)
    tecla = cv2.waitKey(1)& 0xFF

    if cv2.getWindowProperty(nome_janela, cv2.WND_PROP_VISIBLE)<1:
        print("Janela Fechada pelo usuario.")
        break

    if tecla ==ord("s"):
        print("Extraindo texto da imagem.")

        sucesso_encode, buffer = cv2.imencode(".jpg", frame)

    if not sucesso_encode:
        print("Erro ao converter image para JPG.")
        continue

    dados_imagem = buffer.tobytes()

    try:
        resultado = client_azure.analyze(
            image_data = dados_imagem,
            visual_features=[VisualFeatures.READ])
        if resultado.read is None:
            print("\n====Dados Extraidos====")
            for bloco in resultado.read.blocks:
                for linha in bloco.line:
                    texto_extraido = linha.text
                    print(f"LIDO: {texto_extraido}")

                    pontos = linha.bounding_polygon
                    if pontos:
                        pts = np.array(
                            [(p.x,p.y) for p in pontos],
                            np.int32
                        )

                        pts = pts.reshape((-1,1,2))
                        cv2.polylines(
                            frame, 
                            [pts],
                            isClosed=True,
                            color=(0,255,0),
                            thickness=2
                        )

                        x,y = pts[0][0]

                        posicao_y_text = max(y, -10, 20)
                        cv2.putText(
                            frame,
                            texto_extraido,
                            (x,posicao_y_text),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )
                print("======")
                cv2.imshow(nome_janela, frame)
                print("Leitura concluida, pressione qualquer tecla para continuar.")
                cv2.waitKey(0)
            else:
                print("OCR: Nenhum texto detectado.")
    except Exception as erro:
        print(f"ERRO DE API; Falha na leiture: {erro}")
    elif tecla == ord("q"):
    print("Sistema encerrado pelo usuario.")
    break
cap.release()
cv2.destroyAllWindows()
print("Scanner Finalizado")


