import cv2
import numpy as np
import tensorflow.lite as tflite 
from deteccion_Gestos import DetectorManos, identify_gesture

# Ruta al modelo TensorFlow Lite
MODELO_PATH = "modelos/hand_landmark_lite.tflite"

# Inicializar el detector de manos
print("Inicializando detector de manos...")
detector = DetectorManos(MODELO_PATH)

# Configuración de DroidCam
#DROIDCAM_IP = "172.17.0.40"  # Reemplaza con la IP de tu celular
#DROIDCAM_PORT = 4747           # Puerto de DroidCam (por defecto es 4747)
#DROIDCAM_URL = f"http://{DROIDCAM_IP}:{DROIDCAM_PORT}/video"

# Iniciar la cámara remota de DroidCam
#print(f"Conectando a DroidCam en {DROIDCAM_URL}...")
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se pudo conectar a DroidCam. Verifica la IP y el puerto.")
    # Intentar con la cámara local como fallback
    print("Intentando con la cámara local...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir ninguna cámara.")
        exit()

print("Cámara inicializada correctamente.")

# Añadir un contador de frames para reducir la frecuencia de detección
frame_counter = 0
PROCESS_EVERY_N_FRAMES = 2  # Procesar cada 2 frames para mejorar rendimiento

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo recibir el frame. Verifica la conexión.")
        break
    
    # Procesar solo cada N frames para mejorar rendimiento
    frame_counter += 1
    if frame_counter % PROCESS_EVERY_N_FRAMES != 0:
        # Mostrar el frame sin procesar
        cv2.imshow("Detección de Manos y Gestos", frame)
        if cv2.waitKey(1) == ord('q'):
            break
        continue
    
    try:
        # Detectar manos en el frame
        landmarks = detector.detectar_manos(frame)
        
        # Si se detectan landmarks, procesar el gesto
        if landmarks is not None:
            # Dibujar landmarks en el frame
            frame = detector.dibujar_landmarks(frame, landmarks)
            
            # Detectar el gesto
            gesto = identify_gesture(landmarks, lateralidad="Right")
            
            # Mostrar el gesto detectado en el frame
            if gesto:
                cv2.putText(frame, f"Gesto: {gesto}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    except Exception as e:
        print(f"Error al procesar el frame: {e}")
    
    # Mostrar el frame
    cv2.imshow("Detección de Manos y Gestos", frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
