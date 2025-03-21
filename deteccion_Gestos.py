import cv2
import numpy as np
import tensorflow.lite as tflite 

class DetectorManos:
    def __init__(self, modelo_path): 
        # Cargar el modelo TensorFlow Lite
        self.interpreter = tflite.Interpreter(model_path=modelo_path)
        self.interpreter.allocate_tensors()

        # Obtener detalles de entrada y salida
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Imprimir información del modelo para depuración
        print("Input details:", self.input_details)
        print("Output details:", self.output_details)

    def detectar_manos(self, frame):
        # Preprocesar la imagen para el modelo
        # Obtener la forma de entrada esperada
        input_shape = self.input_details[0]['shape']
        input_height, input_width = input_shape[1], input_shape[2]
        
        # Preprocesar la imagen
        input_data = cv2.resize(frame, (input_width, input_height))
        input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(input_data, axis=0)
        input_data = (input_data / 255.0).astype(np.float32)

        # Ejecutar la inferencia
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Obtener las salidas
        # El modelo hand_landmark_lite.tflite normalmente devuelve:
        # - landmarks: coordenadas de los puntos de la mano
        # - handedness: clasificación de la mano (izquierda/derecha)
        # - score: confianza de la detección
        
        # Extraer los landmarks - ajustar índice según la salida del modelo
        landmarks_tensor = self.interpreter.get_tensor(self.output_details[0]['index'])
        print(f"Forma de landmarks: {landmarks_tensor.shape}")
        
        # Verificar si se detectó una mano
        if len(self.output_details) > 2:
            score = self.interpreter.get_tensor(self.output_details[2]['index'])
            print(f"Score: {score}")
            if score[0][0] < 0.5:  # Umbral de confianza
                return None
        
        # Procesar los landmarks para devolverlos en el formato esperado
        # Asumiendo que landmarks_tensor tiene forma [1, 21, 3]
        if landmarks_tensor.shape[0] == 1:
            # Extraer los 21 landmarks de la primera mano detectada
            landmarks_list = []
            for i in range(landmarks_tensor.shape[1]):
                # Extraer las coordenadas x, y, z
                if landmarks_tensor.shape[2] >= 3:
                    x, y, z = landmarks_tensor[0, i, 0], landmarks_tensor[0, i, 1], landmarks_tensor[0, i, 2]
                else:
                    x, y = landmarks_tensor[0, i, 0], landmarks_tensor[0, i, 1]
                    z = 0.0
                landmarks_list.append([x, y, z])
            return landmarks_list
        
        return None

    def dibujar_landmarks(self, frame, landmarks):
        # Dibujar los landmarks en la imagen
        for landmark in landmarks:
            x, y = int(landmark[0] * frame.shape[1]), int(landmark[1] * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
        # Dibujar conexiones entre landmarks para formar la mano
        conexiones = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Pulgar
            (0, 5), (5, 6), (6, 7), (7, 8),  # Índice
            (0, 9), (9, 10), (10, 11), (11, 12),  # Medio
            (0, 13), (13, 14), (14, 15), (15, 16),  # Anular
            (0, 17), (17, 18), (18, 19), (19, 20),  # Meñique
            (5, 9), (9, 13), (13, 17)  # Conexiones entre dedos
        ]
        
        for conexion in conexiones:
            inicio = landmarks[conexion[0]]
            fin = landmarks[conexion[1]]
            punto_inicio = (int(inicio[0] * frame.shape[1]), int(inicio[1] * frame.shape[0]))
            punto_fin = (int(fin[0] * frame.shape[1]), int(fin[1] * frame.shape[0]))
            cv2.line(frame, punto_inicio, punto_fin, (0, 255, 255), 2)
            
        return frame


def identify_gesture(landmarks, lateralidad):
    # Verificar si landmarks existe y tiene la estructura correcta
    if landmarks is None or len(landmarks) < 21:
        return None
    
    try:
        # Pulgar
        base_pulgar = landmarks[1]
        proximal_pulgar = landmarks[2]
        medio_pulgar = landmarks[3]
        punta_pulgar = landmarks[4]
        # Índice
        base_indice = landmarks[5]
        proximal_indice = landmarks[6]
        medio_indice = landmarks[7]
        punta_indice = landmarks[8]
        # Corazón
        base_corazon = landmarks[9]
        proximal_corazon = landmarks[10]
        medio_corazon = landmarks[11]
        punta_corazon = landmarks[12]
        # Anular
        base_anular = landmarks[13]
        proximal_anular = landmarks[14]
        medio_anular = landmarks[15]
        punta_anular = landmarks[16]
        # Meñique
        base_menique = landmarks[17]
        proximal_menique = landmarks[18]
        medio_menique = landmarks[19]
        punta_menique = landmarks[20]

        # Verifica si los dedos están doblados hacia la palma
        si_indice_doblado = punta_indice[1] > base_indice[1] and abs(punta_indice[0] - base_indice[0]) < 0.05
        si_corazon_doblado = punta_corazon[1] > base_corazon[1] and abs(punta_corazon[0] - base_corazon[0]) < 0.05
        si_anular_doblado = punta_anular[1] > base_anular[1] and abs(punta_anular[0] - base_anular[0]) < 0.05
        si_menique_doblado = punta_menique[1] > base_menique[1] and abs(punta_menique[0] - base_menique[0]) < 0.05

        # Dedos levantados
        si_indice_levantado = punta_indice[1] < medio_indice[1]  # Índice levantado
        si_corazon_levantado = punta_corazon[1] < medio_corazon[1]  # Medio levantado
        si_anular_levantado = punta_anular[1] < medio_anular[1]  # Anular levantado
        si_menique_levantado = punta_menique[1] < medio_menique[1]  # Meñique levantado

        # Detección de la letra "A"
        pulgar_extendido_A = punta_pulgar[1] < base_indice[1]
        if lateralidad == "Right":
            pulgar_al_lado_A = punta_pulgar[0] < base_indice[0]  # Pulgar a la izquierda del índice
        elif lateralidad == "Left":
            pulgar_al_lado_A = punta_pulgar[0] > base_indice[0]  # Pulgar a la derecha del índice
        if pulgar_al_lado_A and si_indice_doblado and si_corazon_doblado and si_anular_doblado and si_menique_doblado and pulgar_extendido_A:
            return "A"

        # Detección de la letra "B"
        pulgar_escondido_indice_B = abs(punta_pulgar[1] - base_indice[1]) < 0.06 and abs(punta_pulgar[0] - base_indice[0]) < 0.04
        if pulgar_escondido_indice_B and si_indice_levantado and si_corazon_levantado and si_anular_levantado and si_menique_levantado:
            return "B"

        # Detección de la letra "C"
        estan_dedos_juntos_C = abs(punta_indice[1] - punta_corazon[1]) < 0.03 and abs(punta_corazon[1] - punta_anular[1]) < 0.03 and abs(punta_indice[0] - punta_corazon[0]) < 0.05 and abs(punta_corazon[0] - punta_anular[0]) < 0.3
        profundidad_C = estan_dedos_juntos_C and abs(proximal_indice[2] - proximal_corazon[2]) > 0.02 and abs(proximal_corazon[2] - proximal_anular[2]) > 0.02
        pulgar_gancho_c = abs(punta_pulgar[0] - punta_corazon[0]) < 0.05
        if profundidad_C and pulgar_gancho_c and abs(punta_pulgar[1] - punta_indice[1]) > 0.11:
            return "C"

        # Si no se detecta ningún gesto, devolver None
        return None
    except Exception as e:
        print(f"Error en identify_gesture: {e}")
        return None
