import tensorflow as tf
import pandas as pd

workingPath = "/Users/paul/Documents/Modern Portfolio Theory Data/Data/"

# Liste des devices disponibles
devices = tf.config.list_physical_devices()
print("Devices disponibles :")
for d in devices:
    print(f" - {d.device_type}: {d.name}")

# Vérifier spécifiquement le GPU Apple Metal
for device in tf.config.list_physical_devices():
    print(f" - {device.name}")
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print("\n✅ GPU Metal détecté !")
else:
    print("\n⚠️ Pas de GPU Metal détecté, TensorFlow tournera sur CPU.")

results = pd.read_pickle(workingPath + "results.pkl")
