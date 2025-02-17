import pstats
import matplotlib.pyplot as plt
import os

# Ajusta esta ruta según donde guardaste el archivo de perfilado
profile_path = "perfomance.prof"  # Si está en el mismo directorio del script actual

if not os.path.exists(profile_path):
    print(f"❌ No se encontró el archivo en: {profile_path}")
    exit()

# Cargar los datos de perfilado
stats = pstats.Stats(profile_path)
stats.strip_dirs().sort_stats("cumulative")

# Obtener las 10 funciones más costosas
functions = []
times = []

for func, stat in stats.stats.items():
    functions.append(func[2])  # Nombre de la función
    times.append(stat[3])  # Tiempo total en segundos

# Ordenar por tiempo de ejecución
sorted_indices = sorted(range(len(times)), key=lambda i: times[i], reverse=True)[:10]

# Tomar solo las 10 funciones más lentas
top_functions = [functions[i] for i in sorted_indices]
top_times = [times[i] for i in sorted_indices]

# Crear la gráfica
plt.figure(figsize=(10, 5))
plt.barh(top_functions[::-1], top_times[::-1], color="red")
plt.xlabel("Tiempo de ejecución (segundos)")
plt.ylabel("Funciones")
plt.title("Top 10 Funciones Más Lentas")
plt.show()
